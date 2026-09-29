"""Export a HuggingFace text classifier to ExecuTorch (.pte, XNNPACK) and validate it.

    python export_pte.py <hf_model_id> <out_dir> [--seq-lens 64,128,256,512]

One method per sequence length ("seq_<L>"), like tflite-server's export_tflite.py signatures, the
same fixtures and the same validation.json shape, plus the XNNPACK delegation coverage per method,
the tool versions, and the layout checks et-server relies on.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.runtime import Runtime
from transformers import AutoModelForSequenceClassification, AutoTokenizer

FIXTURES = [
    "My name is John Smith and my SSN is 123-45-6789.",
    "URGENT: verify your account at http://secure-login.example to avoid suspension.",
    "Thanks for the notes from today's standup, talk tomorrow.",
]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class Logits(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def encode(tok, text: str, n: int):
    enc = tok(text, truncation=True, max_length=n, padding="max_length", return_tensors="pt")
    return enc["input_ids"].to(torch.int64), enc["attention_mask"].to(torch.int64)


def coverage(edge_program) -> dict:
    """Nodes left outside XNNPACK delegate calls, per method (0 = fully delegated)."""
    out = {}
    for name in edge_program.methods:
        gm = edge_program.exported_program(name).graph_module
        delegated = sum(1 for n in gm.graph.nodes if n.op == "call_function" and "executorch_call_delegate" in str(n.target))
        leftover = [str(n.target) for n in gm.graph.nodes
                    if n.op == "call_function" and "executorch_call_delegate" not in str(n.target)
                    and "getitem" not in str(n.target)]
        out[name] = {"delegate_calls": delegated, "undelegated_ops": len(leftover),
                     "undelegated_kinds": sorted(set(leftover))[:12]}
    return out


def layout(buffer: bytes, param_bytes: int) -> dict:
    """Where the constants live in the .pte, and a check that they are stored once.

    The XNNPACK weight cache can share packed weights across methods only when the delegated
    constants are named data (one copy, looked up by key) rather than per-method inline blobs.
    """
    pte = deserialize_pte_binary(buffer)
    named = pte.named_data
    named_bytes = sum(len(b.buffer) if hasattr(b, "buffer") else len(b) for b in named.buffers) if named else 0
    constant_bytes = sum(len(c.storage) for c in pte.program.constant_buffer)
    out = {
        "pte_bytes": len(buffer),
        "fp32_param_bytes": param_bytes,
        "named_data_entries": len(named.pte_data) if named else 0,
        "named_data_bytes": named_bytes,
        "constant_buffer_bytes": constant_bytes,
    }
    if out["named_data_entries"] == 0:
        raise SystemExit("export_pte: no named data in the .pte; the XNNPACK weight cache cannot share weights")
    if len(buffer) > 1.2 * param_bytes:
        raise SystemExit(f"export_pte: model.pte is {len(buffer)} bytes, over 1.2x the {param_bytes} fp32 parameter "
                         "bytes; the constants are duplicated per method")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("out")
    ap.add_argument("--seq-lens", default="64,128,256,512")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seq_lens = sorted({int(x) for x in args.seq_lens.split(",")})

    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id).eval()
    wrapped = Logits(model).eval()

    programs = {}
    for n in seq_lens:
        ids, mask = encode(tok, FIXTURES[0], n)
        programs[f"seq_{n}"] = torch.export.export(wrapped, (ids, mask))
    edge = to_edge_transform_and_lower(programs, partitioner=[XnnpackPartitioner()])
    cov = coverage(edge)
    et = edge.to_executorch()
    param_bytes = sum(p.numel() * 4 for p in model.parameters())
    lay = layout(et.buffer, param_bytes)
    (out / "model.pte").write_bytes(et.buffer)

    tok.save_pretrained(out)
    model.config.save_pretrained(out)
    (out / "manifest.json").write_text(json.dumps({
        "task": "text-classification",
        "id2label": {str(k): v for k, v in model.config.id2label.items()},
        "score_normalization": "softmax",
        "token_aggregation": None,
        "max_length": min(seq_lens[-1], int(getattr(tok, "model_max_length", seq_lens[-1]) or seq_lens[-1])),
        "max_sequence_length": seq_lens[-1],
    }, indent=2) + "\n")

    program = Runtime.get().load_program(str(out / "model.pte"))
    max_delta, per = 0.0, []
    for n in seq_lens:
        method = program.load_method(f"seq_{n}")
        for text in FIXTURES:
            ids, mask = encode(tok, text, n)
            with torch.no_grad():
                ref = softmax(wrapped(ids, mask).numpy())[0]
            got = softmax(method.execute([ids, mask])[0].numpy())[0]
            delta = float(np.abs(ref - got).max())
            max_delta = max(max_delta, delta)
            per.append({"signature": f"seq_{n}", "text": text, "max_delta": delta})
    (out / "validation.json").write_text(json.dumps({
        "compared_against": args.model_id,
        "engine": "ExecuTorch runtime (XNNPACK delegate)",
        "versions": {k: version(k) for k in ("executorch", "torch", "transformers")},
        "layout": lay,
        "signatures": [f"seq_{n}" for n in seq_lens],
        "fixtures": len(per),
        "max_score_delta": max_delta,
        "delegation": cov,
        "per_fixture": per,
    }, indent=2) + "\n")
    print(f"model.pte {(out / 'model.pte').stat().st_size} bytes; max_score_delta={max_delta:.2e}")
    print(f"  named data {lay['named_data_entries']} entries / {lay['named_data_bytes']} bytes, "
          f"constant buffer {lay['constant_buffer_bytes']} bytes, fp32 params {param_bytes} bytes")
    for name, c in cov.items():
        print(f"  {name}: {c['delegate_calls']} XNNPACK delegate call(s), {c['undelegated_ops']} op(s) outside XNNPACK {c['undelegated_kinds'][:6]}")
    if max_delta > 1e-6:
        raise SystemExit(f"export_pte: max_score_delta {max_delta:.2e} against PyTorch is over the 1e-6 gate")


if __name__ == "__main__":
    main()
