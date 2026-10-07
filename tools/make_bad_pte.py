"""Write tiny .pte programs whose seq_64 method breaks et-server's text-model input/output contract.

    python make_bad_pte.py <out_dir>

int32_input.pte (input_ids is int32), three_inputs.pte (a token_type_ids input) and
rank3_output.pte (per-token [1, 64, 4] logits). tools/test_server.sh checks that each one makes
et-server exit at startup with one "et-server:" line.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from executorch.exir import to_edge_transform_and_lower

L, LABELS = 64, 4


class Int32Input(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        return (input_ids.to(torch.float32) * attention_mask).sum(dim=1, keepdim=True).repeat(1, LABELS)


class ThreeInputs(torch.nn.Module):
    def forward(self, input_ids, attention_mask, token_type_ids):
        return (input_ids * attention_mask + token_type_ids).to(torch.float32).sum(dim=1, keepdim=True).repeat(1, LABELS)


class Rank3Output(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        return (input_ids * attention_mask).to(torch.float32).unsqueeze(-1).repeat(1, 1, LABELS)


def write(out: Path, name: str, module: torch.nn.Module, inputs: tuple) -> None:
    program = torch.export.export(module.eval(), inputs)
    et = to_edge_transform_and_lower({f"seq_{L}": program}).to_executorch()
    (out / f"{name}.pte").write_bytes(et.buffer)
    print(f"{name}.pte {len(et.buffer)} bytes")


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    ids64 = torch.ones(1, L, dtype=torch.int64)
    write(out, "int32_input", Int32Input(), (torch.ones(1, L, dtype=torch.int32), ids64))
    write(out, "three_inputs", ThreeInputs(), (ids64, ids64, torch.zeros(1, L, dtype=torch.int64)))
    write(out, "rank3_output", Rank3Output(), (ids64, ids64))


if __name__ == "__main__":
    main()
