# et-server

An ExecuTorch text-classification server for [Lemonade](https://github.com/lemonade-sdk/lemonade).
It is [tflite-server](https://github.com/ianbmacdonald/tflite-server), and before that
[ort-server](https://github.com/lemonade-sdk/ort-server), with the inference engine swapped: the same
`/classify` contract, manifest rules, tokenizer handling and model-family allowlist, with an ExecuTorch
`Module` (XNNPACK delegate, optimized CPU kernels) in place of the LiteRT or ONNX Runtime session. A
client, or Lemonade's `/v1/classify`, gets the same request and response shape from any of the three.

It targets small hosts. It links ExecuTorch's static libraries and runs musl-native, for example on a
prplOS gateway, with no glibc bundle.

> Status: **experimental.**

## Model directory

```
model-dir/
  model.pte              # methods seq_<L>: input_ids, attention_mask as int64 [1, L]; output 0 float32 [1, labels]
  tokenizer.json         # the model's HuggingFace tokenizer
  config.json            # stock HF config (model_type is checked against the allowlist)
  tokenizer_config.json  # model_max_length
  manifest.json          # explicit contract, same schema as ort-server (id2label, max_length, ...)
  validation.json        # written by the exporter; not read by the server
```

Each method is the same graph at one fixed sequence length. The exporter writes `seq_64`, `seq_128`,
`seq_256` and `seq_512`. A request is tokenized, truncated to `min(max_length, longest loaded L)`
keeping the trailing `[SEP]`, and run on the smallest loaded method that holds it, padded with an
attention mask of zero.

v0.1.0 serves text classification only, and checks every method at startup: exactly two int64
`[1, L]` inputs in the order `input_ids`, `attention_mask`, and a float32 `[1, labels]` first output
whose width matches `id2label`. Anything else stops the server with one `et-server:` line on stderr.

`tools/export_pte.py <hf_model_id> <out_dir> [--seq-lens 64,128,256,512]` produces such a directory
(run it in a venv with `executorch` 1.5.1). It uses the same fixtures as the lemonade-sdk ONNX exports,
fails unless every method matches PyTorch within 1e-6, and writes a `validation.json` with the score
difference per method and fixture, the tool versions, the XNNPACK delegation coverage and the
constant layout. It also fails if the constants are not stored once, as named data: that is what
lets the XNNPACK weight cache share one packed copy across all methods.

## Run

```bash
et-server --model-path <model-dir> --port <n> [--threads N] [--seq-lens 64,512] [--weight-cache FILE] [--verbose]
```

- `--threads N`: XNNPACK and kernel threads, 1 to 1024 (default: one per hardware thread).
- `--seq-lens 64,512`: load only these methods. Every loaded method shares one packed copy of the
  weights, so a subset saves little memory; it saves load time.
- `--weight-cache FILE`: accepted for command-line compatibility with tflite-server and checked
  (at most 254 bytes, since ExecuTorch truncates longer backend options), but **not used in v0.1.0**:
  the packed weights are shared in memory. An on-disk cache is planned for v0.2.0. A started server
  prints one line saying the flag was accepted and not used.
- `--verbose`: log the methods loaded, the load time, the default thread stack size, and the token
  count, last token id and method of each request.

An unknown argument, a flag with no value, or a `--port` outside 1..65535 fails with the usage message.
A failed start prints exactly one `et-server:` line on stderr and exits 1.

- `GET /health` returns `{"status":"ok","engine":"executorch"}` once every method is loaded and warmed.
- `POST /classify {"input": "...", "top_k": 2}` (or `"text"`) returns `{"labels": {"LABEL_1": 0.98, ...}}`.
  A malformed body (including invalid UTF-8) is 400, a body over 64 KiB is 413, a model fault is 500.
  The whole text is tokenized and the tokens are cut to `max_length`, keeping the trailing `[SEP]`;
  the 64 KiB body cap bounds the tokenizer's work. There is no byte-level clip before tokenizing,
  because whitespace or zero-width padding would then push real content out of view.

It binds 127.0.0.1 only, without `SO_REUSEPORT`, so a second instance on a port in use fails to bind
instead of sharing its traffic. The binary sets a 1 MiB default thread stack (musl's default is 128 KiB). Lemonade reaches it as a local subprocess. At startup it raises its own
`oom_score_adj` to 500, so on a gateway that also runs an LLM the kernel reclaims the classifier first.

Setting `ET_SERVER_XNNPACK_WEIGHT_CACHE=0` turns off the in-memory weight sharing. It exists to
measure what the sharing saves.

## Build

On the ai4 build host layout (`~/build-litert`):

```bash
tools/et-musl-build.sh     # ExecuTorch v1.5.1 static libraries, once; checks the checkout is v1.5.1
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 ./build-prplos-x86_64.sh
./package-release.sh v0.1.0
```

`build-prplos-x86_64.sh` cross-compiles with the prplOS 5.1 gcc 13.3.0 musl toolchain against the raw
ExecuTorch build tree (`EXECUTORCH_BUILD`) and source checkout (`EXECUTORCH_SRC`), strips the binary,
and fails unless it needs only `libc.so`, `libgcc_s.so.1` and `libstdc++.so.6`.

## Tests

```bash
tools/make_bad_pte.py <dir>                               # .pte files that break the method contract
tools/test_server.sh <launcher> <model-dir> [<dir>]       # health, routing, errors, concurrency, startup failures
tools/compare.py ort=URL tflite=URL et=URL [--subset NAME] # score parity against the first server
```

## Measured

DistilBERT phishing classifier (`cybersectony/phishing-email-detection-distilbert_v2.4.1`, fp32,
`model.pte` 268 MB with 4 methods). The musl binary ran through the prplOS 5.1 loader on a Ryzen AI
Max+ 395, pinned to two cores with `--threads 2`, three starts each; the page cache was warm.

| methods loaded | weight sharing | start to `/health` ok | RssAnon | RssFile | VmHWM |
|---|---|---|---|---|---|
| 64, 128, 256, 512 | on (default) | 541-647 ms | 230 MiB | 20 MiB | 257 MiB |
| 64, 512 | on | 424-433 ms | 229 MiB | 20 MiB | 254 MiB |
| 64, 128, 256, 512 | off | 684-709 ms | 723 MiB | 20 MiB | 750 MiB |
| 64, 512 | off | 455-494 ms | 393 MiB | 20 MiB | 418 MiB |

With sharing on, loading all four methods costs about 1 MiB more than two, so all four is the default.

Scores match ort-server 0.3.7 (ONNX Runtime) to within 3.9e-6 on 8 texts, including one truncated at
512 tokens, with the same top label on all of them; the musl ort-server and tflite-server agree with it
to within 7.3e-7 and 1.9e-6 on the same texts (`tools/compare.py`).

## License

Apache-2.0. See `LICENSE` and `NOTICE`; this is a derivative of tflite-server and ort-server.
