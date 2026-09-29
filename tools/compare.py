"""POST the same texts to several /classify servers and compare each with the first.

    python compare.py ort=http://127.0.0.1:18190 tflite=http://127.0.0.1:18192 et=http://127.0.0.1:18193 \
        [--subset et]

The first server is the reference. The gate: every other server gives the same top label on all
texts and a max score delta below 1e-5. A server named in --subset loads only some methods
(et-server --seq-lens), so its truncation or routing can differ; for it only top-label agreement is
reported and gated. Exit status 1 if any gate fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

TEXTS = [
    "My name is John Smith and my SSN is 123-45-6789.",
    "URGENT: verify your account at http://secure-login.example to avoid suspension.",
    "Thanks for the notes from today's standup, talk tomorrow.",
    "Your parcel could not be delivered. Pay the 1.99 EUR redelivery fee here: http://dhl-redelivery.example/pay",
    "Hi team, attached is the Q3 budget spreadsheet. Please review before Friday's meeting.",
    "Félicitations ! Vous avez gagné un iPhone 17 — cliquez ici pour réclamer votre prix 🎁",
    "ok",
    (
        "Dear customer, "
        + "we noticed unusual sign-in activity on your account and need you to confirm your identity. " * 60
        + "Click http://verify.example now."
    ),
]
MAX_DELTA = 1e-5


def post(base: str, text: str):
    req = urllib.request.Request(
        base + "/classify",
        data=json.dumps({"input": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.load(r)
    return body["labels"], (time.time() - t0) * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("servers", nargs="+", help="name=url; the first is the reference")
    ap.add_argument("--subset", action="append", default=[], help="server name that loads a method subset")
    args = ap.parse_args()
    servers = [tuple(s.split("=", 1)) for s in args.servers]
    if len(servers) < 2:
        ap.error("need at least two servers")
    ref_name, ref_url = servers[0]
    worst = {name: 0.0 for name, _ in servers[1:]}
    same = {name: 0 for name, _ in servers[1:]}
    for text in TEXTS:
        ref, ref_ms = post(ref_url, text)
        top_ref = max(ref, key=ref.get)
        cells = [f"{ref_name}={ref[top_ref]:.4f} ({ref_ms:.0f} ms)"]
        for name, url in servers[1:]:
            got, ms = post(url, text)
            delta = max(abs(ref[k] - got.get(k, -1.0)) for k in ref)
            worst[name] = max(worst[name], delta)
            top = max(got, key=got.get)
            same[name] += top == top_ref
            cells.append(f"{name}={got[top]:.4f} {'SAME' if top == top_ref else 'DIFF'} d={delta:.2e} ({ms:.0f} ms)")
        print(f"top={top_ref} " + " | ".join(cells) + f" | {text[:40]!r}")
    ok = True
    for name, _ in servers[1:]:
        if name in args.subset:
            passed = same[name] == len(TEXTS)
            print(f"{name}: top label {same[name]}/{len(TEXTS)} vs {ref_name} (method subset: top label only; "
                  f"max delta {worst[name]:.2e} reported, not gated) -> {'PASS' if passed else 'FAIL'}")
        else:
            passed = same[name] == len(TEXTS) and worst[name] < MAX_DELTA
            print(f"{name}: top label {same[name]}/{len(TEXTS)}, max delta {worst[name]:.2e} vs {ref_name} "
                  f"(gate < {MAX_DELTA:.0e}) -> {'PASS' if passed else 'FAIL'}")
        ok &= passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
