#!/bin/bash
# Functional tests for a built et-server against an exported model directory.
#
#   tools/test_server.sh <launcher> <model-dir> [bad-pte-dir]
#
# <launcher> starts et-server with its arguments appended (a plain binary path, or a wrapper that
# runs the musl binary through the prplOS loader). [bad-pte-dir] holds the model.pte variants
# written by tools/make_bad_pte.py; without it the strict-method tests are skipped.
# PORT (default 18480) is the first of the few consecutive ports used. CONCURRENCY (default 8).
set -uo pipefail
RUN=$1
MODEL=$2
BAD=${3:-}
PORT=${PORT:-18480}
CONC=${CONCURRENCY:-8}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/et-server-test.XXXXXX")
trap 'kill $(jobs -p) 2>/dev/null; rm -rf "$WORK"' EXIT
fails=0
pass() { echo "PASS $*"; }
fail() { echo "FAIL $*"; fails=$((fails + 1)); }

start() {  # start <port> <log> [args...]; sets SPID; waits for /health
  local p=$1 log=$2
  shift 2
  "$RUN" --model-path "$MODEL" --port "$p" "$@" >"$log" 2>&1 &
  SPID=$!
  for _ in $(seq 1 1200); do
    curl -sf "http://127.0.0.1:$p/health" >/dev/null && return 0
    kill -0 "$SPID" 2>/dev/null || return 1
    sleep 0.1
  done
  return 1
}
stop() { kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; }
classify() {  # classify <port> <text>
  python3 -c 'import json,sys; print(json.dumps({"input": sys.argv[1]}))' "$2" |
    curl -s -H 'Content-Type: application/json' -XPOST "http://127.0.0.1:$1/classify" --data-binary @-
}
code() {  # code <port> <body-file>
  # JSON content type: httplib caps form-urlencoded bodies (curl's default) at 8 KiB on its own.
  curl -s -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' -XPOST \
    "http://127.0.0.1:$1/classify" --data-binary @"$2"
}
words() { python3 -c 'import sys; print(" ".join(["hello"] * int(sys.argv[1])))' "$1"; }
mem() {  # mem <pid>
  awk '/^(VmHWM|VmRSS|RssAnon|RssFile):/ {printf "%s %d MiB  ", $1, $2 / 1024}' "/proc/$1/status"
  echo "majflt $(awk '{print $12}' "/proc/$1/stat")"
}

# --- all methods ---------------------------------------------------------------------------
P=$PORT
if start "$P" "$WORK/all.log" --verbose; then
  h=$(curl -s "http://127.0.0.1:$P/health")
  [ "$h" = '{"engine":"executorch","status":"ok"}' ] && pass "health $h" || fail "health $h"
  echo "     memory (all methods): $(mem "$SPID")"

  # Method routing: N words tokenize to N+2 ids ([CLS] ... [SEP]).
  for spec in 18:64 98:128 198:256 598:512; do
    n=${spec%%:*}; want=seq_${spec##*:}
    before=$(wc -l <"$WORK/all.log")
    out=$(classify "$P" "$(words "$n")")
    line=$(tail -n +"$((before + 1))" "$WORK/all.log" | grep -- '->' | tail -1)
    if echo "$out" | grep -q '"labels"' && echo "$line" | grep -q -- "-> $want\$"; then
      pass "route $((n + 2)) tokens: $line"
    else
      fail "route $((n + 2)) tokens (want $want): $line $out"
    fi
  done

  # Error contract.
  printf '{"input": ' >"$WORK/badjson"
  printf '{"top_k": 1}' >"$WORK/noinput"
  printf '{"input": 42}' >"$WORK/nonstring"
  for f in badjson noinput nonstring; do
    c=$(code "$P" "$WORK/$f")
    [ "$c" = 400 ] && pass "$f -> $c" || fail "$f -> $c (want 400)"
  done
  python3 -c 'import json; print(json.dumps({"input": "a" * (1 << 20)}))' >"$WORK/big"
  c=$(code "$P" "$WORK/big")
  [ "$c" = 413 ] && pass "1 MiB body -> $c" || fail "1 MiB body -> $c (want 413)"
  python3 -c 'import json; print(json.dumps({"input": ("phishing é " * 6000).encode()[:60000].decode(errors="ignore")}, ensure_ascii=False))' >"$WORK/60k"
  t0=$(date +%s%N); c=$(code "$P" "$WORK/60k"); t1=$(date +%s%N)
  ms=$(((t1 - t0) / 1000000))
  [ "$c" = 200 ] && [ "$ms" -lt 5000 ] && pass "60 KB text pre-clipped -> $c in $ms ms" ||
    fail "60 KB text -> $c in $ms ms"

  # Concurrency: parallel mixed-length requests must equal the serial answers exactly.
  texts=("$(words 10)" "$(words 100)" "$(words 250)" "$(words 700)")
  for i in "${!texts[@]}"; do classify "$P" "${texts[$i]}" >"$WORK/serial.$i"; done
  for round in 1 2 3; do
    pids=()
    for k in $(seq 0 $((CONC - 1))); do
      i=$((k % ${#texts[@]}))
      classify "$P" "${texts[$i]}" >"$WORK/par.$round.$k" &
      pids+=($!)
    done
    wait "${pids[@]}"
  done
  bad=0
  for round in 1 2 3; do
    for k in $(seq 0 $((CONC - 1))); do
      i=$((k % ${#texts[@]}))
      cmp -s "$WORK/serial.$i" "$WORK/par.$round.$k" || bad=$((bad + 1))
    done
  done
  [ "$bad" = 0 ] && pass "concurrency: 3 x $CONC parallel mixed-length requests equal serial bit for bit" ||
    fail "concurrency: $bad of $((3 * CONC)) parallel responses differ from serial"
  echo "     memory after requests: $(mem "$SPID")"
  stop
else
  fail "server did not start: $(cat "$WORK/all.log")"
fi

# --- --seq-lens 64,512 ------------------------------------------------------------------------
P=$((PORT + 1))
if start "$P" "$WORK/sub.log" --verbose --seq-lens 64,512; then
  grep -q 'lengths 64,512' "$WORK/sub.log" && pass "seq-lens: $(grep 'method(s)' "$WORK/sub.log")" ||
    fail "seq-lens: $(cat "$WORK/sub.log")"
  echo "     memory (64,512): $(mem "$SPID")"
  classify "$P" "$(words 98)" >/dev/null
  line=$(grep -- '->' "$WORK/sub.log" | tail -1)
  echo "$line" | grep -q -- '-> seq_512$' && pass "seq-lens: 100 tokens -> seq_512" || fail "seq-lens routing: $line"
  stop
else
  fail "server with --seq-lens did not start: $(cat "$WORK/sub.log")"
fi

# --- startup failures: one "et-server:" line on stderr, exit 1 --------------------------------
expect_fail() {  # expect_fail <name> <stderr-pattern> <model-dir> [args...]
  local name=$1 pat=$2 dir=$3
  shift 3
  "$RUN" --model-path "$dir" --port $((PORT + 2)) "$@" >"$WORK/f.out" 2>"$WORK/f.err" &
  local pid=$! rc
  for _ in $(seq 1 300); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"; wait "$pid" 2>/dev/null
    fail "$name: still running"
    return
  fi
  wait "$pid"; rc=$?
  local lines
  lines=$(wc -l <"$WORK/f.err")
  if [ "$rc" = 1 ] && [ "$lines" = 1 ] && grep -q "^et-server: .*$pat" "$WORK/f.err"; then
    pass "$name: $(cat "$WORK/f.err")"
  else
    fail "$name: rc=$rc lines=$lines $(head -c 400 "$WORK/f.err")"
  fi
}
variant() {  # variant <name>: a copy of the model dir with symlinked files
  local d=$WORK/m-$1
  mkdir -p "$d"
  for f in "$MODEL"/*; do ln -s "$f" "$d/"; done
  echo "$d"
}
d=$(variant nopte); rm "$d/model.pte"; expect_fail "missing model.pte" "model.pte not found" "$d"
d=$(variant corruptpte); rm "$d/model.pte"; head -c 4096 /dev/urandom >"$d/model.pte"; expect_fail "corrupt model.pte" "cannot load" "$d"
d=$(variant trunctok); rm "$d/tokenizer.json"; head -c 1000 "$MODEL/tokenizer.json" >"$d/tokenizer.json"
expect_fail "truncated tokenizer.json" "not valid JSON" "$d"
d=$(variant noconfig); rm "$d/config.json"; expect_fail "missing config.json" "config.json is missing" "$d"
d=$(variant xlnet); rm "$d/config.json"
python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); c["model_type"]="xlnet"; json.dump(c, open(sys.argv[2],"w"))' \
  "$MODEL/config.json" "$d/config.json"
expect_fail "model_type xlnet" "unsupported model_type .xlnet" "$d"
expect_fail "--weight-cache of 255 bytes" "weight-cache path is 255 bytes" "$MODEL" --weight-cache "/$(python3 -c 'print("c" * 254)')"
d=$(variant missinglen); expect_fail "--seq-lens 64,100" "does not have" "$d" --seq-lens 64,100
if [ -n "$BAD" ]; then
  for v in int32_input three_inputs rank3_output; do
    d=$(variant "$v"); rm "$d/model.pte"; ln -s "$BAD/$v.pte" "$d/model.pte"
    expect_fail "strict method: $v" "method seq_64: " "$d"
  done
else
  echo "SKIP strict method validation (no bad-pte dir)"
fi

echo "failures: $fails"
[ "$fails" = 0 ]
