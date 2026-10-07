#!/usr/bin/env python3
"""Collect the licence texts of the Rust crates linked into a static library, for one target.

    rust_licenses.py <cargo-project-dir> <rust-target> <out-dir> <binary-name> [--cargo PATH]

Walks `cargo metadata --filter-platform <target>` from the root package along normal
(non-build, non-dev) dependencies, not into proc-macro crates, so what runs only in the compiler
(build scripts, derive macros) is left out.
For every crate it copies LICENSE*/LICENCE*/COPYING*/UNLICENSE*/NOTICE* (and the same names
one directory down, which is where onig_sys keeps Oniguruma's COPYING) into
<out-dir>/rust/<name>-<version>/ and writes <out-dir>/rust-crates.txt. Exits 1 if any crate
has no licence file.
"""
import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys

NAMES = re.compile(r"^(LICEN[CS]E|COPYING|UNLICENSE|NOTICE)([-._].*)?$", re.I)
SKIP_DIRS = {"tests", "test", "benches", "examples", "target", ".git", "src"}


def is_proc_macro(pkg):
    """Proc-macro crates (and what only they use) run in the compiler and are not linked."""
    return any("proc-macro" in t["kind"] for t in pkg["targets"])


def licence_files(root: pathlib.Path):
    found = [p for p in sorted(root.iterdir()) if p.is_file() and NAMES.match(p.name)]
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name not in SKIP_DIRS:
            found += [p for p in sorted(d.iterdir()) if p.is_file() and NAMES.match(p.name)]
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("target")
    ap.add_argument("out")
    ap.add_argument("binary")
    ap.add_argument("--cargo", default="cargo")
    a = ap.parse_args()
    meta = json.loads(subprocess.run(
        [a.cargo, "metadata", "--format-version", "1", "--filter-platform", a.target, "--offline"],
        cwd=a.project, check=True, capture_output=True, text=True).stdout)
    pkgs = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    root = meta["resolve"]["root"]
    seen, todo = set(), [root]
    while todo:
        nid = todo.pop()
        if nid in seen:
            continue
        seen.add(nid)
        for dep in nodes[nid]["deps"]:
            if any(k.get("kind") is None for k in dep["dep_kinds"]) and not is_proc_macro(pkgs[dep["pkg"]]):
                todo.append(dep["pkg"])
    seen.discard(root)
    out = pathlib.Path(a.out)
    rust = out / "rust"
    if rust.exists():
        shutil.rmtree(rust)
    lines, missing = [], []
    for pid in sorted(seen, key=lambda i: (pkgs[i]["name"], pkgs[i]["version"])):
        p = pkgs[pid]
        name, ver, lic = p["name"], p["version"], p.get("license") or p.get("license_file") or "see crate"
        src = pathlib.Path(p["manifest_path"]).parent
        files = licence_files(src)
        dest = rust / f"{name}-{ver}"
        dest.mkdir(parents=True)
        for f in files:
            rel = f.relative_to(src)
            shutil.copyfile(f, dest / str(rel).replace("/", "__"))
        if not files:
            missing.append(f"{name}-{ver}")
        lines.append(f"  {name} {ver}  {lic}  [{', '.join(str(f.relative_to(src)) for f in files) or 'NO LICENCE FILE'}]")
    with open(out / "rust-crates.txt", "w") as f:
        f.write(f"Rust crates statically linked into bin/{a.binary} (through the HuggingFace tokenizers C shim,\n"
                f"normal dependencies of {pkgs[root]['name']} for {a.target}); licence texts in rust/<crate>-<version>/:\n")
        f.write("\n".join(lines) + "\n")
    print(f"{len(seen)} crates, {len(missing)} without a licence file{': ' + ' '.join(missing) if missing else ''}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
