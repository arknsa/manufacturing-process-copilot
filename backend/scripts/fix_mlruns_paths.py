"""
Rewrite absolute artifact_uri paths in MLflow meta.yaml files so they
resolve correctly inside the container.

When mlruns/ is created on Windows, each run's meta.yaml stores:
    artifact_uri: file:///D:/some/host/path/mlruns/<exp>/<run>/artifacts

Inside the container the volume is mounted at /app/mlruns, so the stored
path is wrong. This script rewrites every artifact_uri to use the
container-local path. It is idempotent — runs that already point at
/app/mlruns are left unchanged.

Usage:
    python fix_mlruns_paths.py /app/mlruns
"""

import re
import sys
from pathlib import Path


def fix(mlruns_root: Path) -> None:
    pattern = re.compile(r"^(artifact_uri:\s*)file://(/[A-Za-z]:)?(.+)$")
    fixed = 0
    for meta in mlruns_root.rglob("meta.yaml"):
        lines = meta.read_text(encoding="utf-8").splitlines(keepends=True)
        new_lines = []
        changed = False
        for line in lines:
            m = pattern.match(line.rstrip("\n"))
            if m:
                rest = m.group(3)  # path after optional drive letter
                # Normalise Windows backslashes and drive letter
                rest = rest.replace("\\", "/")
                # Re-anchor to /app/mlruns if not already there
                if "/mlruns/" in rest:
                    _, _, tail = rest.partition("/mlruns/")
                    new_uri = f"file:///app/mlruns/{tail}"
                    new_line = f"{m.group(1)}{new_uri}\n"
                    if new_line != line:
                        line = new_line
                        changed = True
            new_lines.append(line)
        if changed:
            meta.write_text("".join(new_lines), encoding="utf-8")
            fixed += 1
    print(f"fix_mlruns_paths: {fixed} meta.yaml file(s) updated in {mlruns_root}")


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/mlruns")
    if not root.exists():
        print(f"fix_mlruns_paths: {root} does not exist — skipping")
        sys.exit(0)
    fix(root)
