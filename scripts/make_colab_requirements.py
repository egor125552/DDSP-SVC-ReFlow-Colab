#!/usr/bin/env python3
from pathlib import Path
import sys

src = Path(sys.argv[1] if len(sys.argv) > 1 else "requirements.txt")
dst = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/ddsp-requirements-colab.txt")

lines = []
seen_numba = False
for raw in src.read_text().splitlines():
    stripped = raw.strip()
    lower = stripped.lower()

    if lower.startswith("numba"):
        seen_numba = True

    if lower.startswith("numpy==1.26.4") and sys.version_info >= (3, 13):
        lines.append("numpy>=2.1,<2.3")
        continue

    lines.append(raw)

if sys.version_info >= (3, 13) and not seen_numba:
    lines.append("numba==0.61.2")

dst.write_text("\n".join(lines) + "\n")

print(f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
if sys.version_info >= (3, 13):
    print("Режим совместимости Python 3.13: numpy>=2.1,<2.3 + numba==0.61.2")
else:
    print("Используются upstream зависимости без замены numpy==1.26.4")
print("Requirements:", dst)
