"""Compile paper/main.typ to paper/paper.pdf.

Usage: uv run python paper/build.py
"""

from pathlib import Path

import typst

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "paper/paper.pdf"
typst.compile(str(ROOT / "paper/main.typ"), output=str(out), root=str(ROOT))
print(f"wrote {out.relative_to(ROOT)}")
