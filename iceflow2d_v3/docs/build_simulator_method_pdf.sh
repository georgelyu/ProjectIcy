#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
snapshot_date="$(python -c 'import json; from pathlib import Path; print(json.loads(Path("simulator_method_source_manifest.json").read_text())["snapshot_date"])')"
pandoc simulator_method_zh.md \
  --from=markdown --standalone --shift-heading-level-by=-1 \
  --toc --toc-depth=2 \
  --pdf-engine=xelatex \
  --include-in-header=simulator_method_pdf_header.tex \
  --variable=documentclass:article \
  --variable=papersize:a4 --variable=fontsize:10pt \
  --variable=geometry:margin=18mm \
  --variable=mainfont:'DejaVu Serif' \
  --variable=sansfont:'DejaVu Sans' \
  --variable=monofont:'DejaVu Sans Mono' \
  --variable=CJKmainfont:'Noto Serif CJK SC' \
  --variable=CJKsansfont:'Noto Sans CJK SC' \
  --variable=CJKmonofont:'Noto Sans CJK SC' \
  --metadata=lang:zh-CN \
  --metadata=title:'IceFlow2D v3 模拟器原理与逐步验证指南' \
  --metadata=date:"源码快照 ${snapshot_date}" \
  --output=simulator_method_zh.pdf
