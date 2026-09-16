#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p .build
for pass in 1 2 3; do
  xelatex -interaction=nonstopmode -halt-on-error -file-line-error \
    -output-directory=.build -jobname=ProjectIcy_code_algorithm_zh main.tex
done
cp .build/ProjectIcy_code_algorithm_zh.pdf ProjectIcy_code_algorithm_zh.pdf
