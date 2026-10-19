LATEXMK ?= latexmk
PDFINFO ?= pdfinfo
PYTHON ?= python3
PAGE_LIMIT ?= 4

.PHONY: all figures verify-figures check clean

all: figures
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
	cp build/main.pdf main.pdf
	$(MAKE) check

figures: figures/connected_reconstruction.pdf figures/gradient_accuracy.pdf

figures/connected_reconstruction.pdf figures/gradient_accuracy.pdf &: scripts/build_method_figures.py
	$(PYTHON) scripts/build_method_figures.py

verify-figures: figures
	$(PYTHON) scripts/verify_gradient_figure.py

check:
	@pages=$$($(PDFINFO) main.pdf | awk '/^Pages:/ {print $$2}'); \
	test -n "$$pages" && test "$$pages" -le $(PAGE_LIMIT) || \
	{ echo "ERROR: expected at most $(PAGE_LIMIT) pages; found $${pages:-unknown}."; exit 1; }; \
	echo "Verified: $$pages pages (limit: $(PAGE_LIMIT), including references)."

clean:
	$(LATEXMK) -c -outdir=build main.tex
