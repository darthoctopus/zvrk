all: zvrk.tex
	latexmk -pdf zvrk.tex -interaction=nonstopmode -shell-escape -synctex=1 -bibtex-cond -quiet -silent -f
clean:
	rm -f zvrk.tex zvrk.fls zvrk.aux zvrk.bbl zvrk.blg zvrk.fdb_latexmk zvrk.synctex.gz zvrk.log zvrk.pdf
zvrk.tex: zvrk.md
	pandoc -f markdown+tex_math_single_backslash-latex_macros -s -o zvrk.tex zvrk.md --number-sections --natbib --highlight-style=breezedark --wrap=preserve
