compile:
	pdflatex main.tex
	bibtex main
	pdflatex main.tex

clean:
	cleantex
