compile:
	pdflatex tesis.tex
	bibtex tesis
	pdflatex tesis.tex

clean:
	cleantex
