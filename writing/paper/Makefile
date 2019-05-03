FILE := paper
#OUT  := build


.PHONY: paper.pdf latexmk clean purge

all: paper.pdf

paper.pdf: paper.tex
        #latexmk -outdir=$(OUT) -shell-escape -interaction=nonstopmode -synctex=1 -bibtex -pdf $(FILE)
	latexmk -synctex=1 -shell-escape -interaction=nonstopmode -pdf $(FILE)
clean:
	latexmk -CA

purge:
	rm -rf $(OUT)

