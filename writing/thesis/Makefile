FILE := main
#OUT  := build


.PHONY: main.pdf latexmk clean purge

all: main.pdf

main.pdf: main.tex
        #latexmk -outdir=$(OUT) -shell-escape -interaction=nonstopmode -synctex=1 -bibtex -pdf $(FILE)
	latexmk -synctex=1 -shell-escape -interaction=nonstopmode -pdf $(FILE)
clean:
	latexmk -CA

purge:
	rm -rf $(OUT)

