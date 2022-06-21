FILE := CP032.tex
#OUT  := build


.PHONY: erratum.pdf latexmk clean purge

all: erratum.pdf

erratum.pdf: CP032.tex
	latexmk -synctex=1 -shell-escape -interaction=nonstopmode -pdf $(FILE)
clean:
	latexmk -CA

purge:
	rm -rf $(OUT)

