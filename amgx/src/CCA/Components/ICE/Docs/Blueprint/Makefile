# ----------------------------------------------------------------- #
#                                                                   #
# This file: Makefile                                               #
#                                                                   #
#                                                                   #
#                                                                   #
# ----------------------------------------------------------------- #
PAPER      = ICE_MM

all : 
	pdflatex $(PAPER).tex
	bibtex $(PAPER)  | tee bibtexlog.txt
	pdflatex $(PAPER).tex
	pdflatex $(PAPER).tex

clean : 
	rm -f $(PAPER).dvi $(PAPER).pdf $(PAPER).ps $(PAPER).toc $(PAPER).out *.bbl *.blg *.bak *.backup *.log *.aux figures/*eps-converted-to.pdf
