# Makefile for source code views.

.PHONY: products tarball

SRC = ../../../src

products:
	if test -d ${SRC} -a \( "$$WITH_CV" = true -o $$MAKELEVEL -eq 1 \);  \
	then \
		test -d codeviews && rm -rf codeviews; \
		doxygen Doxyconfig; \
		mv html codeviews; \
	fi

tarball: products
	tar cf - codeviews | gzip >codeviews.tgz

clean:
	-rm *~

veryclean:
	-rm -rf codeviews
