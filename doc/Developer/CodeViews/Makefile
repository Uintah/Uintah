# Makefile for source code views.

.PHONY: products tarball

SRC = ../../../src

products:
	if test -d ${SRC} -a \( "$$WITH_CV" = true -o $$MAKELEVEL -eq 1 \);  \
	then \
		test -d html && rm -rf html; \
		doxygen Doxyconfig; \
	fi

tarball: products
	tar cf - html | gzip >codeviews.tgz

clean:
	-rm *~

veryclean:
	-rm -rf html
