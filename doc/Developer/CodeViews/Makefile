# Makefile for source code views.

.PHONY: products

SRC = ../../../src
TIMESTAMP = timestamp

products:
	if test -d ${SRC} -a \( "$$WITH_CV" = true -o $$MAKELEVEL -eq 1 \);  \
	then \
		doit=0; \
		if test ! -f ${TIMESTAMP} -o ! -d html; \
		then \
			doit=1; \
		elif ! cvs history -xAMR -al -D "`cat ${TIMESTAMP}`" -z MDT ${SRC}/ | grep -q '^No records selected.'; \
		then \
			doit=1; \
		fi; \
		if test $$doit -eq 1; \
		then \
			date >${TIMESTAMP}; \
			echo "-rm -rf html"; \
			echo "doxygen Doxyconfig"; \
		fi \
	fi

clean:
	-rm *~

veryclean:
	-rm -rf html ${TIMESTAMP}
