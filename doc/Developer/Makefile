# Makefile for Developer directory.
.PHONY: products

PRODUCTS = Guide CodeViews

# Build all docs except code view docs (in CodeViews).  To additionallly
# build code views invoke make this way: make WITH_CV=true
products:
	for p in ${PRODUCTS} ; do cd $$p; ${MAKE}; cd ..; done

clean:
	for p in ${PRODUCTS} ; do cd $$p;  ${MAKE} clean; cd ..; done

veryclean: clean
	for p in ${PRODUCTS} ; do cd $$p; ${MAKE} veryclean; cd ..; done
