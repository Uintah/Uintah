# Makefile for Installation directory.
.PHONY: products

PRODUCTS = Guide

products:
	for p in ${PRODUCTS} ; do cd $$p; ${MAKE}; cd ..; done

clean:
	for p in ${PRODUCTS} ; do cd $$p;  ${MAKE} clean; cd ..; done

veryclean: clean
	for p in ${PRODUCTS} ; do cd $$p; ${MAKE} veryclean; cd ..; done
