#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#
#  The Original Source Code is SCIRun, released March 12, 2001.
#
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
#  University of Utah. All Rights Reserved.
#

# Makefile for Developer directory.
.PHONY: products

PRODUCTS = Guide FAQ Modules CodeViews 

default: index.html products

index.html: index.rhtml ../edition.xml
	eruby --delimiters html $< >$@

# Build all docs except code view docs (in CodeViews).  To additionallly
# build code views invoke make this way: make WITH_CV=true
products:
	for p in ${PRODUCTS} ; do cd $$p; ${MAKE}; cd ..; done

clean:
	for p in ${PRODUCTS} ; do cd $$p;  ${MAKE} clean; cd ..; done

veryclean:
	for p in ${PRODUCTS} ; do cd $$p; ${MAKE} veryclean; cd ..; done
