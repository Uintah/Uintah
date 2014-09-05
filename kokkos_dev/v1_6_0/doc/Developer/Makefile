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

#
# This file uses xalan-java 2.3.1 or higher to process the xml files.
# The environment variable CLASSPATH must point to xalan.jar, xml-api.jar,
#  and xercesImpl.jar.
#

# Makefile for Developer directory.
.PHONY: products

PRODUCTS = Guide Modules CodeViews 

# Build all docs except code view docs (in CodeViews).  To additionallly
# build code views invoke make this way: make WITH_CV=true
products:
	for p in ${PRODUCTS} ; do cd $$p; ${MAKE}; cd ..; done

clean:
	for p in ${PRODUCTS} ; do cd $$p;  ${MAKE} clean; cd ..; done

veryclean: clean
	for p in ${PRODUCTS} ; do cd $$p; ${MAKE} veryclean; cd ..; done
