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

.PHONY: tar tarball

TAR = SCIRunDocs.tar
TARBALL = ${TAR}.gz
PRODUCTS = FAQBook Installation User Developer

# Build all docs except code view docs (in Developer/CodeViews).  To
# additionallly build code views invoke make this way: make WITH_CV=true
default : products

# Build all docs and put them in  a tarball.
tarball : tar
	echo "Making a tarball"
	-rm -f ${TARBALL}
	gzip ${TAR}

# Build all docs and put them in a tar archive.
tar: products
	echo "Making a tar archive"
	-rm -f SCIRunDocs.tar
	cd ..; tar cvf doc/SCIRunDocs.tar \
		doc/index.html \
		doc/.htaccess \
		doc/Installation/index.html \
                doc/Installation/Guide/*.html \
		doc/User/index.html \
		doc/User/Guide/usersguide \
		doc/User/Guide/Figures \
		doc/User/Tutorial \
		doc/User/Guide/usersguide.ps.gz \
		doc/User/Guide/usersguide2.ps.gz \
		doc/Developer/index.html \
                doc/Developer/Guide/*.html \
		doc/Developer/CodeViews/html \
		doc/FAQBook/*.html \
                doc/Utilities/HTML/doc_styles.css \
		doc/Utilities/Figures

# Build all docs.
products:
	for p in ${PRODUCTS} ; do cd $$p; ${MAKE}; cd ..; done

define rmtgz
-rm -f SCIRunDocs.tar.gz; rm -f SCIRunDocs.tar
endef

clean:
	${rmtgz}
	for p in ${PRODUCTS} ; do cd $$p; ${MAKE} clean; cd ..; done

veryclean:
	${rmtgz}
	for p in ${PRODUCTS} ; do cd $$p; ${MAKE} veryclean; cd ..; done
