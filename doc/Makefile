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
PRODUCTS = FAQBook Installation User Developer Documentum

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
	@echo "Making a tar archive"
	-rm -f SCIRunDocs.tar
	cd ..; root=`pwd`; root=`basename $${root}`;\
	cd ..; gnutar --exclude='*CVS*' --create --verbose \
		--file=$${root}/doc/SCIRunDocs.tar \
		$${root}/doc/index.html \
		$${root}/doc/.htaccess \
		$${root}/doc/Documentum/index.html \
		$${root}/doc/Documentum/documentum.html \
		$${root}/doc/Documentum/module-latex \
		$${root}/doc/Documentum/module-xml \
		$${root}/doc/Installation/index.html \
                $${root}/doc/Installation/Guide/*.html \
		$${root}/doc/User/index.html \
		$${root}/doc/User/Guide/usersguide \
		$${root}/doc/User/Guide/Figures \
		$${root}/doc/User/Tutorial \
		$${root}/doc/User/Guide/usersguide.ps.gz \
		$${root}/doc/User/Guide/usersguide2.ps.gz \
		$${root}/doc/Developer/index.html \
                $${root}/doc/Developer/Guide/*.html \
                $${root}/doc/Developer/Guide/dev/Figures \
		$${root}/doc/Developer/Modules/*.html \
		$${root}/doc/Developer/CodeViews/html \
		$${root}/doc/FAQBook/*.html \
                $${root}/doc/Utilities/HTML/*.css \
                $${root}/doc/Utilities/HTML/*.js \
		$${root}/doc/Utilities/Figures \
		$${root}/src/Dataflow/XML/*.html \
		$${root}/src/Dataflow/Modules/*/doc \
		$${root}/src/Packages/*/Dataflow/XML/*.html \
		$${root}/src/Packages/*/Dataflow/TeX/*/*/* \
		$${root}/src/Packages/*/Dataflow/TeX/*/figs

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
