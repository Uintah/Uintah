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

root := ${shell cd ..; basename `pwd` }
TARFILE := ${notdir ${shell cd ..; pwd}}.tar
TARBALL := ${TARFILE}.gz
TARCMD := gnutar --exclude='*CVS*' --exclude='auto' \
--create --verbose --file=${root}/doc/${TARFILE} 
PRODUCTS := Installation User Developer Documentum

# Build all docs except code view docs (in Developer/CodeViews).  To
# additionally build code views invoke make this way: make WITH_CV=true
default : index.html products

# Build all docs and put them in  a tarball.
tarball : tar
	echo "Making a tarball"
	-rm -f ${TARBALL}
	gzip ${TARFILE}

# Construct list of files to be included in the tar archive.  Edit
# tarlist to make changes to this list.
tarlist = ${shell ruby tarlist}

# Build all docs and put them in a tar archive.
tar: index.html products
	@echo "Making a tar archive"
	-rm -f ${TARFILE}
	cd ../..; ${TARCMD} ${tarlist}

index.html: index.rhtml edition.xml
	eruby --delimiters html $< >$@

# Build all docs.
products:
	for p in ${PRODUCTS} ; do cd $$p; ${MAKE}; cd ..; done

# Cleanage
define rmtgz
-rm -f ${TARBALL}; rm -f ${TARFILE}
endef

clean:
	${rmtgz}
	for p in ${PRODUCTS} ; do cd $$p; ${MAKE} clean; cd ..; done

veryclean:
	${rmtgz}
	-rm -f index.html
	for p in ${PRODUCTS} ; do cd $$p; ${MAKE} veryclean; cd ..; done
