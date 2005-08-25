#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#


.PHONY: default tar tarball products clean veryclean

root := ${shell cd ..; basename `pwd` }
tarfile := ${notdir ${shell cd ..; pwd}}.tar
tarball := ${tarfile}.gz
tarlist := tarlist
tarcmd := gnutar --exclude='*CVS*' --exclude='auto' \
--create --file=${root}/doc/${tarfile} --files-from ${root}/doc/${tarlist}
products := User Installation Developer Documentum
edition := ../src/main/sci_version.h

# Build all docs except code view docs (in Developer/CodeViews).  To
# additionally build code views invoke make this way: make WITH_CV=true
default : index.html products

# Build all docs and put them in  a tarball.
tarball : tar
	@echo "Compressing the tar archive"
	gzip ${tarfile}

# Construct list of files to be included in the tar archive.  Edit
# tarlist to make changes to this list.
tarlist = ${shell ruby tarlist}

# Build all docs and put them in a tar archive.
tar: index.html products
	@echo "Making a tar archive"
	-rm -f ${tarfile} ${tarball}
	ruby tarlist.rb >${tarlist}
	cd ../..; ${tarcmd} ${tarlist}

index.html: index.rhtml ${edition}
	eruby --delimiters html $< >$@

# Build all docs.
products:
	for p in ${products} ; do cd $$p; ${MAKE}; cd ..; done

# Cleanage
define rmtgz
-rm -f ${tarball}; rm -f ${tarfile}
endef

clean:
	${rmtgz}
	for p in ${products} ; do cd $$p; ${MAKE} clean; cd ..; done

veryclean:
	${rmtgz}
	-rm -f index.html
	for p in ${products} ; do cd $$p; ${MAKE} veryclean; cd ..; done
