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


# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Core/CCA/datawrapper

SRCS := $(SRCDIR)/Matrix_sidl.cc $(SRCDIR)/MatrixWrap.cc $(SRCDIR)/ColumnMatrixWrap.cc \
	$(SRCDIR)/DenseMatrixWrap.cc $(SRCDIR)/SparseRowMatrixWrap.cc

# Hack until I get the guts to put this in Makefile.in
$(SRCDIR)/Matrix_sidl.o: $(SRCDIR)/Matrix_sidl.cc $(SRCDIR)/Matrix_sidl.h

$(SRCDIR)/Matrix_sidl.cc: $(SRCDIR)/Matrix.sidl $(SIDL_EXE)
	$(SIDL_EXE) -I $(SRCTOP)/Core/CCA/spec/sci.sidl -o $@ $<

$(SRCDIR)/Matrix_sidl.h: $(SRCDIR)/Matrix.sidl $(SIDL_EXE)
	$(SIDL_EXE) -I $(SRCTOP)/Core/CCA/spec/sci.sidl -h -o $@ $<

GENHDRS := $(SRCDIR)/Matrix_sidl.h $(SRCDIR)/MatrixWrap.h $(SRCDIR)/ColumnMatrixWrap.h \
	$(SRCDIR)/DenseMatrixWrap.h $(SRCDIR)/SparseRowMatrixWrap.h 
PSELIBS := Core/CCA/PIDL Core/Thread 

$(SRCDIR)/Matrix_sidl.o: Core/CCA/spec/sci_sidl.h

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

SUBDIRS := $(SRCDIR)/PDEtranslate

include $(SCIRUN_SCRIPTS)/recurse.mk

