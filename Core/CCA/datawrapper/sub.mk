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

# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Core/CCA/datawrapper

SRCS := $(SRCDIR)/Matrix_sidl.cc $(SRCDIR)/MatrixWrap.cc $(SRCDIR)/ColumnMatrixWrap.cc \
	$(SRCDIR)/DenseMatrixWrap.cc $(SRCDIR)/SparseRowMatrixWrap.cc

# Hack until I get the guts to put this in Makefile.in
$(SRCDIR)/Matrix_sidl.o: $(SRCDIR)/Matrix_sidl.cc $(SRCDIR)/Matrix_sidl.h

$(SRCDIR)/Matrix_sidl.cc: $(SRCDIR)/Matrix.sidl $(SIDL_EXE)
	$(SIDL_EXE) -I ../src/Core/CCA/spec/cca.sidl -o $@ $<

$(SRCDIR)/Matrix_sidl.h: $(SRCDIR)/Matrix.sidl $(SIDL_EXE)
	$(SIDL_EXE) -I ../src/Core/CCA/spec/cca.sidl -h -o $@ $<

GENHDRS := $(SRCDIR)/Matrix_sidl.h $(SRCDIR)/MatrixWrap.h $(SRCDIR)/ColumnMatrixWrap.h \
	$(SRCDIR)/DenseMatrixWrap.h $(SRCDIR)/SparseRowMatrixWrap.h 
PSELIBS := Core/CCA/PIDL Core/Thread 

$(SRCDIR)/Matrix_sidl.o: Core/CCA/spec/cca_sidl.h

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
