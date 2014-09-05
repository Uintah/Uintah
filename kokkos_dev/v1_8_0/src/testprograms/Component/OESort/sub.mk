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

SRCDIR := testprograms/Component/OESort

ifeq ($(LARGESOS),yes)
PSELIBS := Core
else
PSELIBS := Core/CCA/Component/CIA Core/CCA/Component/PIDL Core/Thread \
	Core/Exceptions Core/globus_threads Core/CCA/Component/Comm
endif
LIBS := $(MPI_LIBRARY) 

PROGRAM := $(SRCDIR)/OESort
SRCS := $(SRCDIR)/OESort.cc $(SRCDIR)/OESort_sidl.cc \
	$(SRCDIR)/OESort_impl.cc
GENHDRS := $(SRCDIR)/OESort_sidl.h

include $(SCIRUN_SCRIPTS)/program.mk

