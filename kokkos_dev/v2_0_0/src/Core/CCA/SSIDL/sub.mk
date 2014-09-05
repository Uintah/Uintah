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

SRCDIR   := Core/CCA/SSIDL

SRCS     += \
	$(SRCDIR)/sidl_sidl.cc \
	$(SRCDIR)/Object.cc \
	$(SRCDIR)/BaseInterface.cc \
	$(SRCDIR)/BaseException.cc 

# We cannot use the implicit rule for SSIDL, since it needs that
# special -cia flag

# SSIDL has been replaced with sidl.sidl from Babel source
$(SRCDIR)/sidl_sidl.o: $(SRCDIR)/sidl_sidl.cc $(SRCDIR)/sidl_sidl.h

$(SRCDIR)/sidl_sidl.cc: $(SRCDIR)/sidl.sidl $(SIDL_EXE)
	$(SIDL_EXE) -cia -o $@ $<

$(SRCDIR)/sidl_sidl.h: $(SRCDIR)/sidl.sidl $(SIDL_EXE)
	$(SIDL_EXE) -cia -h -o $@ $<

GENHDRS := $(SRCDIR)/sidl_sidl.h

PSELIBS := Core/CCA/PIDL

LIBS := 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

