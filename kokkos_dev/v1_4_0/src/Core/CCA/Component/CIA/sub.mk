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

SRCDIR   := Core/CCA/Component/CIA

SRCS     += \
	$(SRCDIR)/CIA_sidl.cc \
	$(SRCDIR)/Class.cc \
	$(SRCDIR)/ClassNotFoundException.cc \
	$(SRCDIR)/IllegalArgumentException.cc \
	$(SRCDIR)/InstantiationException.cc \
	$(SRCDIR)/Interface.cc \
	$(SRCDIR)/Method.cc \
	$(SRCDIR)/NoSuchMethodException.cc \
	$(SRCDIR)/Object.cc \
	$(SRCDIR)/Throwable.cc 

# We cannot use the implicit rule for CIA, since it needs that
# special -cia flag
$(SRCDIR)/CIA_sidl.o: $(SRCDIR)/CIA_sidl.cc $(SRCDIR)/CIA_sidl.h

$(SRCDIR)/CIA_sidl.cc: $(SRCDIR)/CIA.sidl $(SIDL_EXE)
	$(SIDL_EXE) -cia -o $@ $<

$(SRCDIR)/CIA_sidl.h: $(SRCDIR)/CIA.sidl $(SIDL_EXE)
	$(SIDL_EXE) -cia -h -o $@ $<

GENHDRS := $(SRCDIR)/CIA_sidl.h

PSELIBS := Core/CCA/Component/PIDL
LIBS := $(GLOBUS_LIBS) -lglobus_nexus -lglobus_dc -lglobus_common

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

