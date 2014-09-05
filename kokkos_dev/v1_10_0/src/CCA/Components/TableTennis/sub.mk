
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

SRCDIR   := CCA/Components/TableTennis

SRCS     += \
	$(SRCDIR)/TableTennis_sidl.cc $(SRCDIR)/TableTennis.cc

# Hack until I get the guts to put this in Makefile.in  
$(SRCDIR)/TableTennis_sidl.o: $(SRCDIR)/TableTennis_sidl.cc $(SRCDIR)/TableTennis_sidl.h

$(SRCDIR)/TableTennis_sidl.cc: $(SRCDIR)/TableTennis.sidl $(SIDL_EXE)
	$(SIDL_EXE) -I $(SRCTOP_ABS)/Core/CCA/spec/cca.sidl -o $@ $<

$(SRCDIR)/TableTennis_sidl.h: $(SRCDIR)/TableTennis.sidl $(SIDL_EXE)
	$(SIDL_EXE) -I $(SRCTOP_ABS)/Core/CCA/spec/cca.sidl -h -o $@ $<

GENHDRS := $(SRCDIR)/TableTennis_sidl.h
PSELIBS := Core/CCA/Component/SSIDL Core/CCA/Component/PIDL Core/CCA/Component/Comm\
	Core/CCA/spec Core/Thread Core/Containers Core/Exceptions
QT_LIBDIR := /home/sparker/SCIRun/SCIRun_Thirdparty_32_linux/lib
LIBS := $(QT_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

#include $(SCIRUN_SCRIPTS)/program.mk

$(SRCDIR)/TableTennis.o: Core/CCA/spec/cca_sidl.h
