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

SRCDIR   := CCA/Components/Builder

SRCS     += \
	$(SRCDIR)/Builder.cc $(SRCDIR)/BuilderWindow.cc \
	$(SRCDIR)/QtUtils.cc $(SRCDIR)/NetworkCanvasView.cc \
	$(SRCDIR)/Module.cc \
	$(SRCDIR)/Connection.cc\
	$(SRCDIR)/moc_Module.cc\
	$(SRCDIR)/moc_BuilderWindow.cc $(SRCDIR)/moc_NetworkCanvasView.cc

PSELIBS := Core/CCA/SSIDL Core/CCA/PIDL  Core/CCA/Comm\
	Core/CCA/spec Core/Thread Core/Containers Core/Exceptions
LIBS := $(QT_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

PROGRAM := builder
SRCS := $(SRCDIR)/builder.cc
PSELIBS := CCA/Components/Builder Core/CCA/SSIDL \
	Core/CCA/PIDL Core/Exceptions Core/CCA/spec SCIRun
LIBS := 

include $(SCIRUN_SCRIPTS)/program.mk

$(SRCDIR)/Builder.o: Core/CCA/spec/cca_sidl.h
$(SRCDIR)/BuilderWindow.o: Core/CCA/spec/cca_sidl.h
$(SRCDIR)/moc_BuilderWindow.o: Core/CCA/spec/cca_sidl.h
$(SRCDIR)/builder.o: Core/CCA/spec/cca_sidl.h
