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
	$(SRCDIR)/ModuleCanvasItem.cc \
	$(SRCDIR)/moc_BuilderWindow.cc $(SRCDIR)/moc_NetworkCanvasView.cc

PSELIBS := 
QT_LIBDIR := /home/sparker/SCIRun/SCIRun_Thirdparty_32_linux/lib
LIBS := $(QT_LIBS)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

PROGRAM := builder
SRCS := $(SRCDIR)/builder.cc
PSELIBS := CCA/Components/Builder Core/CCA/Component/CIA Core/Thread \
	Core/Exceptions Core/globus_threads Core/CCA/ccaspec \
	Core/Containers
LIBS := $(GLOBUS_LIBS) -lglobus_nexus -lglobus_dc -lglobus_common -lglobus_io

include $(SCIRUN_SCRIPTS)/program.mk

$(SRCDIR)/Builder.o: Core/CCA/ccaspec/cca_sidl.h
