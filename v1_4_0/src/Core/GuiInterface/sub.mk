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

SRCDIR   := Core/GuiInterface

SRCS     += $(SRCDIR)/GuiManager.cc \
	$(SRCDIR)/GuiServer.cc $(SRCDIR)/Histogram.cc \
	$(SRCDIR)/MemStats.cc $(SRCDIR)/Remote.cc $(SRCDIR)/TCL.cc \
	$(SRCDIR)/TCLInit.cc $(SRCDIR)/TCLTask.cc $(SRCDIR)/GuiVar.cc \
	$(SRCDIR)/ThreadStats.cc \
	$(SRCDIR)/TCLstrbuff.cc \
	$(SRCDIR)/TclObj.cc

ifeq ($(BUILD_SCIRUN2),yes)
SRCS +=	$(SRCDIR)/startTCL.cc
endif

PSELIBS := Core/Exceptions Core/Util Core/Thread \
		Core/Containers Core/TkExtensions
LIBS := $(TCL_LIBRARY) $(ITK_LIBRARY) $(X11_LIBS)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

