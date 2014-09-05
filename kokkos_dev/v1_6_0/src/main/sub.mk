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

SRCDIR   := main
SRCS      := $(SRCDIR)/main.cc

ifeq ($(LARGESOS),yes)
  PSELIBS := Dataflow Core
else
  PSELIBS := Dataflow/Network Core/Containers Core/GuiInterface \
	Core/Thread Core/Exceptions Core/Util Core/TkExtensions
endif

LIBS := 
ifeq ($(NEED_SONAME),yes)
LIBS := $(LIBS) $(XML_LIBRARY) $(TK_LIBRARY) $(DL_LIBRARY) -lz
endif

PROGRAM := $(PROGRAM_PSE)

include $(SCIRUN_SCRIPTS)/program.mk

ifeq ($(BUILD_SCIRUN2),yes)

SRCS      := $(SRCDIR)/newmain.cc

ifeq ($(LARGESOS),yes)
  PSELIBS := Core/CCA/Component
else
  PSELIBS := Core/Exceptions Core/CCA/Component/Comm\
        Core/CCA/Component/PIDL Core/globus_threads Core/CCA/spec \
	SCIRun Core/CCA/Component/CIA Core/Thread
endif

LIBS :=
PROGRAM := sr

$(SRCDIR)/newmain.o: Core/CCA/spec/cca_sidl.h

include $(SCIRUN_SCRIPTS)/program.mk

endif
