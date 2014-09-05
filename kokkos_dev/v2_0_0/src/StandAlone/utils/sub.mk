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

SRCDIR := StandAlone/utils

ifeq ($(LARGESOS),yes)
PSELIBS := Core
else
PSELIBS := Core/Datatypes Core/Util Core/Containers Core/Persistent \
           Core/Exceptions Core/Thread Core/Geometry Core/Math Core/Geom
endif
LIBS := $(PLPLOT_LIBRARY) $(XML_LIBRARY) $(M_LIBRARY)

PROGRAM := $(SRCDIR)/MaskLatVolWithHexVol
SRCS := $(SRCDIR)/MaskLatVolWithHexVol.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TriSurfToTetgen
SRCS := $(SRCDIR)/TriSurfToTetgen.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TetgenToTetVol
SRCS := $(SRCDIR)/TetgenToTetVol.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/SingleTet
SRCS := $(SRCDIR)/SingleTet.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TriSurfToVtk
SRCS := $(SRCDIR)/TriSurfToVtk.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/VtkToTriSurf
SRCS := $(SRCDIR)/VtkToTriSurf.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TriSurfToOBJ
SRCS := $(SRCDIR)/TriSurfToOBJ.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/FieldTextToBin
SRCS := $(SRCDIR)/FieldTextToBin.cc
include $(SCIRUN_SCRIPTS)/program.mk
