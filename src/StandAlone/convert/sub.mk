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

SRCDIR := StandAlone/convert

ifeq ($(LARGESOS),yes)
PSELIBS := Core
else
PSELIBS := Core/Datatypes Core/Containers Core/Persistent Core/Exceptions Core/Thread Core/Geometry Core/Math
endif
LIBS := -lm

PROGRAM := $(SRCDIR)/RawToContourField
SRCS := $(SRCDIR)/RawToContourField.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/RawToLatticeVol
SRCS := $(SRCDIR)/RawToLatticeVol.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/RawToTetVol
SRCS := $(SRCDIR)/RawToTetVol.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/RawToTriSurf
SRCS := $(SRCDIR)/RawToTriSurf.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/ContourFieldToRaw
SRCS := $(SRCDIR)/ContourFieldToRaw.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/LatticeVolToRaw
SRCS := $(SRCDIR)/LatticeVolToRaw.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TetVolToRaw
SRCS := $(SRCDIR)/TetVolToRaw.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TriSurfToRaw
SRCS := $(SRCDIR)/TriSurfToRaw.cc
include $(SCIRUN_SCRIPTS)/program.mk
