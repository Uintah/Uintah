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

SRCDIR := Packages/FieldConverters/StandAlone/convert

ifeq ($(LARGESOS),yes)
PSELIBS := Core Packages/FieldConverters/Core
else
PSELIBS := Core/Datatypes Core/Containers Core/Persistent Core/Exceptions Core/Thread Core/Geometry Core/Math Packages/FieldConverters/Core/Datatypes
endif
LIBS := -lm

PROGRAM := $(SRCDIR)/OldSFRGtoNewLatticeVol
SRCS := $(SRCDIR)/OldSFRGtoNewLatticeVol.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/OldVFRGtoNewLatticeVol
SRCS := $(SRCDIR)/OldVFRGtoNewLatticeVol.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/NewLatticeVolToOldSFRG
SRCS := $(SRCDIR)/NewLatticeVolToOldSFRG.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/OldSFUGtoNewTetVol
SRCS := $(SRCDIR)/OldSFUGtoNewTetVol.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/NewTetVolToOldSFUG
SRCS := $(SRCDIR)/NewTetVolToOldSFUG.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/OldMeshToNewTetVol
SRCS := $(SRCDIR)/OldMeshToNewTetVol.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/OldMeshToNewField
SRCS := $(SRCDIR)/OldMeshToNewField.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/NewTetVolToOldMesh
SRCS := $(SRCDIR)/NewTetVolToOldMesh.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/OldSurfaceToNewTriSurf
SRCS := $(SRCDIR)/OldSurfaceToNewTriSurf.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/NewTriSurfToOldSurface
SRCS := $(SRCDIR)/NewTriSurfToOldSurface.cc
include $(SCIRUN_SCRIPTS)/program.mk
