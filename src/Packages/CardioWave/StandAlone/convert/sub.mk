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

SRCDIR := Packages/CardioWave/StandAlone/convert

ifeq ($(LARGESOS),yes)
PSELIBS := Packages/CardioWave/StandAlone/convert
else
PSELIBS := Core/Datatypes Core/Math Core/Containers Core/Persistent Core/Exceptions Core/Thread Core/Geometry
endif
LIBS := $(M_LIBRARY)

PROGRAM := $(SRCDIR)/CardioWaveToColumnMat
SRCS := $(SRCDIR)/CardioWaveToColumnMat.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/CardioWaveVecToColumnMat
SRCS := $(SRCDIR)/CardioWaveVecToColumnMat.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/CardioWaveToDenseMat
SRCS := $(SRCDIR)/CardioWaveToDenseMat.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/CardioWaveToTwoDenseMats
SRCS := $(SRCDIR)/CardioWaveToTwoDenseMats.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/CardioWaveToLatVolVectorField
SRCS := $(SRCDIR)/CardioWaveToLatVolVectorField.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/ColumnMatToCardioWave
SRCS := $(SRCDIR)/ColumnMatToCardioWave.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/SubsetPts
SRCS := $(SRCDIR)/SubsetPts.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/TestFloodFill
SRCS := $(SRCDIR)/TestFloodFill.cc
include $(SRCTOP)/scripts/program.mk
