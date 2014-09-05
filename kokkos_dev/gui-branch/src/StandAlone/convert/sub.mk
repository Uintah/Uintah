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
PSELIBS := Core/Datatypes Core/Disclosure Core/Containers Core/Persistent \
           Core/Exceptions Core/Thread Core/Geometry Core/Math
endif
LIBS := $(XML_LIBRARY) -lm

PROGRAM := $(SRCDIR)/BugProgram
SRCS := $(SRCDIR)/BugProgram.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/CVRTItoTetVolDirichlet
SRCS := $(SRCDIR)/CVRTItoTetVolDirichlet.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/CVRTItoTetVolGrad
SRCS := $(SRCDIR)/CVRTItoTetVolGrad.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/CVRTItoTetVolPot
SRCS := $(SRCDIR)/CVRTItoTetVolPot.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TetVolToCVRTI
SRCS := $(SRCDIR)/TetVolToCVRTI.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/CVRTItoTriSurfGrad
SRCS := $(SRCDIR)/CVRTItoTriSurfGrad.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/CVRTItoTriSurfPot
SRCS := $(SRCDIR)/CVRTItoTriSurfPot.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TriSurfToCVRTI
SRCS := $(SRCDIR)/TriSurfToCVRTI.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/CVRTItoPointCloud
SRCS := $(SRCDIR)/CVRTItoPointCloud.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/PointCloudToCVRTI
SRCS := $(SRCDIR)/PointCloudToCVRTI.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/RawToContourField
SRCS := $(SRCDIR)/RawToContourField.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/RawToColumnMatrix
SRCS := $(SRCDIR)/RawToColumnMatrix.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/RawToDenseMatrix
SRCS := $(SRCDIR)/RawToDenseMatrix.cc
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
