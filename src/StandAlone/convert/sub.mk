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
PSELIBS := Core/Datatypes Core/Util Core/Containers Core/Persistent \
           Core/Exceptions Core/Thread Core/Geometry Core/Math Core/Geom
endif
LIBS := $(PLPLOT_LIBRARY) $(XML_LIBRARY) $(M_LIBRARY)

PROGRAM := $(SRCDIR)/BugProgram
SRCS := $(SRCDIR)/BugProgram.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/MaskLatVolWithHexVol
SRCS := $(SRCDIR)/MaskLatVolWithHexVol.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/RawToHexVol
SRCS := $(SRCDIR)/RawToHexVol.cc
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

PROGRAM := $(SRCDIR)/CVRTItoTetVolMesh
SRCS := $(SRCDIR)/CVRTItoTetVolMesh.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TetVolToCVRTI
SRCS := $(SRCDIR)/TetVolToCVRTI.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/CVRTItoTriSurfGrad
SRCS := $(SRCDIR)/CVRTItoTriSurfGrad.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/CVRTItoTriSurf
SRCS := $(SRCDIR)/CVRTItoTriSurf.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/CVRTItoTriSurfPot
SRCS := $(SRCDIR)/CVRTItoTriSurfPot.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TriSurfToCVRTI
SRCS := $(SRCDIR)/TriSurfToCVRTI.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TriSurfToTetgen
SRCS := $(SRCDIR)/TriSurfToTetgen.cc
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

PROGRAM := $(SRCDIR)/RawToLatVol
SRCS := $(SRCDIR)/RawToLatVol.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/RawToTetVol
SRCS := $(SRCDIR)/RawToTetVol.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TetgenToTetVol
SRCS := $(SRCDIR)/TetgenToTetVol.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/RawToTriSurf
SRCS := $(SRCDIR)/RawToTriSurf.cc
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
