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


PROGRAM := $(SRCDIR)/CurveFieldToText
SRCS := $(SRCDIR)/CurveFieldToText.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/HexVolFieldToText
SRCS := $(SRCDIR)/HexVolFieldToText.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/PointCloudFieldToText
SRCS := $(SRCDIR)/PointCloudFieldToText.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/QuadSurfFieldToText
SRCS := $(SRCDIR)/QuadSurfFieldToText.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/StructCurveFieldToText
SRCS := $(SRCDIR)/StructCurveFieldToText.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/StructHexVolFieldToText
SRCS := $(SRCDIR)/StructHexVolFieldToText.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/StructQuadSurfFieldToText
SRCS := $(SRCDIR)/StructQuadSurfFieldToText.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TetVolFieldToText
SRCS := $(SRCDIR)/TetVolFieldToText.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TriSurfFieldToText
SRCS := $(SRCDIR)/TriSurfFieldToText.cc
include $(SCIRUN_SCRIPTS)/program.mk



PROGRAM := $(SRCDIR)/ColumnMatrixToText
SRCS := $(SRCDIR)/ColumnMatrixToText.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/DenseMatrixToText
SRCS := $(SRCDIR)/DenseMatrixToText.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/SparseRowMatrixToText
SRCS := $(SRCDIR)/SparseRowMatrixToText.cc
include $(SCIRUN_SCRIPTS)/program.mk



PROGRAM := $(SRCDIR)/ColorMapToText
SRCS := $(SRCDIR)/ColorMapToText.cc
include $(SCIRUN_SCRIPTS)/program.mk






PROGRAM := $(SRCDIR)/TextToCurveField
SRCS := $(SRCDIR)/TextToCurveField.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TextToHexVolField
SRCS := $(SRCDIR)/TextToHexVolField.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TextToPointCloudField
SRCS := $(SRCDIR)/TextToPointCloudField.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TextToQuadSurfField
SRCS := $(SRCDIR)/TextToQuadSurfField.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TextToStructCurveField
SRCS := $(SRCDIR)/TextToStructCurveField.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TextToStructHexVolField
SRCS := $(SRCDIR)/TextToStructHexVolField.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TextToStructQuadSurfField
SRCS := $(SRCDIR)/TextToStructQuadSurfField.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TextToTetVolField
SRCS := $(SRCDIR)/TextToTetVolField.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TextToTriSurfField
SRCS := $(SRCDIR)/TextToTriSurfField.cc
include $(SCIRUN_SCRIPTS)/program.mk



PROGRAM := $(SRCDIR)/TextToColumnMatrix
SRCS := $(SRCDIR)/TextToColumnMatrix.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TextToDenseMatrix
SRCS := $(SRCDIR)/TextToDenseMatrix.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TextToSparseRowMatrix
SRCS := $(SRCDIR)/TextToSparseRowMatrix.cc
include $(SCIRUN_SCRIPTS)/program.mk



PROGRAM := $(SRCDIR)/TextToColorMap
SRCS := $(SRCDIR)/TextToColorMap.cc
include $(SCIRUN_SCRIPTS)/program.mk
