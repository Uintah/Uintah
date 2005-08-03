#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#


# Makefile fragment for this subdirectory

SRCDIR := StandAlone/convert

ifeq ($(LARGESOS),yes)
PSELIBS := Core
else
PSELIBS := Core/Datatypes Core/Util Core/Containers Core/Persistent Core/Util \
           Core/Exceptions Core/Init Core/Thread Core/Geometry Core/Math Core/Geom \
	   Core/ImportExport Core/Basis
endif
LIBS := $(XML_LIBRARY) $(M_LIBRARY)


PROGRAM := $(SRCDIR)/CurveFieldToText
SRCS := $(SRCDIR)/CurveFieldToText.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/HexVolFieldToText
SRCS := $(SRCDIR)/HexVolFieldToText.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/HexVolToVtk
SRCS := $(SRCDIR)/HexVolToVtk.cc
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

PROGRAM := $(SRCDIR)/TextToHexTricubicHmt
SRCS := $(SRCDIR)/TextToHexTricubicHmt.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TextToPointCloudField
SRCS := $(SRCDIR)/TextToPointCloudField.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TextToPointCloudString
SRCS := $(SRCDIR)/TextToPointCloudString.cc
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

PROGRAM := $(SRCDIR)/VTKtoTriSurfField
SRCS := $(SRCDIR)/VTKtoTriSurfField.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/VTKtoHexVolField
SRCS := $(SRCDIR)/VTKtoHexVolField.cc
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

PROGRAM := $(SRCDIR)/TriSurfToVtk
SRCS := $(SRCDIR)/TriSurfToVtk.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TriSurfToOBJ
SRCS := $(SRCDIR)/TriSurfToOBJ.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TriSurfToTetgen
SRCS := $(SRCDIR)/TriSurfToTetgen.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TetgenToTetVol
SRCS := $(SRCDIR)/TetgenToTetVol.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/vff2nrrd
SRCS := $(SRCDIR)/vff2nrrd.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/PictToNrrd
SRCS := $(SRCDIR)/PictToNrrd.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/GeoProbeToNhdr
SRCS := $(SRCDIR)/GeoProbeToNhdr.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/UCSDToHexVol
SRCS := $(SRCDIR)/UCSDToHexVol.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/VistaToNrrd
SRCS := $(SRCDIR)/VistaToNrrd.cc
include $(SCIRUN_SCRIPTS)/program.mk


####################################
# added by C.Wolters, Nov.18 2004:
PROGRAM := $(SRCDIR)/gmvToPts
SRCS := $(SRCDIR)/gmvToPts.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/geoToPts
SRCS := $(SRCDIR)/geoToPts.cc
include $(SCIRUN_SCRIPTS)/program.mk
####################################


PROGRAM := $(SRCDIR)/SampleHexTricubicHmt
SRCS := $(SRCDIR)/SampleHexTricubicHmt.cc
include $(SCIRUN_SCRIPTS)/program.mk
