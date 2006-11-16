#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
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

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Dataflow/Modules/Fields

SRCS     += \
	$(SRCDIR)/ApplyMappingMatrix.cc\
	$(SRCDIR)/CalculateNodeNormals.cc\
	$(SRCDIR)/BuildMappingMatrix.cc\
	$(SRCDIR)/BuildMatrixOfSurfaceNormals.cc\
	$(SRCDIR)/ConvertMLVtoHV.cc\
	$(SRCDIR)/ConvertTVtoMLV.cc\
	$(SRCDIR)/GetCentroidsFromMesh.cc\
	$(SRCDIR)/ConvertMeshCoordinateSystem.cc\
	$(SRCDIR)/ConvertFieldBasis.cc\
	$(SRCDIR)/ConvertFieldDataType.cc\
	$(SRCDIR)/EditMeshBoundingBox.cc\
	$(SRCDIR)/ChooseField.cc\
	$(SRCDIR)/ClipFieldByFunction.cc\
	$(SRCDIR)/ClipFieldToFieldOrWidget.cc\
	$(SRCDIR)/ClipLatVolByIndicesOrWidget.cc\
	$(SRCDIR)/ConvertTet.cc\
	$(SRCDIR)/CoregisterPointClouds.cc\
	$(SRCDIR)/ConvertMatricesToMesh.cc\
	$(SRCDIR)/InterfaceWithCubit.cc\
	$(SRCDIR)/MapFieldDataFromSourceToDestination.cc\
	$(SRCDIR)/GetHexVolSheetBasedOnEdgeIndices.cc\
	$(SRCDIR)/ExtractPlanarSliceFromField.cc\
	$(SRCDIR)/GetFieldBoundary.cc\
	$(SRCDIR)/ReportFieldInfo.cc\
	$(SRCDIR)/ReportFieldGeometryMeasures.cc\
	$(SRCDIR)/SetFieldProperty.cc\
	$(SRCDIR)/GetSliceFromLatVol.cc\
	$(SRCDIR)/ClipRasterFieldByIndices.cc\
	$(SRCDIR)/JoinFields.cc\
	$(SRCDIR)/CalculateGradients.cc\
	$(SRCDIR)/ConvertHexVolToTetVol.cc\
	$(SRCDIR)/MergeFields.cc\
	$(SRCDIR)/InsertHexVolSheetFromTriSurf.cc\
	$(SRCDIR)/MergeTriSurfs.cc\
	$(SRCDIR)/ClipVolumeByIsovalue.cc\
	$(SRCDIR)/ClipVolumeByIsovalueWithRefinement.cc\
	$(SRCDIR)/SwapFieldDataWithMatrixEntries.cc\
	$(SRCDIR)/SwapNodeLocationsWithMatrixEntries.cc\
	$(SRCDIR)/MapFieldDataToNodeCoordinate.cc\
	$(SRCDIR)/MaskLatVol.cc\
	$(SRCDIR)/MaskLatVolWithTriSurf.cc\
	$(SRCDIR)/ConvertLatVolDataFromElemToNode.cc\
	$(SRCDIR)/ConvertLatVolDataFromNodeToElem.cc\
	$(SRCDIR)/CalculateLatVolGradientsAtNodes.cc\
	$(SRCDIR)/ConvertQuadSurfToTriSurf.cc\
	$(SRCDIR)/TransformPlanarMesh.cc\
	$(SRCDIR)/BuildPointCloudToLatVolMappingMatrix.cc\
	$(SRCDIR)/GenerateSinglePointProbeFromField.cc\
	$(SRCDIR)/GeneratePointSamplesFromFieldOrWidget.cc\
	$(SRCDIR)/CreateLatVol.cc\
	$(SRCDIR)/CreateImage.cc\
	$(SRCDIR)/CreateStructHex.cc\
	$(SRCDIR)/ReportScalarFieldStats.cc\
	$(SRCDIR)/ReportSearchGridInfo.cc\
	$(SRCDIR)/GeneratePointSamplesFromField.cc\
	$(SRCDIR)/SelectFieldROIWithBoxWidget.cc\
	$(SRCDIR)/SetFieldOrMeshStringProperty.cc\
	$(SRCDIR)/ConvertMeshToPointCloud.cc\
	$(SRCDIR)/ConvertRasterMeshToStructuredMesh.cc\
	$(SRCDIR)/CalculateFieldData.cc\
	$(SRCDIR)/CalculateFieldData3.cc\
	$(SRCDIR)/CalculateFieldDataCompiled.cc\
	$(SRCDIR)/CalculateFieldDataCompiled2.cc\
	$(SRCDIR)/CalculateFieldDataCompiled3.cc\
	$(SRCDIR)/TransformMeshWithFunction.cc\
	$(SRCDIR)/TransformMeshWithTransform.cc\
	$(SRCDIR)/ConvertMeshToUnstructuredMesh.cc\
	$(SRCDIR)/CalculateVectorMagnitudes.cc\
  $(SRCDIR)/CalculateDistanceToField.cc\
  $(SRCDIR)/CalculateDistanceToFieldBoundary.cc\
  $(SRCDIR)/CalculateSignedDistanceToField.cc\
  $(SRCDIR)/CalculateIsInsideField.cc\
  $(SRCDIR)/CalculateInsideWhichField.cc\
  $(SRCDIR)/ConvertIndicesToFieldData.cc\
  $(SRCDIR)/CreateFieldData.cc\
  $(SRCDIR)/GetFieldData.cc\
  $(SRCDIR)/SetFieldData.cc\
#[INSERT NEW CODE FILE HERE]


PSELIBS := \
	Dataflow/Network         \
	Dataflow/Widgets         \
	Core/Algorithms/Fields   \
	Core/Algorithms/Visualization   \
	Core/Algorithms/Geometry \
	Core/Algorithms/ArrayMath \
	Core/Algorithms/Converter \
	Core/Basis               \
	Core/Datatypes           \
	Core/Exceptions          \
	Core/Geom                \
	Core/Geometry            \
	Core/GeomInterface       \
	Core/Containers          \
	Dataflow/GuiInterface        \
	Core/ImportExport        \
	Core/Math                \
	Core/Persistent          \
	Core/Thread              \
	Dataflow/TkExtensions        \
	Core/Util              


LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(LEX_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY) $(TEEM_LIBRARY)

# Sandia Meshing Library
ifeq ($(HAVE_CAMAL),yes)
   SRCS += $(SRCDIR)/InterfaceWithCamal.cc
   LIBS := $(LIBS) $(CAMAL_LIBRARY) $(F_LIBRARY)
endif

# InterfaceWithTetGen http://tetgen.berlios.de
ifeq ($(HAVE_TETGEN),yes)
  SRCS += $(SRCDIR)/InterfaceWithTetGen.cc
  LIBS := $(LIBS) $(TETGEN_LIBRARY)
endif

# VERDICT Mesh Quality Library
ifeq ($(HAVE_VERDICT),yes)
   SRCS += $(SRCDIR)/ReportMeshQualityMeasures.cc
   LIBS := $(LIBS) $(VERDICT_LIBRARY)
endif

# MESQUITE Mesh Optimization Library
ifeq ($(HAVE_MESQUITE),yes)
   SRCS += $(SRCDIR)/SmoothMesh.cc\
	$(SRCDIR)/MesquiteDomain.cc
   LIBS := $(LIBS) $(MESQUITE_LIBRARY)
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

ifeq ($(LARGESOS),no)
SCIRUN_MODULES := $(SCIRUN_MODULES) $(LIBNAME)
endif

