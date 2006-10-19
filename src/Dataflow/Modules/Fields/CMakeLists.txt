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

# CMakeLists.txt for Dataflow/Modules/Fields

SET(Dataflow_Modules_Fields_SRCS
  ApplyMappingMatrix.cc
  CalculateNodeNormals.cc
  BuildMappingMatrix.cc
  BuildMatrixOfSurfaceNormals.cc
  ConvertMLVtoHV.cc
  ConvertTVtoMLV.cc
  GetCentroidsFromMesh.cc
  ConvertMeshCoordinateSystem.cc
  ConvertFieldBasis.cc
  ConvertFieldDataType.cc
  EditMeshBoundingBox.cc
  ChooseField.cc
  ClipFieldByFunction.cc
  ClipFieldToFieldOrWidget.cc
  ClipLatVolByIndicesOrWidget.cc
  ConvertTet.cc
  CoregisterPointClouds.cc
  ConvertMatricesToMesh.cc
  InterfaceWithCubit.cc
  MapFieldDataFromSourceToDestination.cc
  GetHexVolSheetBasedOnEdgeIndices.cc
  ExtractPlanarSliceFromField.cc
  GetFieldBoundary.cc
  ShowMeshBoundingBox.cc
  FieldFrenet.cc
  ReportFieldInfo.cc
  ReportFieldGeometryMeasures.cc
  SetFieldProperty.cc
  GetSliceFromLatVol.cc
  ClipRasterFieldByIndices.cc
  JoinFields.cc
  CalculateGradients.cc
  ConvertHexVolToTetVol.cc
  MergeFields.cc
  InsertHexVolSheetFromTriSurf.cc
  MergeTriSurfs.cc
  ClipVolumeByIsovalue.cc
  ClipVolumeByIsovalueWithRefinement.cc
  SwapFieldDataWithMatrixEntries.cc
  SwapNodeLocationsWithMatrixEntries.cc
  MapFieldDataToNodeCoordinate.cc
  MaskLatVol.cc
  MaskLatVolWithTriSurf.cc
  ConvertLatVolDataFromElemToNode.cc
  ConvertLatVolDataFromNodeToElem.cc
  NodeCalculateGradients.cc
  ConvertQuadSurfToTriSurf.cc
  TransformPlanarMesh.cc
  BuildPointCloudToLatVolMappingMatrix.cc
  GenerateSinglePointProbeFromField.cc
  ReplaceScalarDataValue.cc
  GeneratePointSamplesFromFieldOrWidget.cc
  CreateLatVol.cc
  CreateImage.cc
  CreateStructHex.cc
  ReportScalarFieldStats.cc
  ScaleFieldData.cc
  ReportSearchGridInfo.cc
  GeneratePointSamplesFromField.cc
  SelectFieldROIWithBoxWidget.cc
  SetFieldOrMeshStringProperty.cc
  ConvertMeshToPointCloud.cc
  ConvertRasterMeshToStructuredMesh.cc
  TransformMeshWithFunctionWithTransform.cc
  CalculateFieldData.cc
  CalculateFieldData2.cc
  CalculateFieldData3.cc
  TransformMeshWithFunction.cc
  ConvertMeshToUnstructuredMesh.cc
  CalculateVectorMagnitudes.cc
)

ADD_LIBRARY(Dataflow_Modules_Fields ${Dataflow_Modules_Fields_SRCS})

TARGET_LINK_LIBRARIES(Dataflow_Modules_Fields
  Dataflow_Network
  Dataflow_Widgets
  Core_Algorithms_Fields
  Core_Algorithms_Visualization
  Core_Algorithms_Geometry
  Core_Basis
  Core_Datatypes
  Core_Exceptions
  Core_Geom
  Core_Geometry
  Core_GeomInterface
  Core_Containers
  Dataflow_GuiInterface
  Core_ImportExport
  Core_Math
  Core_Persistent
  Core_Thread
  Dataflow_TkExtensions
  Core_Util
  ${TK_LIBRARY}
  ${GL_LIBRARY}
  ${LEX_LIBRARY}
  ${M_LIBRARY}
  ${THREAD_LIBRARY}
  ${TEEM_LIBRARY}
)

IF(BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Dataflow_Modules_Fields)
ENDIF(BUILD_SHARED_LIBS)
