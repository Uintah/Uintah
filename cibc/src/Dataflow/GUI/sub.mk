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

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the Core/CCA/Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Core/CCA/Component"
# documentation on how to do it correctly.

# Makefile fragment for this subdirectory

SRCDIR := Dataflow/GUI

SRCS := \
        $(SRCDIR)/AppendMatrix.tcl\
        $(SRCDIR)/CreateLightForViewer.tcl \
        $(SRCDIR)/CollectMatrices.tcl \
        $(SRCDIR)/ArrowWidget.tcl \
        $(SRCDIR)/BaseWidget.tcl \
        $(SRCDIR)/BoxWidget.tcl \
        $(SRCDIR)/BuildMappingMatrix.tcl \
        $(SRCDIR)/BuildNoiseColumnMatrix.tcl \
        $(SRCDIR)/CreateGeometricTransform.tcl \
        $(SRCDIR)/InsertBundlesIntoBundle.tcl \
        $(SRCDIR)/InsertColorMapsIntoBundle.tcl \
        $(SRCDIR)/InsertColorMap2sIntoBundle.tcl \
        $(SRCDIR)/InsertFieldsIntoBundle.tcl \
        $(SRCDIR)/InsertMatricesIntoBundle.tcl \
        $(SRCDIR)/InsertNrrdsIntoBundle.tcl \
        $(SRCDIR)/InsertPathsIntoBundle.tcl \
        $(SRCDIR)/InsertStringsIntoBundle.tcl \
        $(SRCDIR)/ReportBundleInfo.tcl \
        $(SRCDIR)/GetBundlesFromBundle.tcl \
        $(SRCDIR)/GetColorMapsFromBundle.tcl \
        $(SRCDIR)/GetColorMap2sFromBundle.tcl \
        $(SRCDIR)/GetFieldsFromBundle.tcl \
        $(SRCDIR)/GetMatricesFromBundle.tcl \
        $(SRCDIR)/GetNrrdsFromBundle.tcl \
        $(SRCDIR)/GetPathsFromBundle.tcl \
        $(SRCDIR)/GetStringsFromBundle.tcl \
        $(SRCDIR)/ReadBundle.tcl \
        $(SRCDIR)/WriteBundle.tcl \
        $(SRCDIR)/ShowAndEditCameraWidget.tcl \
        $(SRCDIR)/ConvertTVtoMLV.tcl \
        $(SRCDIR)/ConvertMatrixType.tcl \
        $(SRCDIR)/ConvertMeshCoordinateSystem.tcl \
        $(SRCDIR)/ConvertFieldBasis.tcl \
        $(SRCDIR)/ConvertFieldDataType.tcl \
        $(SRCDIR)/EditMeshBoundingBox.tcl \
        $(SRCDIR)/SetTetVolFieldDataValues.tcl \
        $(SRCDIR)/SetFieldDataValues.tcl \
        $(SRCDIR)/ChooseModule.tcl \
        $(SRCDIR)/ChooseColorMap.tcl \
        $(SRCDIR)/ChooseField.tcl \
        $(SRCDIR)/ChooseMatrix.tcl \
        $(SRCDIR)/ClipFieldByFunction.tcl \
        $(SRCDIR)/ClipFieldToFieldOrWidget.tcl \
        $(SRCDIR)/ClipLatVolByIndicesOrWidget.tcl \
        $(SRCDIR)/ReadColorMap2.tcl \
        $(SRCDIR)/WriteColorMap2.tcl \
        $(SRCDIR)/ReadColorMap.tcl \
        $(SRCDIR)/WriteColorMap.tcl \
        $(SRCDIR)/ComboListbox.tcl \
        $(SRCDIR)/ComponentWizard.tcl \
        $(SRCDIR)/Connection.tcl \
        $(SRCDIR)/CoregisterPointClouds.tcl \
        $(SRCDIR)/ConvertMatricesToMesh.tcl \
        $(SRCDIR)/CriticalPointWidget.tcl \
        $(SRCDIR)/CrosshairWidget.tcl \
        $(SRCDIR)/InterfaceWithCubit.tcl \
        $(SRCDIR)/MapFieldDataFromSourceToDestination.tcl \
        $(SRCDIR)/CreateAndEditColorMap.tcl \
        $(SRCDIR)/CreateAndEditCameraPath.tcl \
        $(SRCDIR)/CreateAndEditColorMap2D.tcl \
        $(SRCDIR)/ReportColumnMatrixMisfit.tcl \
				$(SRCDIR)/GetHexVolSheetBasedOnEdgeIndices.tcl \
        $(SRCDIR)/ExtractPlanarSliceFromField.tcl\
        $(SRCDIR)/ShowMeshBoundingBox.tcl\
        $(SRCDIR)/ReportFieldInfo.tcl\
        $(SRCDIR)/ReportFieldGeometryMeasures.tcl \
        $(SRCDIR)/ReadField.tcl \
				$(SRCDIR)/SetFieldProperty.tcl\
        $(SRCDIR)/GetSliceFromLatVol.tcl\
        $(SRCDIR)/ClipRasterFieldByIndices.tcl\
        $(SRCDIR)/WriteField.tcl \
        $(SRCDIR)/FrameWidget.tcl \
        $(SRCDIR)/JoinFields.tcl \
        $(SRCDIR)/GaugeWidget.tcl \
        $(SRCDIR)/CreateViewerAxes.tcl \
        $(SRCDIR)/CreateViewerClockIcon.tcl \
        $(SRCDIR)/CreateStandardColorMaps.tcl \
        $(SRCDIR)/CreateViewerCaption.tcl \
        $(SRCDIR)/ViewGraph.tcl \
        $(SRCDIR)/ToolTipText.tcl \
				$(SRCDIR)/InsertHexVolSheetFromTriSurf.tcl \
        $(SRCDIR)/ExtractIsosurface.tcl \
        $(SRCDIR)/ClipVolumeByIsovalue.tcl \
        $(SRCDIR)/ClipVolumeByIsovalueWithRefinement.tcl \
        $(SRCDIR)/LightWidget.tcl \
        $(SRCDIR)/EvaluateLinAlgBinary.tcl \
        $(SRCDIR)/EvaluateLinAlgUnary.tcl \
        $(SRCDIR)/EvaluateLinAlgGeneral.tcl \
        $(SRCDIR)/Linkedpane.tcl \
        $(SRCDIR)/SwapFieldDataWithMatrixEntries.tcl \
        $(SRCDIR)/MapFieldDataToNodeCoordinate.tcl \
        $(SRCDIR)/MaskLatVol.tcl \
        $(SRCDIR)/ReportMatrixInfo.tcl \
        $(SRCDIR)/ReadMatrix.tcl \
        $(SRCDIR)/GetColumnOrRowFromMatrix.tcl \
        $(SRCDIR)/WriteMatrix.tcl \
        $(SRCDIR)/Module.tcl \
        $(SRCDIR)/NetworkEditor.tcl \
        $(SRCDIR)/ConvertNrrdsToTexture.tcl \
        $(SRCDIR)/ReadPath.tcl \
        $(SRCDIR)/PathWidget.tcl \
        $(SRCDIR)/WritePath.tcl \
        $(SRCDIR)/TransformPlanarMesh.tcl \
        $(SRCDIR)/BuildPointCloudToLatVolMappingMatrix.tcl \
        $(SRCDIR)/PointWidget.tcl \
        $(SRCDIR)/Port.tcl \
        $(SRCDIR)/GenerateSinglePointProbeFromField.tcl \
        $(SRCDIR)/PromptedEntry.tcl \
        $(SRCDIR)/PromptedText.tcl \
        $(SRCDIR)/RefineTetVol.tcl \
        $(SRCDIR)/RescaleColorMap.tcl \
        $(SRCDIR)/RingWidget.tcl \
        $(SRCDIR)/GeneratePointSamplesFromFieldOrWidget.tcl \
        $(SRCDIR)/CreateLatVol.tcl \
        $(SRCDIR)/CreateImage.tcl \
        $(SRCDIR)/CreateStructHex.tcl \
        $(SRCDIR)/ReportScalarFieldStats.tcl \
        $(SRCDIR)/SciDialog.tcl \
        $(SRCDIR)/SciButtonPanel.tcl \
        $(SRCDIR)/SciMoveToCursor.tcl \
				$(SRCDIR)/GeneratePointSamplesFromField.tcl \
        $(SRCDIR)/SelectFieldROIWithBoxWidget.tcl \
        $(SRCDIR)/SetFieldOrMeshStringProperty.tcl \
        $(SRCDIR)/ShowColorMap.tcl \
        $(SRCDIR)/ShowField.tcl \
        $(SRCDIR)/ShowMatrix.tcl \
        $(SRCDIR)/SolveLinearSystem.tcl \
        $(SRCDIR)/StickyLocator.tcl \
        $(SRCDIR)/GenerateStreamLines.tcl \
        $(SRCDIR)/GetSubmatrix.tcl \
        $(SRCDIR)/Subnet.tcl \
        $(SRCDIR)/SynchronizeGeometry.tcl \
        $(SRCDIR)/ConvertFieldsToTexture.tcl \
        $(SRCDIR)/InterfaceWithTetGen.tcl \
        $(SRCDIR)/TimeControls.tcl \
        $(SRCDIR)/Tooltips.tcl \
        $(SRCDIR)/CalculateFieldData.tcl \
        $(SRCDIR)/CalculateFieldData3.tcl \
        $(SRCDIR)/CalculateFieldDataCompiled.tcl \
        $(SRCDIR)/CalculateFieldDataCompiled2.tcl \
        $(SRCDIR)/CalculateFieldDataCompiled3.tcl \
        $(SRCDIR)/TransformMeshWithFunction.tcl \
        $(SRCDIR)/UIvar.tcl \
        $(SRCDIR)/ViewScene.tcl \
        $(SRCDIR)/ViewSlices.tcl \
        $(SRCDIR)/ViewWidget.tcl \
        $(SRCDIR)/ShowTextureSlices.tcl \
        $(SRCDIR)/ShowTextureVolume.tcl \
				$(SRCDIR)/ReadString.tcl \
				$(SRCDIR)/WriteString.tcl \
				$(SRCDIR)/CreateString.tcl\
				$(SRCDIR)/ReportStringInfo.tcl\
				$(SRCDIR)/PrintMatrixIntoString.tcl\
				$(SRCDIR)/PrintStringIntoString.tcl\
				$(SRCDIR)/ShowString.tcl\
				$(SRCDIR)/CreateMatrix.tcl\
				$(SRCDIR)/GetFileName.tcl\
				$(SRCDIR)/GenerateStreamLinesWithPlacementHeuristic.tcl\
        $(SRCDIR)/ReportMatrixColumnMeasure.tcl\
        $(SRCDIR)/ReportMatrixRowMeasure.tcl\
        $(SRCDIR)/ResizeMatrix.tcl\
        $(SRCDIR)/StreamMatrixFromDisk.tcl\
        $(SRCDIR)/CreateDataArray.tcl\
        $(SRCDIR)/CalculateDataArray.tcl\
        $(SRCDIR)/ReplicateDataArray.tcl\
        $(SRCDIR)/ReportDataArrayMeasure.tcl\
        $(SRCDIR)/ReportDataArrayInfo.tcl\
        $(SRCDIR)/MapFieldDataFromElemToNode.tcl\
        $(SRCDIR)/MapFieldDataFromNodeToElem.tcl\
        $(SRCDIR)/SelectAndSetFieldData.tcl\
        $(SRCDIR)/SelectAndSetFieldData3.tcl\
        $(SRCDIR)/CalculateMeshNodes.tcl\
        $(SRCDIR)/CollectFields.tcl\
        $(SRCDIR)/GetDomainBoundary.tcl\
        $(SRCDIR)/ConvertNrrdToField.tcl\
        $(SRCDIR)/ConvertMatrixToField.tcl\
				$(SRCDIR)/CalculateIsInsideField.tcl\
				$(SRCDIR)/CalculateInsideWhichField.tcl\
				#[INSERT NEW TCL FILE HERE]

# MESQUITE Mesh Optimization Library
ifeq ($(HAVE_MESQUITE),yes)
   SRCS += $(SRCDIR)/SmoothMesh.tcl
endif

include $(SCIRUN_SCRIPTS)/tclIndex.mk

SCIRUN_MODULES := $(SCIRUN_MODULES) $(TCLINDEX)
