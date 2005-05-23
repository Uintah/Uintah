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

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the Core/CCA/Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Core/CCA/Component"
# documentation on how to do it correctly.

# Makefile fragment for this subdirectory

SRCDIR := Dataflow/GUI

SRCS := \
        $(SRCDIR)/AddLight.tcl \
        $(SRCDIR)/AppendMatrix.tcl \
        $(SRCDIR)/ArrowWidget.tcl \
        $(SRCDIR)/BaseWidget.tcl \
        $(SRCDIR)/BoxWidget.tcl \
        $(SRCDIR)/BuildMappingMatrix.tcl \
        $(SRCDIR)/BuildNoise.tcl \
        $(SRCDIR)/BuildTransform.tcl \
        $(SRCDIR)/BundleSetBundle.tcl \
        $(SRCDIR)/BundleSetColorMap.tcl \
        $(SRCDIR)/BundleSetColorMap2.tcl \
        $(SRCDIR)/BundleSetField.tcl \
        $(SRCDIR)/BundleSetMatrix.tcl \
        $(SRCDIR)/BundleSetNrrd.tcl \
        $(SRCDIR)/BundleSetPath.tcl \
        $(SRCDIR)/BundleInfo.tcl \
        $(SRCDIR)/BundleGetBundle.tcl \
        $(SRCDIR)/BundleGetColorMap.tcl \
        $(SRCDIR)/BundleGetColorMap2.tcl \
        $(SRCDIR)/BundleGetField.tcl \
        $(SRCDIR)/BundleGetMatrix.tcl \
        $(SRCDIR)/BundleGetNrrd.tcl \
        $(SRCDIR)/BundleGetPath.tcl \
        $(SRCDIR)/BundleReader.tcl \
        $(SRCDIR)/BundleWriter.tcl \
        $(SRCDIR)/Camera.tcl \
        $(SRCDIR)/CastTVtoMLV.tcl \
        $(SRCDIR)/CastMatrix.tcl \
        $(SRCDIR)/ChangeCoordinates.tcl \
        $(SRCDIR)/ChangeFieldBasis.tcl \
        $(SRCDIR)/ChangeFieldDataType.tcl \
        $(SRCDIR)/ChangeFieldBounds.tcl \
        $(SRCDIR)/ChooseColorMap.tcl \
        $(SRCDIR)/ChooseField.tcl \
        $(SRCDIR)/ChooseMatrix.tcl \
        $(SRCDIR)/ClipByFunction.tcl \
        $(SRCDIR)/ClipField.tcl \
        $(SRCDIR)/ClipLattice.tcl \
        $(SRCDIR)/ColorMap2Reader.tcl \
        $(SRCDIR)/ColorMap2Writer.tcl \
        $(SRCDIR)/ColorMapReader.tcl \
        $(SRCDIR)/ColorMapWriter.tcl \
        $(SRCDIR)/ComboListbox.tcl \
        $(SRCDIR)/ComponentWizard.tcl \
        $(SRCDIR)/Connection.tcl \
        $(SRCDIR)/Coregister.tcl \
        $(SRCDIR)/CreateMesh.tcl \
        $(SRCDIR)/CriticalPointWidget.tcl \
        $(SRCDIR)/CrosshairWidget.tcl \
        $(SRCDIR)/CubitInterface.tcl \
        $(SRCDIR)/DirectMapping.tcl \
        $(SRCDIR)/EditColorMap.tcl \
        $(SRCDIR)/EditPath.tcl \
        $(SRCDIR)/EditColorMap2D.tcl \
        $(SRCDIR)/ErrorMetric.tcl \
        $(SRCDIR)/FieldCage.tcl\
        $(SRCDIR)/FieldFrenet.tcl\
        $(SRCDIR)/FieldInfo.tcl\
        $(SRCDIR)/FieldMeasures.tcl \
        $(SRCDIR)/FieldReader.tcl \
        $(SRCDIR)/FieldSlicer.tcl\
        $(SRCDIR)/FieldSubSample.tcl\
        $(SRCDIR)/FieldWriter.tcl \
        $(SRCDIR)/FrameWidget.tcl \
        $(SRCDIR)/GatherFields.tcl \
        $(SRCDIR)/GaugeWidget.tcl \
        $(SRCDIR)/GenAxes.tcl \
        $(SRCDIR)/GenClock.tcl \
        $(SRCDIR)/GenStandardColorMaps.tcl \
        $(SRCDIR)/GenTitle.tcl \
        $(SRCDIR)/ToolTipText.tcl \
        $(SRCDIR)/Isosurface.tcl \
        $(SRCDIR)/IsoClip.tcl \
        $(SRCDIR)/LightWidget.tcl \
        $(SRCDIR)/LinAlgBinary.tcl \
        $(SRCDIR)/LinAlgUnary.tcl \
        $(SRCDIR)/LinearAlgebra.tcl \
        $(SRCDIR)/ManageFieldData.tcl \
        $(SRCDIR)/MapDataToMeshCoord.tcl \
        $(SRCDIR)/MaskLattice.tcl \
        $(SRCDIR)/MatrixInfo.tcl \
        $(SRCDIR)/MatrixReader.tcl \
        $(SRCDIR)/MatrixSelectVector.tcl \
        $(SRCDIR)/MatrixWriter.tcl \
        $(SRCDIR)/Module.tcl \
        $(SRCDIR)/NetworkEditor.tcl \
        $(SRCDIR)/NrrdTextureBuilder.tcl \
        $(SRCDIR)/PathReader.tcl \
        $(SRCDIR)/PathWidget.tcl \
        $(SRCDIR)/PathWriter.tcl \
        $(SRCDIR)/PlanarTransformField.tcl \
        $(SRCDIR)/PointLatticeMap.tcl \
        $(SRCDIR)/PointWidget.tcl \
        $(SRCDIR)/Port.tcl \
        $(SRCDIR)/Probe.tcl \
        $(SRCDIR)/PromptedEntry.tcl \
        $(SRCDIR)/PromptedText.tcl \
        $(SRCDIR)/RefineTetVol.tcl \
        $(SRCDIR)/ReplaceScalarDataValue.tcl \
        $(SRCDIR)/RescaleColorMap.tcl \
        $(SRCDIR)/RingWidget.tcl \
        $(SRCDIR)/SampleField.tcl \
        $(SRCDIR)/SampleLattice.tcl \
        $(SRCDIR)/SamplePlane.tcl \
        $(SRCDIR)/SampleStructHex.tcl \
        $(SRCDIR)/ScalarFieldStats.tcl \
        $(SRCDIR)/SciDialog.tcl \
        $(SRCDIR)/SciButtonPanel.tcl \
        $(SRCDIR)/SciMoveToCursor.tcl \
        $(SRCDIR)/SelectField.tcl \
        $(SRCDIR)/SetProperty.tcl \
        $(SRCDIR)/ShowColorMap.tcl \
        $(SRCDIR)/ShowField.tcl \
        $(SRCDIR)/ShowMatrix.tcl \
        $(SRCDIR)/SolveMatrix.tcl \
        $(SRCDIR)/StreamLines.tcl \
        $(SRCDIR)/Submatrix.tcl \
        $(SRCDIR)/Subnet.tcl \
        $(SRCDIR)/SynchronizeGeometry.tcl \
        $(SRCDIR)/TextureBuilder.tcl \
        $(SRCDIR)/TimeControls.tcl \
        $(SRCDIR)/Tooltips.tcl \
        $(SRCDIR)/TransformData.tcl \
        $(SRCDIR)/TransformData2.tcl \
        $(SRCDIR)/TransformData3.tcl \
        $(SRCDIR)/TransformMesh.tcl \
        $(SRCDIR)/UIvar.tcl \
        $(SRCDIR)/Viewer.tcl \
        $(SRCDIR)/ViewSlices.tcl \
        $(SRCDIR)/ViewWidget.tcl \
        $(SRCDIR)/VolumeSlicer.tcl \
        $(SRCDIR)/VolumeVisualizer.tcl \
#[INSERT NEW TCL FILE HERE]

include $(SCIRUN_SCRIPTS)/tclIndex.mk

SCIRUN_MODULES := $(SCIRUN_MODULES) $(TCLINDEX)
