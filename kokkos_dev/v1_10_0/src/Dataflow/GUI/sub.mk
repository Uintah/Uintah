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

SRCDIR := Dataflow/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/AppendMatrix.tcl \
	$(SRCDIR)/ArrowWidget.tcl \
	$(SRCDIR)/BaseWidget.tcl \
	$(SRCDIR)/BuildInterpolant.tcl \
	$(SRCDIR)/BuildNoise.tcl \
	$(SRCDIR)/BuildTransform.tcl \
	$(SRCDIR)/BoxWidget.tcl \
	$(SRCDIR)/CastTVtoMLV.tcl \
	$(SRCDIR)/CastMatrix.tcl \
	$(SRCDIR)/ChangeCoordinates.tcl \
	$(SRCDIR)/ChangeFieldDataAt.tcl \
	$(SRCDIR)/ChangeFieldDataType.tcl \
	$(SRCDIR)/ChangeFieldBounds.tcl \
	$(SRCDIR)/ClipByFunction.tcl \
	$(SRCDIR)/ClipField.tcl \
	$(SRCDIR)/ClipLattice.tcl \
	$(SRCDIR)/ColorMapReader.tcl \
	$(SRCDIR)/ColorMapWriter.tcl \
	$(SRCDIR)/ComboListbox.tcl \
	$(SRCDIR)/ComponentWizard.tcl \
	$(SRCDIR)/Coregister.tcl \
	$(SRCDIR)/CriticalPointWidget.tcl \
	$(SRCDIR)/CrosshairWidget.tcl \
	$(SRCDIR)/DirectInterpolate.tcl \
	$(SRCDIR)/EditPath.tcl \
	$(SRCDIR)/ErrorMetric.tcl \
	$(SRCDIR)/FieldInfo.tcl\
	$(SRCDIR)/FieldMeasures.tcl \
	$(SRCDIR)/FieldReader.tcl \
	$(SRCDIR)/FieldSlicer.tcl\
	$(SRCDIR)/FieldSubSample.tcl\
	$(SRCDIR)/FieldWriter.tcl \
	$(SRCDIR)/FrameWidget.tcl \
	$(SRCDIR)/GLTextureBuilder.tcl \
	$(SRCDIR)/GaugeWidget.tcl \
	$(SRCDIR)/GenStandardColorMaps.tcl \
	$(SRCDIR)/GenTransferFunc.tcl \
	$(SRCDIR)/Isosurface.tcl \
	$(SRCDIR)/LightWidget.tcl \
	$(SRCDIR)/LinAlgBinary.tcl \
	$(SRCDIR)/LinAlgUnary.tcl \
	$(SRCDIR)/MacroModule.tcl \
	$(SRCDIR)/MapDataToMeshCoord.tcl \
	$(SRCDIR)/MaskLattice.tcl \
	$(SRCDIR)/MatrixReader.tcl \
	$(SRCDIR)/MatrixSelectVector.tcl \
	$(SRCDIR)/MatrixWriter.tcl \
	$(SRCDIR)/Module.tcl \
	$(SRCDIR)/NetworkEditor.tcl \
	$(SRCDIR)/PathReader.tcl \
	$(SRCDIR)/PathWidget.tcl \
	$(SRCDIR)/PathWriter.tcl \
	$(SRCDIR)/PointWidget.tcl \
	$(SRCDIR)/Probe.tcl \
	$(SRCDIR)/PromptedText.tcl \
	$(SRCDIR)/PromptedEntry.tcl \
	$(SRCDIR)/ReplaceScalarDataValue.tcl \
	$(SRCDIR)/RescaleColorMap.tcl \
	$(SRCDIR)/RingWidget.tcl \
	$(SRCDIR)/SampleField.tcl \
	$(SRCDIR)/SamplePlane.tcl \
	$(SRCDIR)/SampleLattice.tcl \
	$(SRCDIR)/ScalarFieldStats.tcl \
	$(SRCDIR)/SetProperty.tcl \
	$(SRCDIR)/ShowField.tcl \
	$(SRCDIR)/SolveMatrix.tcl \
	$(SRCDIR)/StreamLines.tcl \
	$(SRCDIR)/Submatrix.tcl \
	$(SRCDIR)/SynchronizeGeometry.tcl \
	$(SRCDIR)/TexCuttingPlanes.tcl \
	$(SRCDIR)/TextureVolVis.tcl \
	$(SRCDIR)/TransformScalarData.tcl \
	$(SRCDIR)/TransformVectorData.tcl \
	$(SRCDIR)/ViewWidget.tcl \
	$(SRCDIR)/Viewer.tcl \
	$(SRCDIR)/TclStream.tcl \

	$(OBJTOP)/createTclIndex $(SRCTOP)/Dataflow/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex
