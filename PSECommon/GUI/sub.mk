#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := PSECommon/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: $(SRCDIR)/AddWells.tcl $(SRCDIR)/AddWells2.tcl \
	$(SRCDIR)/BitVisualize.tcl $(SRCDIR)/BldTransform.tcl \
	$(SRCDIR)/BoundGrid.tcl $(SRCDIR)/BoxClipSField.tcl \
	$(SRCDIR)/BuildFEMatrix.tcl $(SRCDIR)/ClipField.tcl \
	$(SRCDIR)/ColorMapReader.tcl $(SRCDIR)/ColorMapWriter.tcl \
	$(SRCDIR)/ColumnMatrixReader.tcl $(SRCDIR)/ColumnMatrixWriter.tcl \
	$(SRCDIR)/CuttingPlane.tcl $(SRCDIR)/CuttingPlaneTex.tcl \
	$(SRCDIR)/Downsample.tcl $(SRCDIR)/DukeRawRead.tcl \
	$(SRCDIR)/EditMatrix.tcl $(SRCDIR)/ErrorInterval.tcl \
	$(SRCDIR)/ExtractSubmatrix.tcl $(SRCDIR)/FieldCage.tcl \
	$(SRCDIR)/FieldFilter.tcl $(SRCDIR)/FieldGainCorrect.tcl \
	$(SRCDIR)/FieldMedianFilter.tcl $(SRCDIR)/FieldRGAug.tcl \
	$(SRCDIR)/FieldSeed.tcl $(SRCDIR)/GenAxes.tcl \
	$(SRCDIR)/GenStandardColorMaps.tcl $(SRCDIR)/GenSurface.tcl \
	$(SRCDIR)/GenTransferFunc.tcl $(SRCDIR)/GeomReader.tcl \
	$(SRCDIR)/GeometryReader.tcl $(SRCDIR)/GeometryWriter.tcl \
	$(SRCDIR)/Hedgehog.tcl $(SRCDIR)/HedgehogLitLines.tcl \
	$(SRCDIR)/ImageReader.tcl $(SRCDIR)/IsoMask.tcl \
	$(SRCDIR)/IsoSurface.tcl $(SRCDIR)/IsoSurfaceDW.tcl \
	$(SRCDIR)/IsoSurfaceMRSG.tcl $(SRCDIR)/IsoSurfaceSP.tcl \
	$(SRCDIR)/LabelSurface.tcl $(SRCDIR)/MatMat.tcl \
	$(SRCDIR)/MatSelectVec.tcl $(SRCDIR)/MatVec.tcl \
	$(SRCDIR)/MatrixReader.tcl $(SRCDIR)/MatrixWriter.tcl \
	$(SRCDIR)/MeshReader.tcl $(SRCDIR)/MeshWriter.tcl \
	$(SRCDIR)/MultiSFRGReader.tcl $(SRCDIR)/MultiScalarFieldWriter.tcl \
	$(SRCDIR)/OpenGL_Ex.tcl $(SRCDIR)/PointsReader.tcl \
	$(SRCDIR)/Reader.tcl $(SRCDIR)/RescaleColorMap.tcl \
	$(SRCDIR)/Salmon.tcl $(SRCDIR)/ScalarFieldReader.tcl \
	$(SRCDIR)/ScalarFieldWriter.tcl $(SRCDIR)/SimpSurface.tcl \
	$(SRCDIR)/SimpVolVis.tcl $(SRCDIR)/SolveMatrix.tcl \
	$(SRCDIR)/Streamline.tcl $(SRCDIR)/SurfGen.tcl \
	$(SRCDIR)/SurfInterpVals.tcl $(SRCDIR)/SurfToGeom.tcl \
	$(SRCDIR)/SurfaceReader.tcl $(SRCDIR)/SurfaceWriter.tcl \
	$(SRCDIR)/TetraWriter.tcl $(SRCDIR)/TiffWriter.tcl \
	$(SRCDIR)/TracePath.tcl $(SRCDIR)/TrainSeg2.tcl \
	$(SRCDIR)/TrainSegment.tcl $(SRCDIR)/TranslateSurface.tcl \
	$(SRCDIR)/VectorFieldReader.tcl $(SRCDIR)/VectorFieldWriter.tcl \
	$(SRCDIR)/VectorSeg.tcl $(SRCDIR)/VisualizeMatrix.tcl \
	$(SRCDIR)/VoidStarReader.tcl $(SRCDIR)/VoidStarWriter.tcl \
	$(SRCDIR)/VolRendTexSlices.tcl $(SRCDIR)/VolVis.tcl \
	$(SRCDIR)/WidgetTest.tcl $(SRCDIR)/Writer.tcl \
	$(SRCDIR)/cConjGrad.tcl $(SRCDIR)/cPhase.tcl
	scripts/createTclIndex PSECommon/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

#
# $Log$
# Revision 1.1  2000/03/17 09:26:45  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
