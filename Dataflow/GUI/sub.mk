# Makefile fragment for this subdirectory

SRCDIR := Dataflow/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/BldTransform.tcl \
	$(SRCDIR)/CastField.tcl \
	$(SRCDIR)/ChangeCellType.tcl \
	$(SRCDIR)/ColorMapReader.tcl \
	$(SRCDIR)/ColorMapWriter.tcl \
	$(SRCDIR)/ComponentWizard.tcl \
	$(SRCDIR)/EditPath.tcl \
	$(SRCDIR)/ErrorMetric.tcl \
	$(SRCDIR)/FieldReader.tcl \
	$(SRCDIR)/FieldWriter.tcl \
	$(SRCDIR)/GLTextureBuilder.tcl \
	$(SRCDIR)/GenStandardColorMaps.tcl \
	$(SRCDIR)/GenTransferFunc.tcl \
	$(SRCDIR)/IsoSurface.tcl \
	$(SRCDIR)/LocateNbrhd.tcl \
	$(SRCDIR)/MacroModule.tcl \
	$(SRCDIR)/ManipFields.tcl \
	$(SRCDIR)/MatrixReader.tcl \
	$(SRCDIR)/MatrixWriter.tcl \
	$(SRCDIR)/Module.tcl \
	$(SRCDIR)/NetworkEditor.tcl \
	$(SRCDIR)/PathReader.tcl \
	$(SRCDIR)/PathWriter.tcl \
	$(SRCDIR)/RescaleColorMap.tcl \
	$(SRCDIR)/SeedField.tcl \
	$(SRCDIR)/ShowField.tcl \
	$(SRCDIR)/SolveMatrix.tcl \
	$(SRCDIR)/Streamline.tcl \
	$(SRCDIR)/TexCuttingPlanes.tcl \
	$(SRCDIR)/TextureVolVis.tcl \
	$(SRCDIR)/Viewer.tcl \
	$(SRCDIR)/TclStream.tcl \
	$(SRCDIR)/Rescale.tcl \

	$(SRCTOP)/scripts/createTclIndex $(SRCTOP)/Dataflow/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex
