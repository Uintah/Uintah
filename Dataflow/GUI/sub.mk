# Makefile fragment for this subdirectory

SRCDIR := Dataflow/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/ArrowWidget.tcl \
	$(SRCDIR)/BaseWidget.tcl \
	$(SRCDIR)/BoxWidget.tcl \
	$(SRCDIR)/BuildTransform.tcl \
	$(SRCDIR)/CastField.tcl \
	$(SRCDIR)/ChangeCellType.tcl \
	$(SRCDIR)/ColorMapReader.tcl \
	$(SRCDIR)/ColorMapWriter.tcl \
	$(SRCDIR)/ComboListbox.tcl \
	$(SRCDIR)/ComponentWizard.tcl \
	$(SRCDIR)/CriticalPointWidget.tcl \
	$(SRCDIR)/CrosshairWidget.tcl \
	$(SRCDIR)/EditPath.tcl \
	$(SRCDIR)/ErrorMetric.tcl \
	$(SRCDIR)/FieldReader.tcl \
	$(SRCDIR)/FieldWriter.tcl \
	$(SRCDIR)/FieldSetReader.tcl \
	$(SRCDIR)/FieldSetWriter.tcl \
	$(SRCDIR)/FrameWidget.tcl \
	$(SRCDIR)/GLTextureBuilder.tcl \
	$(SRCDIR)/GaugeWidget.tcl \
	$(SRCDIR)/GenStandardColorMaps.tcl \
	$(SRCDIR)/GenTransferFunc.tcl \
	$(SRCDIR)/Isosurface.tcl \
	$(SRCDIR)/LightWidget.tcl \
	$(SRCDIR)/LocateNbrhd.tcl \
	$(SRCDIR)/MacroModule.tcl \
	$(SRCDIR)/ManageFieldSet.tcl \
	$(SRCDIR)/ManipFields.tcl \
        $(SRCDIR)/ManipMatrix.tcl \
	$(SRCDIR)/MatrixReader.tcl \
	$(SRCDIR)/MatrixWriter.tcl \
	$(SRCDIR)/Module.tcl \
	$(SRCDIR)/NetworkEditor.tcl \
	$(SRCDIR)/PathReader.tcl \
	$(SRCDIR)/PathWidget.tcl \
	$(SRCDIR)/PathWriter.tcl \
	$(SRCDIR)/PointWidget.tcl \
	$(SRCDIR)/PromptedText.tcl \
	$(SRCDIR)/PromptedEntry.tcl \
	$(SRCDIR)/RescaleColorMap.tcl \
	$(SRCDIR)/RingWidget.tcl \
	$(SRCDIR)/ScaledBoxWidget.tcl \
	$(SRCDIR)/ScaledFrameWidget.tcl \
	$(SRCDIR)/SeedField.tcl \
	$(SRCDIR)/ShowField.tcl \
	$(SRCDIR)/SolveMatrix.tcl \
	$(SRCDIR)/StreamLines.tcl \
	$(SRCDIR)/TexCuttingPlanes.tcl \
	$(SRCDIR)/TextureVolVis.tcl \
	$(SRCDIR)/ViewWidget.tcl \
	$(SRCDIR)/Viewer.tcl \
	$(SRCDIR)/TclStream.tcl \
	$(SRCDIR)/Rescale.tcl \

	$(SRCTOP)/scripts/createTclIndex $(SRCTOP)/Dataflow/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex
