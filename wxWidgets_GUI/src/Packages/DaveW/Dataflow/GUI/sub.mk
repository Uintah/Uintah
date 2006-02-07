# Makefile fragment for this subdirectory

SRCDIR := Packages/DaveW/Dataflow/GUI

SRCS := \
	$(SRCDIR)/BldBRDF.tcl \
	$(SRCDIR)/BldScene.tcl \
	$(SRCDIR)/Bundles.tcl \
	$(SRCDIR)/ContourSetReader.tcl \
	$(SRCDIR)/ContourSetWriter.tcl \
	$(SRCDIR)/ContoursToField.tcl \
	$(SRCDIR)/RTrace.tcl \
	$(SRCDIR)/Radiosity.tcl \
	$(SRCDIR)/RayMatrix.tcl \
	$(SRCDIR)/RayTest.tcl \
	$(SRCDIR)/STreeExtractSurf.tcl \
	$(SRCDIR)/SegFldReader.tcl \
	$(SRCDIR)/SegFldWriter.tcl \
	$(SRCDIR)/ShowContours.tcl \
	$(SRCDIR)/ShowTopoSurfTree.tcl \
	$(SRCDIR)/SiReAll.tcl \
	$(SRCDIR)/SiReCrunch.tcl \
	$(SRCDIR)/SiReInput.tcl \
	$(SRCDIR)/SiReOutput.tcl \
	$(SRCDIR)/SigmaSetReader.tcl \
	$(SRCDIR)/SigmaSetWriter.tcl \
	$(SRCDIR)/TensorFieldReader.tcl \
	$(SRCDIR)/TensorFieldWriter.tcl \
#	$(SRCDIR)/XYZtoRGB.tcl\
#[INSERT NEW TCL FILE HERE]

include $(SCIRUN_SCRIPTS)/tclIndex.mk


