#
# Makefile fragment for this subdirectory
#

SRCDIR := SCIRun/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/Binop.tcl $(SRCDIR)/Delaunay.tcl \
	$(SRCDIR)/Edge.tcl $(SRCDIR)/Gauss.tcl \
	$(SRCDIR)/ImageConvolve.tcl $(SRCDIR)/ImageGen.tcl \
	$(SRCDIR)/ImageSel.tcl $(SRCDIR)/MeshInterpVals.tcl \
	$(SRCDIR)/MeshToGeom.tcl $(SRCDIR)/MeshView.tcl \
	$(SRCDIR)/Noise.tcl $(SRCDIR)/Radon.tcl \
	$(SRCDIR)/Segment.tcl $(SRCDIR)/Sharpen.tcl \
	$(SRCDIR)/Snakes.tcl $(SRCDIR)/Subsample.tcl $(SRCDIR)/Ted.tcl \
	$(SRCDIR)/Threshold.tcl $(SRCDIR)/TiffReader.tcl \
	$(SRCDIR)/TiffWriter.tcl $(SRCDIR)/Transforms.tcl $(SRCDIR)/Unop.tcl\
#[INSERT NEW TCL FILE HERE]
	$(SRCTOP)/scripts/createTclIndex $(SRCTOP)/SCIRun/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex
