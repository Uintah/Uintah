#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := SCIRun/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: $(SRCDIR)/Binop.tcl $(SRCDIR)/Delaunay.tcl \
	$(SRCDIR)/Edge.tcl $(SRCDIR)/Gauss.tcl \
	$(SRCDIR)/ImageConvolve.tcl $(SRCDIR)/ImageGen.tcl \
	$(SRCDIR)/ImageSel.tcl $(SRCDIR)/MeshInterpVals.tcl \
	$(SRCDIR)/MeshToGeom.tcl $(SRCDIR)/MeshView.tcl \
	$(SRCDIR)/Noise.tcl $(SRCDIR)/Radon.tcl \
	$(SRCDIR)/Segment.tcl $(SRCDIR)/Sharpen.tcl \
	$(SRCDIR)/Snakes.tcl $(SRCDIR)/Subsample.tcl $(SRCDIR)/Ted.tcl \
	$(SRCDIR)/Threshold.tcl $(SRCDIR)/TiffReader.tcl \
	$(SRCDIR)/TiffWriter.tcl $(SRCDIR)/Transforms.tcl $(SRCDIR)/Unop.tcl
	scripts/createTclIndex SCIRun/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

#
# $Log$
# Revision 1.1  2000/03/17 09:28:58  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
