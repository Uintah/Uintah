#
# Makefile fragment for this subdirectory
# $Id$
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

#
# $Log$
# Revision 1.3  2000/10/24 05:57:50  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.2  2000/03/20 19:38:09  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:58  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
