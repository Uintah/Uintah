#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := Uintah/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: $(SRCDIR)/ChemVis.tcl $(SRCDIR)/PartToGeom.tcl \
	$(SRCDIR)/ParticleGridVisControl.tcl $(SRCDIR)/ParticleViz.tcl \
	$(SRCDIR)/RescaleParticleColorMap.tcl \
	$(SRCDIR)/TecplotFileSelector.tcl $(SRCDIR)/TriangleReader.tcl \
	$(SRCDIR)/application.tcl $(SRCDIR)/cfdGridLines.tcl \
	$(SRCDIR)/test.tcl $(SRCDIR)/GridLines.tcl \
	$(SRCDIR)/NodeHedgehog.tcl $(SRCDIR)/ArchiveReader.tcl
	$(SRCTOP)/scripts/createTclIndex $(SRCTOP)/Uintah/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

#
# $Log$
# Revision 1.4  2000/06/27 16:56:58  bigler
# Added Nodehedgehog.tcl
#
# Revision 1.3  2000/06/22 17:44:07  kuzimmer
# removed commented out code and modified the sub.mk so that the GUI
# comes up.
#
# Revision 1.2  2000/03/20 19:38:34  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:29:56  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
