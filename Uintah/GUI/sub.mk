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
	$(SRCDIR)/test.tcl $(SRCDIR)/GridLines.tcl
	scripts/createTclIndex Uintah/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

#
# $Log$
# Revision 1.1  2000/03/17 09:29:56  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
