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
	$(SRCDIR)/ScalarFieldExtractor.tcl $(SRCDIR)/TimestepSelector.tcl \
	$(SRCDIR)/VectorFieldExtractor.tcl \
	$(SRCDIR)/TensorFieldExtractor.tcl \
	$(SRCDIR)/ParticleFieldExtractor.tcl \
	$(SRCDIR)/RescaleColorMapForParticles.tcl $(SRCDIR)/ParticleVis.tcl \
	$(SRCDIR)/NodeHedgehog.tcl $(SRCDIR)/ArchiveReader.tcl \
	$(SRCDIR)/GridVisualizer.tcl $(SRCDIR)/EigenEvaluator.tcl \
	$(SRCDIR)/ParticleEigenEvaluator.tcl \
	$(SRCDIR)/InPlaneEigenEvaluator.tcl \
	$(SRCDIR)/TensorElementExtractor.tcl
	$(SRCTOP)/scripts/createTclIndex $(SRCTOP)/Uintah/GUI


CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

#
# $Log$
# Revision 1.10.4.1  2000/10/19 05:18:00  sparker
# Merge changes from main branch into csafe_risky1
#
# Revision 1.11  2000/10/17 22:49:35  witzel
# Added ParticleEigenEvaluator.tcl
#
# Revision 1.10  2000/09/20 23:44:50  witzel
# Added TensorElementExtractor.tcl
#
# Revision 1.9  2000/08/28 22:41:53  bigler
# Fixed a typo.  Extra "\" at the end of the dependancies list for tclIndex.
#
# Revision 1.8  2000/08/25 17:29:47  witzel
# Added InPlaneEigenEvaluator.tcl
#
# Revision 1.7  2000/08/22 22:24:53  witzel
# Added EigenEvaluator.tcl
#
# Revision 1.6  2000/08/14 17:31:15  bigler
# Added GridVisualizer UI code
#
# Revision 1.5  2000/07/31 17:45:43  kuzimmer
# Added files and modules for Field Extraction from uda
#
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
