#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := DaveW/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: $(SRCDIR)/Anneal.tcl $(SRCDIR)/BldBRDF.tcl \
	$(SRCDIR)/BldEEGMesh.tcl $(SRCDIR)/BldScene.tcl \
	$(SRCDIR)/Bundles.tcl $(SRCDIR)/ContourSetReader.tcl \
	$(SRCDIR)/ContourSetWriter.tcl $(SRCDIR)/Coregister.tcl \
	$(SRCDIR)/DipoleInSphere.tcl $(SRCDIR)/DipoleMatToGeom.tcl \
	$(SRCDIR)/Downhill_Simplex.tcl $(SRCDIR)/ErrorMetric.tcl \
	$(SRCDIR)/InvEEGSolve.tcl $(SRCDIR)/OptDip.tcl \
	$(SRCDIR)/RTrace.tcl $(SRCDIR)/Radiosity.tcl \
	$(SRCDIR)/RayMatrix.tcl $(SRCDIR)/RayTest.tcl \
	$(SRCDIR)/RescaleSegFld.tcl $(SRCDIR)/SGI_LU.tcl \
	$(SRCDIR)/SGI_Solve.tcl $(SRCDIR)/STreeExtractSurf.tcl \
	$(SRCDIR)/SegFldOps.tcl $(SRCDIR)/SegFldReader.tcl \
	$(SRCDIR)/SegFldWriter.tcl $(SRCDIR)/SelectSurfNodes.tcl \
	$(SRCDIR)/SiReAll.tcl $(SRCDIR)/SiReCrunch.tcl \
	$(SRCDIR)/SiReInput.tcl $(SRCDIR)/SiReOutput.tcl \
	$(SRCDIR)/SigmaSetReader.tcl $(SRCDIR)/SigmaSetWriter.tcl \
	$(SRCDIR)/SurfToVectGeom.tcl $(SRCDIR)/Taubin.tcl \
	$(SRCDIR)/TensorFieldReader.tcl $(SRCDIR)/TensorFieldWriter.tcl \
	$(SRCDIR)/Thermal.tcl $(SRCDIR)/TopoSurfToGeom.tcl \
	$(SRCDIR)/XYZtoRGB.tcl
	scripts/createTclIndex DaveW/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

#
# $Log$
# Revision 1.1  2000/03/17 09:25:25  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
