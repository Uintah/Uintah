#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := DaveW/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/Anneal.tcl $(SRCDIR)/BldBRDF.tcl \
	$(SRCDIR)/BldEEGMesh.tcl $(SRCDIR)/BldScene.tcl \
	$(SRCDIR)/Bundles.tcl $(SRCDIR)/ContourSetReader.tcl \
	$(SRCDIR)/ContourSetWriter.tcl $(SRCDIR)/Coregister.tcl \
	$(SRCDIR)/DipoleInSphere.tcl $(SRCDIR)/DipoleMatToGeom.tcl \
	$(SRCDIR)/DipoleSourceRHS.tcl \
	$(SRCDIR)/Downhill_Simplex3.tcl $(SRCDIR)/ErrorMetric.tcl \
	$(SRCDIR)/InvEEGSolve.tcl $(SRCDIR)/OptDip.tcl \
	$(SRCDIR)/RTrace.tcl $(SRCDIR)/Radiosity.tcl \
	$(SRCDIR)/RayMatrix.tcl $(SRCDIR)/RayTest.tcl \
	$(SRCDIR)/RescaleSegFld.tcl $(SRCDIR)/SGI_LU.tcl \
	$(SRCDIR)/SGI_Solve.tcl $(SRCDIR)/STreeExtractSurf.tcl \
	$(SRCDIR)/SeedDipoles2.tcl \
	$(SRCDIR)/SegFldOps.tcl $(SRCDIR)/SegFldReader.tcl \
	$(SRCDIR)/SegFldWriter.tcl $(SRCDIR)/SelectSurfNodes.tcl \
	$(SRCDIR)/SiReAll.tcl $(SRCDIR)/SiReCrunch.tcl \
	$(SRCDIR)/SiReInput.tcl $(SRCDIR)/SiReOutput.tcl \
	$(SRCDIR)/SigmaSetReader.tcl $(SRCDIR)/SigmaSetWriter.tcl \
	$(SRCDIR)/SurfToVectGeom.tcl $(SRCDIR)/Taubin.tcl \
	$(SRCDIR)/TensorFieldReader.tcl $(SRCDIR)/TensorFieldWriter.tcl \
	$(SRCDIR)/Thermal.tcl $(SRCDIR)/TopoSurfToGeom.tcl \
	$(SRCDIR)/XYZtoRGB.tcl\
#[INSERT NEW TCL FILE HERE]
	$(SRCTOP)/scripts/createTclIndex $(SRCTOP)/DaveW/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

#
# $Log$
# Revision 1.2.2.4  2000/11/01 23:02:19  mcole
# Fix for previous merge from trunk
#
# Revision 1.2.2.2  2000/10/26 14:01:53  moulding
# merge HEAD into FIELD_REDESIGN
#
# Revision 1.4  2000/10/29 03:47:58  dmw
# new GUIs
#
# Revision 1.3  2000/10/24 05:57:07  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.2  2000/03/20 19:36:00  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:25:25  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
