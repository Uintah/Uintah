# Makefile fragment for this subdirectory

SRCDIR := Packages/DaveW/Dataflow/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/Anneal.tcl $(SRCDIR)/BldBRDF.tcl \
	$(SRCDIR)/BldEEGMesh.tcl $(SRCDIR)/BldScene.tcl \
	$(SRCDIR)/Bundles.tcl $(SRCDIR)/ConductivitySearch.tcl \
	$(SRCDIR)/ContourSetReader.tcl \
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
	$(SRCTOP)/scripts/createTclIndex $(SRCTOP)/Packages/DaveW/Dataflow/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

