# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/DaveW/Dataflow/Modules/EEG

SRCS     += \
	$(SRCDIR)/BldEEGMesh.cc\
	$(SRCDIR)/Coregister.cc\
	$(SRCDIR)/RescaleSegFld.cc\
	$(SRCDIR)/SFRGtoSFUG.cc\
	$(SRCDIR)/STreeExtractSurf.cc\
	$(SRCDIR)/SegFldOps.cc\
	$(SRCDIR)/SegFldToSurfTree.cc\
	$(SRCDIR)/SelectSurfNodes.cc\
	$(SRCDIR)/Taubin.cc\
	$(SRCDIR)/Thermal.cc\
	$(SRCDIR)/TopoSurfToGeom.cc\
	$(SRCDIR)/SliceMaker.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Packages/DaveW/Core/Datatypes/General \
	Dataflow/Ports Dataflow/Widgets \
	Dataflow/Network Core/Persistent Core/Exceptions \
	Core/Geom Core/Thread Core/Geometry Core/Math \
	Core/TclInterface Core/Datatypes Core/Containers
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

