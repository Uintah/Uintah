#
# Makefile fragment for this subdirectory
#

# *** NOTE ***
# 
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCIRun/Modules/Mesh

SRCS     += \
	$(SRCDIR)/Delaunay.cc\
	$(SRCDIR)/ExtractMesh.cc\
	$(SRCDIR)/HexMeshToGeom.cc\
	$(SRCDIR)/InsertDelaunay.cc\
	$(SRCDIR)/MakeScalarField.cc\
	$(SRCDIR)/MeshBoundary.cc\
	$(SRCDIR)/MeshFindSurfNodes.cc\
	$(SRCDIR)/MeshInterpVals.cc\
	$(SRCDIR)/MeshNodeComponent.cc\
	$(SRCDIR)/MeshRender.cc\
	$(SRCDIR)/MeshToGeom.cc\
	$(SRCDIR)/MeshView.cc\
	$(SRCDIR)/TransformMesh.cc
#[INSERT NEW MODULE HERE]

PSELIBS := PSECore/Dataflow PSECore/Datatypes PSECore/Widgets \
	SCICore/Geom SCICore/Thread SCICore/Exceptions \
	SCICore/Containers SCICore/Geometry SCICore/Datatypes \
	SCICore/Persistent SCICore/TclInterface SCICore/Math
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk
