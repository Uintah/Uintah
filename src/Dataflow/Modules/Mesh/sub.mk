#
# Makefile fragment for this subdirectory
# $Id$
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
	$(SRCDIR)/MeshInterpVals.cc\
	$(SRCDIR)/MeshRender.cc\
	$(SRCDIR)/MeshToGeom.cc\
	$(SRCDIR)/SliceMeshToGeom.cc\
	$(SRCDIR)/MeshView.cc\
	$(SRCDIR)/SliceMeshToGeom.cc\
#[INSERT NEW MODULE HERE]

PSELIBS := PSECore/Dataflow PSECore/Datatypes PSECore/Widgets \
	SCICore/Geom SCICore/Thread SCICore/Exceptions \
	SCICore/Containers SCICore/Geometry SCICore/Datatypes \
	SCICore/Persistent SCICore/TclInterface SCICore/Math
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.5  2000/09/07 00:12:19  zyp
# MakeScalarField.cc
#
# Revision 1.4  2000/06/08 22:46:36  moulding
# Added a comment note about not messing with the module maker comment lines,
# and how to edit this file by hand.
#
# Revision 1.3  2000/06/07 17:32:59  moulding
# made changes to allow the module maker to add to and edit this file
#
# Revision 1.2  2000/03/20 19:38:13  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:29:14  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
