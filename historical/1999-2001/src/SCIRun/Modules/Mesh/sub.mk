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

#
# $Log$
# Revision 1.10  2000/10/29 04:42:24  dmw
# MeshInterpVals -- fixed a bug
# MeshNodeComponent -- build a columnmatrix of the x/y/z position of the nodes
# MeshFindSurfNodes -- the surface nodes in a mesh
#
# Revision 1.9  2000/10/24 05:57:53  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.8  2000/09/07 16:14:42  dmw
# removed MeshNodeComponent from sub.mk -- it wasnt ready to be committed yet...
#
# Revision 1.7  2000/09/07 06:35:15  dmw
# Removed SliceMeshToGeom.cc reference in sub.mk
#
# Revision 1.6  2000/09/07 00:16:16  zyp
# Removing the SliceMeshToGeom.cc because I accidently added it in my
# last commit.  Also in the last commit I didn't add the comments
# properly, so I will mention them now.  I changed MakeScalarField to be
# able to handle a ColumnMatrix of the size of the number of nodes in a
# mesh or the number of elements and assign the values from the matrix accordingly.
#
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
