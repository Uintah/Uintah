#
# Makefile fragment for this subdirectory
# $Id$
#

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
	$(SRCDIR)/MeshView.cc\
#[INSERT NEW MODULE HERE]

PSELIBS := PSECore/Dataflow PSECore/Datatypes PSECore/Widgets \
	SCICore/Geom SCICore/Thread SCICore/Exceptions \
	SCICore/Containers SCICore/Geometry SCICore/Datatypes \
	SCICore/Persistent SCICore/TclInterface SCICore/Math
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
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
