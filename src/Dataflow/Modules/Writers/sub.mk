#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := PSECommon/Modules/Writers

SRCS     += \
	$(SRCDIR)/ColorMapWriter.cc\
	$(SRCDIR)/ColumnMatrixWriter.cc\
	$(SRCDIR)/GeometryWriter.cc\
	$(SRCDIR)/MatrixWriter.cc\
	$(SRCDIR)/MeshWriter.cc\
	$(SRCDIR)/MultiScalarFieldWriter.cc\
	$(SRCDIR)/ScalarFieldWriter.cc\
	$(SRCDIR)/SurfaceWriter.cc\
	$(SRCDIR)/TetraWriter.cc\
	$(SRCDIR)/VectorFieldWriter.cc\
	$(SRCDIR)/VoidStarWriter.cc\
#[INSERT NEW MODULE HERE]

PSELIBS := PSECore/Dataflow PSECore/Datatypes SCICore/Persistent \
	SCICore/Thread SCICore/Exceptions SCICore/TclInterface \
	SCICore/Geom SCICore/Containers SCICore/Datatypes
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.3  2000/06/07 00:11:42  moulding
# made some modifications that will allow the module make to edit and add
# to this file
#
# Revision 1.2  2000/03/20 19:37:08  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:27:43  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
