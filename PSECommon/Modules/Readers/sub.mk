#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := PSECommon/Modules/Readers

SRCS     += \
	$(SRCDIR)/ColorMapReader.cc\
	$(SRCDIR)/ColumnMatrixReader.cc\
	$(SRCDIR)/DukeRawRead.cc\
	$(SRCDIR)/GeomReader.cc\
	$(SRCDIR)/GeometryReader.cc\
	$(SRCDIR)/ImageReader.cc\
	$(SRCDIR)/MatrixReader.cc\
	$(SRCDIR)/MeshReader.cc\
	$(SRCDIR)/MultiSFRGReader.cc\
	$(SRCDIR)/PointsReader.cc\
	$(SRCDIR)/ScalarFieldReader.cc\
	$(SRCDIR)/SurfaceReader.cc\
	$(SRCDIR)/VectorFieldReader.cc\
	$(SRCDIR)/VoidStarReader.cc\
#[INSERT NEW MODULE HERE]

PSELIBS := PSECore/Dataflow PSECore/Datatypes SCICore/Datatypes \
	SCICore/Persistent SCICore/Exceptions SCICore/Thread \
	SCICore/Containers SCICore/TclInterface SCICore/Geom
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.3  2000/06/07 00:11:39  moulding
# made some modifications that will allow the module make to edit and add
# to this file
#
# Revision 1.2  2000/03/20 19:37:02  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:27:14  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
