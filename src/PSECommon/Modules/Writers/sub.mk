#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := PSECommon/Modules/Writers

SRCS     += $(SRCDIR)/ColorMapWriter.cc $(SRCDIR)/ColumnMatrixWriter.cc \
	$(SRCDIR)/GeometryWriter.cc $(SRCDIR)/MatrixWriter.cc \
	$(SRCDIR)/MeshWriter.cc $(SRCDIR)/MultiScalarFieldWriter.cc \
	$(SRCDIR)/ScalarFieldWriter.cc $(SRCDIR)/SurfaceWriter.cc \
	$(SRCDIR)/TetraWriter.cc $(SRCDIR)/VectorFieldWriter.cc \
	$(SRCDIR)/VoidStarWriter.cc

PSELIBS := PSECore/Dataflow PSECore/Datatypes SCICore/Persistent \
	SCICore/Thread SCICore/Exceptions SCICore/TclInterface \
	SCICore/Geom SCICore/Containers SCICore/Datatypes
LIBS := 

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:27:43  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
