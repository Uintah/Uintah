#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := PSECore/Datatypes

SRCS     += $(SRCDIR)/BooleanPort.cc $(SRCDIR)/ColorMapPort.cc \
	$(SRCDIR)/ColumnMatrixPort.cc $(SRCDIR)/GeometryPort.cc \
	$(SRCDIR)/HexMeshPort.cc $(SRCDIR)/IntervalPort.cc \
	$(SRCDIR)/MatrixPort.cc $(SRCDIR)/MeshPort.cc \
	$(SRCDIR)/ScalarFieldPort.cc $(SRCDIR)/ScaledBoxWidgetData.cc \
	$(SRCDIR)/ScaledBoxWidgetDataPort.cc $(SRCDIR)/SoundPort.cc \
	$(SRCDIR)/SurfacePort.cc $(SRCDIR)/VectorFieldPort.cc \
	$(SRCDIR)/VoidStarPort.cc $(SRCDIR)/cMatrixPort.cc \
	$(SRCDIR)/cVectorPort.cc $(SRCDIR)/SpanTree.cc \
	$(SRCDIR)/SpanPort.cc $(SRCDIR)/templates.cc


PSELIBS := PSECore/Dataflow PSECore/Comm SCICore/Containers \
	SCICore/Thread SCICore/Geom SCICore/Geometry SCICore/Exceptions \
	SCICore/Persistent SCICore/Datatypes
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.3  2000/03/21 03:01:24  sparker
# Partially fixed special_get method in SimplePort
# Pre-instantiated a few key template types, in an attempt to reduce
#   initial compile time and reduce code bloat.
# Manually instantiated templates are in */*/templates.cc
#
# Revision 1.2  2000/03/20 19:37:20  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:27:58  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
