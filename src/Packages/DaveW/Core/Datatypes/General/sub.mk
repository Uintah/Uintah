#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := DaveW/Datatypes/General

SRCS     += $(SRCDIR)/ContourSet.cc $(SRCDIR)/ContourSetPort.cc \
	$(SRCDIR)/ManhattanDist.cc \
	$(SRCDIR)/ScalarTriSurface.cc \
	$(SRCDIR)/SegFld.cc $(SRCDIR)/SegFldPort.cc \
	$(SRCDIR)/SigmaSet.cc $(SRCDIR)/SigmaSetPort.cc \
	$(SRCDIR)/TensorField.cc $(SRCDIR)/TensorFieldBase.cc \
	$(SRCDIR)/TensorFieldPort.cc $(SRCDIR)/TopoSurfTree.cc \
	$(SRCDIR)/VectorFieldMI.cc

PSELIBS := PSECore/Dataflow SCICore/Persistent SCICore/Geometry \
	SCICore/Exceptions SCICore/Datatypes SCICore/Thread \
	SCICore/Containers 
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.3  2000/07/19 00:24:52  samsonov
# *** empty log message ***
#
# Revision 1.2  2000/03/20 19:35:57  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:25:21  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
