#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := PSECommon/Modules/FEM

SRCS     += $(SRCDIR)/ApplyBC.cc $(SRCDIR)/BuildFEMatrix.cc \
	$(SRCDIR)/ComposeError.cc $(SRCDIR)/ErrorInterval.cc \
	$(SRCDIR)/FEMError.cc $(SRCDIR)/MeshRefiner.cc

PSELIBS := PSECore/Dataflow PSECore/Datatypes SCICore/Datatypes \
	SCICore/Persistent SCICore/Thread SCICore/Containers \
	SCICore/Exceptions SCICore/TclInterface SCICore/Geometry
LIBS := 

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:26:53  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
