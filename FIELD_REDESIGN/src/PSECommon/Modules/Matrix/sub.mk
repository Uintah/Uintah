#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := PSECommon/Modules/Matrix

SRCS     += $(SRCDIR)/BldTransform.cc $(SRCDIR)/EditMatrix.cc \
	$(SRCDIR)/ExtractSubmatrix.cc $(SRCDIR)/MatMat.cc \
	$(SRCDIR)/MatSelectVec.cc $(SRCDIR)/MatVec.cc \
	$(SRCDIR)/SolveMatrix.cc $(SRCDIR)/VisualizeMatrix.cc \
	$(SRCDIR)/cConjGrad.cc $(SRCDIR)/cPhase.cc

PSELIBS := PSECore/Dataflow PSECore/Datatypes SCICore/Persistent \
	SCICore/Exceptions SCICore/Thread SCICore/Containers \
	SCICore/TclInterface SCICore/Geometry SCICore/Datatypes \
	SCICore/Util SCICore/Geom SCICore/TkExtensions
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:37:00  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:27:08  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
