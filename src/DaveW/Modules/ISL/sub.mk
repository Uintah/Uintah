#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := DaveW/Modules/ISL

SRCS     += $(SRCDIR)/Downhill_Simplex.cc \
	$(SRCDIR)/LeastSquaresSolve.cc $(SRCDIR)/OptDip.cc \
	$(SRCDIR)/SGI_LU.cc $(SRCDIR)/SGI_Solve.cc

PSELIBS := DaveW/ThirdParty/NumRec DaveW/ThirdParty/OldLinAlg \
	PSECore/Datatypes PSECore/Dataflow SCICore/Datatypes \
	SCICore/Persistent SCICore/Exceptions SCICore/Thread \
	SCICore/Containers SCICore/TclInterface
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:36:12  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:25:48  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
