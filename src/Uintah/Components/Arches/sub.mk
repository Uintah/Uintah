#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Components/Arches

SRCS     += $(SRCDIR)/Arches.cc \
	$(SRCDIR)/NonlinearSolver.cc $(SRCDIR)/PhysicalConstants.cc \
	$(SRCDIR)/Properties.cc
#$(SRCDIR)/Discretization.cc $(SRCDIR)/PicardNonlinearSolver.cc \
#	$(SRCDIR)/PressureSolver.cc
PSELIBS := Uintah/Parallel Uintah/Interface Uintah/Grid Uintah/Exceptions
LIBS := $(XML_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.7  2000/04/13 06:50:52  sparker
# More implementation to get this to work
#
# Revision 1.6  2000/04/12 22:58:30  sparker
# Resolved conflicts
# Making it compile
#
# Revision 1.5  2000/03/22 23:42:52  sparker
# Do not compile it all quite yet
#
# Revision 1.4  2000/03/22 23:41:20  sparker
# Working towards getting arches to compile/run
#
# Revision 1.3  2000/03/21 18:52:50  sparker
# Fixed compilation errors in Arches.cc
# use new problem spec class
#
# Revision 1.2  2000/03/20 19:38:20  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:29:27  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
