#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Components/SimulationController

SRCS     += $(SRCDIR)/SimulationController.cc

PSELIBS := Uintah/Parallel Uintah/Exceptions Uintah/Interface \
	Uintah/Grid SCICore/Geometry SCICore/Thread SCICore/Exceptions
LIBS := $(XML_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.5  2000/05/07 06:02:10  sparker
# Added beginnings of multiple patch support and real dependencies
#  for the scheduler
#
# Revision 1.4  2000/04/19 05:26:13  sparker
# Implemented new problemSetup/initialization phases
# Simplified DataWarehouse interface (not finished yet)
# Made MPM get through problemSetup, but still not finished
#
# Revision 1.3  2000/04/12 23:00:10  sparker
# Start of reading grids
#
# Revision 1.2  2000/03/20 19:38:28  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:29:47  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
