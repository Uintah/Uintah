#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Interface

SRCS     += $(SRCDIR)/CFDInterface.cc $(SRCDIR)/DataWarehouse.cc \
	$(SRCDIR)/MPMInterface.cc $(SRCDIR)/Output.cc \
	$(SRCDIR)/ProblemSpec.cc $(SRCDIR)/ProblemSpecInterface.cc \
	$(SRCDIR)/Scheduler.cc $(SRCDIR)/DWMpiHandler.cc

PSELIBS := Uintah/Parallel Uintah/Grid Uintah/Exceptions SCICore/Thread
LIBS := $(XML_LIBRARY) -lmpi

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.7  2000/05/11 20:10:23  dav
# adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
#
# Revision 1.6  2000/04/12 23:01:55  sparker
# Implemented more of problem spec - added Point and IntVector readers
#
# Revision 1.5  2000/04/11 07:10:54  sparker
# Completing initialization and problem setup
# Finishing Exception modifications
#
# Revision 1.4  2000/03/29 02:00:00  jas
# Filled in the findBlock method.
#
# Revision 1.3  2000/03/23 20:42:24  sparker
# Added copy ctor to exception classes (for Linux/g++)
# Helped clean up move of ProblemSpec from Interface to Grid
#
# Revision 1.2  2000/03/20 19:38:37  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:30:04  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
