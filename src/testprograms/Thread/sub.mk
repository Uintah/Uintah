#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := testprograms/Thread

ifeq ($(LARGESOS),yes)
PSELIBS := SCICore
else
PSELIBS := SCICore/Thread
endif
LIBS := $(THREAD_LIBS)

PROGRAM := $(SRCDIR)/bps
SRCS := $(SRCDIR)/bps.cc

include $(OBJTOP_ABS)/scripts/program.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:31:17  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#