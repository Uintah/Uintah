#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/globus_threads

SRCS     += $(SRCDIR)/globus_external_threads.cc

PSELIBS := SCICore/Thread
LIBS := $(GLOBUS_COMMON)

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:38:02  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:48  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
