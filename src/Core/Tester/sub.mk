#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/Tester

SRCS     += $(SRCDIR)/PerfTest.cc $(SRCDIR)/RigorousTest.cc

PSELIBS := 
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:37:52  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:40  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
