#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/Containers

SRCS     += $(SRCDIR)/Sort.cc $(SRCDIR)/String.cc $(SRCDIR)/PQueue.cc \
	   $(SRCDIR)/TrivialAllocator.cc $(SRCDIR)/BitArray1.cc

PSELIBS := SCICore/Exceptions SCICore/Tester SCICore/Thread

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:28:17  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
