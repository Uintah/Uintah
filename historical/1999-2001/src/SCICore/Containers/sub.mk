#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/Containers

SRCS     += $(SRCDIR)/Sort.cc $(SRCDIR)/String.cc $(SRCDIR)/PQueue.cc \
	$(SRCDIR)/TrivialAllocator.cc $(SRCDIR)/BitArray1.cc \
	$(SRCDIR)/templates.cc $(SRCDIR)/ConsecutiveRangeSet.cc

PSELIBS := SCICore/Exceptions SCICore/Tester SCICore/Thread

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.4  2000/12/04 22:50:24  witzel
# Added ConsecutiveRangeSet
#
# Revision 1.3  2000/03/21 03:01:26  sparker
# Partially fixed special_get method in SimplePort
# Pre-instantiated a few key template types, in an attempt to reduce
#   initial compile time and reduce code bloat.
# Manually instantiated templates are in */*/templates.cc
#
# Revision 1.2  2000/03/20 19:37:33  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:17  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
