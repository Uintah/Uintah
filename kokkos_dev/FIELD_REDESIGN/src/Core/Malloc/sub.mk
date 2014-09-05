#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/Malloc

SRCS     += $(SRCDIR)/Allocator.cc $(SRCDIR)/AllocOS.cc \
	$(SRCDIR)/malloc.cc $(SRCDIR)/new.cc 

SRCS += $(LOCK_IMPL)

PSELIBS := 
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:37:43  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:29  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
