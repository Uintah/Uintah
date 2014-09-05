#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := DaveW/ThirdParty/OldLinAlg

SRCS     += $(SRCDIR)/cuthill.cc $(SRCDIR)/matrix.cc $(SRCDIR)/vector.cc

PSELIBS :=
LIBS :=

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:36:32  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:16  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
