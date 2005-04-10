#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := DaveW/ThirdParty/Nrrd

SRCS     += $(SRCDIR)/arrays.c $(SRCDIR)/err.c $(SRCDIR)/histogram.c \
	$(SRCDIR)/io.c $(SRCDIR)/methods.c $(SRCDIR)/nan.c \
	$(SRCDIR)/subset.c $(SRCDIR)/types.c

PSELIBS :=
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:36:29  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:11  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
