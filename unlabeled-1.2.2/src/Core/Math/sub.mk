#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/Math

SRCS     += $(SRCDIR)/Mat.c $(SRCDIR)/MusilRNG.cc $(SRCDIR)/TrigTable.cc \
	$(SRCDIR)/LinAlg.c $(SRCDIR)/fft.c $(SRCDIR)/hf.c $(SRCDIR)/ssmult.c

PSELIBS := 
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:37:44  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:31  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
