#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/Math

FNSRCDIR	:= $(SRCTOP)/$(SRCDIR)

$(FNSRCDIR)/fnparser.cc\
$(FNSRCDIR)/fnparser.h:	$(FNSRCDIR)/fnparser.y;
	$(YACC) -p fn $(FNSRCDIR)/fnparser.y -o $(FNSRCDIR)/fnparser.cc
	mv -f $(FNSRCDIR)/fnparser.cc.h $(FNSRCDIR)/fnparser.h

$(FNSRCDIR)/fnscanner.cc: $(FNSRCDIR)/fnscanner.l $(FNSRCDIR)/fnparser.cc;
	$(LEX) -Pfn -o$(FNSRCDIR)/fnscanner.cc $(FNSRCDIR)/fnscanner.l

SRCS     += $(SRCDIR)/Mat.c $(SRCDIR)/MusilRNG.cc $(SRCDIR)/TrigTable.cc \
	$(SRCDIR)/LinAlg.c $(SRCDIR)/fft.c $(SRCDIR)/hf.c $(SRCDIR)/ssmult.c \
	$(SRCDIR)/PiecewiseInterp.cc $(SRCDIR)/LinearPWI.cc \
	$(SRCDIR)/function.cc $(SRCDIR)/fnscanner.cc $(SRCDIR)/fnparser.cc

PSELIBS := SCICore/Exceptions SCICore/Containers
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.5  2000/07/26 22:16:49  jehall
# - Removed hardcoded LEX and YACC declarations; these get set portably
#   by the top-level configure script
#
# Revision 1.4  2000/07/23 18:25:47  dahart
# Initial commit of class & support for symbolic functions
#
# Revision 1.3  2000/07/14 23:27:43  samsonov
# Added interpolation classes source files paths
#
# Revision 1.2  2000/03/20 19:37:44  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:31  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
