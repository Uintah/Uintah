# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/Math

FNSRCDIR	:= $(SRCTOP)/$(SRCDIR)

$(FNSRCDIR)/fnparser.cc\
$(FNSRCDIR)/fnparser.h:	$(FNSRCDIR)/fnparser.y;
	$(YACC) -p fn $(FNSRCDIR)/fnparser.y -o $(FNSRCDIR)/fnparser.cc
	mv -f $(FNSRCDIR)/fnparser.cc.h $(FNSRCDIR)/fnparser.h

$(FNSRCDIR)/fnscanner.cc: $(FNSRCDIR)/fnscanner.l $(FNSRCDIR)/fnparser.cc;
	$(LEX) -Pfn -o$(FNSRCDIR)/fnscanner.cc $(FNSRCDIR)/fnscanner.l

SRCS     += $(SRCDIR)/Mat.c $(SRCDIR)/MusilRNG.cc $(SRCDIR)/TrigTable.cc \
	$(SRCDIR)/LinAlg.c $(SRCDIR)/fft.c $(SRCDIR)/hf.c $(SRCDIR)/ssmult.c \
	$(SRCDIR)/PiecewiseInterp.cc $(SRCDIR)/LinearPWI.cc $(SRCDIR)/CubicPWI.cc \
	$(SRCDIR)/function.cc $(SRCDIR)/fnscanner.cc $(SRCDIR)/fnparser.cc


PSELIBS := Core/Exceptions Core/Containers
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

