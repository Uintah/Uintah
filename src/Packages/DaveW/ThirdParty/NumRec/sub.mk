# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/DaveW/ThirdParty/NumRec

SRCS     += $(SRCDIR)/amoeba.cc $(SRCDIR)/amotry.cc $(SRCDIR)/banbks.cc \
	$(SRCDIR)/bandec.cc $(SRCDIR)/banmprv.cc $(SRCDIR)/banmul.cc \
	$(SRCDIR)/protozoa.cc \
	$(SRCDIR)/dpythag.cc $(SRCDIR)/dsvbksb.cc $(SRCDIR)/dsvdcmp.cc \
	$(SRCDIR)/linbcg.cc $(SRCDIR)/nrutil.cc $(SRCDIR)/plgndr.cc

PSELIBS := Packages/DaveW/ThirdParty/OldLinAlg Core/Exceptions

include $(SRCTOP)/scripts/smallso_epilogue.mk

