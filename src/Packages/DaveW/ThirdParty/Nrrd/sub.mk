# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/DaveW/ThirdParty/Nrrd

SRCS     += $(SRCDIR)/arrays.c $(SRCDIR)/err.c $(SRCDIR)/histogram.c \
	$(SRCDIR)/io.c $(SRCDIR)/methods.c $(SRCDIR)/nan.c \
	$(SRCDIR)/subset.c $(SRCDIR)/types.c

PSELIBS :=
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

