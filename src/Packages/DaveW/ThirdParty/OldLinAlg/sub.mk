# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/DaveW/ThirdParty/OldLinAlg

SRCS     += $(SRCDIR)/cuthill.cc $(SRCDIR)/matrix.cc $(SRCDIR)/vector.cc

PSELIBS :=
LIBS :=

include $(SRCTOP)/scripts/smallso_epilogue.mk

