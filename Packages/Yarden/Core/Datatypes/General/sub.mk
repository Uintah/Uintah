# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Yarden/Core/Datatypes/General

SRCS     += $(SRCDIR)/Clock.cc

PSELIBS :=
LIBS :=

include $(SRCTOP)/scripts/smallso_epilogue.mk

