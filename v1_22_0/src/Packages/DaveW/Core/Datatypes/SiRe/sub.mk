# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/DaveW/Core/Datatypes/SiRe

SRCS     += $(SRCDIR)/SiRe

PSELIBS :=
LIBS :=

include $(SRCTOP)/scripts/smallso_epilogue.mk

