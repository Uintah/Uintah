# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/largeso_prologue.mk

SRCDIR := Core/CCA/tools

SUBDIRS := \
	$(SRCDIR)/sidl \

include $(SRCTOP)/scripts/recurse.mk

PSELIBS :=
LIBS :=

include $(SRCTOP_ABS)/scripts/largeso_epilogue.mk


