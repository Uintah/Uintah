# Makefile fragment for this subdirectory

SRCDIR := testprograms

SUBDIRS := $(SRCDIR)/Malloc $(SRCDIR)/Thread
ifeq ($(BUILD_PARALLEL),yes)
SUBDIRS := $(SUBDIRS) $(SRCDIR)/Core/CCA/Component
endif

include $(SRCTOP)/scripts/recurse.mk

