# Makefile fragment for this subdirectory

SRCDIR := testprograms

SUBDIRS := $(SRCDIR)/Malloc $(SRCDIR)/Thread
ifeq ($(BUILD_PARALLEL),yes)
SUBDIRS := $(SUBDIRS) $(SRCDIR)/Component
endif

include $(SRCTOP)/scripts/recurse.mk

