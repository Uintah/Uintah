# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/largeso_prologue.mk

SRCDIR := Core/CCA

SUBDIRS := 

ifeq ($(BUILD_PARALLEL),yes)
SUBDIRS := \
	$(SRCDIR)/tools \
	$(SRCDIR)/Component
endif

include $(SRCTOP)/scripts/recurse.mk

PSELIBS := 

include $(SRCTOP)/scripts/largeso_epilogue.mk

