# Makefile fragment for this subdirectory

SRCDIR := Core/Algorithms/SurfSimplification

SUBDIRS := \
	$(SRCDIR)/qslim \
	$(SRCDIR)/gfx

include $(SRCTOP)/scripts/recurse.mk

