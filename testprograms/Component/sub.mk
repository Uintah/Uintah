# Makefile fragment for this subdirectory

SRCDIR := testprograms/Component

SUBDIRS := $(SRCDIR)/argtest $(SRCDIR)/memstress $(SRCDIR)/mitest \
	$(SRCDIR)/objects $(SRCDIR)/pingpong 

include $(SRCTOP)/scripts/recurse.mk

