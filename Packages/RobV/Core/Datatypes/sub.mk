# Makefile fragment for this subdirectory

SRCDIR := Packages/RobV/Core/Datatypes

SUBDIRS := \
	$(SRCDIR)/MEG

include $(SRCTOP)/scripts/recurse.mk

