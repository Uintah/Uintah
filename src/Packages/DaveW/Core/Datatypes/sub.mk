# Makefile fragment for this subdirectory

SRCDIR := Packages/DaveW/Core/Datatypes

SUBDIRS := $(SRCDIR)/CS684 $(SRCDIR)/General

include $(SRCTOP)/scripts/recurse.mk

