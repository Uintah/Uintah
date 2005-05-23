# Makefile fragment for this subdirectory

SRCDIR := Packages/DaveW/Core/Datatypes

SUBDIRS := $(SRCDIR)/General 
#	$(SRCDIR)/CS684 

include $(SRCTOP)/scripts/recurse.mk

