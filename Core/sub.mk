#Makefile fragment for the Packages/Uintah/Core directory

SRCDIR := Packages/Uintah/Core
SUBDIRS := \
	$(SRCDIR)/Parallel \
	$(SRCDIR)/Math \
	$(SRCDIR)/Exceptions \
	$(SRCDIR)/Grid \
	$(SRCDIR)/ProblemSpec \
	$(SRCDIR)/Datatypes

include $(SRCTOP_ABS)/scripts/recurse.mk

