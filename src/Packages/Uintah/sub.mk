# Makefile fragment for the Packages/Uintah directory

SRCDIR := Packages/Uintah

SUBDIRS := \
	$(SRCDIR)/Core         \
	$(SRCDIR)/Dataflow     \
	$(SRCDIR)/CCA          \
	$(SRCDIR)/StandAlone   \
	$(SRCDIR)/tools        \
	$(SRCDIR)/testprograms

include $(SCIRUN_SCRIPTS)/recurse.mk
