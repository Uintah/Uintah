# Makefile fragment for the Packages/Uintah directory

SRCDIR := Packages/Uintah

SUBDIRS := \
	$(SRCDIR)/Core         \
	$(SRCDIR)/CCA          \
	$(SRCDIR)/StandAlone   \
	$(SRCDIR)/tools        \
	$(SRCDIR)/testprograms

ifeq ($(BUILD_DATAFLOW),yes)
  SUBDIRS += $(SRCDIR)/Dataflow
endif

include $(SCIRUN_SCRIPTS)/recurse.mk
