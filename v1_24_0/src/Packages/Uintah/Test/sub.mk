#Makefile fragment for the Packages/Uintah/Test directory

SRCDIR := Packages/Uintah/Test
SUBDIRS := $(SRCDIR)/helpers

include $(SCIRUN_SCRIPTS)/recurse.mk
