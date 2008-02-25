#Makefile fragment for the Packages/Uintah/R_Tester directory

SRCDIR := Packages/Uintah/R_Tester
SUBDIRS := $(SRCDIR)/helpers

include $(SCIRUN_SCRIPTS)/recurse.mk
