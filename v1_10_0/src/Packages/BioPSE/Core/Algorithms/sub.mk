#Makefile fragment for the Packages/Yarden/Core directory

SRCDIR := Packages/BioPSE/Core/Algorithms

SUBDIRS := \
	$(SRCDIR)/NumApproximation \
	$(SRCDIR)/Forward \

include $(SCIRUN_SCRIPTS)/recurse.mk
