#Makefile fragment for the Packages/BioPSE/Core/Algorithms directory

SRCDIR := Packages/BioPSE/Core/Algorithms

SUBDIRS := \
	$(SRCDIR)/Forward \
	$(SRCDIR)/NumApproximation \

include $(SCIRUN_SCRIPTS)/recurse.mk
