#Makefile fragment for the Packages/Yarden/Core directory

SRCDIR := Packages/BioPSE/Core/Algorithms

SUBDIRS := \
	$(SRCDIR)/NumApproximation \

include $(SRCTOP_ABS)/scripts/recurse.mk
