#Makefile fragment for the Packages/Yarden/Core directory

include $(OBJTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/Yarden/Core/Algorithms
SUBDIRS := \
	$(SRCDIR)/Visualization \

include $(OBJTOP_ABS)/scripts/recurse.mk
