#Makefile fragment for the Packages/Uintah/Dataflow directory

SRCDIR := Packages/Uintah/CCA
SUBDIRS := \
	$(SRCDIR)/Components \
	$(SRCDIR)/Ports \

include $(SRCTOP_ABS)/scripts/recurse.mk

