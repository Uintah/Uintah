#
# Makefile fragment for this subdirectory
#

SRCDIR := Packages/DaveW/ThirdParty

SUBDIRS := \
	$(SRCDIR)/Nrrd \
	$(SRCDIR)/NumRec \
	$(SRCDIR)/OldLinAlg

include $(SRCTOP)/scripts/recurse.mk

