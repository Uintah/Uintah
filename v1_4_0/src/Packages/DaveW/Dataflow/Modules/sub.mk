# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

SRCDIR := Packages/DaveW/Dataflow/Modules

SUBDIRS := \
#	$(SRCDIR)/LeadField\
#	$(SRCDIR)/Contours\
#	$(SRCDIR)/Readers\
#	$(SRCDIR)/SurfTree\
#	$(SRCDIR)/Writers\
#	$(SRCDIR)/CS684\
#	$(SRCDIR)/SiRe\
#	$(SRCDIR)/Tensor\
#[INSERT NEW CATEGORY DIR HERE]

include $(SRCTOP)/scripts/recurse.mk

