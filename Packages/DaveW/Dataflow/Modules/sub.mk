# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

SRCDIR := Packages/DaveW/Dataflow/Modules

SUBDIRS := \
	$(SRCDIR)/CS684\
	$(SRCDIR)/EEG\
	$(SRCDIR)/EGI\
	$(SRCDIR)/FDM\
	$(SRCDIR)/FEM\
	$(SRCDIR)/ISL\
	$(SRCDIR)/MEG\
	$(SRCDIR)/Readers\
	$(SRCDIR)/Tensor\
	$(SRCDIR)/Writers\
#[INSERT NEW CATEGORY DIR HERE]

include $(SRCTOP)/scripts/recurse.mk

