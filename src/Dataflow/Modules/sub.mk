# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

SRCDIR := Dataflow/Modules

SUBDIRS := \
	$(SRCDIR)/Domain\
	$(SRCDIR)/Example\
        $(SRCDIR)/FEM\
        $(SRCDIR)/Fields\
	$(SRCDIR)/Image\
	$(SRCDIR)/Iterators\
        $(SRCDIR)/Matrix\
	$(SRCDIR)/Mesh\
        $(SRCDIR)/Readers\
	$(SRCDIR)/Salmon\
        $(SRCDIR)/Surface\
        $(SRCDIR)/Visualization\
	$(SRCDIR)/Writers\
#[INSERT NEW CATEGORY DIR HERE]

include $(SRCTOP)/scripts/recurse.mk

