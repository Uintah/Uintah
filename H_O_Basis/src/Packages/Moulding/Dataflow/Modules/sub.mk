# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the Core/CCA/Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Core/CCA/Component"
# documentation on how to do it correctly.

SRCDIR := Packages/Moulding/Dataflow/Modules

SUBDIRS := \
	$(SRCDIR)/Visualization\
#[INSERT NEW CATEGORY DIR HERE]

include $(SRCTOP_ABS)/scripts/recurse.mk


