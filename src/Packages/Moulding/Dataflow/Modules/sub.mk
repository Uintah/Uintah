# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

SRCDIR := Moulding/Modules

SUBDIRS := \
	$(SRCDIR)/Visualization\
#[INSERT NEW CATEGORY DIR HERE]

include $(OBJTOP_ABS)/scripts/recurse.mk


