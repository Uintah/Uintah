# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the Core/CCA/Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Core/CCA/Component"
# documentation on how to do it correctly.

SRCDIR := Packages/BioPSE/Dataflow/Modules

SUBDIRS := \
	$(SRCDIR)/LeadField\
#	$(SRCDIR)/Forward\
#	$(SRCDIR)/Inverse\
#[INSERT NEW CATEGORY DIR HERE]

include $(OBJTOP_ABS)/scripts/recurse.mk


