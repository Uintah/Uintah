# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

SRCDIR := Packages/Wangxl/Dataflow/Modules

SUBDIRS := \
	$(SRCDIR)/Surface\
	$(SRCDIR)/Volume\
#[INSERT NEW CATEGORY DIR HERE]

include $(SCIRUN_SCRIPTS)/recurse.mk


