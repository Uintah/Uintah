# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

SRCDIR := Packages/MIT/Dataflow/Modules

SUBDIRS := \
	$(SRCDIR)/DataIO \
	$(SRCDIR)/Bayer \
	$(SRCDIR)/Metropolis \
#[INSERT NEW CATEGORY DIR HERE]

#	$(SRCDIR)/Test\


include $(SCIRUN_SCRIPTS)/recurse.mk


