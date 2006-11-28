# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

SRCDIR := Packages/ModelCreation/Dataflow/Modules

SUBDIRS := \
	$(SRCDIR)/BreakPoint\
	$(SRCDIR)/ChangeFieldData\
	$(SRCDIR)/ChangeMesh\
	$(SRCDIR)/Converter\
	$(SRCDIR)/CreateField\
	$(SRCDIR)/ExampleFields\
	$(SRCDIR)/FieldProperty\
	$(SRCDIR)/FieldsData\
	$(SRCDIR)/FiniteElements\
	$(SRCDIR)/Math\
	$(SRCDIR)/ReportInfo\
	$(SRCDIR)/Script\
	$(SRCDIR)/SelectionMask\
	$(SRCDIR)/Time\
#[INSERT NEW CATEGORY DIR HERE]

include $(SCIRUN_SCRIPTS)/recurse.mk


