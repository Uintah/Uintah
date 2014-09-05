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
	$(SRCDIR)/TensorVectorMath\
	$(SRCDIR)/DataInfo\
	$(SRCDIR)/DataIO\
	$(SRCDIR)/Script\
        $(SRCDIR)/SelectionMask\
	$(SRCDIR)/FieldsData\
	$(SRCDIR)/FieldsCreate\
	$(SRCDIR)/FieldsProperty\
	$(SRCDIR)/FieldsGeometry\
	$(SRCDIR)/FieldsExample\
	$(SRCDIR)/FiniteElements\
	$(SRCDIR)/Converter\
	$(SRCDIR)/Math\
	$(SRCDIR)/DataStreaming\
	$(SRCDIR)/Time\
#[INSERT NEW CATEGORY DIR HERE]

include $(SCIRUN_SCRIPTS)/recurse.mk


