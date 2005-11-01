# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

SRCDIR := Packages/ModelCreation/Dataflow/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/BundleReader.tcl\
	$(SRCDIR)/BundleWriter.tcl\
	$(SRCDIR)/ClipFieldByFunction.tcl\
	$(SRCDIR)/ColorMapReader.tcl\
	$(SRCDIR)/ColorMapWriter.tcl\
	$(SRCDIR)/ColorMap2Reader.tcl\
	$(SRCDIR)/ColorMap2Writer.tcl\
	$(SRCDIR)/ComputeDataArray.tcl\
	$(SRCDIR)/ComputeFieldData.tcl\
	$(SRCDIR)/ComputeFieldsData.tcl\
	$(SRCDIR)/CreateFieldData.tcl\
	$(SRCDIR)/DataArrayInfo.tcl\
	$(SRCDIR)/FieldDataElemToNode.tcl\
	$(SRCDIR)/FieldDataNodeToElem.tcl\
	$(SRCDIR)/FieldInfo.tcl\
	$(SRCDIR)/FieldReader.tcl\
	$(SRCDIR)/FieldWriter.tcl\
	$(SRCDIR)/MatrixInfo.tcl\
	$(SRCDIR)/MatrixReader.tcl\
	$(SRCDIR)/MatrixWriter.tcl\
	$(SRCDIR)/ParameterList.tcl\
	$(SRCDIR)/ParameterListMatrix.tcl\
	$(SRCDIR)/ParameterListString.tcl\
	$(SRCDIR)/PathReader.tcl\
	$(SRCDIR)/PathWriter.tcl\
	$(SRCDIR)/ReplicateDataArray.tcl\
	$(SRCDIR)/StringReader.tcl\
	$(SRCDIR)/StringWriter.tcl\
	$(SRCDIR)/FieldGetMatrixProperty.tcl\
	$(SRCDIR)/FieldGetStringProperty.tcl\
	$(SRCDIR)/FieldSetMatrixProperty.tcl\
	$(SRCDIR)/FieldSetStringProperty.tcl\
#[INSERT NEW TCL FILE HERE]

	$(OBJTOP)/createTclIndex $(SRCTOP)/Packages/ModelCreation/Dataflow/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex


