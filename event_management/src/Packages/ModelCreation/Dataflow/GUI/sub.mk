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
  $(SRCDIR)/DomainBoundary.tcl\
	$(SRCDIR)/ComputeDataArray.tcl\
	$(SRCDIR)/ComputeFieldData.tcl\
	$(SRCDIR)/ComputeFieldsData.tcl\
	$(SRCDIR)/ComputeFieldNodes.tcl\
	$(SRCDIR)/ComputeFieldsNodes.tcl\
	$(SRCDIR)/CreateFieldData.tcl\
	$(SRCDIR)/DataArrayInfo.tcl\
	$(SRCDIR)/FieldDataElemToNode.tcl\
	$(SRCDIR)/FieldDataNodeToElem.tcl\
	$(SRCDIR)/FieldInfo.tcl\
	$(SRCDIR)/FieldReader.tcl\
	$(SRCDIR)/FieldWriter.tcl\
	$(SRCDIR)/FieldGetMatrixProperty.tcl\
	$(SRCDIR)/FieldGetStringProperty.tcl\
	$(SRCDIR)/FieldSetMatrixProperty.tcl\
	$(SRCDIR)/FieldSetStringProperty.tcl\
	$(SRCDIR)/LinkFieldBoundary.tcl\
	$(SRCDIR)/MatrixInfo.tcl\
	$(SRCDIR)/MatrixReader.tcl\
	$(SRCDIR)/MatrixToField.tcl\
	$(SRCDIR)/MatrixWriter.tcl\
	$(SRCDIR)/MergeFields.tcl\
	$(SRCDIR)/MergeNodes.tcl\
	$(SRCDIR)/NrrdToField.tcl\
	$(SRCDIR)/ParameterList.tcl\
	$(SRCDIR)/ParameterListMatrix.tcl\
	$(SRCDIR)/ParameterListString.tcl\
	$(SRCDIR)/PathReader.tcl\
	$(SRCDIR)/PathWriter.tcl\
	$(SRCDIR)/ReplicateDataArray.tcl\
	$(SRCDIR)/ResizeMatrix.tcl\
	$(SRCDIR)/SelectAndSetFieldData.tcl\
	$(SRCDIR)/SelectAndSetFieldsData.tcl\
	$(SRCDIR)/SelectByFieldData.tcl\
	$(SRCDIR)/SelectByFieldsData.tcl\
	$(SRCDIR)/SphericalSurface.tcl\
	$(SRCDIR)/StringReader.tcl\
	$(SRCDIR)/StringWriter.tcl\
	$(SRCDIR)/ScaleField.tcl\
	$(SRCDIR)/DataArrayMeasure.tcl\
	$(SRCDIR)/StreamMatrix.tcl\
	$(SRCDIR)/ApplyRowOperation.tcl\
	$(SRCDIR)/ApplyColumnOperation.tcl\
	$(SRCDIR)/CollectFields.tcl\
	$(SRCDIR)/SetFieldData.tcl\
  $(SRCDIR)/TimeToWeights.tcl\
#[INSERT NEW TCL FILE HERE]

	$(OBJTOP)/createTclIndex $(SRCTOP)/Packages/ModelCreation/Dataflow/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex


