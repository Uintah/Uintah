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
	$(SRCDIR)/AppendMatrix.tcl\
	$(SRCDIR)/CalculateFieldData.tcl\
	$(SRCDIR)/CalculateFieldsData.tcl\
	$(SRCDIR)/CalculateInsideWhichField.tcl\
	$(SRCDIR)/CalculateIsInsideField.tcl\
	$(SRCDIR)/CalculateLinkBetweenOppositeFieldBoundaries.tcl\
  $(SRCDIR)/CalculateMeshNodes.tcl\
	$(SRCDIR)/ClipFieldByFunction.tcl\
	$(SRCDIR)/CollectFields.tcl\
	$(SRCDIR)/ConvertMatrixToField.tcl\
	$(SRCDIR)/ConvertNrrdToField.tcl\
	$(SRCDIR)/ConvertTimeToWeightedIndices.tcl\
	$(SRCDIR)/CreateFieldData.tcl\
	$(SRCDIR)/GenerateSphericalSurface.tcl\
	$(SRCDIR)/GetDomainBoundary.tcl\
	$(SRCDIR)/GetMatrixFromFieldProperties.tcl\
	$(SRCDIR)/GetStringFromFieldProperties.tcl\
	$(SRCDIR)/InsertBundleBreakPoint.tcl\
	$(SRCDIR)/InsertFieldBreakPoint.tcl\
	$(SRCDIR)/InsertMatrixBreakPoint.tcl\
	$(SRCDIR)/InsertMatrixIntoFieldProperties.tcl\
	$(SRCDIR)/InsertStringBreakPoint.tcl\
  $(SRCDIR)/InsertStringIntoFieldProperties.tcl\
  $(SRCDIR)/ManageParameterMatrix.tcl\
	$(SRCDIR)/ManageParameterString.tcl\
	$(SRCDIR)/MapCurrentDensityOntoField.tcl\
	$(SRCDIR)/MapFieldDataFromElemToNode.tcl\
	$(SRCDIR)/MapFieldDataFromNodeToElem.tcl\
	$(SRCDIR)/MapFieldDataGradientOntoField.tcl\
	$(SRCDIR)/MapFieldDataOntoFieldNodes.tcl\
	$(SRCDIR)/MergeFields.tcl\
	$(SRCDIR)/MergeMeshes.tcl\
	$(SRCDIR)/MergeMeshNodes.tcl\
	$(SRCDIR)/ModalMapping.tcl\
	$(SRCDIR)/ParameterList.tcl\
	$(SRCDIR)/ReportDataArrayInfo.tcl\
	$(SRCDIR)/ReportFieldDataMeasures.tcl\
	$(SRCDIR)/ReportFieldInfo.tcl\
	$(SRCDIR)/ReportMatrixColumnMeasure.tcl\
	$(SRCDIR)/ReportMatrixInfo.tcl\
	$(SRCDIR)/ReportMatrixRowMeasure.tcl\
	$(SRCDIR)/ResizeMatrix.tcl\
	$(SRCDIR)/ScaleFieldMeshAndData.tcl\
	$(SRCDIR)/SelectAndSetFieldData.tcl\
	$(SRCDIR)/SelectAndSetFieldsData.tcl\
  $(SRCDIR)/SelectByFieldData.tcl\
	$(SRCDIR)/SelectByFieldsData.tcl\
	$(SRCDIR)/SetFieldData.tcl\
#[INSERT NEW TCL FILE HERE]

	$(OBJTOP)/createTclIndex $(SRCTOP)/Packages/ModelCreation/Dataflow/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex


