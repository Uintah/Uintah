# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/ModelCreation/Dataflow/Modules/ChangeFieldData

SRCS     += \
	$(SRCDIR)/CalculateDistanceToFieldBoundary.cc\
	$(SRCDIR)/CalculateDistanceToField.cc\
	$(SRCDIR)/CalculateFieldData.cc\
	$(SRCDIR)/CalculateFieldsData.cc\
	$(SRCDIR)/CalculateInsideWhichField.cc\
	$(SRCDIR)/CalculateIsInsideField.cc\
	$(SRCDIR)/CalculateSignedDistanceToField.cc\
	$(SRCDIR)/ConvertIndicesToFieldData.cc\
	$(SRCDIR)/ConvertMappingMatrixToFieldData.cc\
	$(SRCDIR)/CreateFieldData.cc\
	$(SRCDIR)/GetFieldData.cc\
	$(SRCDIR)/MapCurrentDensityOntoField.cc\
	$(SRCDIR)/MapFieldDataFromElemToNode.cc\
	$(SRCDIR)/MapFieldDataFromNodeToElem.cc\
	$(SRCDIR)/MapFieldDataGradientOntoField.cc\
	$(SRCDIR)/MapFieldDataOntoFieldNodes.cc\
	$(SRCDIR)/ReportFieldDataMeasures.cc\
	$(SRCDIR)/SelectAndSetFieldData.cc\
	$(SRCDIR)/SelectAndSetFieldsData.cc\
	$(SRCDIR)/SetFieldData.cc\
	$(SRCDIR)/ModalMapping.cc\
	$(SRCDIR)/SelectByFieldsData.cc\
	$(SRCDIR)/SelectByFieldData.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Dataflow/Network \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Dataflow/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/GeomInterface Dataflow/TkExtensions \
        Packages/ModelCreation/Core/Datatypes \
        Packages/ModelCreation/Core/Fields \
        Core/Algorithms/Converter \
        Core/Algorithms/Fields \
        Core/Algorithms/ArrayMath

LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


