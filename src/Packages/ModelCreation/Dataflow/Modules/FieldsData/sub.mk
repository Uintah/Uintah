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

SRCDIR   := Packages/ModelCreation/Dataflow/Modules/FieldsData

SRCS     += \
	$(SRCDIR)/ComputeFieldData.cc\
	$(SRCDIR)/ComputeFieldsData.cc\
	$(SRCDIR)/CreateFieldData.cc\
	$(SRCDIR)/DistanceToBoundary.cc\
	$(SRCDIR)/DistanceToField.cc\
	$(SRCDIR)/FieldDataElemToNode.cc\
	$(SRCDIR)/FieldDataNodeToElem.cc\
	$(SRCDIR)/IsInsideField.cc\
	$(SRCDIR)/SelectAndSetFieldData.cc\
	$(SRCDIR)/SelectAndSetFieldsData.cc\
	$(SRCDIR)/SelectByFieldData.cc\
	$(SRCDIR)/SelectByFieldsData.cc\
	$(SRCDIR)/SignedDistanceToField.cc\
	$(SRCDIR)/MappingMatrixToField.cc\
	$(SRCDIR)/GetFieldData.cc\
  $(SRCDIR)/SetFieldData.cc\
	$(SRCDIR)/IndicesToData.cc\
	$(SRCDIR)/NodalMapping.cc\
	$(SRCDIR)/ModalMapping.cc\
	$(SRCDIR)/CurrentDensityMapping.cc\
	$(SRCDIR)/IsInsideFields.cc\
	$(SRCDIR)/GradientModalMapping.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Dataflow/Network \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/GeomInterface Core/TkExtensions \
        Packages/ModelCreation/Core/Algorithms \
        Packages/ModelCreation/Core/Datatypes \
        Packages/ModelCreation/Core/Fields \
        Core/Algorithms/Converter \
        Core/Algorithms/Fields 
        
LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


