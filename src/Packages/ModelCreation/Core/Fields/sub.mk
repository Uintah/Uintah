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

SRCDIR   := Packages/ModelCreation/Core/Fields

SRCS     += $(SRCDIR)/ConvertToTetVol.cc\
            $(SRCDIR)/ConvertToTriSurf.cc\
            $(SRCDIR)/CompartmentBoundary.cc\
            $(SRCDIR)/ClipBySelectionMask.cc\
            $(SRCDIR)/DistanceToField.cc\
            $(SRCDIR)/FieldsAlgo.cc\
            $(SRCDIR)/ExampleFields.cc\
            $(SRCDIR)/FieldDataNodeToElem.cc\
            $(SRCDIR)/FieldDataElemToNode.cc\
            $(SRCDIR)/SplitFieldByElementData.cc\
            $(SRCDIR)/MergeFields.cc\
            $(SRCDIR)/GetFieldData.cc\
            $(SRCDIR)/SetFieldData.cc\
            $(SRCDIR)/MappingMatrixToField.cc\
            $(SRCDIR)/TransformField.cc\
            $(SRCDIR)/ToPointCloud.cc\
            $(SRCDIR)/Unstructure.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS :=  Core/Datatypes Core/Util Core/Containers \
            Core/Exceptions Core/Thread Core/GuiInterface \
            Core/Geom Core/Geometry Dataflow/Network \
            Packages/ModelCreation/Core/Algorithms \
            Packages/ModelCreation/Core/Datatypes \
            Packages/ModelCreation/Core/Util \
            Dataflow/Modules/Fields \
            Core/Algorithms/Fields \
            Core/Persistent \
            Core/Basis
LIBS :=

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
