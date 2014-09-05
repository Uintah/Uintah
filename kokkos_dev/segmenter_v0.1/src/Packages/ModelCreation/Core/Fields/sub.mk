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

SRCS     += $(SRCDIR)/ApplyMappingMatrix.cc\
            $(SRCDIR)/BuildMembraneTable.cc\
            $(SRCDIR)/ClearAndChangeFieldBasis.cc\
            $(SRCDIR)/ConvertToTetVol.cc\
            $(SRCDIR)/ConvertToTriSurf.cc\
            $(SRCDIR)/CompartmentBoundary.cc\
            $(SRCDIR)/ClipBySelectionMask.cc\
            $(SRCDIR)/DistanceToField.cc\
            $(SRCDIR)/DistanceField.cc\
            $(SRCDIR)/FieldsAlgo.cc\
            $(SRCDIR)/ExampleFields.cc\
            $(SRCDIR)/FieldBoundary.cc\
            $(SRCDIR)/FieldDataNodeToElem.cc\
            $(SRCDIR)/FieldDataElemToNode.cc\
            $(SRCDIR)/GetFieldData.cc\
            $(SRCDIR)/GetFieldInfo.cc\
            $(SRCDIR)/IsInsideField.cc\
            $(SRCDIR)/LinkFieldBoundary.cc\
            $(SRCDIR)/LinkToCompGridByDomain.cc\
            $(SRCDIR)/MergeFields.cc\
            $(SRCDIR)/Precompile.cc\
            $(SRCDIR)/ScaleField.cc\
            $(SRCDIR)/SplitByConnectedRegion.cc\
            $(SRCDIR)/SplitFieldByDomain.cc\
            $(SRCDIR)/SelectionMask.cc\
            $(SRCDIR)/SetFieldData.cc\
            $(SRCDIR)/MappingMatrixToField.cc\
            $(SRCDIR)/TransformField.cc\
            $(SRCDIR)/ToPointCloud.cc\
            $(SRCDIR)/Unstructure.cc\
            $(SRCDIR)/TriSurfPhaseFilter.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS :=  Core/Datatypes Core/Util Core/Containers \
            Core/Exceptions Core/Thread Core/GuiInterface \
            Core/Geom Core/Geometry \
            Packages/ModelCreation/Core/Converter \
            Core/Algorithms/Util \
            Core/Persistent \
            Core/Basis Core/Bundle

LIBS :=     $(TEEM_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
