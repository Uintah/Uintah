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

SRCDIR   := Core/Algorithms/Fields


SRCS     += $(SRCDIR)/ApplyMappingMatrix.cc\
            $(SRCDIR)/GetBoundingBox.cc\
            $(SRCDIR)/ClearAndChangeFieldBasis.cc\
            $(SRCDIR)/ClipFieldByFunction.cc\
            $(SRCDIR)/ConvertToTetVol.cc\
            $(SRCDIR)/ConvertToTriSurf.cc\
            $(SRCDIR)/DomainBoundary.cc\
            $(SRCDIR)/ClipAtIndeces.cc\
            $(SRCDIR)/ClipBySelectionMask.cc\
            $(SRCDIR)/IndicesToData.cc\
            $(SRCDIR)/DistanceField.cc\
            $(SRCDIR)/FieldsAlgo.cc\
            $(SRCDIR)/FieldBoundary.cc\
            $(SRCDIR)/FieldDataNodeToElem.cc\
            $(SRCDIR)/FieldDataElemToNode.cc\
            $(SRCDIR)/FindClosestNode.cc\
            $(SRCDIR)/FindClosestNodeByValue.cc\
            $(SRCDIR)/GetFieldData.cc\
            $(SRCDIR)/GetFieldDataMinMax.cc\
            $(SRCDIR)/GetFieldInfo.cc\
            $(SRCDIR)/GetFieldMeasure.cc\
            $(SRCDIR)/IsInsideField.cc\
            $(SRCDIR)/LinkFieldBoundary.cc\
            $(SRCDIR)/LinkToCompGrid.cc\
            $(SRCDIR)/LinkToCompGridByDomain.cc\
            $(SRCDIR)/GatherFields.cc\
            $(SRCDIR)/MergeFields.cc\
            $(SRCDIR)/MergeMeshes.cc\
            $(SRCDIR)/RemoveUnusedNodes.cc\
            $(SRCDIR)/ScaleField.cc\
            $(SRCDIR)/SplitByConnectedRegion.cc\
            $(SRCDIR)/SplitFieldByDomain.cc\
            $(SRCDIR)/SetFieldData.cc\
            $(SRCDIR)/MappingMatrixToField.cc\
            $(SRCDIR)/Mapping.cc\
            $(SRCDIR)/CurrentDensityMapping.cc\
            $(SRCDIR)/CalculateFieldData.cc\
            $(SRCDIR)/TransformMeshWithTransform.cc\
            $(SRCDIR)/ToPointCloud.cc\
            $(SRCDIR)/Unstructure.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS :=  Core/Datatypes Core/Util Core/Containers \
            Core/Exceptions Core/Thread Dataflow/GuiInterface \
            Core/Geom Core/Geometry \
            Core/Algorithms/Converter \
            Core/Algorithms/Util \
            Core/Persistent \
            Core/Basis Core/Bundle

LIBS :=     $(TEEM_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
