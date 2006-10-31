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


SRCS     += \
            $(SRCDIR)/ApplyMappingMatrix.cc\
            $(SRCDIR)/CalculateFieldData.cc\
            $(SRCDIR)/CalculateIsInsideField.cc\
            $(SRCDIR)/CalculateLinkBetweenOppositeFieldBoundaries.cc\
            $(SRCDIR)/ClearAndChangeFieldBasis.cc\
            $(SRCDIR)/ClipAtIndeces.cc\
            $(SRCDIR)/ClipBySelectionMask.cc\
            $(SRCDIR)/ClipFieldByFunction.cc\
            $(SRCDIR)/ConvertIndicesToFieldData.cc\
            $(SRCDIR)/ConvertMappingMatrixToFieldData.cc\
            $(SRCDIR)/ConvertMeshToTetVol.cc\
            $(SRCDIR)/ConvertMeshToTriSurf.cc\
            $(SRCDIR)/ConvertToUnstructuredMesh.cc\
            $(SRCDIR)/CreateLinkBetweenMeshAndCompGridByDomain.cc\
            $(SRCDIR)/CreateLinkBetweenMeshAndCompGrid.cc\
            $(SRCDIR)/DistanceField.cc\
            $(SRCDIR)/FieldsAlgo.cc\
            $(SRCDIR)/FindClosestNodeByValue.cc\
            $(SRCDIR)/FindClosestNode.cc\
            $(SRCDIR)/GatherFields.cc\
            $(SRCDIR)/GetBoundingBox.cc\
            $(SRCDIR)/GetDomainBoundary.cc\
            $(SRCDIR)/GetFieldBoundary.cc\
            $(SRCDIR)/GetFieldData.cc\
            $(SRCDIR)/GetFieldDataMinMax.cc\
            $(SRCDIR)/GetFieldInfo.cc\
            $(SRCDIR)/GetFieldMeasure.cc\
            $(SRCDIR)/MapCurrentDensityOntoField.cc\
            $(SRCDIR)/MapFieldDataFromElemToNode.cc\
            $(SRCDIR)/MapFieldDataFromNodeToElem.cc\
            $(SRCDIR)/Mapping.cc\
            $(SRCDIR)/MergeFields.cc\
            $(SRCDIR)/MergeMeshes.cc\
            $(SRCDIR)/RemoveUnusedNodes.cc\
            $(SRCDIR)/ScaleFieldMeshAndData.cc\
            $(SRCDIR)/SetFieldData.cc\
            $(SRCDIR)/SplitAndMergeFieldByDomain.cc\
            $(SRCDIR)/SplitByConnectedRegion.cc\
            $(SRCDIR)/ToPointCloud.cc\
            $(SRCDIR)/TransformMeshWithTransform.cc\
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
