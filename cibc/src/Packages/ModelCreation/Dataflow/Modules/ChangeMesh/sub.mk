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

SRCDIR   := Packages/ModelCreation/Dataflow/Modules/ChangeMesh

SRCS     += \
	$(SRCDIR)/CalculateLinkBetweenOppositeFieldBoundaries.cc\
	$(SRCDIR)/CalculateMeshNodes.cc\
	$(SRCDIR)/ConvertMeshToPointCloud.cc\
	$(SRCDIR)/ConvertMeshToTetVol.cc\
	$(SRCDIR)/ConvertMeshToTriSurf.cc\
	$(SRCDIR)/ConvertToUnstructuredMesh.cc\
	$(SRCDIR)/FindClosestNodeIndexByValue.cc\
	$(SRCDIR)/FindClosestNodeIndex.cc\
	$(SRCDIR)/GetBoundingBox.cc\
	$(SRCDIR)/MergeMeshNodes.cc\
	$(SRCDIR)/ScaleFieldMeshAndData.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Dataflow/Network \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Dataflow/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/GeomInterface Dataflow/TkExtensions \
        Packages/ModelCreation/Core/Datatypes \
        Core/Algorithms/Fields \
        Core/Algorithms/Converter \
        Core/Algorithms/ArrayMath \

LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


