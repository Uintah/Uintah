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

SRCDIR   := Packages/ModelCreation/Dataflow/Modules/FieldsGeometry

SRCS     += \
 	$(SRCDIR)/ComputeFieldNodes.cc\
  	$(SRCDIR)/ComputeFieldsNodes.cc\
	$(SRCDIR)/ConvertToPointCloud.cc\
	$(SRCDIR)/ConvertToTetVol.cc\
	$(SRCDIR)/ConvertToTriSurf.cc\
	$(SRCDIR)/MergeNodes.cc\
	$(SRCDIR)/LinkFieldBoundary.cc\
	$(SRCDIR)/Unstructure.cc\
	$(SRCDIR)/ScaleField.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Dataflow/Network \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/GeomInterface Core/TkExtensions \
        Packages/ModelCreation/Core/Algorithms \
        Packages/ModelCreation/Core/Datatypes \
        Core/Algorithms/Fields \
        Core/Algorithms/Converter \
        
LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


