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

SRCDIR   := Packages/Fusion/Dataflow/Modules/DataIO

INCLUDES += $(MDSPLUS_INCLUDE) $(HDF5_INCLUDE)

SRCS     += \
	$(SRCDIR)/FusionFieldReader.cc\
	$(SRCDIR)/FusionFieldSetReader.cc\
	$(SRCDIR)/MDSPlusFieldReader.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Dataflow/Network Dataflow/Ports \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/TkExtensions \
	Packages/Fusion/Core/ThirdParty \
	Packages/Teem/Core/Datatypes

LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(MDSPLUS_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


