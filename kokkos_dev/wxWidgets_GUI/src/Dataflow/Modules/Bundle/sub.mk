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

SRCDIR   := Dataflow/Modules/Bundle

SRCS += $(SRCDIR)/BundleSetBundle.cc\
        $(SRCDIR)/BundleSetField.cc\
        $(SRCDIR)/BundleSetMatrix.cc\
        $(SRCDIR)/BundleSetNrrd.cc\
        $(SRCDIR)/BundleSetColorMap.cc\
        $(SRCDIR)/BundleSetColorMap2.cc\
        $(SRCDIR)/BundleSetPath.cc\
        $(SRCDIR)/BundleSetString.cc\
        $(SRCDIR)/BundleInfo.cc\
        $(SRCDIR)/BundleGetBundle.cc\
        $(SRCDIR)/BundleGetField.cc\
        $(SRCDIR)/BundleGetMatrix.cc\
        $(SRCDIR)/BundleGetNrrd.cc\
        $(SRCDIR)/BundleGetColorMap.cc\
        $(SRCDIR)/BundleGetColorMap2.cc\
        $(SRCDIR)/BundleGetPath.cc\
        $(SRCDIR)/BundleGetString.cc\
        $(SRCDIR)/BundleMerge.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Dataflow/Network Dataflow/Ports \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/GeomInterface Core/TkExtensions \
        Core/Volume Core/Bundle

LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

ifeq ($(LARGESOS),no)
SCIRUN_MODULES := $(SCIRUN_MODULES) $(LIBNAME)
endif

