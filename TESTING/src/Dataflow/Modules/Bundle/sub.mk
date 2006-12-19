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

SRCS += $(SRCDIR)/InsertBundlesIntoBundle.cc\
        $(SRCDIR)/InsertFieldsIntoBundle.cc\
        $(SRCDIR)/InsertMatricesIntoBundle.cc\
        $(SRCDIR)/InsertNrrdsIntoBundle.cc\
        $(SRCDIR)/InsertColorMapsIntoBundle.cc\
        $(SRCDIR)/InsertColorMap2sIntoBundle.cc\
        $(SRCDIR)/InsertPathsIntoBundle.cc\
        $(SRCDIR)/InsertStringsIntoBundle.cc\
        $(SRCDIR)/ReportBundleInfo.cc\
        $(SRCDIR)/GetBundlesFromBundle.cc\
        $(SRCDIR)/GetFieldsFromBundle.cc\
        $(SRCDIR)/GetMatricesFromBundle.cc\
        $(SRCDIR)/GetNrrdsFromBundle.cc\
        $(SRCDIR)/GetColorMapsFromBundle.cc\
        $(SRCDIR)/GetColorMap2sFromBundle.cc\
        $(SRCDIR)/GetPathsFromBundle.cc\
        $(SRCDIR)/GetStringsFromBundle.cc\
        $(SRCDIR)/JoinBundles.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := \
	Core/Datatypes     \
	Dataflow/Network   \
        Core/Persistent    \
	Core/Containers    \
	Core/Util          \
	Core/Exceptions    \
	Core/Thread        \
	Dataflow/GuiInterface  \
        Core/Geom          \
	Core/Datatypes     \
	Core/Geometry      \
        Core/GeomInterface \
	Dataflow/TkExtensions  \
        Core/Volume        \
	Core/Bundle

LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

ifeq ($(LARGESOS),no)
  SCIRUN_MODULES := $(SCIRUN_MODULES) $(LIBNAME)
endif

