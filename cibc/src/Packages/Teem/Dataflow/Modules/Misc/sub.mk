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

SRCDIR   := Packages/Teem/Dataflow/Modules/Misc

SRCS     += \
        $(SRCDIR)/BuildDerivedNrrdWithGage.cc\
        $(SRCDIR)/BuildNrrdGradient.cc\
        $(SRCDIR)/BuildNrrdGradientAndMagnitude.cc\
        $(SRCDIR)/ClassifyHeadTissuesFromMultiMRI.cc\
        $(SRCDIR)/ChooseNrrd.cc\
        $(SRCDIR)/ReportNrrdInfo.cc\
        $(SRCDIR)/SelectTimeStepFromNrrd.cc\
        $(SRCDIR)/SetNrrdProperty.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := \
        Core/Containers    \
        Core/Datatypes     \
        Core/Exceptions    \
        Core/Geom          \
        Core/GeomInterface \
        Core/Geometry      \
        Dataflow/GuiInterface  \
        Core/Persistent    \
        Core/Thread        \
        Core/Util          \
        Dataflow/TkExtensions  \
        Dataflow/Network   

LIBS := $(TEEM_LIBRARY) $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


ifeq ($(LARGESOS),no)
  TEEM_MODULES := $(TEEM_MODULES) $(LIBNAME)
endif
