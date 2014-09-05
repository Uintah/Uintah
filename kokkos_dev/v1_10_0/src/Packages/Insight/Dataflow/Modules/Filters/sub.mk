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

INCLUDES += $(INSIGHT_INCLUDE)
SRCDIR   := Packages/Insight/Dataflow/Modules/Filters

SRCS     += \
	$(SRCDIR)/DiscreteGaussianImageFilter.cc\
	$(SRCDIR)/CannySegmentationLevelSetImageFilter.cc\
	$(SRCDIR)/UChar2DToFloat2D.cc\
	$(SRCDIR)/GradientAnisotropicDiffusionImageFilter.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Packages/Insight/Core/Datatypes \
	Core/Datatypes Dataflow/Network Dataflow/Ports \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/TkExtensions Core/GeomInterface

LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY) $(INSIGHT_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


