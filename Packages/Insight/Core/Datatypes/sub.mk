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
SRCDIR   := Packages/Insight/Core/Datatypes

SRCS     += $(SRCDIR)/ITKDatatype.cc \

#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Persistent Core/Exceptions Core/Containers \
	Core/Thread Core/Geometry Core/GuiInterface \
	Core/Math Core/Geom Core/Util Core/Datatypes 

LIBS := $(GL_LIBRARY) $(M_LIBRARY) $(INSIGHT_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

