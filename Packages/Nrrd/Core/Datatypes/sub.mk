# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the Core/CCA/Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Core/CCA/Component"
# documentation on how to do it correctly.

include $(SRCTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Nrrd/Core/Datatypes

SRCS     += \
	$(SRCDIR)/NrrdData.cc \
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Core/Persistent Core/Exceptions Core/Containers \
	Core/Thread Core/Geometry Core/Geom Core/TclInterface \
	Core/Math Core/Util

LIBS := $(NRRD_LIBRARY) -lnrrd -lbiff -lair

INCLUDES += -I./Packages $(NRRD_INCLUDE)

include $(SRCTOP_ABS)/scripts/smallso_epilogue.mk

