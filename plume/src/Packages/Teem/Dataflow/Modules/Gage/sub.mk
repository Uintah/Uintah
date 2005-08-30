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

SRCDIR   := Packages/Teem/Dataflow/Modules/Gage

SRCS     += \
	$(SRCDIR)/GageProbe.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := \
	Core/Containers    \
	Core/Datatypes     \
        Core/Exceptions    \
        Core/Geom          \
        Core/GeomInterface \
	Core/Geometry      \
	Core/GuiInterface  \
        Core/Persistent    \
	Core/Thread        \
	Core/Util          \
	Core/TkExtensions  \
	Dataflow/Network   \
	Dataflow/Ports

LIBS := $(TEEM_LIBRARY) $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


