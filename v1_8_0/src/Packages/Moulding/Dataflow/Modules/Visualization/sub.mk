# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

include $(SRCTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Moulding/Dataflow/Modules/Visualization

SRCS     += \
	$(SRCDIR)/VolumeRender.cc\
	$(SRCDIR)/StreamLines.cc\
	$(SRCDIR)/GenField.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Dataflow/Network Dataflow/Ports \
	Dataflow/Widgets\
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/TkExtensions
LIBS := $(TK_LIBRARY) $(GL_LIBS) $(M_LIBRARY)

include $(SRCTOP_ABS)/scripts/smallso_epilogue.mk


