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

SRCDIR   := Packages/DDDAS/Dataflow/Modules/DataIO

SRCS     += \
	$(SRCDIR)/StreamReader.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Dataflow/Network \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Dataflow/GuiInterface \
        Core/Geom Core/Geometry Dataflow/TkExtensions \


INCLUDES += $(LIBGEOTIFF_INCLUDE)
LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY) $(LIBGEOTIFF_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


