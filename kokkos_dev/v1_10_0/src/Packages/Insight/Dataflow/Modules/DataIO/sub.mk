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
SRCDIR   := Packages/Insight/Dataflow/Modules/DataIO

SRCS     += \
	$(SRCDIR)/ImageFileReader.cc\
	$(SRCDIR)/ImageFileWriter.cc\
	$(SRCDIR)/ImageReaderUChar2D.cc \
	$(SRCDIR)/Switch.cc\
	$(SRCDIR)/ImageReaderFloat2D.cc\
	$(SRCDIR)/ImageReaderFloat3D.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Packages/Insight/Core/Datatypes Dataflow/Network \
	Packages/Insight/Dataflow/Ports \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/TkExtensions Core/GeomInterface

LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY) $(INSIGHT_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


