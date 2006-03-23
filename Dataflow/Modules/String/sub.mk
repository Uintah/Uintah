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

SRCDIR   := Dataflow/Modules/String

SRCS     += \
	$(SRCDIR)/CreateString.cc\
	$(SRCDIR)/StringInfo.cc\
	$(SRCDIR)/JoinStrings.cc\
	$(SRCDIR)/SprintfString.cc\
	$(SRCDIR)/SprintfMatrix.cc\
	$(SRCDIR)/GetFileName.cc\
	$(SRCDIR)/SplitFileName.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := \
	Core/Containers    \
	Core/Datatypes     \
        Core/Exceptions    \
        Core/Geom          \
	Core/Geometry      \
        Core/GeomInterface \
	Core/GuiInterface  \
        Core/Persistent    \
	Core/Thread        \
	Core/TkExtensions  \
	Core/Util          \
	Dataflow/Network   

LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

ifeq ($(LARGESOS),no)
SCIRUN_MODULES := $(SCIRUN_MODULES) $(LIBNAME)
endif

