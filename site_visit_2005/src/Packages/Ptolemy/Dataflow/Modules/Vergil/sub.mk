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

SRCDIR   := Packages/Ptolemy/Dataflow/Modules/Vergil

SRCS     += \
            $(SRCDIR)/VergilApplication.cc \
#[INSERT NEW CODE FILE HERE]


PSELIBS := Dataflow/Network Dataflow/Ports \
           Core/Datatypes Core/Persistent Core/Containers Core/Util \
           Core/Exceptions Core/Thread Core/GuiInterface \
           Core/TkExtensions \
           Packages/Ptolemy/Core/jni 

#LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)
LIBS := $(LIBS) $(TK_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


