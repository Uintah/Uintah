# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the Core/CCA/Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Core/CCA/Component"
# documentation on how to do it correctly.

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := Packages/BioPSE/Dataflow/Modules/Forward

SRCS     += \
#	$(SRCDIR)/DipoleInSphere.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Dataflow/Network \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/TclInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/TkExtensions
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk


