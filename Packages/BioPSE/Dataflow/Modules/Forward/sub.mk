# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the Core/CCA/Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Core/CCA/Component"
# documentation on how to do it correctly.

include $(SRCTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := Packages/BioPSE/Dataflow/Modules/Forward


SRCS     += \
	$(SRCDIR)/ApplyFEMCurrentSource.cc\
	$(SRCDIR)/BuildFEMatrix.cc\
	$(SRCDIR)/DipoleInSphere.cc\
#[INSERT NEW CODE FILE HERE]
#	$(SRCDIR)/DipoleInSphere.cc\


PSELIBS := Packages/BioPSE/Core/Datatypes Core/Datatypes \
	Dataflow/Network Dataflow/Ports \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/TkExtensions
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP_ABS)/scripts/smallso_epilogue.mk


