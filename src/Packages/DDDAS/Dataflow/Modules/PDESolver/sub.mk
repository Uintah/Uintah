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

SRCDIR   := Packages/DDDAS/Dataflow/Modules/PDESolver

SRCS     += \
	$(SRCDIR)/FEM.cc\
#[INSERT NEW CODE FILE HERE]

AGGIEFEMDIR := $(HOME)/AggieFEM
INCLUDES += -I$(AGGIEFEMDIR)

PSELIBS := Core/Datatypes Dataflow/Network Dataflow/Ports \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/GeomInterface Core/TkExtensions
LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY) \
	-L$(AGGIEFEMDIR)/linalg -llinalg \
	-L$(AGGIEFEMDIR)/mesh -lmesh \
	-L$(AGGIEFEMDIR)/fem -lfem \
	-L$(AGGIEFEMDIR)/general -lgeneral


include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


