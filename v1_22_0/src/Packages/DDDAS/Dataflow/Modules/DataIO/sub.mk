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
	$(SRCDIR)/Reader.cc\
	$(SRCDIR)/StreamReader.cc\
	$(SRCDIR)/SendIntermediateTest.cc\
	$(SRCDIR)/Mesh3dReader.cc\
#	$(SRCDIR)/NetConnector.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Dataflow/Network Dataflow/Ports \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/Geometry Core/TkExtensions \
	Packages/DDDAS/Core/Datatypes Packages/DDDAS/Core/Utils \
#        Core/Util/Comm \

#INCLUDES += $(SCISOCK_INCLUDE)
LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY) #$(SCISOCK_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


