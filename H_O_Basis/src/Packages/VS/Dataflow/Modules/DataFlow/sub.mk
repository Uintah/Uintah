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

SRCDIR   := Packages/VS/Dataflow/Modules/DataFlow

SRCS     += \
	$(SRCDIR)/HotBox.cc\
	$(SRCDIR)/VS_SCI_HotBox.cc\
	$(SRCDIR)/labelmaps.cc\
	$(SRCDIR)/soapC.cc\
	$(SRCDIR)/soapClient.cc\
	$(SRCDIR)/stdsoap2.cc
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Dataflow/Network Dataflow/Ports \
        Dataflow/Widgets Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry  Core/GeomInterface \
        Core/TkExtensions Dataflow/Modules/Fields Dataflow/XMLUtil
LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY) $(XML_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


