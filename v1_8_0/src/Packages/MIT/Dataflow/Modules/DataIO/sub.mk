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

SRCDIR   := Packages/MIT/Dataflow/Modules/DataIO

SRCS     += \
	$(SRCDIR)/MeasurementsReader.cc\
	$(SRCDIR)/DistributionReader.cc\
	$(SRCDIR)/MetropolisWriter.cc\
	$(SRCDIR)/MetropolisReader.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Packages/MIT/Core/Datatypes \
	Core/Datatypes Dataflow/Network Dataflow/Ports \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/TkExtensions
LIBS := $(TK_LIBRARY) $(GL_LIBS) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


