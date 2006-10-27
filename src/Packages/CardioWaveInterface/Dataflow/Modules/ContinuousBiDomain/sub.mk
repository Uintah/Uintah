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

SRCDIR   := Packages/CardioWaveInterface/Dataflow/Modules/ContinuousBiDomain

SRCS     += \
	$(SRCDIR)/CBDAddMembrane.cc\
	$(SRCDIR)/CBDAddReference.cc\
	$(SRCDIR)/CBDAddStimulus.cc\
	$(SRCDIR)/CBDCreateDomain.cc\
	$(SRCDIR)/CBDGenerateSimulation.cc\
	$(SRCDIR)/CBDSetupSimulation.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Dataflow/Network \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Dataflow/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/GeomInterface Dataflow/TkExtensions \
        Core/Bundle \
        Core/Algorithms/Converter \
        Core/Algorithms/Fields \
        Core/Algorithms/Math \
        Packages/CardioWaveInterface/Core/XML \
        Packages/CardioWaveInterface/Core/Model

LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


