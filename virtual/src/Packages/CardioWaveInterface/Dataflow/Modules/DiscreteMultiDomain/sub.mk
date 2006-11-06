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

SRCDIR   := Packages/CardioWaveInterface/Dataflow/Modules/DiscreteMultiDomain

SRCS     += \
	$(SRCDIR)/DMDAddDomainElectrodes.cc\
	$(SRCDIR)/DMDAddMembrane.cc\
	$(SRCDIR)/DMDAddMembraneElectrodes.cc\
	$(SRCDIR)/DMDAddReference.cc\
	$(SRCDIR)/DMDAddStimulus.cc\
	$(SRCDIR)/DMDAddStimulusSeries.cc\
	$(SRCDIR)/DMDCreateDomain.cc\
	$(SRCDIR)/DMDGenerateSimulation.cc\
	$(SRCDIR)/DMDSetupSimulation.cc\
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
				Core/OS \
        Packages/CardioWaveInterface/Core/XML \
        Packages/CardioWaveInterface/Core/Model

LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


