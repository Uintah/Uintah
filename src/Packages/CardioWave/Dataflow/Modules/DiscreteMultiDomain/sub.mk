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

SRCDIR   := Packages/CardioWave/Dataflow/Modules/DiscreteMultiDomain

SRCS     += \
	$(SRCDIR)/DMDAddMembrane.cc\
	$(SRCDIR)/DMDCreateDomain.cc\
	$(SRCDIR)/DMDCreateSimulator.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Dataflow/Network Dataflow/Ports \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/GeomInterface Core/TkExtensions \
        Core/Bundle \
        Packages/ModelCreation/Core/Converter \
        Packages/ModelCreation/Core/Fields \
        Packages/ModelCreation/Core/Numeric \
        Packages/CardioWave/Core/XML \
        
        
LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


