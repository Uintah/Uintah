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

SRCDIR   := Packages/ModelCreation/Core/DataIO

SRCS     += $(SRCDIR)/DataIOAlgo.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS :=  Core/Datatypes Core/Util Core/Containers \
            Core/Exceptions Core/Thread Core/GuiInterface \
            Core/Geom Core/Geometry Dataflow/Network \
            Packages/ModelCreation/Core/Algorithms \
            Packages/ModelCreation/Core/Util \
            Dataflow/Modules/Fields \
            Core/Algorithms/Fields \
            Core/Algorithms/Util \
            Core/Persistent \
            Core/Basis Core/Bundle \
            Core/ImportExport \
            
LIBS :=

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
