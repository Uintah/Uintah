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

SRCDIR   := Packages/CardiacVis/Core/Algorithms/Fields

SRCS     += $(SRCDIR)/TriSurfPhaseFilter.cc\
            $(SRCDIR)/TracePoints.cc\
            $(SRCDIR)/FieldsAlgo.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS :=  Core/Datatypes Core/Util Core/Containers \
            Core/Exceptions Core/Thread Dataflow/GuiInterface \
            Core/Basis Core/Bundle Core/Persistent \
            Core/Geom Core/Geometry \
            Core/Algorithms/Util \
            

LIBS :=     $(TEEM_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
