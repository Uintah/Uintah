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

SRCDIR   := Core/Algorithms/Math

SRCS     += $(SRCDIR)/MathAlgo.cc \
            $(SRCDIR)/BasicIntegrators.cc \
            $(SRCDIR)/BuildFEMatrix.cc \
            $(SRCDIR)/CreateFEDirichletBC.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS :=  Core/Datatypes Core/Util Core/Containers      \
            Core/Exceptions Core/Thread Dataflow/GuiInterface \
            Core/Geom Core/Geometry \
            Core/Algorithms/Util    \
            Core/Persistent         \
            Core/Basis Core/Bundle Core/Math

LIBS := $(TEEM_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
