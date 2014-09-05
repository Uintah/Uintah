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

SRCS     += $(SRCDIR)/MathAlgo.cc\
            $(SRCDIR)/BuildFEMatrix.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS :=  Core/Datatypes Core/Util Core/Containers      \
            Core/Exceptions Core/Thread Core/GuiInterface \
            Core/Geom Core/Geometry \
            Core/Algorithms/Util    \
            Core/Persistent         \
            Core/Basis Core/Bundle

LIBS :=

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
