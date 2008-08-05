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
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Algorithms/Util \
           Core/Basis           \
           Core/Bundle          \
	   Core/Containers      \
	   Core/Datatypes       \
           Core/Exceptions      \
           Core/Geometry        \
           Core/Persistent      \
	   Core/Thread          \
	   Core/Util

ifeq ($BUILD_DATAFLOW,yes)
  PSELIBS += Core/Geom Core/GuiInterface
endif

LIBS :=

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
