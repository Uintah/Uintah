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

SRCDIR   := Packages/MIT/Core/Datatypes

SRCS     += \
	$(SRCDIR)/MetropolisData.cc
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Persistent Core/Exceptions Core/Containers \
	Core/Util Core/Datatypes Core/Geom
LIBS :=

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

