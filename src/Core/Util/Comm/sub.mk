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

SRCDIR   := Core/Util/Comm

SRCS     += \
	$(SRCDIR)/NetInterface.cc \
	$(SRCDIR)/NetConnection.cc \
#[INSERT NEW CODE FILE HERE]
 
INCLUDES += $(SCISOCK_INCLUDE)
PSELIBS :=
LIBS := $(SCISOCK_LIBRARY)
 
include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

