# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Wangxl/Core/ThirdParty

SRCS     += \
	$(SRCDIR)/triangle.c \
#[INSERT NEW CODE FILE HERE]

PSELIBS :=
LIBS := $(M_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

