#
# Makefile fragment for this subdirectory
#

# *** NOTE ***
# 
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := PSECommon/Modules/Example

SRCS     += \
	$(SRCDIR)/GeomPortTest.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := PSECore/Dataflow PSECore/Datatypes SCICore/Containers \
	SCICore/Thread SCICore/Geom SCICore/TclInterface
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk
