# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/DaveW/Dataflow/Modules/EGI

SRCS     += \
	$(SRCDIR)/DipoleInSphere.cc\
#	$(SRCDIR)/Anneal.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Dataflow/Ports Dataflow/Network Core/Containers \
	Core/Persistent Core/Exceptions Core/Thread \
	Core/TclInterface Core/Datatypes
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

