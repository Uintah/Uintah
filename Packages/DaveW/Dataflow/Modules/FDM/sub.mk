# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/DaveW/Dataflow/Modules/FDM

SRCS     += \
	$(SRCDIR)/BuildFDField.cc\
	$(SRCDIR)/BuildFDMatrix.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Packages/DaveW/Core/Datatypes/General \
	Core/Datatypes Dataflow/Network \
	Core/TclInterface Core/Persistent Core/Exceptions \
	Dataflow/Ports Core/Thread Core/Containers
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

