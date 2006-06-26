# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/DaveW/Dataflow/Modules/Readers

SRCS     += \
	$(SRCDIR)/ContourSetReader.cc\
	$(SRCDIR)/SegFldReader.cc\
	$(SRCDIR)/SigmaSetReader.cc\
#[INSERT NEW CODE FILE HERE]

#	$(SRCDIR)/TensorFieldReader.cc\

PSELIBS := Packages/DaveW/Core/Datatypes/General Dataflow/Network \
	Dataflow/Ports Core/Exceptions Core/Thread Core/Containers \
	Core/GuiInterface Core/Persistent Core/Datatypes
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

