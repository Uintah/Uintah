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
	$(SRCDIR)/GenesisMatrixReader.cc\
	$(SRCDIR)/SegFldReader.cc\
	$(SRCDIR)/SigmaSetReader.cc\
	$(SRCDIR)/TensorFieldReader.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := DaveW/Datatypes/General Dataflow/Network Core/Datatypes \
	Core/Exceptions Core/Thread Core/Containers \
	Core/TclInterface Core/Persistent
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

