# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Butson/Dataflow/Modules/Genesis


SRCS     += \
	$(SRCDIR)/GenesisMatrixReader.cc\
#[INSERT NEW MODULE HERE]


PSELIBS := Dataflow/Network \
	Dataflow/Ports Core/Persistent Core/Exceptions Core/Thread \
	Core/Datatypes Core/GuiInterface Core/Containers \
	Core/Geom
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

