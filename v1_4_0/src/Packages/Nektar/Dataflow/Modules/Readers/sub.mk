# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Nektar/Dataflow/Modules/Readers

SRCS     += \
	$(SRCDIR)/ICPackages/NektarReader.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Dataflow/Network Core/Datatypes Core/Datatypes \
	Core/Persistent Core/Exceptions Core/Thread \
	Core/Containers Core/GuiInterface Core/Geom
LIBS := $(NEKTAR_LIBRARY)  

include $(SRCTOP)/scripts/smallso_epilogue.mk

