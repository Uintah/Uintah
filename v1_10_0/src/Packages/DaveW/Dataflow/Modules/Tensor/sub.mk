# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/DaveW/Dataflow/Modules/Tensor

SRCS     += \
	$(SRCDIR)/Bundles.cc\
	$(SRCDIR)/Flood.cc\
	$(SRCDIR)/TensorAccessFields.cc\
	$(SRCDIR)/TensorAnisotropy.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Packages/DaveW/Core/Datatypes/General Dataflow/Ports \
	Dataflow/Network Dataflow/Widgets Core/Persistent Core/Geometry \
	Core/Math Core/Exceptions Core/Datatypes \
	Core/Thread Core/Geom Core/Containers \
	Core/GuiInterface 
LIBS := $(M_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

