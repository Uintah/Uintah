# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Dataflow/Modules/FEM

SRCS     += \
	$(SRCDIR)/BuildFEMatrix.cc
#	$(SRCDIR)/ApplyBC.cc\
#	$(SRCDIR)/ComposeError.cc\
#        $(SRCDIR)/ErrorInterval.cc\
#	$(SRCDIR)/FEMError.cc\
#        $(SRCDIR)/MeshRefiner.cc
#[INSERT NEW CODE FILE HERE]

PSELIBS := Dataflow/Network Dataflow/Ports Core/Datatypes Core/Datatypes \
	Core/Persistent Core/Thread Core/Containers \
	Core/Exceptions Core/TclInterface Core/Geometry
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk
