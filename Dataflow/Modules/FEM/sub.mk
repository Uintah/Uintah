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

SRCDIR   := PSECommon/Modules/FEM

SRCS     += \
	$(SRCDIR)/ApplyBC.cc\
	$(SRCDIR)/BuildFEMatrix.cc\
	$(SRCDIR)/ComposeError.cc\
        $(SRCDIR)/ErrorInterval.cc\
	$(SRCDIR)/FEMError.cc\
        $(SRCDIR)/MeshRefiner.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := PSECore/Dataflow PSECore/Datatypes SCICore/Datatypes \
	SCICore/Persistent SCICore/Thread SCICore/Containers \
	SCICore/Exceptions SCICore/TclInterface SCICore/Geometry
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk
