# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Dataflow/Modules/Matrix

SRCS     += \
	$(SRCDIR)/BldTransform.cc\
	$(SRCDIR)/EditMatrix.cc\
	$(SRCDIR)/ExtractSubmatrix.cc\
	$(SRCDIR)/MatMat.cc\
	$(SRCDIR)/MatSelectVec.cc\
	$(SRCDIR)/MatVec.cc\
	$(SRCDIR)/SolveMatrix.cc\
	$(SRCDIR)/VecVec.cc\
	$(SRCDIR)/VisualizeMatrix.cc\
	$(SRCDIR)/cConjGrad.cc\
	$(SRCDIR)/cPhase.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Dataflow/Network Core/Datatypes Core/Persistent \
	Core/Exceptions Core/Thread Core/Containers \
	Core/TclInterface Core/Geometry Core/Datatypes \
	Core/Util Core/Geom Core/TkExtensions \
	Dataflow/Widgets
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk
