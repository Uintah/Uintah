# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/DaveW/Dataflow/Modules/ISL

SRCS     += \
	$(SRCDIR)/ConductivitySearch.cc\
	$(SRCDIR)/Downhill_Simplex3.cc\
	$(SRCDIR)/LeastSquaresSolve.cc\
	$(SRCDIR)/OptDip.cc\
	$(SRCDIR)/SGI_LU.cc\
	$(SRCDIR)/SGI_Solve.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Packages/DaveW/ThirdParty/NumRec \
	Packages/DaveW/ThirdParty/OldLinAlg \
	Dataflow/Ports Dataflow/Network Core/Datatypes \
	Core/Persistent Core/Exceptions Core/Math Core/Thread \
	Core/Containers Core/TclInterface Core/Geometry
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

