# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/DaveW/Dataflow/Modules/FEM

SRCS     += \
	$(SRCDIR)/CStoGeom.cc\
	$(SRCDIR)/CStoSFRG.cc\
	$(SRCDIR)/DipoleMatToGeom.cc\
	$(SRCDIR)/DipoleSourceRHS.cc\
	$(SRCDIR)/ErrorMetric.cc\
	$(SRCDIR)/FieldFromBasis.cc\
	$(SRCDIR)/RecipBasis.cc\
	$(SRCDIR)/RemapVector.cc\
	$(SRCDIR)/SeedDipoles2.cc\
	$(SRCDIR)/VecSplit.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := DaveW/Datatypes/General Dataflow/Widgets Dataflow/Ports \
	Dataflow/Network Core/Persistent Core/Exceptions \
	Core/Datatypes Core/Thread Core/TclInterface \
	Core/Geom Core/Containers Core/Geometry Core/Math
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

