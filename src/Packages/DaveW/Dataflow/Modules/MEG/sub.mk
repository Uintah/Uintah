# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/DaveW/Dataflow/Modules/MEG


SRCS     += \
	$(SRCDIR)/EleValuesToMatLabFile.cc\
	$(SRCDIR)/FieldCurl.cc\
	$(SRCDIR)/MakeCurrentDensityField.cc\
	$(SRCDIR)/MagneticFieldAtPoints.cc\
	$(SRCDIR)/MagneticScalarField.cc\
	$(SRCDIR)/NegateGradient.cc\
	$(SRCDIR)/SurfToVectGeom.cc\
	$(SRCDIR)/ForwardMEG.cc \
#[INSERT NEW MODULE HERE]


PSELIBS := DaveW/Datatypes/General Dataflow/Network Core/Datatypes \
	Core/Persistent Core/Exceptions Core/Thread \
	Core/Datatypes Core/TclInterface Core/Containers \
	Core/Geom
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

