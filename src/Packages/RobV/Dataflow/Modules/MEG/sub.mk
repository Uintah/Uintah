# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/RobV/Dataflow/Modules/MEG


SRCS     += \
	$(SRCDIR)/EleValuesToMatLabFile.cc\
	$(SRCDIR)/FieldCurl.cc\
	$(SRCDIR)/MagneticFieldAtPoints.cc\
	$(SRCDIR)/MagneticScalarField.cc\
	$(SRCDIR)/MakeCurrentDensityField.cc\
	$(SRCDIR)/NegateGradient.cc\
	$(SRCDIR)/SurfToVectGeom.cc\
#	$(SRCDIR)/ForwardMEG.cc \
#[INSERT NEW MODULE HERE]


PSELIBS := Packages/RobV/Core/Datatypes/MEG Dataflow/Network \
	Dataflow/Ports Core/Persistent Core/Exceptions Core/Thread \
	Core/Datatypes Core/GuiInterface Core/Containers \
	Core/Geom
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

