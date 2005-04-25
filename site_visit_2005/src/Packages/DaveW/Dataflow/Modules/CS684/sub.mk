# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/DaveW/Dataflow/Modules/CS684

SRCS     += \
	$(SRCDIR)/BldBRDF.cc\
	$(SRCDIR)/BldScene.cc\
	$(SRCDIR)/RTrace.cc\
	$(SRCDIR)/Radiosity.cc\
	$(SRCDIR)/RayMatrix.cc\
	$(SRCDIR)/RayTest.cc\
	$(SRCDIR)/XYZtoRGB.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Packages/DaveW/Core/Datatypes/CS684 Dataflow/Widgets \
	Dataflow/Network Dataflow/Ports Core/Containers Core/Exceptions \
	Core/GuiInterface Core/Thread Core/Persistent \
	Core/Geom Core/Geometry Core/Datatypes Core/Util \
	Core/TkExtensions
LIBS := $(TK_LIBRARY) $(GL_LIBS)

include $(SRCTOP)/scripts/smallso_epilogue.mk




