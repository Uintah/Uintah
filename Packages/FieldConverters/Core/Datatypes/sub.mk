# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/FieldConverters/Core/Datatypes

SRCS     += \
	$(SRCDIR)/Mesh.cc \
	$(SRCDIR)/Surface.cc $(SRCDIR)/TriSurface.cc $(SRCDIR)/SurfTree.cc \
	$(SRCDIR)/ScalarField.cc $(SRCDIR)/ScalarFieldRGBase.cc \
	$(SRCDIR)/ScalarFieldRG.cc $(SRCDIR)/ScalarFieldUG.cc \
	$(SRCDIR)/VectorField.cc $(SRCDIR)/VectorFieldRG.cc \
	$(SRCDIR)/VectorFieldUG.cc
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Core/Persistent Core/Exceptions Core/Containers \
	Core/Thread Core/Geometry Core/Math
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

