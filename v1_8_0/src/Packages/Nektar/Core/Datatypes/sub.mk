# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Nektar/Core/Datatypes

SRCS += $(SRCDIR)/Packages/NektarScalarField.cc \
	$(SRCDIR)/Packages/NektarVectorField.cc  \
	$(SRCDIR)/Packages/NektarScalarFieldPort.cc \
	$(SRCDIR)/Packages/NektarVectorFieldPort.cc  

PSELIBS := Core/Persistent Core/Exceptions Core/Containers \
	Core/Thread Core/Geometry Core/Geom Core/GuiInterface \
	Core/Math
LIBS := $(NEKTAR_LIBRARY) $(M_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

clean::

