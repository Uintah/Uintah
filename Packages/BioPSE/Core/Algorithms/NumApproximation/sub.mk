# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/BioPSE/Core/Algorithms/NumApproximation

SRCS += \
	$(SRCDIR)/BuildFEMatrix.cc \
	$(SRCDIR)/BuildBEMatrix.cc \
	$(SRCDIR)/BuildHexFEMatrix.cc \
	$(SRCDIR)/BuildTriFEMatrix.cc \
	$(SRCDIR)/ReferenceElement.cc \

PSELIBS := Core/Datatypes Core/Persistent Core/Exceptions Core/Containers \
	Core/Thread Core/Geometry Core/Geom Core/GuiInterface \
	Core/Math Core/Util 

LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
