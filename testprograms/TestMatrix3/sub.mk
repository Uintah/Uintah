# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := Packages/Uintah/testprograms/TestMatrix3

SRCS := $(SRCDIR)/testmatrix3.cc

PSELIBS := Packages/Uintah/Interface Packages/Uintah/Grid Packages/Uintah/Parallel \
	Uintah/Exceptions Core/Exceptions Core/Thread \
	Core/Geometry PSECore/XMLUtil Uintah/Math \
	Uintah/Core/CCA/Components/MPM Uintah/testprograms/TestSuite

LIBS := $(XML_LIBRARY) -lmpi -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

