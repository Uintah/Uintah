# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/testprograms/TestFastMatrix

SRCS := $(SRCDIR)/testfastmatrix.cc

PSELIBS := \
	Packages/Uintah/Core/Math
PROGRAM := $(SRCDIR)/testfastmatrix

LIBS := $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

