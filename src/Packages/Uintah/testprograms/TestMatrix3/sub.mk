#
# Makefile fragment for this subdirectory
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := Uintah/testprograms/TestMatrix3

SRCS := $(SRCDIR)/testmatrix3.cc

PSELIBS := Uintah/Interface Uintah/Grid Uintah/Parallel \
	Uintah/Exceptions SCICore/Exceptions SCICore/Thread \
	SCICore/Geometry PSECore/XMLUtil Uintah/Math \
	Uintah/Components/MPM Uintah/testprograms/TestSuite

LIBS := $(XML_LIBRARY) -lmpi

include $(SRCTOP)/scripts/smallso_epilogue.mk

