# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := Packages/Uintah/testprograms/TestConsecutiveRangeSet

SRCS := $(SRCDIR)/TestConsecutiveRangeSet.cc

PSELIBS := Core/Exceptions Core/Thread \
	Core/Containers Uintah/testprograms/TestSuite

LIBS := $(XML_LIBRARY) -lmpi

include $(SRCTOP)/scripts/smallso_epilogue.mk

