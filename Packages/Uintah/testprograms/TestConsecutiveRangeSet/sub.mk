# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := Packages/Uintah/testprograms/TestConsecutiveRangeSet

SRCS := $(SRCDIR)/TestConsecutiveRangeSet.cc

PSELIBS := \
	Core/Exceptions \
	Core/Thread \
	Core/Containers \
	Packages/Uintah/testprograms/TestSuite

LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

