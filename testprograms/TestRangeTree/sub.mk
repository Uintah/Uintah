# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := Packages/Uintah/testprograms/TestRangeTree

SRCS := $(SRCDIR)/TestRangeTree.cc

PSELIBS := \
	Core/Exceptions \
	Core/Thread \
	Core/Containers \
	Packages/Uintah/testprograms/TestSuite

LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk
