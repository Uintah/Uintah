# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := Packages/Uintah/testprograms/TestSuite

SRCS := $(SRCDIR)/Test.cc $(SRCDIR)/Suite.cc $(SRCDIR)/SuiteTree.cc

PSELIBS := Core/Exceptions
LIBS :=

include $(SRCTOP)/scripts/smallso_epilogue.mk
