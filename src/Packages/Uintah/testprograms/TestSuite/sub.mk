# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Packages/Uintah/testprograms/TestSuite

SRCS := $(SRCDIR)/Test.cc $(SRCDIR)/Suite.cc $(SRCDIR)/SuiteTree.cc

PSELIBS := Core/Exceptions
LIBS :=

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
