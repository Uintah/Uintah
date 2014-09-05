#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := Uintah/testprograms/TestSuite

SRCS := $(SRCDIR)/Test.cc $(SRCDIR)/Suite.cc $(SRCDIR)/SuiteTree.cc

PSELIBS := SCICore/Exceptions
LIBS :=

include $(SRCTOP)/scripts/smallso_epilogue.mk
