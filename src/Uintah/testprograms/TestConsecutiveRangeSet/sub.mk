#
# Makefile fragment for this subdirectory
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := Uintah/testprograms/TestConsecutiveRangeSet

SRCS := $(SRCDIR)/TestConsecutiveRangeSet.cc

PSELIBS := SCICore/Exceptions SCICore/Thread \
	SCICore/Containers Uintah/testprograms/TestSuite

LIBS := $(XML_LIBRARY) -lmpi

include $(SRCTOP)/scripts/smallso_epilogue.mk

