SRCDIR := Packages/Uintah/testprograms

SUBDIRS := $(SRCDIR)/TestSuite \
	$(SRCDIR)/TestMatrix3 \
	$(SRCDIR)/TestConsecutiveRangeSet

include $(SRCTOP)/scripts/recurse.mk

PROGRAM := $(SRCDIR)/RunTests

SRCS	= $(SRCDIR)/RunTests.cc

PSELIBS := Packages/Uintah/testprograms/TestSuite \
	Packages/Uintah/testprograms/TestMatrix3 \
	Packages/Uintah/testprograms/TestConsecutiveRangeSet

LIBS := $(XML_LIBRARY) -lm

include $(SRCTOP)/scripts/program.mk
