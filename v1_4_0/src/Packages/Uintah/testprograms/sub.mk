SRCDIR := Packages/Uintah/testprograms

SUBDIRS := $(SRCDIR)/TestSuite \
	$(SRCDIR)/TestMatrix3 \
	$(SRCDIR)/TestConsecutiveRangeSet \
	$(SRCDIR)/TestRangeTree

include $(SCIRUN_SCRIPTS)/recurse.mk

PROGRAM := $(SRCDIR)/RunTests

SRCS	= $(SRCDIR)/RunTests.cc

PSELIBS := Packages/Uintah/testprograms/TestSuite \
	Packages/Uintah/testprograms/TestMatrix3 \
	Packages/Uintah/testprograms/TestConsecutiveRangeSet \
	Packages/Uintah/testprograms/TestRangeTree

LIBS := $(XML_LIBRARY) -lm

include $(SCIRUN_SCRIPTS)/program.mk
