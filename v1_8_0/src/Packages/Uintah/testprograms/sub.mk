SRCDIR := Packages/Uintah/testprograms

SUBDIRS := $(SRCDIR)/TestSuite \
	$(SRCDIR)/TestFastMatrix \
	$(SRCDIR)/TestMatrix3 \
	$(SRCDIR)/TestConsecutiveRangeSet \
	$(SRCDIR)/TestRangeTree \
	$(SRCDIR)/TestBoxGrouper

include $(SCIRUN_SCRIPTS)/recurse.mk

PROGRAM := $(SRCDIR)/RunTests

SRCS	= $(SRCDIR)/RunTests.cc

PSELIBS := Packages/Uintah/testprograms/TestSuite \
	Packages/Uintah/testprograms/TestMatrix3 \
	Packages/Uintah/testprograms/TestConsecutiveRangeSet \
	Packages/Uintah/testprograms/TestRangeTree \
	Packages/Uintah/testprograms/TestBoxGrouper

LIBS := $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk
