SRCDIR := Uintah/testprograms

SUBDIRS := $(SRCDIR)/TestSuite \
	$(SRCDIR)/TestMatrix3

include $(SRCTOP)/scripts/recurse.mk

PROGRAM := $(SRCDIR)/RunTests

SRCS	= $(SRCDIR)/RunTests.cc

PSELIBS := Uintah/testprograms/TestSuite \
	Uintah/testprograms/TestMatrix3

LIBS := -lm

include $(SRCTOP)/scripts/program.mk
