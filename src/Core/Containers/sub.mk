# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/Containers

SRCS     += $(SRCDIR)/Sort.cc $(SRCDIR)/String.cc $(SRCDIR)/PQueue.cc \
	$(SRCDIR)/TrivialAllocator.cc $(SRCDIR)/BitArray1.cc \
	$(SRCDIR)/templates.cc $(SRCDIR)/ConsecutiveRangeSet.cc

PSELIBS := Core/Exceptions Core/Tester Core/Thread

include $(SRCTOP)/scripts/smallso_epilogue.mk

