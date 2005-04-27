# Makefile fragment for this subdirectory


SRCDIR   := Packages/Uintah/Core/Util

SRCS     += \
	$(SRCDIR)/RefCounted.cc \
	$(SRCDIR)/Util.cc

PSELIBS := \
	Core/Thread \
	Core/Exceptions


