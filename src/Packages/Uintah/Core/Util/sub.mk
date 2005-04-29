# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Core/Util

SRCS     += \
	$(SRCDIR)/RefCounted.cc \
	$(SRCDIR)/Util.cc

PSELIBS := \
	Core/Thread \
	Core/Exceptions

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

