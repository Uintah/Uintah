# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/Util

SRCS     += $(SRCDIR)/Debug.cc $(SRCDIR)/Timer.cc \
	$(SRCDIR)/soloader.cc $(SRCDIR)/DebugStream.cc

PSELIBS := Core/Containers Core/Exceptions
LIBS := $(DL_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

