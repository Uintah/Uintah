# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/Util

SRCS     += \
	$(SRCDIR)/Debug.cc \
	$(SRCDIR)/soloader.cc \
	$(SRCDIR)/DebugStream.cc \
	$(SRCDIR)/Timer.cc \
	$(SRCDIR)/sci_system.c \

PSELIBS := Core/Containers Core/Exceptions
LIBS := $(DL_LIBRARY) $(THREAD_LIBS)

include $(SRCTOP)/scripts/smallso_epilogue.mk

