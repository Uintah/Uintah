# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/Exceptions

SRCS     += $(SRCDIR)/ArrayIndexOutOfBounds.cc \
	    $(SRCDIR)/AssertionFailed.cc \
	    $(SRCDIR)/DimensionMismatch.cc \
	    $(SRCDIR)/ErrnoException.cc \
	    $(SRCDIR)/Exception.cc \
	    $(SRCDIR)/FileNotFound.cc \
	    $(SRCDIR)/InternalError.cc

PSELIBS := 
LIBS := $(TRACEBACK_LIB)

include $(SRCTOP)/scripts/smallso_epilogue.mk

