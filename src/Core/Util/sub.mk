#
# Makefile fragment for this subdirectory
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/Util

SRCS     += $(SRCDIR)/Debug.cc $(SRCDIR)/Timer.cc \
	$(SRCDIR)/soloader.cc $(SRCDIR)/DebugStream.cc

PSELIBS := SCICore/Containers SCICore/Exceptions
LIBS := $(DL_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

