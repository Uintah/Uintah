# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/globus_threads

SRCS     += $(SRCDIR)/globus_external_threads.cc

PSELIBS := Core/Thread
LIBS := $(GLOBUS_COMMON)

include $(SRCTOP)/scripts/smallso_epilogue.mk

