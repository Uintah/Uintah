# Makefile fragment for this subdirectory

SRCDIR := testprograms/Thread

ifeq ($(LARGESOS),yes)
PSELIBS := Core
else
PSELIBS := Core/Thread
endif
LIBS := $(THREAD_LIBS)

PROGRAM := $(SRCDIR)/bps
SRCS := $(SRCDIR)/bps.cc

include $(SRCTOP)/scripts/program.mk

