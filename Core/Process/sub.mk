# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/Process

SRCS     += $(SRCDIR)/ProcessManager.cc

PSELIBS := Core/Thread
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

