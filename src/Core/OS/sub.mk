# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/OS

SRCS     += $(SRCDIR)/Dir.cc $(SRCDIR)/sock.cc

PSELIBS := Core/Exceptions
LIBS :=

include $(SRCTOP)/scripts/smallso_epilogue.mk

