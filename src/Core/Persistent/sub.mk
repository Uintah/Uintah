# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/Persistent

SRCS     += $(SRCDIR)/Persistent.cc $(SRCDIR)/Pstreams.cc

PSELIBS := Core/Containers
LIBS := -lz

include $(SRCTOP)/scripts/smallso_epilogue.mk

