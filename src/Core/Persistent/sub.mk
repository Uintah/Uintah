# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/Persistent

SRCS     += $(SRCDIR)/Persistent.cc $(SRCDIR)/Pstreams.cc

PSELIBS := Core/Containers
LIBS := $(GZ_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

