# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Dataflow/XMLUtil

SRCS     += $(SRCDIR)/SimpleErrorHandler.cc $(SRCDIR)/XMLUtil.cc

PSELIBS := Core/Containers
LIBS := $(XML_LIBRARY) $(GZ_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

