# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Dataflow/Datatypes/Image

SRCS     += $(SRCDIR)/ImagePort.cc

PSELIBS := PSECore/Dataflow Core/Containers Core/Thread
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

