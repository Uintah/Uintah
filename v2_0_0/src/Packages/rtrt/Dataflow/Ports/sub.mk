# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Packages/rtrt/Dataflow/Ports

SRCS     += $(SRCDIR)/ScenePort.cc

PSELIBS := \
	Core/Containers

LIBS := 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


