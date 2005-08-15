# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Packages/rtrt/Dataflow/Ports

SRCS     += $(SRCDIR)/ScenePort.cc

PSELIBS := \
	Core/Containers Core/Thread Core/Persistent Core/Datatypes \
	Dataflow/Network Dataflow/Ports

LIBS := 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


