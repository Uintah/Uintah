# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone/tools/graphview

PROGRAM := $(SRCDIR)/graphview

SRCS := $(SRCDIR)/graphview.cc	$(SRCDIR)/GV_TaskGraph.cc 	\
	$(SRCDIR)/DaVinci.cc

PSELIBS := \
	Packages/Uintah/CCA/Ports \
	Dataflow/XMLUtil \
	Core/Exceptions  \
	Core/Thread 

LIBS := $(XML_LIBRARY)

include $(SRCTOP)/scripts/program.mk

