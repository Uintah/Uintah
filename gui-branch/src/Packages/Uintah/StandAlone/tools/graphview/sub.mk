# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone/tools/graphview

PROGRAM := $(SRCDIR)/graphview

SRCS := $(SRCDIR)/graphview.cc	$(SRCDIR)/GV_TaskGraph.cc 	\
	$(SRCDIR)/DaVinci.cc

PSELIBS := \
	Packages/Uintah/CCA/Ports \
	Packages/Uintah/Core/Disclosure \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Exceptions \
	Dataflow/XMLUtil \
	Core/Exceptions  \
	Core/Geometry \
	Core/Thread 

LIBS := $(XML_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

