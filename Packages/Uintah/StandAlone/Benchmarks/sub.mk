# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone/Benchmarks

##############################################
# No subdirectories yet, maybe in the future
#
#SUBDIRS := 
#
#include $(SCIRUN_SCRIPTS)/recurse.mk
#


##############################################
# Simple Math Benchmark

SRCS    := $(SRCDIR)/SimpleMath.cc

PROGRAM := $(SRCDIR)/SimpleMath

ifeq ($(LARGESOS),yes)
  PSELIBS := Datflow Packages/Uintah
else
  PSELIBS := \
        Packages/Uintah/Core/Exceptions    \
        Packages/Uintah/Core/Grid          \
        Packages/Uintah/Core/Math          \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/ProblemSpec   \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/DataArchive   \
	Packages/Uintah/CCA/Ports          \
        Core/Exceptions  \
        Core/Geometry    \
        Core/Thread      \
        Core/Util        \
        Core/OS          \
        Core/Containers
endif

LIBS    := $(XML_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

SimpleMath: prereqs Packages/Uintah/StandAlone/Benchmarks/SimpleMath
