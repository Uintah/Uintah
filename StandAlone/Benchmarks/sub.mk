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
        Packages/Uintah/Core		\
        Packages/Uintah/CCA/Components/ProblemSpecification \
        Core/Exceptions  \
        Core/Geometry    \
        Core/Geom	\
        Core/Persistent  \
        Core/Util        \
        Core/Thread      \
	Core/Datatypes	\
        Core/Containers
endif

LIBS    := $(XML_LIBRARY) $(MPI_LIBRARY) $(Z_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

SimpleMath: prereqs Packages/Uintah/StandAlone/Benchmarks/SimpleMath
