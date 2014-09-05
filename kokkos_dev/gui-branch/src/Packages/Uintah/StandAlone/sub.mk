# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone

SUBDIRS := $(SRCDIR)/tools

include $(SCIRUN_SCRIPTS)/recurse.mk

SRCS := $(SRCDIR)/sus.cc

PROGRAM := Packages/Uintah/StandAlone/sus
ifeq ($(LARGESOS),yes)
  PSELIBS := Packages/Uintah
else

  PSELIBS := \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/Parallel \
	Packages/Uintah/Core/Math \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Disclosure \
	Packages/Uintah/Core/Exceptions \
	Packages/Uintah/CCA/Ports \
	Packages/Uintah/CCA/Components/MPM \
	Packages/Uintah/CCA/Components/MPMICE \
	Packages/Uintah/CCA/Components/DataArchiver \
	Packages/Uintah/CCA/Components/SimulationController \
	Packages/Uintah/CCA/Components/Schedulers \
	Packages/Uintah/CCA/Components/ProblemSpecification \
	Packages/Uintah/CCA/Components/ICE \
	Packages/Uintah/CCA/Components/Examples \
	Packages/Uintah/CCA/Components/Arches \
	Packages/Uintah/CCA/Components/MPMArches \
	Dataflow/Network \
	Core/Exceptions  \
	Core/Thread      \
	Core/Geometry    \
	Core/Util

endif

LIBS := $(XML_LIBRARY) $(TAU_LIBRARY) $(MPI_LIBRARY) $(GL_LIBS) $(FLIBS) \
	$(PETSC_LIBS)

include $(SCIRUN_SCRIPTS)/program.mk

SRCS := $(SRCDIR)/puda.cc
PROGRAM := Packages/Uintah/StandAlone/puda
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
	Packages/Uintah/CCA/Ports          \
	Dataflow/XMLUtil \
	Core/Exceptions  \
	Core/Geometry    \
	Core/Thread      \
	Core/Util        \
	Core/OS          \
	Core/Containers
endif
LIBS 	:= $(XML_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

SRCS := $(SRCDIR)/compare_uda.cc
PROGRAM := Packages/Uintah/StandAlone/compare_uda
ifeq ($(LARGESOS),yes)
PSELIBS := Datflow Packages/Uintah
else
PSELIBS := \
	Packages/Uintah/Core/Exceptions    \
	Packages/Uintah/Core/Grid          \
	Packages/Uintah/Core/Disclosure    \
	Packages/Uintah/Core/Math          \
	Packages/Uintah/Core/ProblemSpec   \
	Packages/Uintah/CCA/Ports          \
	Dataflow/XMLUtil \
	Core/Exceptions  \
	Core/Geometry    \
	Core/Thread      \
	Core/Util        \
	Core/OS          \
	Core/Containers
endif
LIBS 	:= $(XML_LIBRARY) $(MPI_LIBRARY) -lm

include $(SCIRUN_SCRIPTS)/program.mk

SRCS := $(SRCDIR)/restart_merger.cc
PROGRAM := Packages/Uintah/StandAlone/restart_merger
ifeq ($(LARGESOS),yes)
PSELIBS := Datflow Packages/Uintah
else
PSELIBS := \
	Packages/Uintah/Core/Exceptions \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Parallel \
	Packages/Uintah/Core/Disclosure \
	Packages/Uintah/CCA/Ports \
	Packages/Uintah/CCA/Components/DataArchiver \
	Packages/Uintah/CCA/Components/ProblemSpecification \
	Dataflow/XMLUtil \
	Core/Exceptions  \
	Core/Geometry    \
	Core/Thread      \
	Core/Util        \
	Core/OS          \
	Core/Containers
endif
LIBS 	:= $(XML_LIBRARY) -lm

include $(SCIRUN_SCRIPTS)/program.mk


