# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone

SUBDIRS := $(SRCDIR)/tools

include $(SCIRUN_SCRIPTS)/recurse.mk

SRCS := $(SRCDIR)/sus.cc

ifeq ($(CC),newmpxlc)
AIX_LIBS := \
        Core/Datatypes    \
        Dataflow/Comm     \
        Dataflow/Network  \
        Dataflow/Ports    \
        Dataflow/XMLUtil  \
        Core/GuiInterface \
        Core/Malloc       \
        Core/Math         \
        Core/OS           \
        Core/Persistent   \
        Core/Geom         \
        Core/GeomInterface\
        Core/Containers   \
        Core/TkExtensions \
        Packages/Uintah/CCA/Components/HETransformation \
        Packages/Uintah/CCA/Components/Arches/Mixing \
        Packages/Uintah/CCA/Components/Arches_fortran
endif


PROGRAM := Packages/Uintah/StandAlone/sus
ifeq ($(LARGESOS),yes)
  PSELIBS := Packages/Uintah
else

  PSELIBS := \
        Core/Exceptions   \
        Core/Thread       \
        Core/Geometry     \
        Core/Util         \
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
        Packages/Uintah/CCA/Components/PatchCombiner \
        $(AIX_LIBS)
endif

ifeq ($(CC),newmpxlc)
LIBS := \
        -L$(TCL_LIB_DIR) $(TCL_LIB_FLAG) $(TK_LIB_FLAG) $(ITCL_LIB_FLAG) \
        $(ITK_LIB_FLAG) $(BLT_LIB_FLAG) \
        $(XML_LIB_FLAG) \
        $(TAU_LIBRARY) \
        $(GL_LIBS) $(GZ_LIBRARY) \
        $(THREAD_LIBS) \
        $(FLIBS) \
        $(MPI_LIBRARY) \
        $(PETSC_LIBS) \
        -lld -lm
else
LIBS := $(XML_LIBRARY) $(TAU_LIBRARY) $(MPI_LIBRARY) $(FLIBS) \
        $(PETSC_LIBS)
endif

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
LIBS    := $(XML_LIBRARY) $(MPI_LIBRARY) -lm

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
LIBS    := $(XML_LIBRARY) $(MPI_LIBRARY) -lm

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
LIBS    := $(XML_LIBRARY) -lm

include $(SCIRUN_SCRIPTS)/program.mk

# Convenience targets for Specific executables (use make sus)
sus: prereqs Packages/Uintah/StandAlone/sus

puda: prereqs Packages/Uintah/StandAlone/puda

compare_uda: prereqs Packages/Uintah/StandAlone/compare_uda
