# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone

SUBDIRS := \
	$(SRCDIR)/tools \
	$(SRCDIR)/Benchmarks

include $(SCIRUN_SCRIPTS)/recurse.mk

##############################################
# sus

SRCS := $(SRCDIR)/sus.cc

ifeq ($(CC),newmpxlc)
  AIX_LIBRARY := \
        Dataflow/XMLUtil  \
        Core/Malloc       \
        Core/Math         \
        Core/Containers   \
	Core/Persistent   \
	Core/OS		  \
        Packages/Uintah/CCA/Components/HETransformation \
        Packages/Uintah/CCA/Components/Arches/Mixing \
        Packages/Uintah/CCA/Components/Arches/fortran \
        Packages/Uintah/CCA/Components/Arches/Radiation \
        Packages/Uintah/CCA/Components/Arches/Radiation/fortran
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
        Packages/Uintah/Core/DataArchive   \
        Packages/Uintah/Core/Grid \
        Packages/Uintah/Core/Parallel \
        Packages/Uintah/Core/Math \
        Packages/Uintah/Core/Disclosure \
        Packages/Uintah/Core/Exceptions \
        Packages/Uintah/CCA/Ports \
	Packages/Uintah/CCA/Components/Models \
        Packages/Uintah/CCA/Components/MPM \
        Packages/Uintah/CCA/Components/MPMICE \
        Packages/Uintah/CCA/Components/DataArchiver \
        Packages/Uintah/Core/ProblemSpec \
        Packages/Uintah/CCA/Components/SimulationController \
        Packages/Uintah/CCA/Components/Schedulers \
        Packages/Uintah/CCA/Components/ProblemSpecification \
        Packages/Uintah/CCA/Components/Solvers \
        Packages/Uintah/CCA/Components/ICE \
        Packages/Uintah/CCA/Components/Examples \
        Packages/Uintah/CCA/Components/Arches \
        Packages/Uintah/CCA/Components/MPMArches \
        Packages/Uintah/CCA/Components/PatchCombiner \
        $(AIX_LIBRARY)
endif

ifeq ($(CC),newmpxlc)
  LIBS := \
        $(XML_LIBRARY) \
	$(Z_LIBRARY) \
        $(THREAD_LIBRARY) \
        $(F_LIBRARY) \
        $(PETSC_LIBRARY) \
        $(HYPRE_LIBRARY) \
        $(BLAS_LIBRARY) \
        $(LAPACK_LIBRARY) \
        $(MPI_LIBRARY) \
	$(X_LIBRARY) \
        -lld $(M_LIBRARY)
else
  LIBS := $(XML_LIBRARY) $(F_LIBRARY) $(HYPRE_LIBRARY) \
          $(CANTERA_LIBRARY) \
          $(PETSC_LIBRARY) $(BLAS_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)
endif

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# puda

SRCS := $(SRCDIR)/puda.cc
PROGRAM := Packages/Uintah/StandAlone/puda

ifeq ($(LARGESOS),yes)
  PSELIBS := Datflow Packages/Uintah
else
  PSELIBS := \
        Packages/Uintah/Core/Exceptions    \
        Packages/Uintah/Core/Grid          \
        Packages/Uintah/Core/Math          \
        Packages/Uintah/Core/Parallel      \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/ProblemSpec   \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/DataArchive   \
	Packages/Uintah/CCA/Ports          \
        Packages/Uintah/CCA/Components/ProblemSpecification \
        Dataflow/XMLUtil \
        Core/Exceptions  \
        Core/Persistent  \
        Core/Geometry    \
        Core/Thread      \
        Core/Util        \
        Core/OS          \
        Core/Containers
endif

LIBS    := $(XML_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(Z_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# timeextract

SRCS := $(SRCDIR)/timeextract.cc
PROGRAM := Packages/Uintah/StandAlone/timeextract

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# uda2nrrd

ifeq ($(findstring teem, $(TEEM_LIBRARY)),teem)
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
	Packages/Uintah/Core/Parallel   \
	Packages/Uintah/CCA/Ports          \
	Packages/Uintah/CCA/Components/ProblemSpecification \
	Dataflow/XMLUtil  \
	Core/Persistent   \
	Core/Datatypes    \
        Core/Geometry    \
        Core/Thread      \
        Core/Util        \
	Core/Math        \
	Core/Exceptions  \
        Core/Containers
endif

LIBS := $(XML_LIBRARY) $(TEEM_LIBRARY) $(Z_LIBRARY) $(M_LIBRARY)

SRCS := $(SRCDIR)/uda2nrrd.cc
PROGRAM := Packages/Uintah/StandAlone/uda2nrrd

include $(SCIRUN_SCRIPTS)/program.mk

endif

##############################################
# compare_uda

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

##############################################
# slb

SRCS := $(SRCDIR)/slb.cc
PROGRAM := Packages/Uintah/StandAlone/slb

ifeq ($(LARGESOS),yes)
  PSELIBS := Datflow Packages/Uintah
else
  PSELIBS := \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/Parallel \
	Packages/Uintah/Core/Exceptions \
	Packages/Uintah/Core/Math \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/CCA/Ports \
	Packages/Uintah/CCA/Components/ProblemSpecification \
	Core/Exceptions \
        Core/Geometry
endif

LIBS    := $(XML_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# restart_merger

SRCS := $(SRCDIR)/restart_merger.cc
PROGRAM := Packages/Uintah/StandAlone/restart_merger
ifeq ($(LARGESOS),yes)
PSELIBS := Datflow Packages/Uintah
else
PSELIBS := \
        Packages/Uintah/Core/Exceptions \
        Packages/Uintah/Core/Grid \
        Packages/Uintah/Core/Parallel \
        Packages/Uintah/Core/Disclosure \
        Packages/Uintah/Core/DataArchive \
	Packages/Uintah/CCA/Ports \
        Packages/Uintah/CCA/Components/DataArchiver \
        Packages/Uintah/Core/ProblemSpec \
        Packages/Uintah/CCA/Components/ProblemSpecification \
        Core/Exceptions  \
        Core/Geometry    \
        Core/Thread      \
        Core/Util        \
        Core/OS          \
        Core/Containers
endif
LIBS    := $(XML_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# gambitFileReader

SRCS := $(SRCDIR)/gambitFileReader.cc
PROGRAM := Packages/Uintah/StandAlone/gambitFileReader

include $(SCIRUN_SCRIPTS)/program.mk

# Convenience targets for Specific executables 
uintah: sus \
        puda \
        compare_uda \
        restart_merger \
        gambitFileReader \
        link_inputs
                
            
link_inputs: 
	@( ln -s $(SRCTOP_ABS)/Packages/Uintah/StandAlone/inputs Packages/Uintah/StandAlone/inputs)

TOP_ABS:= $(SRCTOP_ABS)/..
faster_gmake: 
	@( $(SRCTOP_ABS)/Packages/Uintah/Test/helpers/useFakeArches.pl $(TOP_ABS)) 

sus: prereqs Packages/Uintah/StandAlone/sus

puda: prereqs Packages/Uintah/StandAlone/puda

compare_uda: prereqs Packages/Uintah/StandAlone/compare_uda

restart_merger: prereqs Packages/Uintah/StandAlone/restart_merger

gambitFileReader: prereqs Packages/Uintah/StandAlone/gambitFileReader

slb: prereqs Packages/Uintah/StandAlone/slb