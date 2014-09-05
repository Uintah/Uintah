# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone

SUBDIRS := \
	$(SRCDIR)/tools       \
	$(SRCDIR)/compare_mms \
	$(SRCDIR)/Benchmarks

include $(SCIRUN_SCRIPTS)/recurse.mk

##############################################
# sus

SRCS := $(SRCDIR)/sus.cc

ifeq ($(IS_AIX),yes)
  AIX_LIBRARY := \
        Core/XMLUtil  \
        Core/Malloc       \
        Core/Math         \
        Core/Containers   \
	Core/Persistent   \
	Core/OS		  \
	Packages/Uintah/Core/Math			\
        Packages/Uintah/Core/GeometryPiece              \
	Packages/Uintah/CCA/Components/Parent		\
	Packages/Uintah/CCA/Components/SwitchingCriteria\
        Packages/Uintah/CCA/Components/Arches/Mixing    \
        Packages/Uintah/CCA/Components/Arches/fortran   \
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
        Core/Math         \
        Packages/Uintah/Core/DataArchive \
        Packages/Uintah/Core/Grid        \
        Packages/Uintah/Core/Parallel    \
        Packages/Uintah/Core/Labels      \
        Packages/Uintah/Core/Util        \
        Packages/Uintah/Core/Math        \
        Packages/Uintah/Core/Disclosure  \
        Packages/Uintah/Core/Exceptions  \
        Packages/Uintah/CCA/Ports        \
	Packages/Uintah/CCA/Components/Parent \
	Packages/Uintah/CCA/Components/Models \
        Packages/Uintah/CCA/Components/MPM    \
        Packages/Uintah/CCA/Components/MPMICE \
        Packages/Uintah/CCA/Components/DataArchiver  \
        Packages/Uintah/CCA/Components/LoadBalancers \
        Packages/Uintah/CCA/Components/Regridder     \
        Packages/Uintah/Core/ProblemSpec             \
        Packages/Uintah/CCA/Components/SimulationController \
        Packages/Uintah/CCA/Components/Schedulers           \
        Packages/Uintah/CCA/Components/ProblemSpecification \
        Packages/Uintah/CCA/Components/Solvers              \
        Packages/Uintah/CCA/Components/ICE           \
        Packages/Uintah/CCA/Components/Examples      \
        Packages/Uintah/CCA/Components/Arches        \
        Packages/Uintah/CCA/Components/MPMArches     \
        Packages/Uintah/CCA/Components/PatchCombiner \
        $(AIX_LIBRARY)
endif

ifeq ($(IS_AIX),yes)
  LIBS := \
	$(TCL_LIBRARY) \
	$(TEEM_LIBRARY) \
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
        Packages/Uintah/Core/Util          \
        Packages/Uintah/Core/Math          \
        Packages/Uintah/Core/Parallel      \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/ProblemSpec   \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/DataArchive   \
	Packages/Uintah/CCA/Ports          \
        Packages/Uintah/CCA/Components/ProblemSpecification \
        Core/XMLUtil \
        Core/Exceptions  \
        Core/Persistent  \
        Core/Geometry    \
        Core/Thread      \
        Core/Util        \
        Core/OS          \
        Core/Containers
endif

LIBS    := $(XML_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(Z_LIBRARY) $(TEEM_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# timeextract

SRCS := $(SRCDIR)/timeextract.cc
PROGRAM := Packages/Uintah/StandAlone/timeextract

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# extractV

SRCS := $(SRCDIR)/extractV.cc
PROGRAM := Packages/Uintah/StandAlone/extractV

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# extractF

SRCS := $(SRCDIR)/extractF.cc
PROGRAM := Packages/Uintah/StandAlone/extractF

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# partextract

SRCS := $(SRCDIR)/partextract.cc
PROGRAM := Packages/Uintah/StandAlone/partextract

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# parvarRange

SRCS := $(SRCDIR)/partvarRange.cc
PROGRAM := Packages/Uintah/StandAlone/partvarRange

ifeq ($(LARGESOS),yes)
  PSELIBS := Datflow Packages/Uintah
else
  PSELIBS := \
        Packages/Uintah/Core/Exceptions    \
        Packages/Uintah/Core/Grid          \
        Packages/Uintah/Core/Util          \
        Packages/Uintah/Core/Math          \
        Packages/Uintah/Core/Parallel      \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/ProblemSpec   \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/DataArchive   \
	Packages/Uintah/CCA/Ports          \
        Packages/Uintah/CCA/Components/ProblemSpecification \
        Core/XMLUtil \
        Core/Exceptions  \
        Core/Persistent  \
        Core/Geometry    \
        Core/Thread      \
        Core/Util        \
        Core/OS          \
        Core/Containers
endif

LIBS    := $(XML_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(Z_LIBRARY) $(TEEM_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# lineextract
                                                                                
SRCS := $(SRCDIR)/lineextract.cc
PROGRAM := Packages/Uintah/StandAlone/lineextract
                                                                                
include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# uda2nrrd

# ifeq ($(findstring teem, $(TEEM_LIBRARY)),teem)
# ifeq ($(LARGESOS),yes)
#   PSELIBS := Datflow Packages/Uintah
# else
#   PSELIBS := \
#         Packages/Uintah/Core/Exceptions    \
#         Packages/Uintah/Core/Grid          \
#         Packages/Uintah/Core/Util          \
#         Packages/Uintah/Core/Math          \
#         Packages/Uintah/Core/Disclosure    \
#         Packages/Uintah/Core/ProblemSpec   \
#         Packages/Uintah/Core/Disclosure    \
#         Packages/Uintah/Core/DataArchive   \
# 	Packages/Uintah/Core/Parallel   \
# 	Packages/Uintah/CCA/Ports          \
# 	Packages/Uintah/CCA/Components/ProblemSpecification \
# 	Core/XMLUtil  \
# 	Core/Persistent   \
# 	Core/Datatypes    \
#         Core/Geometry    \
#         Core/Thread      \
#         Core/Util        \
# 	Core/Math        \
# 	Core/Exceptions  \
#         Core/Containers
# endif

# LIBS := $(XML_LIBRARY) $(TEEM_LIBRARY) $(PNG_LIBRARY) $(Z_LIBRARY) $(M_LIBRARY) $(MPI_LIBRARY)

# SRCS := $(SRCDIR)/uda2nrrd.cc
# PROGRAM := Packages/Uintah/StandAlone/uda2nrrd

# include $(SCIRUN_SCRIPTS)/program.mk

# endif

##############################################
# compare_uda

SRCS := $(SRCDIR)/compare_uda.cc
PROGRAM := Packages/Uintah/StandAlone/compare_uda

ifeq ($(LARGESOS),yes)
  PSELIBS := Datflow Packages/Uintah
else
  PSELIBS := \
        Core/XMLUtil  \
        Packages/Uintah/Core/Exceptions    \
        Packages/Uintah/Core/Grid          \
        Packages/Uintah/Core/Util          \
        Packages/Uintah/Core/Parallel \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/Math          \
        Packages/Uintah/Core/ProblemSpec   \
        Packages/Uintah/Core/DataArchive   \
	Packages/Uintah/CCA/Ports          \
        Packages/Uintah/CCA/Components/ProblemSpecification \
        Core/Exceptions  \
	Core/Persistent   \
        Core/Geometry    \
        Core/Thread      \
        Core/Util        \
        Core/OS          \
        Core/Containers
endif

LIBS    := $(XML_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(TEEM_LIBRARY)

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
	Packages/Uintah/Core/Util \
	Packages/Uintah/Core/GeometryPiece \
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

SRCS := $(SRCDIR)/async_mpi_test.cc
PROGRAM := Packages/Uintah/StandAlone/async_mpi_test
PSELIBS := \
	Core/Thread \
	Packages/Uintah/Core/Parallel
LIBS    := $(XML_LIBRARY) $(M_LIBRARY) $(MPI_LIBRARY)
 
include $(SCIRUN_SCRIPTS)/program.mk
 
##############################################

SRCS := $(SRCDIR)/restart_merger.cc
PROGRAM := Packages/Uintah/StandAlone/restart_merger
ifeq ($(LARGESOS),yes)
PSELIBS := Datflow Packages/Uintah
else
PSELIBS := \
        Packages/Uintah/Core/Exceptions \
        Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/Util \
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
LIBS    := $(XML_LIBRARY) $(M_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# gambitFileReader

SRCS := $(SRCDIR)/gambitFileReader.cc
PROGRAM := Packages/Uintah/StandAlone/gambitFileReader

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# Uintah
# Convenience targets for Specific executables 
uintah: sus \
        puda \
        dumpfields \
        compare_uda \
        restart_merger \
        partextract \
        partvarRange \
	async_mpi_test \
        extractV \
        extractF \
        gambitFileReader \
        lineextract \
        timeextract \
        link_inputs \
        link_regression_tester
                
###############################################
# pfs

SRCS := $(SRCDIR)/pfs.cc
PROGRAM := Packages/Uintah/StandAlone/pfs

ifeq ($(LARGESOS),yes)
  PSELIBS := Datflow Packages/Uintah
else
  PSELIBS := \
      Packages/Uintah/Core/Grid \
      Packages/Uintah/Core/Util \
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

###############################################

link_inputs: 
	@( if ! test -L Packages/Uintah/StandAlone/inputs; then \
               echo "Creating link to inputs directory." ; \
	       ln -sf $(SRCTOP_ABS)/Packages/Uintah/StandAlone/inputs Packages/Uintah/StandAlone/inputs; \
	   fi )

link_regression_tester: 
	@( if ! test -L Packages/Uintah/StandAlone/run_RT; then \
               echo "Creating link to regression_tester script." ; \
	       ln -sf $(SRCTOP_ABS)/Packages/Uintah/scripts/regression_tester Packages/Uintah/StandAlone/run_RT; \
	   fi )

TOP_ABS:= $(SRCTOP_ABS)/..
faster_gmake: 
	@( $(SRCTOP_ABS)/Packages/Uintah/scripts/useFakeArches.pl $(TOP_ABS)) 
fake_arches: 
	@( $(SRCTOP_ABS)/Packages/Uintah/scripts/useFakeArches.pl $(TOP_ABS)) 

sus: prereqs Packages/Uintah/StandAlone/sus

puda: prereqs Packages/Uintah/StandAlone/puda

dumpfields: prereqs Packages/Uintah/StandAlone/tools/dumpfields/dumpfields

compare_uda: prereqs Packages/Uintah/StandAlone/compare_uda

restart_merger: prereqs Packages/Uintah/StandAlone/restart_merger

partextract: prereqs Packages/Uintah/StandAlone/partextract

partvarRange: prereqs Packages/Uintah/StandAlone/partvarRange

async_mpi_test: prereqs Packages/Uintah/StandAlone/async_mpi_test

extractV: prereqs Packages/Uintah/StandAlone/extractV

extractF: prereqs Packages/Uintah/StandAlone/extractF

gambitFileReader: prereqs Packages/Uintah/StandAlone/gambitFileReader

slb: prereqs Packages/Uintah/StandAlone/slb

pfs: prereqs Packages/Uintah/StandAlone/pfs

timeextract: Packages/Uintah/StandAlone/timeextract

lineextract: Packages/Uintah/StandAlone/lineextract
