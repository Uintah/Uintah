# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone

SUBDIRS := \
        $(SRCDIR)/tools       \
        $(SRCDIR)/compare_mms \
        $(SRCDIR)/Benchmarks

include $(SCIRUN_SCRIPTS)/recurse.mk

##############################################
# sus

# The following variables are used by the Fake* scripts... please
# do not modify...
#
COMPONENTS      = Packages/Uintah/CCA/Components
CA              = Packages/Uintah/CCA/Components/Arches
#DUMMY_LIB      = Packages/Uintah/CCA/Components/Dummy
ARCHES_SUB_LIBS = $(CA)/Mixing $(CA)/fortran $(CA)/Radiation $(CA)/Radiation/fortran
ARCHES_LIBS     = $(COMPONENTS)/Arches $(COMPONENTS)/MPMArches
MPM_LIB         = Packages/Uintah/CCA/Components/MPM
ICE_LIB         = Packages/Uintah/CCA/Components/ICE
MPMICE_LIB      = Packages/Uintah/CCA/Components/MPMICE


SRCS := $(SRCDIR)/sus.cc

ifeq ($(IS_AIX),yes)
  AIX_LIBRARY := \
        Core/XMLUtil  \
        Core/Malloc       \
        Core/Math         \
        Core/Containers   \
        Core/Persistent   \
        Core/OS           \
        Packages/Uintah/Core/Math                        \
        Packages/Uintah/Core/GeometryPiece               \
        Packages/Uintah/CCA/Components/Parent            \
        Packages/Uintah/CCA/Components/SwitchingCriteria \
	Packages/Uintah/CCA/Components/OnTheFlyAnalysis  \
        Packages/Uintah/CCA/Components/Examples          \
        $(DUMMY_LIB)                                     \
        $(ARCHES_LIBS)                                   \
        $(MPM_LIB)                                       \
        $(ICE_LIB)                                       \
        $(MPMICE_LIB)                                    \
        Packages/Uintah/CCA/Components/PatchCombiner     \
        $(DUMMY_LIB)                                     \
        $(ARCHES_SUB_LIBS)
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
	Core/Persistent   \
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
        Packages/Uintah/CCA/Components/DataArchiver  \
        Packages/Uintah/CCA/Components/LoadBalancers \
        Packages/Uintah/CCA/Components/Regridder     \
        Packages/Uintah/Core/ProblemSpec             \
        Packages/Uintah/CCA/Components/SimulationController \
        Packages/Uintah/CCA/Components/Schedulers           \
        Packages/Uintah/CCA/Components/ProblemSpecification \
        Packages/Uintah/CCA/Components/Solvers              \
        $(AIX_LIBRARY)
endif

ifeq ($(IS_AIX),yes)
  LIBS := \
        $(TCL_LIBRARY) \
        $(TEEM_LIBRARY) \
        $(XML2_LIBRARY) \
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
  LIBS := $(XML2_LIBRARY) $(F_LIBRARY) $(HYPRE_LIBRARY) \
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

LIBS    := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(Z_LIBRARY) $(TEEM_LIBRARY) $(F_LIBRARY)

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
# extractS

SRCS := $(SRCDIR)/extractS.cc
PROGRAM := Packages/Uintah/StandAlone/extractS

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

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# selectpart

SRCS := $(SRCDIR)/selectpart.cc
PROGRAM := Packages/Uintah/StandAlone/selectpart

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

LIBS    := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(Z_LIBRARY) $(TEEM_LIBRARY) $(F_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# lineextract

SRCS := $(SRCDIR)/lineextract.cc
PROGRAM := Packages/Uintah/StandAlone/lineextract

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
        Packages/Uintah/Core/Util          \
        Packages/Uintah/Core/Math          \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/ProblemSpec   \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/DataArchive   \
        Packages/Uintah/Core/Parallel      \
        Packages/Uintah/CCA/Ports          \
        Packages/Uintah/CCA/Components/ProblemSpecification \
        Core/Basis        \
        Core/Exceptions   \
        Core/Containers   \
        Core/Datatypes    \
        Core/Geometry     \
        Core/Math         \
        Core/Persistent   \
        Core/Thread       \
        Core/Util         \
        Core/XMLUtil
endif

LIBS := $(XML2_LIBRARY) $(TEEM_LIBRARY) $(PNG_LIBRARY) $(Z_LIBRARY) $(M_LIBRARY) $(MPI_LIBRARY) $(F_LIBRARY)

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
        Core/Containers  \
        Core/Geometry    \
        Core/OS          \
        Core/Persistent  \
        Core/Thread      \
        Core/Util
endif

LIBS    := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(TEEM_LIBRARY) $(F_LIBRARY)

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

LIBS    := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(F_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# async_mpi_test.cc

SRCS := $(SRCDIR)/async_mpi_test.cc
PROGRAM := Packages/Uintah/StandAlone/async_mpi_test
PSELIBS := \
        Core/Thread \
        Packages/Uintah/Core/Parallel

LIBS    := $(XML2_LIBRARY) $(M_LIBRARY) $(MPI_LIBRARY) $(F_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# restart_merger

SRCS := $(SRCDIR)/restart_merger.cc
PROGRAM := Packages/Uintah/StandAlone/restart_merger
ifeq ($(LARGESOS),yes)
PSELIBS := Datflow Packages/Uintah
else
PSELIBS := \
        Packages/Uintah/Core/GeometryPiece \
        Packages/Uintah/Core/Exceptions    \
        Packages/Uintah/Core/Grid          \
        Packages/Uintah/Core/Util          \
        Packages/Uintah/Core/Parallel      \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/DataArchive   \
        Packages/Uintah/CCA/Ports          \
        Packages/Uintah/CCA/Components/DataArchiver         \
        Packages/Uintah/Core/ProblemSpec                    \
        Packages/Uintah/CCA/Components/ProblemSpecification \
        Core/Exceptions  \
        Core/Geometry    \
        Core/Thread      \
        Core/Util        \
        Core/OS          \
        Core/Containers
endif
LIBS    := $(XML2_LIBRARY) $(M_LIBRARY) $(MPI_LIBRARY) $(F_LIBRARY)

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
        selectpart \
	async_mpi_test \
        extractV \
        extractF \
        extractS \
        pfs \
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

LIBS    := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(F_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

###############################################
# pfs 2 - Steve Maas' version

SRCS := $(SRCDIR)/pfs2.cc
PROGRAM := Packages/Uintah/StandAlone/pfs2

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

LIBS    := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

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

faster_gmake:
	@( $(SRCTOP_ABS)/Packages/Uintah/scripts/useFakeArches.sh $(OBJTOP_ABS) on)
fake_arches:
	@( $(SRCTOP_ABS)/Packages/Uintah/scripts/useFakeArches.sh $(OBJTOP_ABS) on)

sus: prereqs Packages/Uintah/StandAlone/sus

puda: prereqs Packages/Uintah/StandAlone/puda

dumpfields: prereqs Packages/Uintah/StandAlone/tools/dumpfields/dumpfields

compare_uda: prereqs Packages/Uintah/StandAlone/compare_uda

uda2nrrd: prereqs Packages/Uintah/StandAlone/uda2nrrd

restart_merger: prereqs Packages/Uintah/StandAlone/restart_merger

partextract: prereqs Packages/Uintah/StandAlone/partextract

partvarRange: prereqs Packages/Uintah/StandAlone/partvarRange

selectpart: prereqs Packages/Uintah/StandAlone/selectpart

async_mpi_test: prereqs Packages/Uintah/StandAlone/async_mpi_test

extractV: prereqs Packages/Uintah/StandAlone/extractV

extractF: prereqs Packages/Uintah/StandAlone/extractF

extractS: prereqs Packages/Uintah/StandAlone/extractS

gambitFileReader: prereqs Packages/Uintah/StandAlone/gambitFileReader

slb: prereqs Packages/Uintah/StandAlone/slb

pfs: prereqs Packages/Uintah/StandAlone/pfs

pfs2: prereqs Packages/Uintah/StandAlone/pfs2

timeextract: Packages/Uintah/StandAlone/timeextract

lineextract: Packages/Uintah/StandAlone/lineextract
