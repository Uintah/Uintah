# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone

SUBDIRS := \
        $(SRCDIR)/tools       \
        $(SRCDIR)/Benchmarks

include $(SCIRUN_SCRIPTS)/recurse.mk

##############################################
# sus

# The following variables are used by the Fake* scripts... please
# do not modify...
#
COMPONENTS      = Packages/Uintah/CCA/Components
CA              = Packages/Uintah/CCA/Components/Arches
ifeq ($(BUILD_ARCHES),yes)
  ARCHES_SUB_LIBS = $(CA)/Mixing $(CA)/fortran $(CA)/Radiation $(CA)/Radiation/fortran
  ifeq ($(BUILD_MPM),yes)
    MPMARCHES_LIB    = $(COMPONENTS)/MPMArches
  endif
  ARCHES_LIBS        = $(COMPONENTS)/Arches
endif
ifeq ($(BUILD_MPM),yes)
  MPM_LIB            = Packages/Uintah/CCA/Components/MPM
  ifeq ($(BUILD_ICE),yes)
    MPMICE_LIB       = Packages/Uintah/CCA/Components/MPMICE
  endif
endif
ifeq ($(BUILD_ICE),yes)
  ICE_LIB            = Packages/Uintah/CCA/Components/ICE
endif


SRCS := $(SRCDIR)/sus.cc

ifeq ($(IS_AIX),yes)
  SET_AIX_LIB := yes
endif

ifeq ($(IS_REDSTORM),yes)
  SET_AIX_LIB := yes
endif

ifeq ($(SET_AIX_LIB),yes)
  AIX_LIBRARY := \
        Core/Containers   \
        Core/Malloc       \
        Core/Math         \
        Core/OS           \
        Core/Persistent   \
        Core/Thread       \
        Core/XMLUtil      \
        Packages/Uintah/Core/IO                          \
        Packages/Uintah/Core/Math                        \
        Packages/Uintah/Core/GeometryPiece               \
        Packages/Uintah/CCA/Components/Parent            \
        Packages/Uintah/CCA/Components/SwitchingCriteria \
        Packages/Uintah/CCA/Components/OnTheFlyAnalysis  \
        Packages/Uintah/CCA/Components/Schedulers           \
        Packages/Uintah/CCA/Components/SimulationController \
        Packages/Uintah/CCA/Components/Solvers              \
        Packages/Uintah/CCA/Components/Examples          \
        $(ARCHES_LIBS)                                   \
        $(MPMARCHES_LIB)                                 \
        $(MPM_LIB)                                       \
        $(ICE_LIB)                                       \
        $(MPMICE_LIB)                                    \
        Packages/Uintah/CCA/Components/PatchCombiner     \
        $(ARCHES_SUB_LIBS)
endif

PROGRAM := Packages/Uintah/StandAlone/sus

ifeq ($(LARGESOS),yes)
  PSELIBS := Packages/Uintah
else
  PSELIBS := \
        Core/Containers   \
        Core/Exceptions   \
        Core/Geometry     \
        Core/Math         \
        Core/Persistent   \
        Core/Thread       \
        Core/Util         \
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

ifeq ($(SET_AIX_LIB),yes)
  LIBS := \
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
        $(M_LIBRARY)
else
  LIBS := $(XML2_LIBRARY) $(F_LIBRARY) $(HYPRE_LIBRARY)      \
          $(CANTERA_LIBRARY) $(ZOLTAN_LIBRARY)               \
          $(PETSC_LIBRARY) $(BLAS_LIBRARY) $(LAPACK_LIBRARY) \
          $(MPI_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY)
endif

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
        Core/Containers  \
        Core/Exceptions  \
        Core/Geometry    \
        Core/OS          \
        Core/Persistent  \
        Core/Thread      \
        Core/Util        \
        Core/XMLUtil     
endif

LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(Z_LIBRARY) \
	$(TEEM_LIBRARY) $(F_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# compare_uda

SRCS    := $(SRCDIR)/compare_uda.cc
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

LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(TEEM_LIBRARY) $(F_LIBRARY)

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
        mpi_test \
        extractV \
        extractF \
        extractS \
        pfs \
        pfs2 \
        gambitFileReader \
        lineextract \
        timeextract \
        faceextract \
        link_inputs \
        link_tools \
        link_regression_tester

###############################################

link_inputs:
	@( if ! test -L Packages/Uintah/StandAlone/inputs; then \
               echo "Creating link to inputs directory." ; \
	       ln -sf $(SRCTOP_ABS)/Packages/Uintah/StandAlone/inputs Packages/Uintah/StandAlone/inputs; \
	   fi )
          
link_orderAccuracy:
	@( if ! test -L Packages/Uintah/StandAlone/orderAccuracy; then \
               echo "Creating link to orderAccuracy directory." ; \
	       ln -sf $(SRCTOP_ABS)/Packages/Uintah/orderAccuracy Packages/Uintah/StandAlone; \
	   fi )          
          
link_tools:
	@( if ! test -L Packages/Uintah/StandAlone/puda; then \
               echo "Creating link to all the tools." ; \
	       ln -sf $(OBJTOP_ABS)/Packages/Uintah/StandAlone/tools/puda/puda $(OBJTOP_ABS)/Packages/Uintah/StandAlone/puda; \
              ln -sf $(OBJTOP_ABS)/Packages/Uintah/StandAlone/tools/extractors/lineextract $(OBJTOP_ABS)/Packages/Uintah/StandAlone/lineextract; \
              ln -sf $(OBJTOP_ABS)/Packages/Uintah/StandAlone/tools/extractors/timeextract $(OBJTOP_ABS)/Packages/Uintah/StandAlone/timeextract; \
	   fi )
link_regression_tester:
	@( if ! test -L Packages/Uintah/StandAlone/run_RT; then \
               echo "Creating link to regression_tester script." ; \
	       ln -sf $(SRCTOP_ABS)/Packages/Uintah/scripts/regression_tester Packages/Uintah/StandAlone/run_RT; \
	   fi )

# The REDSTORM portion of the following command somehow prevents Make, on Redstorm,
# from running a bogus compile line of sus...
#
# This is the bogus line:
#
# cc -Minline -O3 -fastsse -fast   -Minform=severe -DREDSTORM  -Llib -lgmalloc \
#       sus.o prereqs Packages/Uintah/StandAlone/sus   -o sus
#
# Notice that is using the generic 'cc' compiler, and only a subset of
# the CFLAGS, and the bogus 'prereqs' target that has not been
# expanded...  This happens after _successfully_ running the real link
# line for sus... I have no idea why it is being triggered, but this
# hack seems to prevent the 2nd 'compilation' from running...
#
sus: prereqs Packages/Uintah/StandAlone/sus
ifeq ($(IS_REDSTORM),yes)
	@echo "Built sus"
endif

tools: puda dumpfields compare_uda uda2nrrd restart_merger partextract partvarRange selectpart async_mpi_test mpi_test extractV extractF extractS gambitFileReader slb pfs pfs2 timeextract faceextract lineextract compare_mms compare_scalar

puda: prereqs Packages/Uintah/StandAlone/tools/puda/puda

dumpfields: prereqs Packages/Uintah/StandAlone/tools/dumpfields/dumpfields

compare_uda: prereqs Packages/Uintah/StandAlone/compare_uda

restart_merger: prereqs Packages/Uintah/StandAlone/restart_merger

partextract: prereqs Packages/Uintah/StandAlone/tools/extractors/partextract

partvarRange: prereqs Packages/Uintah/StandAlone/partvarRange

selectpart: prereqs Packages/Uintah/StandAlone/selectpart

async_mpi_test: prereqs Packages/Uintah/StandAlone/tools/mpi_test/async_mpi_test

mpi_test: prereqs Packages/Uintah/StandAlone/tools/mpi_test/mpi_test

extractV: prereqs Packages/Uintah/StandAlone/tools/extractors/extractV

extractF: prereqs Packages/Uintah/StandAlone/tools/extractors/extractF

extractS: prereqs Packages/Uintah/StandAlone/tools/extractors/extractS

gambitFileReader: prereqs Packages/Uintah/StandAlone/gambitFileReader

slb: prereqs Packages/Uintah/StandAlone/slb

pfs: prereqs Packages/Uintah/StandAlone/tools/pfs/pfs

pfs2: prereqs Packages/Uintah/StandAlone/tools/pfs/pfs2

timeextract: Packages/Uintah/StandAlone/tools/extractors/timeextract

faceextract: Packages/Uintah/StandAlone/tools/extractors/faceextract

lineextract: Packages/Uintah/StandAlone/tools/extractors/lineextract

compare_mms: Packages/Uintah/StandAlone/tools/compare_mms/compare_mms

compare_scalar: Packages/Uintah/StandAlone/tools/compare_mms/compare_scalar

mpi_test: Packages/Uintah/StandAlone/tools/mpi_test/mpi_test
