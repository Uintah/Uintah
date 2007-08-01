# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone/tools/extractors

# The following variables are used by the Fake* scripts... please
# do not modify...
#
COMPONENTS      = Packages/Uintah/CCA/Components
CA              = Packages/Uintah/CCA/Components/Arches
DUMMY_LIB       = Packages/Uintah/CCA/Components/Dummy
#ARCHES_SUB_LIBS= $(CA)/Mixing $(CA)/fortran $(CA)/Radiation $(CA)/Radiation/fortran
#ARCHES_LIBS    = $(COMPONENTS)/Arches $(COMPONENTS)/MPMArches
MPM_LIB         = Packages/Uintah/CCA/Components/MPM
ICE_LIB         = Packages/Uintah/CCA/Components/ICE
MPMICE_LIB      = Packages/Uintah/CCA/Components/MPMICE

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

##############################################
# timeextract

SRCS := $(SRCDIR)/timeextract.cc
PROGRAM :=  $(SRCDIR)/timeextract

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# faceextract

SRCS := $(SRCDIR)/faceextract.cc
PROGRAM :=  $(SRCDIR)/faceextract

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# extractV

SRCS := $(SRCDIR)/extractV.cc
PROGRAM := $(SRCDIR)/extractV

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# extractF

SRCS := $(SRCDIR)/extractF.cc
PROGRAM := $(SRCDIR)/extractF

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# extractS

SRCS := $(SRCDIR)/extractS.cc
PROGRAM := $(SRCDIR)/extractS

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# partextract

SRCS := $(SRCDIR)/partextract.cc
PROGRAM := $(SRCDIR)/partextract

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# lineextract

SRCS := $(SRCDIR)/lineextract.cc
PROGRAM := $(SRCDIR)/lineextract

include $(SCIRUN_SCRIPTS)/program.mk


