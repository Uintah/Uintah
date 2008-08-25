# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone/tools/extractors

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
        Packages/Uintah/Core/ProblemSpec             \
        Packages/Uintah/CCA/Components/ProblemSpecification
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
          $(PETSC_LIBRARY) $(BLAS_LIBRARY) $(LAPACK_LIBRARY) \
          $(MPI_LIBRARY) $(M_LIBRARY)
endif

##############################################
# timeextract

SRCS    := $(SRCDIR)/timeextract.cc
PROGRAM := $(SRCDIR)/timeextract

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# faceextract

SRCS    := $(SRCDIR)/faceextract.cc
PROGRAM := $(SRCDIR)/faceextract

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# extractV

SRCS    := $(SRCDIR)/extractV.cc
PROGRAM := $(SRCDIR)/extractV

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# extractF

SRCS    := $(SRCDIR)/extractF.cc
PROGRAM := $(SRCDIR)/extractF

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# extractS

SRCS    := $(SRCDIR)/extractS.cc
PROGRAM := $(SRCDIR)/extractS

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# partextract

SRCS    := $(SRCDIR)/partextract.cc
PROGRAM := $(SRCDIR)/partextract

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# lineextract

SRCS    := $(SRCDIR)/lineextract.cc
PROGRAM := $(SRCDIR)/lineextract

include $(SCIRUN_SCRIPTS)/program.mk
