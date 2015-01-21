# Makefile fragment for this subdirectory

# include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR  := Packages/Uintah/CCA/Components

ADIR = $(SRCDIR)/Arches

# The following variables are used by the Fake* scripts... please
# do not modify...
#

ifeq ($(BUILD_MPM),yes)
  MPM         := $(SRCDIR)/MPM
  ifeq ($(BUILD_ICE),yes)
    MPMICE    := $(SRCDIR)/MPMICE
  endif
endif
ifeq ($(BUILD_ICE),yes)
  ICE         := $(SRCDIR)/ICE
endif
ifeq ($(BUILD_ARCHES),yes)
  ARCHES      := $(SRCDIR)/Arches $(ADIR)/fortran $(ADIR)/Mixing $(ADIR)/Radiation $(ADIR)/Radiation/fortran
  ifeq ($(BUILD_MPM),yes)
    MPMARCHES := $(SRCDIR)/MPMArches
  endif
endif

SUBDIRS := \
        $(SRCDIR)/DataArchiver \
        $(SRCDIR)/Examples \
        $(SRCDIR)/Models \
        $(SRCDIR)/LoadBalancers \
        $(SRCDIR)/Schedulers \
        $(SRCDIR)/Regridder \
        $(SRCDIR)/SimulationController \
        $(MPM)            \
        $(ICE)            \
        $(MPMICE)         \
        $(ARCHES)         \
        $(MPMARCHES)      \
        $(SRCDIR)/ProblemSpecification \
        $(SRCDIR)/PatchCombiner \
        $(SRCDIR)/Solvers \
        $(SRCDIR)/SwitchingCriteria \
        $(SRCDIR)/OnTheFlyAnalysis \
        $(SRCDIR)/Parent

include $(SCIRUN_SCRIPTS)/recurse.mk

