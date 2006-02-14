# Makefile fragment for this subdirectory

# include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR  := Packages/Uintah/CCA/Components

ADIR = $(SRCDIR)/Arches

# The following variables are used by the Fake* scripts... please
# do not modify...
#
MPM            = $(SRCDIR)/MPM
MPMICE         = $(SRCDIR)/MPMICE
ICE            = $(SRCDIR)/ICE
ARCHES         = $(SRCDIR)/Arches $(ADIR)/fortran $(ADIR)/Mixing $(ADIR)/Radiation $(ADIR)/Radiation/fortran
MPMARCHES      = $(SRCDIR)/MPMArches
#DUMMY_LIB     = $(SRCDIR)/Dummy

SUBDIRS := \
        $(SRCDIR)/DataArchiver \
        $(SRCDIR)/Examples \
        $(SRCDIR)/Models \
        $(SRCDIR)/LoadBalancers \
        $(SRCDIR)/Schedulers \
        $(SRCDIR)/Regridder \
        $(SRCDIR)/SimulationController \
        $(DUMMY_LIB)      \
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

