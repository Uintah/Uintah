# Makefile fragment for this subdirectory

# include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR  := Packages/Uintah/CCA/Components

SUBDIRS := \
        $(SRCDIR)/DataArchiver \
        $(SRCDIR)/Examples \
        $(SRCDIR)/Models \
        $(SRCDIR)/LoadBalancers \
        $(SRCDIR)/Schedulers \
        $(SRCDIR)/Regridder \
        $(SRCDIR)/SimulationController \
        $(SRCDIR)/MPM \
        $(SRCDIR)/ICE \
        $(SRCDIR)/MPMICE \
        $(SRCDIR)/Dummy \
        $(SRCDIR)/ProblemSpecification \
        $(SRCDIR)/PatchCombiner \
        $(SRCDIR)/Solvers \
        $(SRCDIR)/SwitchingCriteria \
        $(SRCDIR)/Parent

include $(SCIRUN_SCRIPTS)/recurse.mk

