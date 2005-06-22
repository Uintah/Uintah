# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

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
        $(SRCDIR)/MPMArches \
        $(SRCDIR)/Arches \
        $(SRCDIR)/Arches/fortran \
        $(SRCDIR)/Arches/Mixing \
        $(SRCDIR)/Arches/Radiation \
        $(SRCDIR)/Arches/Radiation/fortran \
        $(SRCDIR)/ProblemSpecification \
        $(SRCDIR)/PatchCombiner \
        $(SRCDIR)/Solvers \
        $(SRCDIR)/Switcher 

include $(SCIRUN_SCRIPTS)/recurse.mk

SRCS    := $(SRCDIR)/ComponentFactory.cc 

PSELIBS := \
        Core/Exceptions \
        Core/Util       \
        Packages/Uintah/CCA/Components/Arches \
        Packages/Uintah/CCA/Components/Examples \
        Packages/Uintah/CCA/Components/ICE      \
        Packages/Uintah/CCA/Components/MPM      \
        Packages/Uintah/CCA/Components/MPMArches   \
        Packages/Uintah/CCA/Components/MPMICE   \
        Packages/Uintah/CCA/Components/Switcher \
        Packages/Uintah/CCA/Ports        \
        Packages/Uintah/Core/Exceptions  \
        Packages/Uintah/Core/Parallel    \
        Packages/Uintah/Core/ProblemSpec \
        Packages/Uintah/Core/Util        

LIBS    := $(XML_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
