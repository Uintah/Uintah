# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone/compare_mms

SRCS := $(SRCDIR)/compare_mms.cc $(SRCDIR)/MMS1.cc

PROGRAM := Packages/Uintah/StandAlone/compare_mms/compare_mms

ifeq ($(LARGESOS),yes)
  PSELIBS := Packages/Uintah
else

  PSELIBS := \
        Core/Containers   \
        Core/Exceptions   \
        Core/Geometry     \
        Core/Math         \
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
        Packages/Uintah/CCA/Components/PatchCombiner
endif

LIBS := $(XML_LIBRARY) $(F_LIBRARY) $(HYPRE_LIBRARY) \
        $(CANTERA_LIBRARY) \
        $(PETSC_LIBRARY) $(BLAS_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk


