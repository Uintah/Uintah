# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/MPMArches

SRCS     += $(SRCDIR)/MPMArches.cc \
	$(SRCDIR)/MPMArchesLabel.cc \
	$(SRCDIR)/CutCellInfo.cc

SUBDIRS := $(SRCDIR)/fortran 
include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := \
	Packages/Uintah/CCA/Ports          \
	Packages/Uintah/Core/Grid          \
	Packages/Uintah/Core/Parallel      \
	Packages/Uintah/Core/ProblemSpec   \
	Packages/Uintah/Core/Exceptions    \
	Packages/Uintah/Core/Math          \
	Packages/Uintah/CCA/Components/MPM \
	Packages/Uintah/CCA/Components/Arches \
	Packages/Uintah/CCA/Components/Arches/fortran \
	Core/Exceptions \
	Core/Thread     \
	Core/Geometry   

LIBS := $(XML_LIBRARY) $(PETSC_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(F_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

$(SRCDIR)/MPMArches.o: $(SRCDIR)/fortran/collect_drag_cc_fort.h
$(SRCDIR)/MPMArches.o: $(SRCDIR)/fortran/collect_scalar_fctocc_fort.h
$(SRCDIR)/MPMArches.o: $(SRCDIR)/fortran/energy_exchange_term_fort.h
$(SRCDIR)/MPMArches.o: $(SRCDIR)/fortran/interp_centertoface_fort.h
$(SRCDIR)/MPMArches.o: $(SRCDIR)/fortran/momentum_exchange_term_continuous_cc_fort.h
$(SRCDIR)/MPMArches.o: $(SRCDIR)/fortran/pressure_force_fort.h
$(SRCDIR)/MPMArches.o: $(SRCDIR)/fortran/read_complex_geometry_fort.h
