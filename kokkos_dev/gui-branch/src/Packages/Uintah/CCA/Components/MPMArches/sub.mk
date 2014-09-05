# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/MPMArches

SRCS     += \
	$(SRCDIR)/MPMArches.cc \
	$(SRCDIR)/MPMArchesLabel.cc

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
	Core/Exceptions \
	Core/Thread     \
	Core/Datatypes  \
	Core/Geometry   \
	Dataflow/XMLUtil

LIBS := $(PETSC_LIBS) $(XML_LIBRARY) $(MPI_LIBRARY) -lm $(FLIBS)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

$(SRCDIR)/MPMArches.o: $(SRCDIR)/fortran/collect_drag_cc_fort.h
$(SRCDIR)/MPMArches.o: $(SRCDIR)/fortran/interp_centertoface_fort.h
$(SRCDIR)/MPMArches.o: $(SRCDIR)/fortran/momentum_exchange_term_continuous_cc_fort.h
$(SRCDIR)/MPMArches.o: $(SRCDIR)/fortran/pressure_force_fort.h
