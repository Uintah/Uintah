# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/MPMArches

SRCS     += \
	$(SRCDIR)/MPMArches.cc \
	$(SRCDIR)/MPMArchesLabel.cc

ifneq ($(CC_DEPEND_REGEN),-MD)
# The fortran code doesn't work under g++ yet
SUBDIRS := $(SRCDIR)/fortran 

include $(SRCTOP)/scripts/recurse.mk
FLIB := -lftn
endif

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

LIBS := $(PETSC_LIBS) $(XML_LIBRARY) $(FLIB) $(MPI_LIBRARY) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk
