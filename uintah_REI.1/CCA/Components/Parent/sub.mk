# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR  := Packages/Uintah/CCA/Components/Parent

SRCS    := $(SRCDIR)/Switcher.cc \
	   $(SRCDIR)/ComponentFactory.cc 

# The following variables are used by the Fake* scripts... please
# do not modify...
#
COMPONENTS = Packages/Uintah/CCA/Components
ifneq ($(IS_WIN),yes)
# disable ARCHES on windows for now, as we don't know what to do about fortran yet..
# don't indent these, or fake* will probably fail
#ARCHES= $(COMPONENTS)/Arches $(COMPONENTS)/MPMArches
endif
ICE    = $(COMPONENTS)/ICE
MPM    = $(COMPONENTS)/MPM
MPMICE = $(COMPONENTS)/MPMICE
DUMMY  = $(COMPONENTS)/Dummy

PSELIBS := \
	Core/Exceptions \
	Core/Util \
	Core/Geometry \
        Packages/Uintah/CCA/Ports        \
	Packages/Uintah/Core/Disclosure  \
        Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Math	 \
        Packages/Uintah/Core/Parallel    \
        Packages/Uintah/Core/ProblemSpec \
        Packages/Uintah/Core/Util        \
        $(DUMMY)  \
        $(ARCHES) \
        $(ICE)    \
        $(MPM)    \
        $(MPMICE) \
        $(COMPONENTS)/Examples             \
	$(COMPONENTS)/PatchCombiner        \
	$(COMPONENTS)/ProblemSpecification \
	$(COMPONENTS)/Solvers              \
	$(COMPONENTS)/SwitchingCriteria

LIBS    := $(XML2_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
