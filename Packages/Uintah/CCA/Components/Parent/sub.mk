# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR  := Packages/Uintah/CCA/Components/Parent

SRCS    := $(SRCDIR)/Switcher.cc \
	   $(SRCDIR)/ComponentFactory.cc 


PSELIBS := \
	Core/Exceptions \
	Core/Util \
        Packages/Uintah/CCA/Ports        \
	Packages/Uintah/Core/Disclosure  \
        Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/Grid        \
        Packages/Uintah/Core/Parallel    \
        Packages/Uintah/Core/ProblemSpec \
        Packages/Uintah/Core/Util        \
        Packages/Uintah/CCA/Components/Arches \
        Packages/Uintah/CCA/Components/Examples \
        Packages/Uintah/CCA/Components/ICE      \
        Packages/Uintah/CCA/Components/MPM      \
        Packages/Uintah/CCA/Components/MPMArches   \
        Packages/Uintah/CCA/Components/MPMICE  

LIBS    := $(XML_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
