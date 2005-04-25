# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/MPMICE

SRCS     += \
	$(SRCDIR)/MPMICE.cc \
	$(SRCDIR)/MPMICERF.cc \
       $(SRCDIR)/MPMICEDebug.cc \

PSELIBS := \
	Packages/Uintah/CCA/Ports          \
	Packages/Uintah/Core/Disclosure    \
	Packages/Uintah/Core/Grid          \
	Packages/Uintah/Core/Util          \
	Packages/Uintah/Core/Labels        \
	Packages/Uintah/Core/Parallel      \
	Packages/Uintah/Core/ProblemSpec   \
	Packages/Uintah/Core/Exceptions    \
	Packages/Uintah/Core/Math          \
	Packages/Uintah/CCA/Components/MPM \
	Packages/Uintah/CCA/Components/ICE \
	Core/Exceptions \
	Core/Thread     \
	Core/Geometry   \
	Core/Util

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

