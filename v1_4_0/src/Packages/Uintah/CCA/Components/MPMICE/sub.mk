# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/MPMICE

SRCS     += \
	$(SRCDIR)/MPMICE.cc \
	$(SRCDIR)/MPMICERF.cc \
	$(SRCDIR)/MPMICELabel.cc \
       $(SRCDIR)/MPMICEDebug.cc \

PSELIBS := \
	Packages/Uintah/CCA/Ports          \
	Packages/Uintah/Core/Grid          \
	Packages/Uintah/Core/Parallel      \
	Packages/Uintah/Core/ProblemSpec   \
	Packages/Uintah/Core/Exceptions    \
	Packages/Uintah/Core/Math          \
	Packages/Uintah/CCA/Components/MPM \
	Packages/Uintah/CCA/Components/ICE \
	Core/Exceptions \
	Core/Thread     \
	Core/Datatypes  \
	Core/Geometry   \
	Dataflow/XMLUtil

LIBS := $(XML_LIBRARY) -lm

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

