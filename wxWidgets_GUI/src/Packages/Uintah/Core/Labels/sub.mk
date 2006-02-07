# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Core/Labels

SRCS     += \
	$(SRCDIR)/ICELabel.cc \
	$(SRCDIR)/MPMLabel.cc \
	$(SRCDIR)/MPMICELabel.cc

PSELIBS := \
	Core/Exceptions \
	Core/Util \
	Core/Geometry \
	Packages/Uintah/Core/Disclosure \
	Packages/Uintah/Core/Exceptions \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/Util \
	Packages/Uintah/Core/Math \
	Packages/Uintah/Core/ProblemSpec \

LIBS := $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

