# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/MPMArches

SRCS     += \
	$(SRCDIR)/MPMArches.cc \
	$(SRCDIR)/MPMArchesLabel.cc

PSELIBS := \
	Packages/Uintah/CCA/Ports          \
	Packages/Uintah/Core/Grid          \
	Packages/Uintah/Core/Parallel      \
	Packages/Uintah/Core/Exceptions    \
	Packages/Uintah/Core/Math          \
	Packages/Uintah/CCA/Components/MPM \
	Packages/Uintah/CCA/Components/Arches \
	Core/Exceptions \
	Core/Thread     \
	Core/Datatypes  \
	Core/Geometry   \
	Dataflow/XMLUtil

LIBS := $(XML_LIBRARY) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk
