# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Dataflow/Modules/Selectors

SRCS     += \
	$(SRCDIR)/TimestepSelector.cc       \
	$(SRCDIR)/FieldExtractor.cc   \
	$(SRCDIR)/ScalarFieldExtractor.cc   \
	$(SRCDIR)/VectorFieldExtractor.cc   \
	$(SRCDIR)/TensorFieldExtractor.cc   \
	$(SRCDIR)/ParticleFieldExtractor.cc \
#[INSERT NEW CODE FILE HERE]

PSELIBS := \
	Packages/Uintah/Core/Datatypes     \
	Packages/Uintah/CCA/Ports          \
	Packages/Uintah/Core/Grid          \
	Packages/Uintah/Core/Math          \
	Packages/Uintah/Core/Disclosure    \
	Packages/Uintah/Core/ProblemSpec   \
	Packages/Uintah/Core/Exceptions    \
	Dataflow/Network  \
	Dataflow/Ports    \
	Dataflow/XMLUtil  \
	Core/Containers   \
	Core/Disclosure   \
	Core/Persistent   \
	Core/Exceptions   \
	Core/GuiInterface \
	Core/Thread       \
	Core/Datatypes    \
	Core/Geom         \
	Core/Util         \
	Core/Geometry
LIBS := $(XML_LIBRARY) $(MPI_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

