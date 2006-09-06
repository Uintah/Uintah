# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Dataflow/Modules/Selectors

SRCS     += \
	$(SRCDIR)/TimestepSelector.cc       \
	$(SRCDIR)/FieldExtractor.cc   \
	$(SRCDIR)/ScalarFieldExtractor.cc   \
	$(SRCDIR)/VectorFieldExtractor.cc   \
	$(SRCDIR)/TensorFieldExtractor.cc   \
	$(SRCDIR)/ParticleFieldExtractor.cc 
#[INSERT NEW CODE FILE HERE]

PSELIBS := \
	Packages/Uintah/Core/Datatypes     \
	Packages/Uintah/Core/DataArchive   \
	Packages/Uintah/CCA/Ports          \
	Packages/Uintah/Core/Grid          \
	Packages/Uintah/Core/Util          \
	Packages/Uintah/Core/Math          \
	Packages/Uintah/Core/Disclosure    \
	Packages/Uintah/Core/ProblemSpec   \
	Packages/Uintah/Core/Parallel      \
	Packages/Uintah/Core/Exceptions    \
	Dataflow/Network  \
	Core/Basis        \
	Core/Containers   \
	Core/Persistent   \
	Core/Exceptions   \
	Dataflow/GuiInterface \
	Core/Thread       \
	Core/Datatypes    \
	Core/Geom         \
	Core/Util         \
	Core/Geometry     \
	Core/GeomInterface
LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk


ifeq ($(LARGESOS),no)
UINTAH_SCIRUN := $(UINTAH_SCIRUN) $(LIBNAME)
endif
