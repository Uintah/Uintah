# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Dataflow/Modules/Operators

SRCS     += \
	$(SRCDIR)/ScalarFieldOperator.cc \
	$(SRCDIR)/ScalarFieldBinaryOperator.cc \
	$(SRCDIR)/ScalarFieldAverage.cc \
	$(SRCDIR)/TensorFieldOperator.cc \
	$(SRCDIR)/TensorParticlesOperator.cc \
	$(SRCDIR)/ParticleEigenEvaluator.cc \
	$(SRCDIR)/EigenEvaluator.cc \
	$(SRCDIR)/VectorFieldOperator.cc \
	$(SRCDIR)/VectorParticlesOperator.cc \
[INSERT NEW CODE FILE HERE]

PSELIBS := \
	Packages/Uintah/Core/Datatypes     \
	Packages/Uintah/Core/DataArchive     \
	Packages/Uintah/Core/Disclosure     \
	Packages/Uintah/CCA/Ports          \
	Packages/Uintah/Core/Grid          \
	Packages/Uintah/Core/Math          \
	Packages/Uintah/Core/ProblemSpec   \
	Packages/Uintah/Core/Exceptions    \
	Dataflow/Network  \
	Dataflow/Ports    \
	Core/Containers   \
	Core/Persistent   \
	Core/Exceptions   \
	Core/GuiInterface \
	Core/Thread       \
	Core/Datatypes    \
	Core/Geom         \
	Core/Geometry     \
	Core/GeomInterface \
	Core/Malloc
LIBS := $(XML_LIBRARY) $(M_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

ifeq ($(LARGESOS),no)
UINTAH_SCIRUN := $(UINTAH_SCIRUN) $(LIBNAME)
endif

