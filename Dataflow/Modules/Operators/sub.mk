# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Dataflow/Modules/Operators

SRCS     += \
	$(SRCDIR)/ScalarFieldOperator.cc \
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
	Core/Geometry
LIBS := $(XML_LIBRARY) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

