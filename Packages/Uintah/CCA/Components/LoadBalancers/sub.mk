# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/LoadBalancers

SRCS += \
	$(SRCDIR)/LoadBalancerCommon.cc \
	$(SRCDIR)/LoadBalancerFactory.cc \
	$(SRCDIR)/NirvanaLoadBalancer.cc \
	$(SRCDIR)/RoundRobinLoadBalancer.cc \
	$(SRCDIR)/ParticleLoadBalancer.cc \
	$(SRCDIR)/SimpleLoadBalancer.cc \
	$(SRCDIR)/SingleProcessorLoadBalancer.cc

PSELIBS := \
	Packages/Uintah/CCA/Components/ProblemSpecification \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Util        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/CCA/Ports        \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/CCA/Components/ProblemSpecification \
	Core/Geometry                    \
	Core/Containers                  \
	Core/Thread                      \
	Core/Exceptions                  \
	Core/Util

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

