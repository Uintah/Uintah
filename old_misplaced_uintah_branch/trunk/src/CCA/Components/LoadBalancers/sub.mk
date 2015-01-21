# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/LoadBalancers

SRCS += \
	$(SRCDIR)/LoadBalancerCommon.cc \
	$(SRCDIR)/LoadBalancerFactory.cc \
	$(SRCDIR)/NirvanaLoadBalancer.cc \
	$(SRCDIR)/RoundRobinLoadBalancer.cc \
	$(SRCDIR)/DynamicLoadBalancer.cc \
	$(SRCDIR)/SimpleLoadBalancer.cc \
	$(SRCDIR)/SingleProcessorLoadBalancer.cc 

PSELIBS := \
	Packages/Uintah/CCA/Components/ProblemSpecification \
	Packages/Uintah/CCA/Ports        \
	Packages/Uintah/Core/DataArchive \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Util        \
	Core/Containers                  \
	Core/Exceptions                  \
	Core/Geometry                    \
	Core/Thread                      \
	Core/Util

LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

