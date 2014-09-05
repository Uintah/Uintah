# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Schedulers

SRCS += $(SRCDIR)/templates.cc \
	$(SRCDIR)/MixedScheduler.cc \
	$(SRCDIR)/MPIScheduler.cc $(SRCDIR)/MessageLog.cc \
	$(SRCDIR)/NullScheduler.cc \
	$(SRCDIR)/OnDemandDataWarehouse.cc \
	$(SRCDIR)/RoundRobinLoadBalancer.cc \
	$(SRCDIR)/SendState.cc \
	$(SRCDIR)/SimpleLoadBalancer.cc \
	$(SRCDIR)/SingleProcessorScheduler.cc \
	$(SRCDIR)/SingleProcessorLoadBalancer.cc \
	$(SRCDIR)/TaskGraph.cc \
	$(SRCDIR)/ThreadPool.cc

PSELIBS := \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/CCA/Ports \
	Packages/Uintah/Core/Parallel \
	Packages/Uintah/Core/Exceptions \
	Core/Thread \
	Core/Exceptions \
	Core/Util \
	Dataflow/XMLUtil
LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) $(VAMPIR_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

