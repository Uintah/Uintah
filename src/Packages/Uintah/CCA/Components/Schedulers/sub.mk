# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Schedulers

SRCS += \
	$(SRCDIR)/DetailedTasks.cc \
	$(SRCDIR)/MPIScheduler.cc \
	$(SRCDIR)/MessageLog.cc \
	$(SRCDIR)/MixedScheduler.cc \
	$(SRCDIR)/NullScheduler.cc \
	$(SRCDIR)/OnDemandDataWarehouse.cc \
	$(SRCDIR)/RoundRobinLoadBalancer.cc \
	$(SRCDIR)/SchedulerCommon.cc \
	$(SRCDIR)/SendState.cc \
	$(SRCDIR)/SimpleLoadBalancer.cc \
	$(SRCDIR)/SimpleScheduler.cc \
	$(SRCDIR)/SingleProcessorLoadBalancer.cc \
	$(SRCDIR)/SingleProcessorScheduler.cc \
	$(SRCDIR)/TaskGraph.cc \
	$(SRCDIR)/ThreadPool.cc \
	$(SRCDIR)/templates.cc

PSELIBS := \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/CCA/Ports \
	Packages/Uintah/Core/Parallel \
	Packages/Uintah/Core/Exceptions \
	Core/Geometry \
	Core/Thread \
	Core/Exceptions \
	Core/Util \
	Dataflow/XMLUtil
LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) $(VAMPIR_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

