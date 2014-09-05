# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Schedulers

SRCS += \
	$(SRCDIR)/DetailedTasks.cc \
	$(SRCDIR)/MemoryLog.cc \
	$(SRCDIR)/MPIScheduler.cc \
	$(SRCDIR)/MessageLog.cc \
	$(SRCDIR)/MixedScheduler.cc \
	$(SRCDIR)/NirvanaLoadBalancer.cc \
	$(SRCDIR)/NullScheduler.cc \
	$(SRCDIR)/OnDemandDataWarehouse.cc \
	$(SRCDIR)/Relocate.cc \
	$(SRCDIR)/RoundRobinLoadBalancer.cc \
	$(SRCDIR)/SchedulerCommon.cc \
	$(SRCDIR)/SendState.cc \
	$(SRCDIR)/SimpleLoadBalancer.cc \
	$(SRCDIR)/SimpleScheduler.cc \
	$(SRCDIR)/SingleProcessorLoadBalancer.cc \
	$(SRCDIR)/SingleProcessorScheduler.cc \
	$(SRCDIR)/TaskGraph.cc \
	$(SRCDIR)/ThreadPool.cc \
	$(SRCDIR)/GhostOffsetVarMap.cc \
	$(SRCDIR)/DeniedAccess.cc \
	$(SRCDIR)/IncorrectAllocation.cc \
	$(SRCDIR)/Util.cc \
	$(SRCDIR)/templates.cc

PSELIBS := \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/CCA/Ports        \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/Exceptions  \
	Core/Geometry                    \
	Core/Thread                      \
	Core/Exceptions                  \
	Core/Util                        \
	Dataflow/XMLUtil

LIBS := $(XML_LIBRARY) $(TAU_LIBRARY) $(MPI_LIBRARY) $(VAMPIR_LIBRARY) $(PERFEX_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

