# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/Thread

SRCS     += $(SRCDIR)/Guard.cc $(SRCDIR)/MutexPool.cc \
	$(SRCDIR)/ParallelBase.cc $(SRCDIR)/Runnable.cc \
	$(SRCDIR)/Thread.cc $(SRCDIR)/ThreadError.cc \
	$(SRCDIR)/SimpleReducer.cc $(SRCDIR)/ThreadGroup.cc \
	$(SRCDIR)/Thread_unix.cc $(SRCDIR)/ThreadPool.cc \
	$(SRCDIR)/WorkQueue.cc 

SRCS += $(TIME_IMPL) $(THREAD_IMPL)

PSELIBS := Core/Exceptions
LIBS := $(THREAD_LIBS) $(TRACEBACK_LIB)

include $(SRCTOP)/scripts/smallso_epilogue.mk

