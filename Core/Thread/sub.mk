#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#

# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Core/Thread

SRCS     += $(SRCDIR)/Guard.cc $(SRCDIR)/MutexPool.cc \
	$(SRCDIR)/ParallelBase.cc $(SRCDIR)/Runnable.cc \
	$(SRCDIR)/Thread.cc $(SRCDIR)/ThreadError.cc \
	$(SRCDIR)/SimpleReducer.cc $(SRCDIR)/ThreadGroup.cc \
	$(SRCDIR)/Thread_unix.cc $(SRCDIR)/ThreadPool.cc \
	$(SRCDIR)/WorkQueue.cc 

SRCS += $(TIME_IMPL) $(THREAD_IMPL)

PSELIBS := Core/Exceptions
LIBS := $(THREAD_LIBRARY) $(TRACEBACK_LIB) $(SEMAPHORE_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

