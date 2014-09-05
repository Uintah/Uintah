#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/Thread

SRCS     += $(SRCDIR)/Guard.cc $(SRCDIR)/MutexPool.cc \
	$(SRCDIR)/ParallelBase.cc $(SRCDIR)/Runnable.cc \
	$(SRCDIR)/Thread.cc $(SRCDIR)/ThreadError.cc \
	$(SRCDIR)/SimpleReducer.cc $(SRCDIR)/ThreadGroup.cc \
	$(SRCDIR)/Thread_unix.cc $(SRCDIR)/ThreadPool.cc \
	$(SRCDIR)/WorkQueue.cc 

SRCS += $(TIME_IMPL) $(THREAD_IMPL)

PSELIBS := SCICore/Exceptions
LIBS := $(THREAD_LIBS) $(TRACEBACK_LIB)

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.3  2000/03/23 10:21:27  sparker
# Use libexc to print out stack straces on the SGI
# Added "name" method to ThreadError to match exception base class
# Fixed a compiler warning in Thread_irix.cc
#
# Revision 1.2  2000/03/20 19:37:54  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:42  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
