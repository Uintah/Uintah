# Generated automatically from Makefile.in by configure.
# Makefile for Threads

include ../make.include

SRCTOP  := ..
SRCDIR  := .
OBJTOP  := /home/sci/simpson/sv/scirun_upgrade/SCIRun/src/Packages/CollabVis/Standalone/Server

OBJTOP_ABS    := $(shell cd $(OBJTOP) ; pwd)
SRCTOP_ABS    := $(shell cd $(SRCTOP) ; pwd)
ifeq ($(OBJTOP),.)
LIBDIR := lib/
else
LIBDIR := $(OBJTOP_ABS)/lib/
endif
LIBDIR_ABS    := $(shell mkdir -p $(LIBDIR); cd $(LIBDIR) ; pwd)

VPATH := $(SRCDIR)

INCLUDES += -I$(SRCTOP) $(XML_INCLUDE)

INCLUDES := $(sort $(INCLUDES))
#SOFLAGS := $(SOFLAGS) -rpath ${LIBDIR}

LINUXOBJS = Guard.o	MutexPool.o ParallelBase.o  \
	Runnable.o SimpleReducer.o Thread.o ThreadError.o ThreadGroup.o \
	ThreadPool.o  Thread_pthreads.o Thread_unix.o \
	Time_unix.o WorkQueue.o

IRIXOBJS = Guard.o	MutexPool.o ParallelBase.o  \
	Runnable.o SimpleReducer.o Thread.o ThreadError.o ThreadGroup.o \
	ThreadPool.o  Thread_irix.o Thread_unix.o \

BASE_OBJS = 	Guard.o	MutexPool.o ParallelBase.o  \
	Runnable.o SimpleReducer.o Thread.o ThreadError.o ThreadGroup.o \
	ThreadPool.o WorkQueue.o Thread_unix.o 

OTHER_SRCS = Thread_pthreads.cc Time_unix.cc # 

OBJS =  $(BASE_OBJS) $(OTHER_SRCS:.cc=.o)


SOLIB = $(LIBDIR)/libThread.so

LINUXLIBS = -lpthread
IRIXLIBS  = -lfetchop

LIBS = -lpthread   

all:	$(SOLIB)


$(SOLIB):	$(OBJS)
	$(CXX) $(SOFLAGS)	$(OBJS) -o $(SOLIB) $(LIBS)


clean:
	/bin/rm -f $(OBJS) $(SOLIB)


# -include $(OBJS:.o=.d)
