#
# Makefile for sparselb src
#
SHELL = /bin/sh
SPARSELIB_DIR = ../..
include $(SPARSELIB_DIR)/makefile.def

# put MV++ library in the same place as SparseLib++ library
#
MV_LIB_A = ../../lib/libmv.a

SRCS 	= \
	  mvvd.cc	\
	  mvmd.cc \
	  mvvf.cc	\
	  mvmf.cc \
	  mvvdio.cc	\
	  mvblasd.cc \
	  mvblasf.cc \
	  mvblasi.cc \
	  mvvi.cc \
	  mvmi.cc \
	  mvvc.cc \
	  mvmc.cc \
	  mvvcio.cc \
	  mvblasc.cc


OBJS    =  \
	  mvvd.o	\
	  mvmd.o \
	  mvblasd.o \
	  mvvf.o	\
	  mvmf.o \
	  mvblasi.o \
	  mvblasf.o \
	  mvvi.o \
	  mvmi.o \
	  mvvdio.o	\
	  mvvc.o \
	  mvmc.o \
	  mvvcio.o \
	  mvblasc.o


all: lib

srcs: $(SRCS)
lib: $(MV_LIB_A)

#$(MV_LIB_A) : $(OBJS)
#	@ ar rv $(MV_LIB_A) $?
#	@ case x$(HASRANLIB) in xt ) echo ranlib; ranlib $(MV_LIB_A);; esac


$(MV_LIB_A) : $(OBJS)
	@ ar rv $(MV_LIB_A) $?
	@ case x$(HASRANLIB) in xt ) echo ranlib; ranlib $(MV_LIB_A);; esac

clean: 
	/bin/rm -f $(OBJS) 

wipe:
	/bin/rm -f $(OBJS) $(MV_LIB_A) 
