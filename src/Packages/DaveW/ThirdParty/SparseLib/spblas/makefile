#
# Makefile for sparselb src
#

SHELL = /bin/sh

SPARSELIB_DIR = ..
include $(SPARSELIB_DIR)/makefile.def 


SRCS 	= spmm.cc spsm.cc
OBJS	= spmm.o  spsm.o

all: lib

lib: $(SPARSEBLAS_A)

$(SPARSEBLAS_A) : $(OBJS)
	@ ar rv $(SPARSEBLAS_A) $?
	@ case x$(HASRANLIB) in xt ) echo ranlib; ranlib $(SPARSEBLAS_A);; esac

clean: 
	/bin/rm -f $(OBJS) 

wipe:
	/bin/rm -f $(OBJS) $(SPARSEBLAS_A)

#spblas.$(OBJ_EXT) : spblas.cc \
#	$(SPARSELIB_INCLUDE_DIR)/spblas.h
