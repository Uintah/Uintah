#
# Makefile for sparselb src
#

SHELL = /bin/sh

SPARSELIB_DIR = ..
include $(SPARSELIB_DIR)/makefile.def 


SRCS = 	\
	  compcol_double.cc		\
	  comprow_double.cc		\
	  coord_double.cc		\
	  iotext_double.cc		\
	  diagpre_double.cc		\
	  icpre_double.cc		\
	  ilupre_double.cc		\
	  qsort_double.cc		\
	  qsort_int.cc			\
	  iohb_double.cc		\
	  iohb.c


OBJ = \
	  compcol_double.o		\
	  comprow_double.o		\
	  coord_double.o		\
	  iotext_double.o		\
	  diagpre_double.o		\
	  icpre_double.o		\
	  ilupre_double.o		\
	  qsort_double.o		\
	  qsort_int.o			\
	  iohb_double.o		    \
	  iohb.o

all: lib

srcs: $(SRCS)

lib: $(SPARSELIB_A)

$(SPARSELIB_A) : $(OBJ) 
	@ ar rv $(SPARSELIB_A) $?
	@ case x$(HASRANLIB) in xt ) echo ranlib; ranlib $(SPARSELIB_A);; esac

#iohb.o : iohb.c
#	$(CC) $(CCFLAGS) -c iohb.c

clean: 
	/bin/rm -f $(OBJ) 

wipe: 
	/bin/rm -f $(OBJ) $(SPARSELIB_A)

compcol_double.$(OBJ_EXT) : compcol_double.cc  \
  $(SPARSELIB_INCLUDE_DIR)/comprow_double.h $(SPARSELIB_INCLUDE_DIR)/coord_double.h 

comprow_double.$(OBJ_EXT) : comprow_double.cc  \
  $(SPARSELIB_INCLUDE_DIR)/comprow_double.h $(SPARSELIB_INCLUDE_DIR)/coord_double.h 

coord_double.$(OBJ_EXT) : coord_double.cc  \
  $(SPARSELIB_INCLUDE_DIR)/comprow_double.h $(SPARSELIB_INCLUDE_DIR)/coord_double.h 

iotext_double.$(OBJ_EXT) : iotext_double.cc  \
  $(SPARSELIB_INCLUDE_DIR)/compcol_double.h \
  $(SPARSELIB_INCLUDE_DIR)/comprow_double.h $(SPARSELIB_INCLUDE_DIR)/coord_double.h 
   
iohb_double.$(OBJ_EXT) : iohb_double.cc  \
  $(SPARSELIB_INCLUDE_DIR)/minstr.h $(SPARSELIB_INCLUDE_DIR)/compcol_double.h \
  $(SPARSELIB_INCLUDE_DIR)/comprow_double.h $(SPARSELIB_INCLUDE_DIR)/coord_double.h 
   
   
diagpre.$(OBJ_EXT) : diagpre.cc  \
  $(SPARSELIB_INCLUDE_DIR)/diagpre.h \
  $(SPARSELIB_INCLUDE_DIR)/comprow_double.h $(SPARSELIB_INCLUDE_DIR)/compcol_double.h 

icpre.$(OBJ_EXT) : icpredouble.cc  \
  $(SPARSELIB_INCLUDE_DIR)/icpre_double.h  \
  $(SPARSELIB_INCLUDE_DIR)/comprow.h $(SPARSELIB_INCLUDE_DIR)/compcol.h \
  $(SPARSELIB_INCLUDE_DIR)/qsort.h  

ilupre.$(OBJ_EXT) : ilupre.cc  \
  $(SPARSELIB_INCLUDE_DIR)/ilupre.h \
  $(SPARSELIB_INCLUDE_DIR)/comprow.h $(SPARSELIB_INCLUDE_DIR)/compcol.h \
  $(SPARSELIB_INCLUDE_DIR)/qsort.h  


