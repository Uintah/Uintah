#*****************************************************************************
# Makefile for stand-alone C interface to Hypre 1.9.0b FAC/AMG to solve the
# pressure equation in implicit AMR-ICE.
#
# Revision history:
# Oren Livne    19-JUL-2005       Created.
#*****************************************************************************

##################################################################
# General compilation and linking flag definitions
##################################################################

PACKAGE    = HypreStandAlone
HYPRE_SRC  = ../hypre-1.9.0b/src
CC         = g++
HYPRE_DEFS = -DHYPRE_TIMING -DHAVE_CONFIG_H
HYPRE_INC  = -I$(HYPRE_SRC)/hypre/include $(HYPRE_DEFS)
HYPRE_LIB  = -L$(HYPRE_SRC)/hypre/lib -lHYPRE_sstruct_ls -lHYPRE_sstruct_mv -lHYPRE_struct_ls -lHYPRE_struct_mv -lHYPRE_parcsr_ls -lHYPRE_DistributedMatrixPilutSolver -lHYPRE_ParaSails -lHYPRE_Euclid -lHYPRE_MatrixMatrix -lHYPRE_DistributedMatrix -lHYPRE_IJ_mv -lHYPRE_parcsr_mv -lHYPRE_seq_mv -lHYPRE_krylov -lHYPRE_utilities -llapack -lg2c
MPI_LIBS   = -lmpi -llam -lutil -lpthread -ldl

##################################################################
# Targets
##################################################################
HYPRE_DRIVERS            = mydriver.c

HYPRE_DRIVER_OBJS        = ${HYPRE_DRIVERS:.c=.o}
HYPRE_DRIVER_EXECS       = ${HYPRE_DRIVERS:.c=}

all: $(HYPRE_DRIVER_EXECS)

install:

clean:
	rm -f *~
	rm -f #*# 
	rm -f *.o 
	rm -f *.d 
	rm -f so_locations 
	rm -f ${HYPRE_DRIVER_EXECS}
	@ echo "Project cleaned."

distrib:
	tar zcvf $(PACKAGE).tar.gz Makefile *.c* *.h

##################################################################
# Rules
##################################################################

.c.o: $<
	${CC} -c $< ${HYPRE_INC}

.cc.o: $<
	${CC} -c $< ${HYPRE_INC}

mydriver: mydriver.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${HYPRE_LIB} ${MPI_LIBS} 

mpitest : mpitest.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${HYPRE_LI} ${MPI_LIBS}

-include $(patsubst %.o,%.d, $(HYPRE_DRIVER_OBJS))
