#*****************************************************************************
# Makefile for stand-alone C interface to Hypre 1.9.0b FAC/AMG to solve the
# pressure equation in implicit AMR-ICE. Taken from HYPRE_src/test/Makefile.
#*****************************************************************************

##################################################################
# General compilation and linking flag definitions
##################################################################

HYPRE_SRC                = ../hypre-1.9.0b/src

##################################################################
HYPRE_DRIVERS            = sstruct_fac_oren.c

HYPRE_DRIVER_OBJS        = ${HYPRE_DRIVERS:.c=.o}
HYPRE_DRIVER_EXECS       = ${HYPRE_DRIVERS:.c=}

all: sstruct_fac_oren test

clean:
	rm -f *.o *~
	rm -f ${HYPRE_DRIVER_EXECS}

##################################################################
# Rules
##################################################################

CC = g++
MPI_LIBS = -lmpi -llam -lutil -lpthread -ldl

HYPRE_INC = -I/scratch/SCIRun/src/Packages/Uintah/CCA/Components/ICE/PressureSolve/hypre-1.9.0b/src/hypre/include
HYPRE_LIB = -L/scratch/SCIRun/src/Packages/Uintah/CCA/Components/ICE/PressureSolve/hypre-1.9.0b/src/hypre/lib -lHYPRE

#sstruct_fac_orig: sstruct_fac_orig.o
#	@echo  "Building" $@ "... "
#	${CC} -o $@ $@.o ${SSTRUCT_LFLAGS} ${MPI_LIBS}


.c.o: $<
	${CC} -c $< ${HYPRE_INC}

.cc.o: $<
	${CC} -c $< ${HYPRE_INC}

sstruct_fac_oren: sstruct_fac_oren.o
	${CC} -o $@ $@.o ${HYPRE_LIB} ${MPI_LIBS} 

test : test.o
	${CC} -o $@ $@.o ${HYPRE_LI} ${MPI_LIBS}
