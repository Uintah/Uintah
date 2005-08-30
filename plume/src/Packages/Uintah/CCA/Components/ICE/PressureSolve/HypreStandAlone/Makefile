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
HYPRE_SRC  = /scratch/SCIRun/hypre-1.9.0b/src
#HYPRE_SRC  = /home/csafe/livne/hypre-1.9.0b/src
CC         = g++
CCFLAGS    = #-O2 -MD -Wall # Allow .d files, output all warnings
CCFLAGS    = -ggdb -MD -Wall # Allow .d files, output all warnings
HYPRE_DEFS = -DHYPRE_TIMING -DHAVE_CONFIG_H
MPI_INC    = -I/usr/local/lam-mpi/include
MPI_LIB    = -L/usr/local/lam-mpi/lib -lmpi -llam -lutil -lpthread -ldl
HYPRE_INC  = -I$(HYPRE_SRC)/hypre/include $(HYPRE_DEFS)
HYPRE_LIB  = -L$(HYPRE_SRC)/hypre/lib -lHYPRE_sstruct_ls -lHYPRE_sstruct_mv -lHYPRE_struct_ls -lHYPRE_struct_mv -lHYPRE_parcsr_ls -lHYPRE_DistributedMatrixPilutSolver -lHYPRE_ParaSails -lHYPRE_Euclid -lHYPRE_MatrixMatrix -lHYPRE_DistributedMatrix -lHYPRE_IJ_mv -lHYPRE_parcsr_mv -lHYPRE_seq_mv -lHYPRE_krylov -lHYPRE_utilities -llapack -lg2c

# Home cygwin configuration
#HYPRE_SRC  = /home/livne/hypre-1.9.0b/src
#MPI_LIB    =
#HYPRE_LIB  = -L$(HYPRE_SRC)/hypre/lib -lHYPRE_sstruct_ls -lHYPRE_sstruct_mv -lHYPRE_struct_ls -lHYPRE_struct_mv -lHYPRE_parcsr_ls -lHYPRE_DistributedMatrixPilutSolver -lHYPRE_ParaSails -lHYPRE_Euclid -lHYPRE_MatrixMatrix -lHYPRE_DistributedMatrix -lHYPRE_IJ_mv -lHYPRE_parcsr_mv -lHYPRE_seq_mv -lHYPRE_krylov -lHYPRE_utilities -lg2c

##################################################################
# Targets
##################################################################
TARGET1          = mydriver
TARGET1_OBJS     = \
Box.o \
Error.o \
Hierarchy.o \
IntMatrix.o \
Level.o \
mydriver.o \
Param.o \
Patch.o \
SolverAMG.o \
Solver.o \
SolverFAC.o \
TestLinear.o \
util.o \
DebugStream.o \
Side.o

TARGET2          = mpitest
TARGET2_OBJS     = \
mpitest.o \

TARGET3          = vectortest
TARGET3_OBJS     = \
vectortest.o \
Error.o \
IntMatrix.o \
util.o \
Box.o \
Side.o \
DebugStream.o 

all: $(TARGET1)

install:

clean:
	rm -f *~
	rm -f #*# 
	rm -f *.o 
	rm -f *.d 
	rm -f so_locations 
	rm -f $(TARGET1)
	rm -f $(TARGET2)
	rm -f $(TARGET3)
	@ echo "Project cleaned."

distrib:
	tar zcvf $(PACKAGE).tar.gz Makefile *.c* *.h

##################################################################
# Rules
##################################################################

.cc.o: $<
	${CC} ${CCFLAGS} ${MPI_INC} ${HYPRE_INC} -c $<

$(TARGET1) : $(TARGET1_OBJS)
	@echo  "Building" $@ "... "
	${CC} -o $@ $(TARGET1_OBJS) ${HYPRE_LIB} ${MPI_LIB} 

$(TARGET2) : $(TARGET2_OBJS)
	@echo  "Building" $@ "... "
	${CC} -o $@ $(TARGET2_OBJS) ${HYPRE_LIB} ${MPI_LIB}

$(TARGET3) : $(TARGET3_OBJS)
	@echo  "Building" $@ "... "
	${CC} -o $@ $(TARGET3_OBJS) ${HYPRE_LIB} ${MPI_LIB}

-include $(patsubst %.o,%.d, $(TARGET1_OBJS) $(TARGET2_OBJS) $(TARGET3_OBJS))
