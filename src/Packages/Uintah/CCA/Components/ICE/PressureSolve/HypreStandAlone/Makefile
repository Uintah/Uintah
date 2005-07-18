#*****************************************************************************
# Makefile for stand-alone C interface to Hypre 1.9.0b FAC/AMG to solve the
# pressure equation in implicit AMR-ICE. Taken from HYPRE_src/test/Makefile.
#*****************************************************************************

##################################################################
# General compilation and linking flag definitions
##################################################################

HYPRE_SRC                = ..

include $(HYPRE_SRC)/config/Makefile.config

CINCLUDES                = ${INCLUDES} ${MPIINCLUDE}
CDEFS                    = -DHYPRE_TIMING

C_COMPILE_FLAGS          = \
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I${HYPRE_BUILD_DIR}/include\
 ${CINCLUDES}\
 ${CDEFS}

MPILIBFLAGS              = ${MPILIBDIRS} ${MPILIBS} ${MPIFLAGS} 
LIBFLAGS                 = ${LIBDIRS} ${LIBS}
LDLIBFLAGS               = ${LDLIBDIRS} ${LDLIBS} ${LIBM}
LAPACKLIBFLAGS           = ${LAPACKLIBDIRS} ${LAPACKLIBS}
BLASLIBFLAGS             = ${BLASLIBDIRS} ${BLASLIBS}

LFLAGS                   =\
 -L${HYPRE_BUILD_DIR}/lib\
 -lHYPRE_sstruct_ls\
 -lHYPRE_sstruct_mv\
 -lHYPRE_struct_ls\
 -lHYPRE_struct_mv\
 -lHYPRE_parcsr_ls\
 -lHYPRE_DistributedMatrixPilutSolver \
 -lHYPRE_ParaSails \
 -lHYPRE_Euclid \
 -lHYPRE_MatrixMatrix \
 -lHYPRE_DistributedMatrix \
 -lHYPRE_IJ_mv\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 ${LAPACKLIBFLAGS} \
 ${BLASLIBFLAGS} \
 ${MPILIBFLAGS} ${LIBFLAGS} ${FLIBS} ${LDLIBFLAGS}

##################################################################
# Sstruct linking flag definitions
##################################################################
SSTRUCT_LFLAGS           =\
 -L${HYPRE_BUILD_DIR}/lib\
 -lHYPRE_sstruct_ls\
 -lHYPRE_sstruct_mv\
 -lHYPRE_struct_ls\
 -lHYPRE_struct_mv\
 -lHYPRE_parcsr_ls\
 -lHYPRE_DistributedMatrixPilutSolver \
 -lHYPRE_ParaSails \
 -lHYPRE_Euclid \
 -lHYPRE_MatrixMatrix \
 -lHYPRE_DistributedMatrix \
 -lHYPRE_IJ_mv\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 ${LAPACKLIBFLAGS} \
 ${BLASLIBFLAGS} \
 ${MPILIBFLAGS} ${LIBFLAGS} ${FLIBS} ${LDLIBFLAGS}

##################################################################
# Targets
##################################################################
HYPRE_STANDARD_DRIVERS   =\
 sstruct_fac_oren.c\
 sstruct_fac_orig.c

HYPRE_DRIVERS_F77        = ${HYPRE_STANDARD_DRIVERS}
HYPRE_DRIVER_OBJS        = ${HYPRE_DRIVERS:.c=.o}
HYPRE_DRIVER_EXECS       = ${HYPRE_DRIVERS:.c=}

all: ${HYPRE_DRIVER_EXECS}

install:

clean:
	rm -f *.o

distclean: clean
	rm -f ${HYPRE_DRIVER_EXECS}

##################################################################
# Rules
##################################################################

sstruct_fac_orig: sstruct_fac_orig.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${SSTRUCT_LFLAGS}

sstruct_fac: sstruct_fac_oren.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${SSTRUCT_LFLAGS}
