# $Id$ 

CFLAGS	         = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' ${BS_INCLUDE}
FFLAGS	         = 
LOCDIR           = ${HOME}/adaptive/csafe/pcgmg/petsc

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

MGF = pset.o psol.o setCoarserOperator.o setLevelPtrs.o		\
      setLevelShapes.o setShape.o setOperator.o setval.o 	\
      gf_copy.o copy_to_gridfcn.o copy_from_gridfcn.o normalize.o \
      setHomogeneousFlowBC.o relax.o gemv.o inject.o interp.o \
      gf_axpy.o stencilMult.o

pcgmg: pcg_main.o multigridShell.o stencilShell.o $(MGF) chkopts
	-${CLINKER} -o pcgmg pcg_main.o multigridShell.o stencilShell.o \
                       $(MGF) ${PETSC_LIB} -lfortran

remake:
	make clean; make pcgmg

multigridShell.o:	multigridShell.c multigridShell.h
stencilShell.o:		stencilShell.c stencilShell.h

pset.o:		pset.f space.h grids.h variables.h operators.h
psol.o:		psol.f space.h grids.h variables.h operators.h
stencilMult.o:	stencilMult.f space.h grids.h variables.h 
