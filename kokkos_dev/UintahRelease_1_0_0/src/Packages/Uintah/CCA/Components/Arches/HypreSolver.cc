//----- HypreSolver.cc ----------------------------------------------

#include <fstream> // work around compiler bug with RHEL 3

#include <Packages/Uintah/CCA/Components/Arches/HypreSolver.h>
#include <Core/Containers/Array1.h>
#include <Core/Thread/Time.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/CCA/Components/Arches/PressureSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/StencilMatrix.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Exceptions/PetscError.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities.h"
#include "HYPRE_struct_ls.h"
#include "krylov.h"
#include "struct_mv.h"

#undef CHKERRQ
#define CHKERRQ(x) if(x) throw PetscError(x, __FILE__, __FILE__, __LINE__);

using namespace std;
using namespace Uintah;
using namespace SCIRun;


// ****************************************************************************
// Default constructor for HypreSolver
// ****************************************************************************
HypreSolver::HypreSolver(const ProcessorGroup* myworld)
   : d_myworld(myworld)
{
}

// ****************************************************************************
// Destructor
// ****************************************************************************
HypreSolver::~HypreSolver()
{
  finalizeSolver();
}

// ****************************************************************************
// Problem setup
// ****************************************************************************
void 
HypreSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("LinearSolver");
  db->getWithDefault("ksptype", d_kspType, "cg");
  if (d_kspType == "smg")
    d_kspType = "0";
  else
    if (d_kspType == "pfmg")
      d_kspType = "1";
    else
      if (d_kspType == "cg")
	{
	  db->getWithDefault("pctype", d_pcType, "pfmg");
	  if (d_pcType == "smg")
	    d_kspType = "10";
	  else
	    if (d_pcType == "pfmg")
	      d_kspType = "11";
	    else
	      if (d_pcType == "jacobi")
		d_kspType = "17";
	      else
		if (d_pcType == "none")
		  d_kspType = "19";
	}
  db->getWithDefault("max_iter", d_maxSweeps, 75);
  db->getWithDefault("res_tol", d_residual, 1.0e-8);
}


// ****************************************************************************
// Set up the grid structure
// ***************************************************************************
void
HypreSolver::gridSetup(const ProcessorGroup*,
		       const Patch* patch)

{
  int nx, ny, nz;
  int bx, by, bz;
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  nx = idxHi.x() - idxLo.x() + 1;
  ny = idxHi.y() - idxLo.y() + 1;
  nz = idxHi.z() - idxLo.z() + 1;
     
  for (int i = 0; i < 6; i++)
    {    
      d_A_num_ghost[i] = 0;
    }

  d_volume  = nx*ny*nz;    //number of nodes per processor
  bx = 1;
  by = 1;
  bz = 1;
  d_dim = 3;
  d_stencilSize = 4;
  d_nblocks = bx*by*bz;           //number of blocks per processor, now is set to 1
  d_stencilIndices = hypre_CTAlloc(int, d_stencilSize);
  d_offsets = hypre_CTAlloc(int*, d_stencilSize);   //Allocating memory for 7 point stencil but since I'm using symmetry, only 4 is needed
  d_offsets[0] = hypre_CTAlloc(int, 3); //Allocating memory for 3 d_dimension indexing
  d_offsets[0][0] = 0;            //setting the location of each stencil.
  d_offsets[0][1] = 0;            //First index is the stencil number.
  d_offsets[0][2] = -1;           //Second index is the [0,1,2]=[i,j,k]
  d_offsets[1] = hypre_CTAlloc(int, 3);
  d_offsets[1][0] = 0; 
  d_offsets[1][1] = -1; 
  d_offsets[1][2] = 0; 
  d_offsets[2] = hypre_CTAlloc(int, 3);
  d_offsets[2][0] = -1; 
  d_offsets[2][1] = 0; 
  d_offsets[2][2] = 0; 
  d_offsets[3] = hypre_CTAlloc(int, 3);
  d_offsets[3][0] = 0; 
  d_offsets[3][1] = 0; 
  d_offsets[3][2] = 0;
     
  d_ilower = hypre_CTAlloc(int*, d_nblocks);
  d_iupper = hypre_CTAlloc(int*, d_nblocks);

  for (int i = 0; i < d_nblocks; i++)
    {
      d_ilower[i] = hypre_CTAlloc(int, d_dim);
      d_iupper[i] = hypre_CTAlloc(int, d_dim);
    }
  
  for (int i = 0; i < d_dim; i++)
    {
     d_A_num_ghost[2*i] = 1;
     d_A_num_ghost[2*i + 1] = 1;
    }
  
  /* compute d_ilower and d_iupper from (p,q,r), (bx,by,bz), and (nx,ny,nz) */
  int ib = 0;
  
  for (int iz = 0; iz < bz; iz++) {
    for (int iy = 0; iy < by; iy++) {
      for (int ix = 0; ix < bx; ix++) {
	  d_ilower[ib][0] = idxLo.x();
	  d_iupper[ib][0] = idxHi.x();
	  d_ilower[ib][1] = idxLo.y();
	  d_iupper[ib][1] = idxHi.y();
	  d_ilower[ib][2] = idxLo.z();
	  d_iupper[ib][2] = idxHi.z();
	  ib++;
      }
    }
  }
#if 0
  ib = 0;  
  for (int iz = 0; iz < bz; iz++) {
    for (int iy = 0; iy < by; iy++) {
      for (int ix = 0; ix < bx; ix++) {
	printf("  d_ilower[%d](i,j,k)= (%d, %d, %d)\n",ib, d_ilower[ib][0], d_ilower[ib][1],   d_ilower[ib][2]);
	printf("  d_iupper[%d](i,j,k)= (%d, %d, %d)\n",ib, d_iupper[ib][0], d_iupper[ib][1],   d_iupper[ib][2]);
	ib++;
      }
    }
  }
#endif
  
  HYPRE_StructGridCreate(MPI_COMM_WORLD, d_dim, &d_grid);

  for (int ib = 0; ib < d_nblocks; ib++)
    {
      HYPRE_StructGridSetExtents(d_grid, d_ilower[ib], d_iupper[ib]);
    }
 
  const Level* level = patch->getLevel();
  IntVector periodic_vector = level->getPeriodicBoundaries();
  int periodic[3];
  periodic[0] = periodic_vector.x();
  periodic[1] = periodic_vector.y();
  periodic[2] = periodic_vector.z();
  HYPRE_StructGridSetPeriodic(d_grid, periodic);
  HYPRE_StructGridAssemble(d_grid);  

  /*-----------------------------------------------------------
   * Set up the stencil structure
   *-----------------------------------------------------------*/
  HYPRE_StructStencilCreate(d_dim, d_stencilSize, &d_stencil);
   
  for (int s = 0; s < d_stencilSize; s++)
    {
      HYPRE_StructStencilSetElement(d_stencil, s, d_offsets[s]);
    }

}

// ****************************************************************************
// Fill linear parallel matrix
// ****************************************************************************
void 
HypreSolver::setPressMatrix(const ProcessorGroup* pc,
			    const Patch* patch,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars,
			    const ArchesLabel*)
{ 
  double start_time = Time::currentSeconds();
  gridSetup(pc, patch);
  /*-----------------------------------------------------------
   * Set up the matrix structure
   *-----------------------------------------------------------*/

  HYPRE_StructMatrixCreate(MPI_COMM_WORLD, d_grid, d_stencil, &d_A);
  HYPRE_StructMatrixSetSymmetric(d_A, 1);
  HYPRE_StructMatrixSetNumGhost(d_A, d_A_num_ghost);
  HYPRE_StructMatrixInitialize(d_A); 

   /*-----------------------------------------------------------
    * Set up the linear system (b & x)
    *-----------------------------------------------------------*/
   HYPRE_StructVectorCreate(MPI_COMM_WORLD, d_grid, &d_b);
   HYPRE_StructVectorInitialize(d_b);
   HYPRE_StructVectorCreate(MPI_COMM_WORLD, d_grid, &d_x);
   HYPRE_StructVectorInitialize(d_x);
  
  int i, s;
 
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  d_value = hypre_CTAlloc(double, (d_stencilSize)*d_volume);
  
  /* Set the coefficients for the grid */
  i = 0;
  for (s = 0; s < (d_stencilSize); s++)
    {
      d_stencilIndices[s] = s;
    }
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	d_value[i] = -constvars->pressCoeff[Arches::AB][IntVector(colX,colY,colZ)]; //[0,0,-1]
	d_value[i+1] = -constvars->pressCoeff[Arches::AS][IntVector(colX,colY,colZ)]; //[0,-1,0]
	d_value[i+2] = -constvars->pressCoeff[Arches::AW][IntVector(colX,colY,colZ)]; //[-1,0,0]
	d_value[i+3] = constvars->pressCoeff[Arches::AP][IntVector(colX,colY,colZ)]; //[0,0,0]

#if 0
	cerr << "["<<colX<<","<<colY<<","<<colZ<<"]"<<endl;  
	cerr << "value[AB]=" << d_value[i] << endl;
	cerr << "value[AS]=" << d_value[i+1] << endl;
	cerr << "value[AW]=" << d_value[i+2] << endl;
	cerr << "value[AP]=" << d_value[i+3] << endl;
#endif
	    i = i + d_stencilSize;
      }
    }
  }
  
  for (int ib = 0; ib < d_nblocks; ib++)
    {
      HYPRE_StructMatrixSetBoxValues(d_A, d_ilower[ib], d_iupper[ib], d_stencilSize,
				     d_stencilIndices, d_value);
    }


  HYPRE_StructMatrixAssemble(d_A);
  //cerr << "Matrix Assemble time = " << Time::currentSeconds()-start_time << endl;

#if 0
  HYPRE_StructMatrixPrint("driver.out.A", d_A, 0);
#endif

  hypre_TFree(d_value);

  // assemble right hand side and solution vector
  d_value = hypre_CTAlloc(double, d_volume);
 
  i = 0;
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	d_value[i] = constvars->pressNonlinearSrc[IntVector(colX,colY,colZ)]; 
	//cerr << "b[" << i << "] =" << d_value[i] << endl;
	i++;
      }
    }
  }
    
  for (int ib = 0; ib < d_nblocks; ib++)
    {
      HYPRE_StructVectorSetBoxValues(d_b, d_ilower[ib], d_iupper[ib], d_value);
    }
  
  i = 0;
  // Set up the initial guess
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	d_value[i] = vars->pressure[IntVector(colX, colY, colZ)];
	//cerr << "x0[" << i << "] =" << d_value[i] << endl;
	i++;;
      }
    }
  }
    
  for (int ib = 0; ib < d_nblocks; ib++)
    {
      HYPRE_StructVectorSetBoxValues(d_x, d_ilower[ib], d_iupper[ib], d_value);
    }

  HYPRE_StructVectorAssemble(d_b);

#if 0
  HYPRE_StructVectorPrint("driver.out.b", d_b, 0);
#endif
  
  HYPRE_StructVectorAssemble(d_x);

#if 0
  HYPRE_StructVectorPrint("driver.out.x0", d_x, 0);  
#endif
  
  hypre_TFree(d_value);

  int me = d_myworld->myrank();
  if(me == 0) {
    cerr << "Time in HYPRE Assemble: " << Time::currentSeconds()-start_time << " seconds\n";
  }
}

bool
HypreSolver::pressLinearSolve()
{
  /*-----------------------------------------------------------
   * Solve the system using CG
   *-----------------------------------------------------------*/
  /* I have set this up so you can change to use different
     solvers and preconditioners without re-compiling.  Just
     change the ksptype in the ups files to different numbers:
     0  = SMG
     1  = PFMG
     10 = SMG as the preconditoner and CG as the solver
     11 = PFMG as the preconditioner and CG as the solver (default)
     17 = Jacobi as the preconditioner and CG as the solver
     19 = CG as the solver with no preconditioner
  */
     
  HYPRE_StructVector tmp;  
  int num_iterations;
  int n_pre, n_post, skip;
  double sum_b, iprod, final_res_norm;

  /*Calculating initial norm*/
  HYPRE_StructVectorCreate(MPI_COMM_WORLD, d_grid, &tmp);
  HYPRE_StructVectorInitialize(tmp);  
  hypre_StructCopy(d_b,tmp);
  hypre_StructMatvec(1.0, d_A, d_x, -1.0,tmp);
  iprod = hypre_StructInnerProd(tmp,tmp);
  init_norm = sqrt(iprod);
  HYPRE_StructVectorDestroy(tmp);

  /*Calculating sum of RHS*/
  iprod = hypre_StructInnerProd(d_b,d_b);
  sum_b = sqrt(iprod);

  
  n_pre = 1;
  n_post = 1;
  skip = 1;
  HYPRE_StructSolver solver, precond;

  
  int me = d_myworld->myrank();
  double start_time = Time::currentSeconds();
  
  if (d_kspType == "0") {
    /*Solve the system using SMG*/
    HYPRE_StructSMGCreate(MPI_COMM_WORLD, &solver);
    HYPRE_StructSMGSetMemoryUse(solver, 0);
    HYPRE_StructSMGSetMaxIter(solver, d_maxSweeps);
    HYPRE_StructSMGSetTol(solver, d_residual);
    HYPRE_StructSMGSetRelChange(solver, 0);
    HYPRE_StructSMGSetNumPreRelax(solver, n_pre);
    HYPRE_StructSMGSetNumPostRelax(solver, n_post);
    HYPRE_StructSMGSetLogging(solver, 1);
    HYPRE_StructSMGSetup(solver, d_A, d_b, d_x);
    HYPRE_StructSMGSolve(solver, d_A, d_b, d_x);
    
    HYPRE_StructSMGGetNumIterations(solver, &num_iterations);
    HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
    HYPRE_StructSMGDestroy(solver);
    //    cerr << "SMG Solve time = " << Time::currentSeconds()-start_time << endl;
  }
  else if (d_kspType == "1") {
    /*Solve the system using PFMG*/
    HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &solver);
    HYPRE_StructPFMGSetMaxIter(solver, d_maxSweeps);
    HYPRE_StructPFMGSetTol(solver, d_residual);
    HYPRE_StructPFMGSetRelChange(solver, 0);
    /* weighted Jacobi = 1; red-black GS = 2 */
    HYPRE_StructPFMGSetRelaxType(solver, 1);
    HYPRE_StructPFMGSetNumPreRelax(solver, n_pre);
    HYPRE_StructPFMGSetNumPostRelax(solver, n_post);
    HYPRE_StructPFMGSetSkipRelax(solver, skip);
    /*HYPRE_StructPFMGSetDxyz(solver, dxyz);*/
    HYPRE_StructPFMGSetLogging(solver, 1);
    HYPRE_StructPFMGSetup(solver, d_A, d_b, d_x);
    HYPRE_StructPFMGSolve(solver, d_A, d_b, d_x);
    
    HYPRE_StructPFMGGetNumIterations(solver, &num_iterations);
    HYPRE_StructPFMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
    HYPRE_StructPFMGDestroy(solver);
    //    cerr << "PFMG Solve time = " << Time::currentSeconds()-start_time << endl;
  }
  else {
    HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);
    HYPRE_PCGSetMaxIter( (HYPRE_Solver)solver, d_maxSweeps);
    HYPRE_PCGSetTol( (HYPRE_Solver)solver, d_residual);
    HYPRE_PCGSetTwoNorm( (HYPRE_Solver)solver, 1 );
    HYPRE_PCGSetRelChange( (HYPRE_Solver)solver, 0 );
    HYPRE_PCGSetLogging( (HYPRE_Solver)solver, 1 );
 
    
    if (d_kspType == "10") {
      /* use symmetric SMG as preconditioner */
      HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
      HYPRE_StructSMGSetMemoryUse(precond, 0);
      HYPRE_StructSMGSetMaxIter(precond, 1);
      HYPRE_StructSMGSetTol(precond, d_residual);
      HYPRE_StructSMGSetZeroGuess(precond);
      HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
      HYPRE_StructSMGSetNumPostRelax(precond, n_post);
      HYPRE_StructSMGSetLogging(precond, 0);
      HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
			   (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve,
			   (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup,
			   (HYPRE_Solver) precond);
      //      cerr << "SMG Precond time = " << Time::currentSeconds()-start_time << endl;
    }
  
    else if (d_kspType == "11") {  
      /* use symmetric PFMG as preconditioner */
      HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond);
      HYPRE_StructPFMGSetMaxIter(precond, 1);
      HYPRE_StructPFMGSetTol(precond, d_residual);
      HYPRE_StructPFMGSetZeroGuess(precond);
      /* weighted Jacobi = 1; red-black GS = 2 */
      HYPRE_StructPFMGSetRelaxType(precond, 1);
      HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
      HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
      HYPRE_StructPFMGSetSkipRelax(precond, skip);
      /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
      HYPRE_StructPFMGSetLogging(precond, 0);
      HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
			   (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
			   (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
			   (HYPRE_Solver) precond);
      //      cerr << "PFMG Precond time = " << Time::currentSeconds()-start_time << endl;
    }
    else if (d_kspType == "17") {
      /* use two-step Jacobi as preconditioner */
      HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &precond);
      HYPRE_StructJacobiSetMaxIter(precond, 2);
      HYPRE_StructJacobiSetTol(precond, d_residual);
      HYPRE_StructJacobiSetZeroGuess(precond);
      HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
			   (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSolve,
			   (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSetup,
			   (HYPRE_Solver) precond);
      //      cerr << "Jacobi Precond time = " << Time::currentSeconds()-start_time << endl;
    }
    
    //double dummy_start = Time::currentSeconds();
    HYPRE_PCGSetup
      ( (HYPRE_Solver)solver, (HYPRE_Matrix)d_A, (HYPRE_Vector)d_b, (HYPRE_Vector)d_x );
    //cerr << "PCG Setup time = " << Time::currentSeconds()-dummy_start << endl;

    //dummy_start = Time::currentSeconds();
    HYPRE_PCGSolve
      ( (HYPRE_Solver)solver, (HYPRE_Matrix)d_A, (HYPRE_Vector)d_b, (HYPRE_Vector)d_x);
    //cerr << "PCG Solve time = " << Time::currentSeconds()-dummy_start << endl;
    
    HYPRE_PCGGetNumIterations( (HYPRE_Solver)solver, &num_iterations );
    HYPRE_PCGGetFinalRelativeResidualNorm( (HYPRE_Solver)solver, &final_res_norm );
    HYPRE_StructPCGDestroy(solver);

    if (d_kspType == "10") {
      HYPRE_StructSMGDestroy(precond);
    }
    else if (d_kspType == "11") {
      HYPRE_StructPFMGDestroy(precond);
    }
    else if (d_kspType == "17") {
      HYPRE_StructJacobiDestroy(precond);
    }
  }
  if(me == 0) {
    cerr << "hypre: final_res_norm: " << final_res_norm << ", iterations: " << num_iterations << ", solver time: " << Time::currentSeconds()-start_time << " seconds\n";
    cerr << "Init Norm: " << init_norm << " Error reduced by: " <<  final_res_norm/(init_norm+1.0e-20) << endl;
    cerr << "Sum of RHS vector: " << sum_b << endl;
  }
  if (((final_res_norm/(init_norm+1.0e-20) < 1.0) && (final_res_norm < 2.0))||
     ((final_res_norm<d_residual)&&(init_norm<d_residual)))
    return true;
  else
    return false;
}


void
HypreSolver::copyPressSoln(const Patch* patch, ArchesVariables* vars)
{
  // copy solution vector back into the array
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  double* xvec;
  xvec = hypre_CTAlloc(double, d_volume);
 
  for (int ib = 0; ib < d_nblocks; ib++)
    {
      HYPRE_StructVectorGetBoxValues(d_x, d_ilower[ib], d_iupper[ib], xvec);
    }
  
#if 0
  HYPRE_StructVectorPrint("driver.out.x", d_x, 0);
#endif
  
  int i = 0;
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
  	vars->pressure[IntVector(colX, colY, colZ)] = xvec[i];
	//cerr << "xvec[" << i << "] = " << xvec[i] << endl;
	i++;
      }
    }
  }

  hypre_TFree(xvec);
}
  
void
HypreSolver::destroyMatrix() 
{
  /*-----------------------------------------------------------
   * Finalize things
   *-----------------------------------------------------------*/
  int i;
  HYPRE_StructGridDestroy(d_grid);
  HYPRE_StructStencilDestroy(d_stencil);
  HYPRE_StructMatrixDestroy(d_A);
  HYPRE_StructVectorDestroy(d_b);
  HYPRE_StructVectorDestroy(d_x);
   
  for (i = 0; i < d_nblocks; i++)
    {
      hypre_TFree(d_iupper[i]);
      hypre_TFree(d_ilower[i]);
     }
  hypre_TFree(d_ilower);
  hypre_TFree(d_iupper);
  hypre_TFree(d_stencilIndices);
  
  for ( i = 0; i < d_stencilSize; i++)
    hypre_TFree(d_offsets[i]);
  hypre_TFree(d_offsets);
  
  hypre_FinalizeMemoryDebug();
}

void HypreSolver::finalizeSolver()
{
  
}
