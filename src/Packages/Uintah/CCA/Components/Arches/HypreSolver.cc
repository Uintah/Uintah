//----- HypreSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/debug.h>
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
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
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
#define CHKERRQ(x) if(x) throw PetscError(x, __FILE__);

using namespace std;
using namespace Uintah;
using namespace SCIRun;

#include <Packages/Uintah/CCA/Components/Arches/fortran/rescal_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/underelax_fort.h>

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
  db->getWithDefault("underrelax", d_underrelax, 1.0);
  db->getWithDefault("res_tol", d_residual, 1.0e-8);
}


// ****************************************************************************
// Actual compute of pressure residual
// ****************************************************************************
void 
HypreSolver::computePressResidual(const ProcessorGroup*,
				 const Patch* patch,
				 DataWarehouseP&,
				 DataWarehouseP&,
				 ArchesVariables* vars)
{
#ifdef ARCHES_PRES_DEBUG
  cerr << " Before Pressure Compute Residual : " << endl;
  IntVector l = vars->residualPressure.getWindow()->getLowIndex();
  IntVector h = vars->residualPressure.getWindow()->getHighIndex();
  for (int ii = l.x(); ii < h.x(); ii++) {
    cerr << "residual for ii = " << ii << endl;
    for (int jj = l.y(); jj < h.y(); jj++) {
      for (int kk = l.z(); kk < h.z(); kk++) {
	cerr.width(14);
	cerr << vars->residualPressure[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "Resid Press = " << vars->residPress << " Trunc Press = " <<
    vars->truncPress << endl;
#endif
  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call
  fort_rescal(idxLo, idxHi, vars->pressure, vars->residualPressure,
	      vars->pressCoeff[Arches::AE], vars->pressCoeff[Arches::AW], 
	      vars->pressCoeff[Arches::AN], vars->pressCoeff[Arches::AS], 
	      vars->pressCoeff[Arches::AT], vars->pressCoeff[Arches::AB], 
	      vars->pressCoeff[Arches::AP], vars->pressNonlinearSrc,
	      vars->residPress, vars->truncPress);

#ifdef ARCHES_PRES_DEBUG
  cerr << " After Pressure Compute Residual : " << endl;
  for (int ii = l.x(); ii < h.x(); ii++) {
    cerr << "residual for ii = " << ii << endl;
    for (int jj = l.y(); jj < h.y(); jj++) {
      for (int kk = l.z(); kk < h.z(); kk++) {
	cerr.width(14);
	cerr << vars->residualPressure[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "Resid Press = " << vars->residPress << " Trunc Press = " <<
    vars->truncPress << endl;
#endif
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
// Actual calculation of order of magnitude term for pressure equation
// ****************************************************************************
void 
HypreSolver::computePressOrderOfMagnitude(const ProcessorGroup* ,
				const Patch* ,
				DataWarehouseP& ,
				DataWarehouseP& , ArchesVariables* )
{

}

// ****************************************************************************
// Actual compute of pressure underrelaxation
// ****************************************************************************
void 
HypreSolver::computePressUnderrelax(const ProcessorGroup*,
				   const Patch* patch,
				    ArchesVariables* vars,
				    ArchesConstVariables* constvars)
{
  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call
  fort_underelax(idxLo, idxHi, constvars->pressure,
		 vars->pressCoeff[Arches::AP],
		 vars->pressNonlinearSrc, d_underrelax);

#ifdef ARCHES_PRES_DEBUG
  cerr << " After Pressure Underrelax : " << endl;
  cerr << " Underrelaxation coefficient: " << d_underrelax << '\n';
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "pressure for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(14);
	cerr << vars->pressure[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << " After Pressure Underrelax : " << endl;
  for (int ii = domLong.x(); ii <= domHing.x(); ii++) {
    cerr << "pressure AP for ii = " << ii << endl;
    for (int jj = domLong.y(); jj <= domHing.y(); jj++) {
      for (int kk = domLong.z(); kk <= domHing.z(); kk++) {
	cerr.width(14);
	cerr << (vars->pressCoeff[Arches::AP])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << " After Pressure Underrelax : " << endl;
  for (int ii = domLong.x(); ii <= domHing.x(); ii++) {
    cerr << "pressure SU for ii = " << ii << endl;
    for (int jj = domLong.y(); jj <= domHing.y(); jj++) {
      for (int kk = domLong.z(); kk <= domHing.z(); kk++) {
	cerr.width(14);
	cerr << vars->pressNonlinearSrc[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
#endif
#undef ARCHES_PRES_DEBUG
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
  if ((final_res_norm/(init_norm+1.0e-20) < 1.0) && (final_res_norm < 2.0))
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

void 
HypreSolver::pressLisolve(const ProcessorGroup*,
			 const Patch*,
			 DataWarehouseP&,
			 DataWarehouseP&,
			 ArchesVariables*,
			 const ArchesLabel*)
{

}


void HypreSolver::finalizeSolver()
{
  
}

//****************************************************************************
// Actual compute of Velocity residual
//****************************************************************************

void 
HypreSolver::computeVelResidual(const ProcessorGroup* ,
			       const Patch* patch,
			       DataWarehouseP& ,
			       DataWarehouseP& , 
			       int index, ArchesVariables* vars)
{
  // Get the patch bounds and the variable bounds
  IntVector domLo;
  IntVector domHi;
  IntVector idxLo;
  IntVector idxHi;

  switch (index) {
  case Arches::XDIR:
    domLo = vars->uVelocity.getFortLowIndex();
    domHi = vars->uVelocity.getFortHighIndex();
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();
    //fortran call

  fort_rescal(idxLo, idxHi, vars->uVelocity, vars->residualUVelocity,
	      vars->uVelocityCoeff[Arches::AE], 
	      vars->uVelocityCoeff[Arches::AW], 
	      vars->uVelocityCoeff[Arches::AN], 
	      vars->uVelocityCoeff[Arches::AS], 
	      vars->uVelocityCoeff[Arches::AT], 
	      vars->uVelocityCoeff[Arches::AB], 
	      vars->uVelocityCoeff[Arches::AP], 
	      vars->uVelNonlinearSrc, vars->residUVel,
	      vars->truncUVel);
#ifdef ARCHES_VEL_DEBUG
    cerr << " After U Velocity Compute Residual : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "u residual for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->residualUVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "Resid U Vel = " << vars->residUVel << " Trunc U Vel = " <<
      vars->truncUVel << endl;
#endif

    break;
  case Arches::YDIR:
    domLo = vars->vVelocity.getFortLowIndex();
    domHi = vars->vVelocity.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
    //fortran call

  fort_rescal(idxLo, idxHi, vars->vVelocity, vars->residualVVelocity,
	      vars->vVelocityCoeff[Arches::AE], 
	      vars->vVelocityCoeff[Arches::AW], 
	      vars->vVelocityCoeff[Arches::AN], 
	      vars->vVelocityCoeff[Arches::AS], 
	      vars->vVelocityCoeff[Arches::AT], 
	      vars->vVelocityCoeff[Arches::AB], 
	      vars->vVelocityCoeff[Arches::AP], 
	      vars->vVelNonlinearSrc, vars->residVVel,
	      vars->truncVVel);

#ifdef ARCHES_VEL_DEBUG
    cerr << " After V Velocity Compute Residual : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "v residual for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->residualVVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "Resid V Vel = " << vars->residVVel << " Trunc V Vel = " <<
      vars->truncVVel << endl;
#endif

    break;
  case Arches::ZDIR:
    domLo = vars->wVelocity.getFortLowIndex();
    domHi = vars->wVelocity.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
    //fortran call

  fort_rescal(idxLo, idxHi, vars->wVelocity, vars->residualWVelocity,
	      vars->wVelocityCoeff[Arches::AE], 
	      vars->wVelocityCoeff[Arches::AW], 
	      vars->wVelocityCoeff[Arches::AN], 
	      vars->wVelocityCoeff[Arches::AS], 
	      vars->wVelocityCoeff[Arches::AT], 
	      vars->wVelocityCoeff[Arches::AB], 
	      vars->wVelocityCoeff[Arches::AP], 
	      vars->wVelNonlinearSrc, vars->residWVel,
	      vars->truncWVel);

#ifdef ARCHES_VEL_DEBUG
    cerr << " After W Velocity Compute Residual : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "w residual for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->residualWVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "Resid W Vel = " << vars->residWVel << " Trunc W Vel = " <<
      vars->truncWVel << endl;
#endif

    break;
  default:
    throw InvalidValue("Invalid index in LinearSolver for velocity");
  }
}


//****************************************************************************
// Actual calculation of order of magnitude term for Velocity equation
//****************************************************************************
void 
HypreSolver::computeVelOrderOfMagnitude(const ProcessorGroup* ,
				const Patch* ,
				DataWarehouseP& ,
				DataWarehouseP& , ArchesVariables* )
{

  //&vars->truncUVel
  //&vars->truncVVel
  //&vars->truncWVel

}



//****************************************************************************
// Velocity Underrelaxation
//****************************************************************************
void 
HypreSolver::computeVelUnderrelax(const ProcessorGroup* ,
				  const Patch* patch,
				  int index, ArchesVariables* vars,
				  ArchesConstVariables* constvars)
{
  // Get the patch bounds and the variable bounds
  IntVector domLo;
  IntVector domHi;
  IntVector domLong;
  IntVector domHing;
  IntVector idxLo;
  IntVector idxHi;

  switch (index) {
  case Arches::XDIR:
    domLo = constvars->uVelocity.getFortLowIndex();
    domHi = constvars->uVelocity.getFortHighIndex();
    domLong = vars->uVelocityCoeff[Arches::AP].getFortLowIndex();
    domHing = vars->uVelocityCoeff[Arches::AP].getFortHighIndex();
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();
    fort_underelax(idxLo, idxHi, constvars->uVelocity,
		   vars->uVelocityCoeff[Arches::AP], vars->uVelNonlinearSrc,
		   d_underrelax);

#ifdef ARCHES_VEL_DEBUG
    cerr << " After U Vel Underrelax : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "U Vel for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->uVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << " After U Vel Underrelax : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "U Vel AP for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << (vars->uVelocityCoeff[Arches::AP])[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << " After U Vel Underrelax : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "U Vel SU for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->uVelNonlinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

    break;
    case Arches::YDIR:
    domLo = constvars->vVelocity.getFortLowIndex();
    domHi = constvars->vVelocity.getFortHighIndex();
    domLong = vars->vVelocityCoeff[Arches::AP].getFortLowIndex();
    domHing = vars->vVelocityCoeff[Arches::AP].getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
    fort_underelax(idxLo, idxHi, constvars->vVelocity,
		   vars->vVelocityCoeff[Arches::AP], vars->vVelNonlinearSrc,
		   d_underrelax);

#ifdef ARCHES_VEL_DEBUG
    cerr << " After V Vel Underrelax : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "V Vel for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->vVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << " After V Vel Underrelax : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "V Vel AP for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << (vars->vVelocityCoeff[Arches::AP])[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << " After V Vel Underrelax : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "V Vel SU for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->vVelNonlinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

    break;
    case Arches::ZDIR:
    domLo = constvars->wVelocity.getFortLowIndex();
    domHi = constvars->wVelocity.getFortHighIndex();
    domLong = vars->wVelocityCoeff[Arches::AP].getFortLowIndex();
    domHing = vars->wVelocityCoeff[Arches::AP].getFortHighIndex();
    idxLo = patch->getSFCZFORTLowIndex();
    idxHi = patch->getSFCZFORTHighIndex();
    fort_underelax(idxLo, idxHi, constvars->wVelocity,
		   vars->wVelocityCoeff[Arches::AP], vars->wVelNonlinearSrc,
		   d_underrelax);

#ifdef ARCHES_VEL_DEBUG
    cerr << " After W Vel Underrelax : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "W Vel for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->wVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << " After W Vel Underrelax : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "W Vel AP for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << (vars->wVelocityCoeff[Arches::AP])[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << " After W Vel Underrelax : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "W Vel SU for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->wVelNonlinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

    break;
  default:
    throw InvalidValue("Invalid index in LinearSolver for velocity");
  }
}


//****************************************************************************
// Velocity Solve
//****************************************************************************
void 
HypreSolver::velocityLisolve(const ProcessorGroup*,
			     const Patch*,
			     int, double,
			     ArchesVariables*,
			     CellInformation*,
			     const ArchesLabel*)
{
  // Get the patch bounds and the variable bounds
#if 0
  IntVector domLo;
  IntVector domHi;
  IntVector idxLo;
  IntVector idxHi;
  // for explicit solver
  IntVector domLoDen = vars->old_density.getFortLowIndex();
  IntVector domHiDen = vars->old_density.getFortHighIndex();
  
  IntVector Size;

  Array1<double> e1;
  Array1<double> f1;
  Array1<double> e2;
  Array1<double> f2;
  Array1<double> e3;
  Array1<double> f3;

  sum_vartype resid;
  sum_vartype trunc;

  double nlResid;
  double trunc_conv;

  int velIter = 0;
  double velResid = 0.0;
  double theta = 0.5;

  switch (index) {
  case Arches::XDIR:
    domLo = vars->uVelocity.getFortLowIndex();
    domHi = vars->uVelocity.getFortHighIndex();
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();

    Size = domHi - domLo + IntVector(1,1,1);

    e1.resize(Size.x());
    f1.resize(Size.x());
    e2.resize(Size.y());
    f2.resize(Size.y());
    e3.resize(Size.z());
    f3.resize(Size.z());

    old_dw->get(resid, lab->d_uVelResidPSLabel);
    old_dw->get(trunc, lab->d_uVelTruncPSLabel);

    nlResid = resid;
    trunc_conv = trunc*1.0E-7;
#if implicit_defined
    do {
      //fortran call for lineGS solver
      FORT_LINEGS(domLo.get_pointer(), domHi.get_pointer(),
		  idxLo.get_pointer(), idxHi.get_pointer(),
		  vars->uVelocity.getPointer(),
		  vars->uVelocityCoeff[Arches::AE].getPointer(), 
		  vars->uVelocityCoeff[Arches::AW].getPointer(), 
		  vars->uVelocityCoeff[Arches::AN].getPointer(), 
		  vars->uVelocityCoeff[Arches::AS].getPointer(), 
		  vars->uVelocityCoeff[Arches::AT].getPointer(), 
		  vars->uVelocityCoeff[Arches::AB].getPointer(), 
		  vars->uVelocityCoeff[Arches::AP].getPointer(), 
		  vars->uVelNonlinearSrc.getPointer(),
		  e1.get_objs(), f1.get_objs(), e2.get_objs(), f2.get_objs(),
		  e3.get_objs(), f3.get_objs(), &theta);

      computeVelResidual(pc, patch, old_dw, new_dw, index, vars);
      velResid = vars->residUVel;
      ++velIter;
    } while((velIter < d_maxSweeps)&&((velResid > d_residual*nlResid)||
				      (velResid > trunc_conv)));
#else
    fort_explicit(idxLo, idxHi, vars->uVelocity, vars->old_uVelocity,
		  vars->uVelocityCoeff[Arches::AE], 
		  vars->uVelocityCoeff[Arches::AW], 
		  vars->uVelocityCoeff[Arches::AN], 
		  vars->uVelocityCoeff[Arches::AS], 
		  vars->uVelocityCoeff[Arches::AT], 
		  vars->uVelocityCoeff[Arches::AB], 
		  vars->uVelocityCoeff[Arches::AP], 
		  vars->uVelNonlinearSrc, vars->old_density,
		  cellinfo->sewu, cellinfo->sns, cellinfo->stb, delta_t);

#ifdef ARCHES_VEL_DEBUG
    cerr << " After U Vel Explicit solve : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "U Vel for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->uVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

    vars->residUVel = 1.0E-7;
    vars->truncUVel = 1.0;
#endif
    break;
  case Arches::YDIR:
    domLo = vars->vVelocity.getFortLowIndex();
    domHi = vars->vVelocity.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();

    Size = domHi - domLo + IntVector(1,1,1);

    e1.resize(Size.x());
    f1.resize(Size.x());
    e2.resize(Size.y());
    f2.resize(Size.y());
    e3.resize(Size.z());
    f3.resize(Size.z());

    old_dw->get(resid, lab->d_vVelResidPSLabel);
    old_dw->get(trunc, lab->d_vVelTruncPSLabel);

    nlResid = resid;
    trunc_conv = trunc*1.0E-7;
#if implicit_defined

    do {
      //fortran call for lineGS solver
      FORT_LINEGS(domLo.get_pointer(), domHi.get_pointer(),
		  idxLo.get_pointer(), idxHi.get_pointer(),
		  vars->vVelocity.getPointer(),
		  vars->vVelocityCoeff[Arches::AE].getPointer(), 
		  vars->vVelocityCoeff[Arches::AW].getPointer(), 
		  vars->vVelocityCoeff[Arches::AN].getPointer(), 
		  vars->vVelocityCoeff[Arches::AS].getPointer(), 
		  vars->vVelocityCoeff[Arches::AT].getPointer(), 
		  vars->vVelocityCoeff[Arches::AB].getPointer(), 
		  vars->vVelocityCoeff[Arches::AP].getPointer(), 
		  vars->vVelNonlinearSrc.getPointer(),
		  e1.get_objs(), f1.get_objs(), e2.get_objs(), f2.get_objs(),
		  e3.get_objs(), f3.get_objs(), &theta);

      computeVelResidual(pc, patch, old_dw, new_dw, index, vars);
      velResid = vars->residVVel;
      ++velIter;
    } while((velIter < d_maxSweeps)&&((velResid > d_residual*nlResid)||
				      (velResid > trunc_conv)));
    cerr << "After v Velocity solve " << velIter << " " << velResid << endl;
    cerr << "After v Velocity solve " << nlResid << " " << trunc_conv <<  endl;
#else
    fort_explicit(idxLo, idxHi, vars->vVelocity, vars->old_vVelocity,
		  vars->vVelocityCoeff[Arches::AE], 
		  vars->vVelocityCoeff[Arches::AW], 
		  vars->vVelocityCoeff[Arches::AN], 
		  vars->vVelocityCoeff[Arches::AS], 
		  vars->vVelocityCoeff[Arches::AT], 
		  vars->vVelocityCoeff[Arches::AB], 
		  vars->vVelocityCoeff[Arches::AP], 
		  vars->vVelNonlinearSrc, vars->old_density,
		  cellinfo->sew, cellinfo->snsv, cellinfo->stb, delta_t);

#ifdef ARCHES_VEL_DEBUG
    cerr << " After V Vel Explicit solve : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "V Vel for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->vVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

    vars->residVVel = 1.0E-7;
    vars->truncVVel = 1.0;
#endif
    break;
  case Arches::ZDIR:
    domLo = vars->wVelocity.getFortLowIndex();
    domHi = vars->wVelocity.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();

    Size = domHi - domLo + IntVector(1,1,1);

    e1.resize(Size.x());
    f1.resize(Size.x());
    e2.resize(Size.y());
    f2.resize(Size.y());
    e3.resize(Size.z());
    f3.resize(Size.z());

    old_dw->get(resid, lab->d_wVelResidPSLabel);
    old_dw->get(trunc, lab->d_wVelTruncPSLabel);

    nlResid = resid;
    trunc_conv = trunc*1.0E-7;
#if implicit_defined
    do {
      //fortran call for lineGS solver
      FORT_LINEGS(domLo.get_pointer(), domHi.get_pointer(),
		  idxLo.get_pointer(), idxHi.get_pointer(),
		  vars->wVelocity.getPointer(),
		  vars->wVelocityCoeff[Arches::AE].getPointer(), 
		  vars->wVelocityCoeff[Arches::AW].getPointer(), 
		  vars->wVelocityCoeff[Arches::AN].getPointer(), 
		  vars->wVelocityCoeff[Arches::AS].getPointer(), 
		  vars->wVelocityCoeff[Arches::AT].getPointer(), 
		  vars->wVelocityCoeff[Arches::AB].getPointer(), 
		  vars->wVelocityCoeff[Arches::AP].getPointer(), 
		  vars->wVelNonlinearSrc.getPointer(),
		  e1.get_objs(), f1.get_objs(), e2.get_objs(), f2.get_objs(),
		  e3.get_objs(), f3.get_objs(), &theta);

      computeVelResidual(pc, patch, old_dw, new_dw, index, vars);
      velResid = vars->residWVel;
      ++velIter;
    } while((velIter < d_maxSweeps)&&((velResid > d_residual*nlResid)||
				      (velResid > trunc_conv)));
    cerr << "After w Velocity solve " << velIter << " " << velResid << endl;
    cerr << "After w Velocity solve " << nlResid << " " << trunc_conv <<  endl;
#else
    fort_explicit(idxLo, idxHi, vars->wVelocity, vars->old_wVelocity,
		  vars->wVelocityCoeff[Arches::AE], 
		  vars->wVelocityCoeff[Arches::AW], 
		  vars->wVelocityCoeff[Arches::AN], 
		  vars->wVelocityCoeff[Arches::AS], 
		  vars->wVelocityCoeff[Arches::AT], 
		  vars->wVelocityCoeff[Arches::AB], 
		  vars->wVelocityCoeff[Arches::AP], 
		  vars->wVelNonlinearSrc, vars->old_density,
		  cellinfo->sew, cellinfo->sns, cellinfo->stbw, delta_t);

#ifdef ARCHES_VEL_DEBUG
    cerr << " After W Vel Explicit solve : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "W Vel for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->wVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

    vars->residWVel = 1.0E-7;
    vars->truncWVel = 1.0;
#endif
    break;
  default:
    throw InvalidValue("Invalid index in LinearSolver for velocity");
  }
#endif
}

//****************************************************************************
// Calculate Scalar residuals
//****************************************************************************
void 
HypreSolver::computeScalarResidual(const ProcessorGroup* ,
				  const Patch* patch,
				  DataWarehouseP& ,
				  DataWarehouseP& , 
				  int,
				  ArchesVariables* vars)
{
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  fort_rescal(idxLo, idxHi, vars->scalar, vars->residualScalar,
	      vars->scalarCoeff[Arches::AE], vars->scalarCoeff[Arches::AW], 
	      vars->scalarCoeff[Arches::AN], vars->scalarCoeff[Arches::AS], 
	      vars->scalarCoeff[Arches::AT], vars->scalarCoeff[Arches::AB], 
	      vars->scalarCoeff[Arches::AP], vars->scalarNonlinearSrc,
	      vars->residScalar, vars->truncScalar);
}


//****************************************************************************
// Actual calculation of order of magnitude term for Scalar equation
//****************************************************************************
void 
HypreSolver::computeScalarOrderOfMagnitude(const ProcessorGroup* ,
				const Patch* ,
				DataWarehouseP& ,
				DataWarehouseP& , ArchesVariables* )
{

  //&vars->truncScalar

}

//****************************************************************************
// Scalar Underrelaxation
//****************************************************************************
void 
HypreSolver::computeScalarUnderrelax(const ProcessorGroup* ,
				    const Patch* patch,
				    int,
				    ArchesVariables* vars,
				    ArchesConstVariables* constvars)
{
  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call
  fort_underelax(idxLo, idxHi, constvars->scalar,
		 vars->scalarCoeff[Arches::AP], vars->scalarNonlinearSrc,
		 d_underrelax);
}

//****************************************************************************
// Scalar Solve
//****************************************************************************
void 
HypreSolver::scalarLisolve(const ProcessorGroup*,
			  const Patch*,
			  int, double,
			  ArchesVariables*,
			  ArchesConstVariables*,
			  CellInformation*)
{
#if 0
  // Get the patch bounds and the variable bounds
  IntVector domLo = vars->scalar.getFortLowIndex();
  IntVector domHi = vars->scalar.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  // for explicit solver
  IntVector domLoDen = vars->old_density.getFortLowIndex();
  IntVector domHiDen = vars->old_density.getFortHighIndex();

  Array1<double> e1;
  Array1<double> f1;
  Array1<double> e2;
  Array1<double> f2;
  Array1<double> e3;
  Array1<double> f3;

  IntVector Size = domHi - domLo + IntVector(1,1,1);

  e1.resize(Size.x());
  f1.resize(Size.x());
  e2.resize(Size.y());
  f2.resize(Size.y());
  e3.resize(Size.z());
  f3.resize(Size.z());

  sum_vartype resid;
  sum_vartype trunc;

  old_dw->get(resid, lab->d_scalarResidLabel);
  old_dw->get(trunc, lab->d_scalarTruncLabel);

  double nlResid = resid;
  double trunc_conv = trunc*1.0E-7;
  double theta = 0.5;
  int scalarIter = 0;
  double scalarResid = 0.0;
#if implict_defined
  do {
    //fortran call for lineGS solver
    FORT_LINEGS(domLo.get_pointer(), domHi.get_pointer(),
		idxLo.get_pointer(), idxHi.get_pointer(),
		vars->scalar.getPointer(),
		vars->scalarCoeff[Arches::AE].getPointer(), 
		vars->scalarCoeff[Arches::AW].getPointer(), 
		vars->scalarCoeff[Arches::AN].getPointer(), 
		vars->scalarCoeff[Arches::AS].getPointer(), 
		vars->scalarCoeff[Arches::AT].getPointer(), 
		vars->scalarCoeff[Arches::AB].getPointer(), 
		vars->scalarCoeff[Arches::AP].getPointer(), 
		vars->scalarNonlinearSrc.getPointer(),
		e1.get_objs(), f1.get_objs(), e2.get_objs(), f2.get_objs(),
		e3.get_objs(), f3.get_objs(), &theta);
    computeScalarResidual(pc, patch, old_dw, new_dw, index, vars);
    scalarResid = vars->residScalar;
    ++scalarIter;
  } while((scalarIter < d_maxSweeps)&&((scalarResid > d_residual*nlResid)||
				      (scalarResid > trunc_conv)));
  cerr << "After scalar " << index <<" solve " << scalarIter << " " << scalarResid << endl;
  cerr << "After scalar " << index <<" solve " << nlResid << " " << trunc_conv <<  endl;
#endif
    fort_explicit(idxLo, idxHi, vars->scalar, vars->old_scalar,
		  vars->scalarCoeff[Arches::AE], 
		  vars->scalarCoeff[Arches::AW], 
		  vars->scalarCoeff[Arches::AN], 
		  vars->scalarCoeff[Arches::AS], 
		  vars->scalarCoeff[Arches::AT], 
		  vars->scalarCoeff[Arches::AB], 
		  vars->scalarCoeff[Arches::AP], 
		  vars->scalarNonlinearSrc, vars->old_density,
		  cellinfo->sew, cellinfo->sns, cellinfo->stb, delta_t);

#ifdef ARCHES_VEL_DEBUG
    cerr << " After Scalar Explicit solve : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "Scalar for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->scalar[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif


    vars->residScalar = 1.0E-7;
    vars->truncScalar = 1.0;
#endif
}


void 
HypreSolver::computeEnthalpyUnderrelax(const ProcessorGroup* ,
				       const Patch* patch,
				       ArchesVariables* vars,
				       ArchesConstVariables* constvars)
{
  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call
  fort_underelax(idxLo, idxHi, constvars->enthalpy,
		 vars->scalarCoeff[Arches::AP], vars->scalarNonlinearSrc,
		 d_underrelax);
}

//****************************************************************************
// Scalar Solve
//****************************************************************************
void 
HypreSolver::enthalpyLisolve(const ProcessorGroup*,
			     const Patch*,
			     double,
			     ArchesVariables*,
			     ArchesConstVariables*,
			     CellInformation*)
{
}














































