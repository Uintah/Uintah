/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//----- HypreSolver.cc ----------------------------------------------

#include <fstream> // work around compiler bug with RHEL 3

#include <CCA/Components/Arches/HypreSolver.h>
#include <Core/Thread/Time.h>
#include <CCA/Components/Arches/Arches.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include "_hypre_utilities.h"
#include "HYPRE_struct_ls.h"
#include "krylov.h"
#include "_hypre_struct_mv.h"

#undef CHKERRQ
#define CHKERRQ(x) if(x) throw PetscError(x, __FILE__, __FILE__, __LINE__);

using namespace std;
using namespace Uintah;


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
  ProblemSpecP db = params->findBlock("Parameters");

  if(!db) {
    ostringstream warn;
    warn << "INPUT FILE ERROR: ARCHES:PressureSolver: missing <Parameters> tag \n";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__); 
  } 
  
  string solver;
  string preconditioner = "-9";
  
  db->getWithDefault("solver", solver, "cg");
  
  if (solver == "smg"){
    d_solverType = "0";
  }
  
  if (solver == "pfmg"){
    d_solverType = "1";
  }

  db->getWithDefault("preconditioner", preconditioner, "pfmg");
 
  if (solver == "cg"){
    // preconditioners  
    if (preconditioner == "smg")
      d_solverType = "10";
    else if (preconditioner == "pfmg")
      d_solverType = "11";
    else if (preconditioner == "jacobi")
      d_solverType = "17";
    else if (preconditioner == "none"){
      d_solverType = "19";
    }
  }
  
  //__________________________________
  //bulletproofing
  
  string test = "bad";
  string test2 = "bad";
  db->get("ksptype",test);
  db->get("pctype", test2);
  
  if (test != "bad" || test2 != "bad"){
    ostringstream warn;
    warn << "INPUT FILE ERROR: ARCHES: using a depreciated linear solver option \n"
         << "change  <ksptype>   to    <solver> \n"
         << "change  <pctype>    to    <preconditioner> \n"<< endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);    
  }
  
  if(solver != "cg" && preconditioner != "-9" && d_myworld->myrank() == 0 ){
    cout << "-----------------------------------------------\n";
    cout << " WARNING: Linear solver options \n";
    cout << " The preconditioner ("<<preconditioner<< ") only works with the cg solver\n";
    cout << "-----------------------------------------------\n";
  }
  
  if(solver != "cg" && solver != "smg" && solver != "pfmg"){
    ostringstream warn;
    warn << "INPUT FILE ERROR: ARCHES: unknown linear solve type ("<<solver<<") \n"
         << "Valid Options:  cg, smg, or pfmg"<< endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  
  if(solver == "cg" && 
    preconditioner != "pfmg"   && preconditioner != "smg" &&
    preconditioner != "jacobi" && preconditioner != "none"){
     
    ostringstream warn;
    warn << "INPUT FILE ERROR: ARCHES: unknown preconditioner type ("<<preconditioner<<") \n"
         << "Valid Options:  smg, pfmg, jacobi, none"<< endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  
  db->getWithDefault("maxiterations", d_maxSweeps, 75);
  db->getWithDefault("tolerance",     d_stored_residual, 1.0e-8);
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
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();

  nx = idxHi.x() - idxLo.x() + 1;
  ny = idxHi.y() - idxLo.y() + 1;
  nz = idxHi.z() - idxLo.z() + 1;

#if 0  
  //__________________________________
  //  bulletproofing      -This sucks Todd
  if( fmodf(nx,2) !=0 || fmodf(ny,2) != 0 || fmodf(nz,2) != 0){
    ostringstream warn;
    warn << "INPUT FILE ERROR: ARCHES: hypre pressure solver. \n"
         << "This solver only works on a grid with an even number of cells in each directon on a patch\n"
         << "Patch: " << patch->getID() << " cells: (" << nx << ","<< ny <<","<< nz <<")" ;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);  
  }
#endif
     
  for (int i = 0; i < 6; i++){    
    d_A_num_ghost[i] = 0;
  }

  d_volume  = nx*ny*nz;    //number of nodes per processor
  bx = 1;
  by = 1;
  bz = 1;
  d_dim = 3;

  d_nblocks = bx*by*bz;           //number of blocks per processor, now is set to 1
     
  d_ilower = hypre_CTAlloc(int*, d_nblocks);
  d_iupper = hypre_CTAlloc(int*, d_nblocks);

  for (int i = 0; i < d_nblocks; i++){
    d_ilower[i] = hypre_CTAlloc(int, d_dim);
    d_iupper[i] = hypre_CTAlloc(int, d_dim);
  }
  
  for (int i = 0; i < d_dim; i++){
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

  for (int ib = 0; ib < d_nblocks; ib++){
    HYPRE_StructGridSetExtents(d_grid, d_ilower[ib], d_iupper[ib]);
  }
 
  const Level* level = patch->getLevel();
  IntVector periodic_vector = level->getPeriodicBoundaries();
  IntVector low, high;
  level->findCellIndexRange(low, high);
  IntVector range = high-low;
  int periodic[3];
  periodic[0] = periodic_vector.x() * range.x();
  periodic[1] = periodic_vector.y() * range.y();
  periodic[2] = periodic_vector.z() * range.z();
  HYPRE_StructGridSetPeriodic(d_grid, periodic);
  HYPRE_StructGridAssemble(d_grid);  

  /*-----------------------------------------------------------
   * Set up the stencil structure
   *-----------------------------------------------------------*/
  d_stencilSize = 4;
  d_stencilIndices = hypre_CTAlloc(int, d_stencilSize);
  int offsets[4][3] = {{0,0,0},
                      {-1,0,0},
                      {0,-1,0},
                      {0,0,-1}};
   
   
  HYPRE_StructStencilCreate(d_dim, d_stencilSize, &d_stencil);
   
  for (int s = 0; s < d_stencilSize; s++){
    HYPRE_StructStencilSetElement(d_stencil, s, offsets[s]);
  }
}

// ****************************************************************************
// Fill linear parallel matrix
// ****************************************************************************
void 
HypreSolver::setMatrix(const ProcessorGroup* pc,
                       const Patch* patch,
                       CCVariable<Stencil7>& coeff)
{ 
  gridSetup(pc, patch);
  /*-----------------------------------------------------------
   * Set up the matrix structure
   *-----------------------------------------------------------*/

  HYPRE_StructMatrixCreate(MPI_COMM_WORLD, d_grid, d_stencil, &d_A);
  HYPRE_StructMatrixSetSymmetric(d_A, 1);
  HYPRE_StructMatrixSetNumGhost(d_A, d_A_num_ghost);
  HYPRE_StructMatrixInitialize(d_A); 
 
  double *A = hypre_CTAlloc(double, (d_stencilSize)*d_volume);
  
  /* Set the coefficients for the grid */
  int i = 0;
  int s;
  for (s = 0; s < (d_stencilSize); s++){
    d_stencilIndices[s] = s;
  }
  
  for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
    IntVector c = *iter;
    A[i]   =  coeff[c].p; //[0, 0, 0]
    A[i+1] = coeff[c].w; //[-1,0, 0]
    A[i+2] = coeff[c].s; //[0,-1, 0]
    A[i+3] = coeff[c].b; //[0 ,0,-1]

    i = i + d_stencilSize;
   
  }
  
  for (int ib = 0; ib < d_nblocks; ib++){
    HYPRE_StructMatrixSetBoxValues(d_A, d_ilower[ib], d_iupper[ib], d_stencilSize,
                                    d_stencilIndices, A);
  }

  HYPRE_StructMatrixAssemble(d_A);
  hypre_TFree(A);

}
// ****************************************************************************
// Fill linear parallel matrix
// ****************************************************************************
void 
HypreSolver::setRHS_X(const ProcessorGroup* pc,
                      const Patch* patch,
                      CCVariable<double>& guess,
                      constCCVariable<double>& rhs, 
                      bool construct_A )
{ 
   // gridSetup(pc, patch);
   /*-----------------------------------------------------------
    * Set up the linear system (b & x)
    *-----------------------------------------------------------*/
   if ( construct_A ) { 
     // These objects should only be constructed if A is constructed. 
     // Otherwise they are reused. 
     HYPRE_StructVectorCreate(MPI_COMM_WORLD, d_grid, &d_b);
     HYPRE_StructVectorInitialize(d_b);
     HYPRE_StructVectorCreate(MPI_COMM_WORLD, d_grid, &d_x);
     HYPRE_StructVectorInitialize(d_x);
   }
 
  /* Set the coefficients for the grid */
  int i = 0;

  // assemble right hand side and solution vector
  double * B = hypre_CTAlloc(double, d_volume);
  double * X = hypre_CTAlloc(double, d_volume);

  // B
  i = 0;
  for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
    IntVector c = *iter;
    B[i] = rhs[c];
    i++;
  }
    
  for (int ib = 0; ib < d_nblocks; ib++){
    HYPRE_StructVectorSetBoxValues(d_b, d_ilower[ib], d_iupper[ib], B);
  }

  // X
  i = 0;
  for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
    IntVector c = *iter;
    X[i] = guess[c];
    i++;
  }
    
  for (int ib = 0; ib < d_nblocks; ib++){
    HYPRE_StructVectorSetBoxValues(d_x, d_ilower[ib], d_iupper[ib], X);
  }

  HYPRE_StructVectorAssemble(d_b);  
  HYPRE_StructVectorAssemble(d_x);
  
  hypre_TFree(X);
  hypre_TFree(B);
}
//______________________________________________________________________
//
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
  d_residual = d_stored_residual / sum_b;
  double zero_residual = 0.0;

  n_pre = 1;
  n_post = 1;
  skip = 1;
  HYPRE_StructSolver solver, precond;

  int me = d_myworld->myrank();
  double start_time = Time::currentSeconds();
  
  if (d_solverType == "0") {
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
  }
  else if (d_solverType == "1") {
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
  }
  else {
    HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);
    HYPRE_PCGSetMaxIter( (HYPRE_Solver)solver, d_maxSweeps);
    HYPRE_PCGSetTol( (HYPRE_Solver)solver, d_residual);
    HYPRE_PCGSetTwoNorm( (HYPRE_Solver)solver, 1 );
    HYPRE_PCGSetRelChange( (HYPRE_Solver)solver, 0 );
    HYPRE_PCGSetLogging( (HYPRE_Solver)solver, 1 );
 
    
    if (d_solverType == "10") {
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
    }
  
    else if (d_solverType == "11") {  
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
    }
    else if (d_solverType == "17") {
      /* use two-step Jacobi as preconditioner */
      HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &precond);
      HYPRE_StructJacobiSetMaxIter(precond, 2);
      HYPRE_StructJacobiSetTol(precond, zero_residual);
      HYPRE_StructJacobiSetZeroGuess(precond);
      HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                           (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSolve,
                           (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSetup,
                           (HYPRE_Solver) precond);
    }
    
    HYPRE_PCGSetup
      ( (HYPRE_Solver)solver, (HYPRE_Matrix)d_A, (HYPRE_Vector)d_b, (HYPRE_Vector)d_x );

    HYPRE_PCGSolve
      ( (HYPRE_Solver)solver, (HYPRE_Matrix)d_A, (HYPRE_Vector)d_b, (HYPRE_Vector)d_x);
    
    HYPRE_PCGGetNumIterations( (HYPRE_Solver)solver, &num_iterations );
    HYPRE_PCGGetFinalRelativeResidualNorm( (HYPRE_Solver)solver, &final_res_norm );
    HYPRE_StructPCGDestroy(solver);

    if (d_solverType == "10") {
      HYPRE_StructSMGDestroy(precond);
    }
    else if (d_solverType == "11") {
      HYPRE_StructPFMGDestroy(precond);
    }
    else if (d_solverType == "17") {
      HYPRE_StructJacobiDestroy(precond);
    }
  }
  if(me == 0) {
    final_res_norm *= sum_b;          
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

//______________________________________________________________________
// copy solution vector back into the array
void
HypreSolver::copyPressSoln(const Patch* patch, ArchesVariables* vars)
{
  double* xvec;
  xvec = hypre_CTAlloc(double, d_volume);
 
  for (int ib = 0; ib < d_nblocks; ib++){
    HYPRE_StructVectorGetBoxValues(d_x, d_ilower[ib], d_iupper[ib], xvec);
  }
  
  int i = 0;
  for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
    IntVector c = *iter;
    vars->pressure[c] = xvec[i];
    i++;
  }
  hypre_TFree(xvec);
}
 
//______________________________________________________________________
//  
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
   
  for (i = 0; i < d_nblocks; i++){
    hypre_TFree(d_iupper[i]);
    hypre_TFree(d_ilower[i]);
  }
  hypre_TFree(d_ilower);
  hypre_TFree(d_iupper);
  hypre_TFree(d_stencilIndices);

  hypre_FinalizeMemoryDebug();
}

//______________________________________________________________________
//
void HypreSolver::print(const string& desc, const int timestep, const int step){

  char A_fname[100],B_fname[100], X_fname[100];
  
  sprintf(B_fname,"output/b.%s.%i.%i",desc.c_str(), timestep, step);
  sprintf(X_fname,"output/x.%s.%i.%i",desc.c_str(), timestep, step);
  sprintf(A_fname,"output/A.%s.%i.%i",desc.c_str(), timestep, step);
  
  HYPRE_StructMatrixPrint(A_fname, d_A, 0);
  
  HYPRE_StructVectorPrint(B_fname, d_b, 0);
  
  HYPRE_StructVectorPrint(X_fname, d_x, 0);  
}


void HypreSolver::finalizeSolver()
{
  
}
