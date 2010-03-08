/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


//----- PetscSolver.cc ----------------------------------------------

#include <CCA/Components/Arches/PetscSolver.h>
#include <Core/Thread/Time.h>
#include <CCA/Components/Arches/Arches.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/UintahPetscError.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>


#undef CHKERRQ
#define CHKERRQ(x) if(x) throw UintahPetscError(x, __FILE__, __FILE__, __LINE__);

using namespace std;
using namespace Uintah;
using namespace SCIRun;


// ****************************************************************************
// Default constructor for PetscSolver
// ****************************************************************************
PetscSolver::PetscSolver(const ProcessorGroup* myworld)
   : d_myworld(myworld)
{
}

// ****************************************************************************
// Destructor
// ****************************************************************************
PetscSolver::~PetscSolver()
{
  finalizeSolver();
}

// ****************************************************************************
// Problem setup
// ****************************************************************************
void 
PetscSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("parameters");
  
  if(!db) {
    ostringstream warn;
    warn << "INPUT FILE ERROR: ARCHES:PressureSolver: missing <parameters> tag \n";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__); 
  }
  
  //__________________________________
  //bulletproofing
  string test = "none";
  string test2 = "none";
  db->get("ksptype",test);
  db->get("pctype", test2);
  
  if (test != "none" || test2 != "none"){
    ostringstream warn;
    warn << "INPUT FILE ERROR: ARCHES: using a depreciated linear solver option \n"
         << "change  <ksptype>   to    <solver> \n"
         << "change  <pctype>    to    <preconditioner> \n"<< endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);    
  }
  
  db->require("preconditioner",d_pcType);
  db->require("solver",        d_solverType);
  db->require("max_iter",      d_maxSweeps);
  db->getWithDefault("res_tol", d_residual, 1.0e-7);
  
  if (d_pcType == "asm"){
    db->require("overlap",d_overlap);
  }
  
  if (d_pcType == "ilu"){
    db->require("fill",d_fill);
  }
   
  //__________________________________
  //bulletproofing
  if(d_solverType != "cg" ){
    ostringstream warn;
    warn << "INPUT FILE ERROR: ARCHES: unknown linear solve type ("<<d_solverType<<") \n"
         << "Valid PETSC Option:  cg"<< endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  
  if(d_pcType != "asm"   && d_pcType != "ilu" && d_pcType != "jacobi"){
    ostringstream warn;
    warn << "INPUT FILE ERROR: ARCHES: unknown PETSC preconditioner type ("<<d_pcType<<") \n"
         << "Valid Options:  smg, pfmg, jacobi, none"<< endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  
  int argc = 4;
  char** argv;
  argv = scinew char*[argc];
  argv[0] = const_cast<char*>("PetscSolver::problemSetup");
  //argv[1] = "-on_error_attach_debugger";
  argv[1] = const_cast<char*>("-no_signal_handler");
  argv[2] = const_cast<char*>("-log_exclude_actions");
  argv[3] = const_cast<char*>("-log_exclude_objects");
  int ierr = PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  if(ierr){
    throw UintahPetscError(ierr, "PetscInitialize", __FILE__, __LINE__);
  }
//  ierr = PetscOptionsSetValue("-log_exclude_actions", "");
//  if(ierr)
//    throw UintahPetscError(ierr, "PetscExcludeActions");
//  ierr = PetscOptionsSetValue("-log_exclude_objects", "");
//  if(ierr)
//    throw UintahPetscError(ierr, "PetscExcludeObjects");
  delete argv;
}

/*______________________________________________________________________
 Creates a mapping from cell coordinates, IntVector(x,y,z), 
 to global matrix coordinates.  The matrix is laid out as 
 follows:

 Proc 0 patches
    patch 0 cells
    patch 1 cells
    ...
 Proc 1 patches
    patch 0 cells
    patch 1 cells
    ...
 ...

 Thus the entrance at cell xyz provides the global index into the
 matrix for that cells entry.  And each processor owns a 
 consecutive block of those rows.  In order to translate a 
 cell position to the processors local position (needed when using 
 a local array) the global index
 of the processors first patch must be subtracted from the global
 index of the cell in question.  This will provide a zero-based index 
 into each processors data.
//______________________________________________________________________*/
void 
PetscSolver::matrixCreate(const PatchSet* perproc_patches,
                          const PatchSubset* mypatches)
{
  // for global index get a petsc index that
  // make it a data memeber
  int numProcessors = d_myworld->size();
  ASSERTEQ(numProcessors, perproc_patches->size());

  // number of patches for each processor
  vector<int> numCells(numProcessors, 0);
  vector<int> startIndex(numProcessors);
  int totalCells = 0;
  
  //loop through patches and compute the the d_petscGlobalStart for each patch
  for(int s=0;s<perproc_patches->size();s++){
    startIndex[s]=totalCells;
    int mytotal = 0;
    const PatchSubset* patchsub = perproc_patches->getSubset(s);
    
    for(int ps=0;ps<patchsub->size();ps++){
      const Patch* patch = patchsub->get(ps);
      
      IntVector plowIndex  = patch->getFortranCellLowIndex();
      IntVector phighIndex = patch->getFortranCellHighIndex()+IntVector(1,1,1);

      long nc = (phighIndex[0]-plowIndex[0])*
                (phighIndex[1]-plowIndex[1])*
                (phighIndex[2]-plowIndex[2]);
      d_petscGlobalStart[patch]=totalCells;
      totalCells += nc;
      mytotal    += nc;
    }
    numCells[s] = mytotal;
  }
  

  //__________________________________
  //for each patch
  for(int p=0;p<mypatches->size();p++){
    const Patch* patch=mypatches->get(p);
    
    int ngc = 1;
    IntVector lowIndex   = patch->getExtraCellLowIndex(ngc);
    IntVector highIndex  = patch->getExtraCellHighIndex(ngc);
    
    Array3<int> l2g(lowIndex, highIndex);
    l2g.initialize(-1234);
    long totalCells = 0;
    const Level* level = patch->getLevel();
    Patch::selectType neighbors;
    
    //get neighboring patches (which includes this patch)
    level->selectPatches(lowIndex, highIndex, neighbors);
    //for each neighboring patch and myself
    
    for(int i=0;i<neighbors.size();i++){
      const Patch* neighbor = neighbors[i];

      //intersect my patch with my neighbor patch
      IntVector plow  = neighbor->getFortranCellLowIndex();
      IntVector phigh = neighbor->getFortranCellHighIndex()+IntVector(1,1,1);
      IntVector low   = Max(lowIndex, plow);
      IntVector high  = Min(highIndex, phigh);

      if( ( high.x() < low.x() ) || ( high.y() < low.y() ) || ( high.z() < low.z() ) ){
        throw InternalError("Patch doesn't overlap?", __FILE__, __LINE__);
      }
      //set petscglobilIndex equal to the starting global index for the neighbor patch
      int petscglobalIndex = d_petscGlobalStart[neighbor];
      IntVector dcells = phigh-plow;
      IntVector start = low-plow;
      
      //offset the global index by to the intersecting range
      petscglobalIndex += start.z()*dcells.x()*dcells.y()
                        + start.y()*dcells.x() + start.x();
                         
      //for each node in intersecting range
      for (int colZ = low.z(); colZ < high.z(); colZ ++) {
        int idx_slab = petscglobalIndex;
        petscglobalIndex += dcells.x()*dcells.y();

        for (int colY = low.y(); colY < high.y(); colY ++) {
          int idx = idx_slab;
          idx_slab += dcells.x();
          for (int colX = low.x(); colX < high.x(); colX ++) {
            //set the local to global mapping 
            l2g[IntVector(colX, colY, colZ)] = idx++;
          }
        }
      }
      IntVector d = high-low;
      totalCells+=d.x()*d.y()*d.z();
    }
    d_petscLocalToGlobal[patch].copyPointer(l2g);
  }
  
  //__________________________________
  //  Now create the matrix
  int me = d_myworld->myrank();
  int numlrows      = numCells[me];
  int numlcolumns   = numlrows;
  int globalrows    = (int)totalCells;
  int globalcolumns = (int)totalCells;
  int d_nz = 7;
  int o_nz = 6;
  int ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD, numlrows, numlcolumns, globalrows,
                             globalcolumns, d_nz, PETSC_NULL, o_nz, PETSC_NULL, &A);
      
  if(ierr){
    throw UintahPetscError(ierr, "MatCreateMPIAIJ", __FILE__, __LINE__);
  }
  
  /* 
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
     */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,numlrows, globalrows,&d_x);
  if(ierr)
    throw UintahPetscError(ierr, "VecCreateMPI", __FILE__, __LINE__);
  
  ierr = VecSetFromOptions(d_x);
  if(ierr)
    throw UintahPetscError(ierr, "VecSetFromOptions", __FILE__, __LINE__);
  
  ierr = VecDuplicate(d_x,&d_b);
  if(ierr)
    throw UintahPetscError(ierr, "VecDuplicate(d_b)", __FILE__, __LINE__);
  
  ierr = VecDuplicate(d_x,&d_u);
  if(ierr)
    throw UintahPetscError(ierr, "VecDuplicate(d_u)", __FILE__, __LINE__);
}

// ****************************************************************************
// Fill linear parallel matrix
// ****************************************************************************
void 
PetscSolver::setPressMatrix(const ProcessorGroup* ,
                            const Patch* patch,
                            ArchesVariables* vars,
                            ArchesConstVariables* constvars,
                            const ArchesLabel*)
{
  double solve_start = Time::currentSeconds();

  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
     Compute the matrix and right-hand-side vector that define
     the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* 
     Create parallel matrix, specifying only its global dimensions.
     When using MatCreate(), the matrix format can be specified at
     runtime. Also, the parallel partitioning of the matrix is
     determined by PETSc at runtime.

     Performance tuning note:  For problems of substantial size,
     preallocation of matrix memory is crucial for attaining good 
     performance.  Since preallocation is not possible via the generic
     matrix creation routine MatCreate(), we recommend for practical 
     problems instead to use the creation routine for a particular matrix
     format, e.g.,
     MatCreateMPIAIJ() - parallel AIJ (compressed sparse row)
     MatCreateMPIBAIJ() - parallel block AIJ
     See the matrix chapter of the users manual for details.
     */
  int ierr;
  int col[7];
  double value[7];
  // fill matrix for internal patches
  // make sure that sizeof(d_petscIndex) is the last patch, i.e., appears last in the
  // petsc matrix
  IntVector lowIndex = patch->getExtraCellLowIndex(Arches::ONEGHOSTCELL);
  IntVector highIndex = patch->getExtraCellHighIndex(Arches::ONEGHOSTCELL);

  Array3<int> l2g(lowIndex, highIndex);
  l2g.copy(d_petscLocalToGlobal[patch]);

#if 0
  //if ((patchNumber != 0)&&(patchNumber != sizeof(d_petscIndex)-1)) 
#endif
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        col[0] = l2g[IntVector(colX,colY,colZ-1)];  //ab
        col[1] = l2g[IntVector(colX, colY-1, colZ)]; // as
        col[2] = l2g[IntVector(colX-1, colY, colZ)]; // aw
        col[3] = l2g[IntVector(colX, colY, colZ)]; //ap
        col[4] = l2g[IntVector(colX+1, colY, colZ)]; // ae
        col[5] = l2g[IntVector(colX, colY+1, colZ)]; // an
        col[6] = l2g[IntVector(colX, colY, colZ+1)]; // at
        value[0] = -constvars->pressCoeff[IntVector(colX,colY,colZ)].b;
        value[1] = -constvars->pressCoeff[IntVector(colX,colY,colZ)].s;
        value[2] = -constvars->pressCoeff[IntVector(colX,colY,colZ)].w;
        value[3] = constvars->pressCoeff[IntVector(colX,colY,colZ)].p;
        value[4] = -constvars->pressCoeff[IntVector(colX,colY,colZ)].e;
        value[5] = -constvars->pressCoeff[IntVector(colX,colY,colZ)].n;
        value[6] = -constvars->pressCoeff[IntVector(colX,colY,colZ)].t;
        int row = col[3];
        ierr = MatSetValues(A,1,&row,7,col,value,INSERT_VALUES);
        if(ierr)
          throw UintahPetscError(ierr, "MatSetValues", __FILE__, __LINE__);
      }
    }
  }

  // assemble right hand side and solution vector
  double vecvalueb, vecvaluex;
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        vecvalueb = constvars->pressNonlinearSrc[IntVector(colX,colY,colZ)];
        vecvaluex = vars->pressure[IntVector(colX, colY, colZ)];
        int row = l2g[IntVector(colX, colY, colZ)];   
        ierr = VecSetValue(d_b, row, vecvalueb, INSERT_VALUES);
        if(ierr)
          throw UintahPetscError(ierr, "VecSetValue", __FILE__, __LINE__);

        ierr = VecSetValue(d_x, row, vecvaluex, INSERT_VALUES);
        if(ierr)
          throw UintahPetscError(ierr, "VecSetValue", __FILE__, __LINE__);
      }
    }
  }
  int me = d_myworld->myrank();
  if(me == 0) {
    cerr << "Time in PETSC Assemble: " << Time::currentSeconds()-solve_start << " seconds\n";
  }
}

//______________________________________________________________________
//
bool
PetscSolver::pressLinearSolve()
{
  double solve_start = Time::currentSeconds();
  KSP solver;
  PC peqnpc; // pressure eqn pc

  int ierr;
  //__________________________________
  //             A
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  if(ierr){
    throw UintahPetscError(ierr, "MatAssemblyBegin", __FILE__, __LINE__);
  }
  
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
  if(ierr){
    throw UintahPetscError(ierr, "MatAssemblyEnd", __FILE__, __LINE__);
  }
  //__________________________________
  //            B
  ierr = VecAssemblyBegin(d_b);
  if(ierr){
    throw UintahPetscError(ierr, "VecAssemblyBegin", __FILE__, __LINE__);
  }
  
  ierr = VecAssemblyEnd(d_b);
  if(ierr){
    throw UintahPetscError(ierr, "VecAssemblyEnd", __FILE__, __LINE__);
  }
  //__________________________________
  //            X
  ierr = VecAssemblyBegin(d_x);
  if(ierr){
    throw UintahPetscError(ierr, "VecAssemblyBegin", __FILE__, __LINE__);
  }
  
  ierr = VecAssemblyEnd(d_x);
  if(ierr){
    throw UintahPetscError(ierr, "VecAssemblyEnd", __FILE__, __LINE__);
  }
  
  // compute the initial error
  double neg_one = -1.0;
  double sum_b;
  ierr = VecSum(d_b, &sum_b);
  Vec u_tmp;
  
  ierr = VecDuplicate(d_x,&u_tmp);
  if(ierr){
    throw UintahPetscError(ierr, "VecDuplicate", __FILE__, __LINE__);
  }
  
  ierr = MatMult(A, d_x, u_tmp);
  if(ierr){
    throw UintahPetscError(ierr, "MatMult", __FILE__, __LINE__);
  }
  
#if (PETSC_VERSION_MAJOR == 2 && PETSC_VERSION_MINOR == 2)
  ierr = VecAXPY(&neg_one, d_b, u_tmp);
#else
  ierr = VecAXPY(u_tmp,neg_one,d_b);
#endif
  if(ierr){ 
    throw UintahPetscError(ierr, "VecAXPY", __FILE__, __LINE__);
  }
  
  ierr  = VecNorm(u_tmp,NORM_2,&init_norm);
  if(ierr){
    throw UintahPetscError(ierr, "VecNorm", __FILE__, __LINE__);
  }
  
  ierr = VecDestroy(u_tmp);
  if(ierr){
    throw UintahPetscError(ierr, "VecDestroy", __FILE__, __LINE__);
  }
  /* debugging - steve */
  double norm;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
     Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = KSPCreate(PETSC_COMM_WORLD,&solver);
  if(ierr)
    throw UintahPetscError(ierr, "KSPCreate", __FILE__, __LINE__);
    
    
  ierr = KSPSetOperators(solver,A,A,DIFFERENT_NONZERO_PATTERN);
  if(ierr)
    throw UintahPetscError(ierr, "KSPSetOperators", __FILE__, __LINE__);
  
  
  ierr = KSPGetPC(solver, &peqnpc);
  if(ierr)
    throw UintahPetscError(ierr, "KSPGetPC", __FILE__, __LINE__);
  
  if (d_pcType == "jacobi") {
    ierr = PCSetType(peqnpc, PCJACOBI);
    if(ierr)
      throw UintahPetscError(ierr, "PCSetType", __FILE__, __LINE__);
  }
  else if (d_pcType == "asm") {
    ierr = PCSetType(peqnpc, PCASM);
    if(ierr)
      throw UintahPetscError(ierr, "PCSetType", __FILE__, __LINE__);
    
    ierr = PCASMSetOverlap(peqnpc, d_overlap);
    if(ierr)
      throw UintahPetscError(ierr, "PCASMSetOverlap", __FILE__, __LINE__);
  }
  else if (d_pcType == "ilu") {
    ierr = PCSetType(peqnpc, PCILU);
    if(ierr)
      throw UintahPetscError(ierr, "PCSetType", __FILE__, __LINE__);
#if (PETSC_VERSION_MAJOR == 2)
  #if (PETSC_VERSION_MINOR == 3 && PETSC_VERSION_SUBMINOR >= 1) // 2.3.1
    ierr = PCFactorSetFill(peqnpc, d_fill);
    if(ierr)
      throw UintahPetscError(ierr, "PCFactorSetFill", __FILE__, __LINE__);
  #else
    ierr = PCILUSetFill(peqnpc, d_fill);
    if(ierr)
      throw UintahPetscError(ierr, "PCILUSetFill", __FILE__, __LINE__);
  #endif
#else //3.*.*
    ierr = PCFactorSetFill(peqnpc, d_fill);
    if(ierr)
      throw UintahPetscError(ierr, "PCFactorSetFill", __FILE__, __LINE__);
#endif
  }
  else {
    ierr = PCSetType(peqnpc, PCBJACOBI);
    if(ierr)
      throw UintahPetscError(ierr, "PCSetType", __FILE__, __LINE__);
  }
  if (d_solverType == "cg") {
    ierr = KSPSetType(solver, KSPCG);
    if(ierr)
      throw UintahPetscError(ierr, "KSPSetType", __FILE__, __LINE__);
  }
  else {
    ierr = KSPSetType(solver, KSPGMRES);
    if(ierr)
      throw UintahPetscError(ierr, "KSPSetType", __FILE__, __LINE__);
  }
  ierr = KSPSetTolerances(solver, 1.0e-50, d_residual, PETSC_DEFAULT, PETSC_DEFAULT);
  if(ierr)
    throw UintahPetscError(ierr, "KSPSetTolerances", __FILE__, __LINE__);

  // set null space for preconditioner
  // change for a newer version
#ifdef NULL_MATRIX
  PCNullSpace nullsp;
  ierr = PCNullSpaceCreate(PETSC_COMM_WORLD, 1, 0, PETSC_NULL, &nullsp); 
  ierr = PCNullSpaceAttach(peqnpc, nullsp);
  ierr = PCNullSpaceDestroy(nullsp);
#endif
  ierr = KSPSetInitialGuessNonzero(solver, PETSC_TRUE);
  // ierr = KSPSetInitialGuessNonzero(solver);
  if(ierr)
    throw UintahPetscError(ierr, "KSPSetInitialGuessNonzero", __FILE__, __LINE__);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
     Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  int its;
  ierr = KSPSolve(solver,d_b,d_x);
  if(ierr)
    throw UintahPetscError(ierr, "KSPSolve", __FILE__, __LINE__);

  ierr = KSPGetIterationNumber(solver,&its);
  if (ierr)
    throw UintahPetscError(ierr, "KSPGetIterationNumber", __FILE__, __LINE__);

  int me = d_myworld->myrank();

  ierr = VecNorm(d_x,NORM_1,&norm);
  if(ierr)
    throw UintahPetscError(ierr, "VecNorm", __FILE__,  __LINE__);

  // check the error
  ierr = MatMult(A, d_x, d_u);
  if(ierr)
    throw UintahPetscError(ierr, "MatMult", __FILE__, __LINE__);
#if (PETSC_VERSION_MAJOR==2 && PETSC_VERSION_MINOR == 2)
  ierr = VecAXPY(&neg_one, d_b, d_u);
#else
  ierr = VecAXPY(d_u,neg_one,d_b);
#endif
  if(ierr)
    throw UintahPetscError(ierr, "VecAXPY", __FILE__, __LINE__);
  ierr  = VecNorm(d_u,NORM_2,&norm);
  if(ierr)
    throw UintahPetscError(ierr, "VecNorm", __FILE__, __LINE__);
    
  if(me == 0) {
    cerr << "KSPSolve: Norm of error: " << norm << ", iterations: " << its << ", solver time: " << Time::currentSeconds()-solve_start << " seconds\n";
    cerr << "Init Norm: " << init_norm << " Error reduced by: " << norm/(init_norm+1.0e-20) << endl;
    cerr << "Sum of RHS vector: " << sum_b << endl;
  }
  
  ierr =  KSPDestroy(solver);
  if (ierr)
    throw UintahPetscError(ierr, "KSPDestroy", __FILE__, __LINE__);

  if ((norm/(init_norm+1.0e-20) < 1.0)&& (norm < 2.0))
    return true;
  else
    return false;
}

//______________________________________________________________________
//
void
PetscSolver::copyPressSoln(const Patch* patch, ArchesVariables* vars)
{
  // copy solution vector back into the array
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();
  double* xvec;
  int ierr;
  PetscInt begin, end;
  //get the ownership range so we know where the local indicing on this processor begins
  VecGetOwnershipRange(d_x, &begin, &end);
  ierr = VecGetArray(d_x, &xvec);
  
  if(ierr)
    throw UintahPetscError(ierr, "VecGetArray", __FILE__, __LINE__);
  Array3<int> l2g = d_petscLocalToGlobal[patch];
  
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        //subtract the begining index from the global index to get to the local array index
        int row = l2g[IntVector(colX, colY, colZ)]-begin;
        ASSERTRANGE(l2g[IntVector(colX, colY, colZ)] ,begin,end);
        vars->pressure[IntVector(colX, colY, colZ)] = xvec[row];
      }
    }
  }
  
#if 0
  cerr << "Print computed pressure" << endl;
  vars->pressure.print(cerr);
#endif
  ierr = VecRestoreArray(d_x, &xvec);
  if(ierr)
    throw UintahPetscError(ierr, "VecRestoreArray", __FILE__, __LINE__);
}

//______________________________________________________________________
//   Free work space.  All PETSc objects should be destroyed when they are no longer needed
void
PetscSolver::destroyMatrix() 
{
  int ierr;
  ierr = VecDestroy(d_u);
  if(ierr)
    throw UintahPetscError(ierr, "VecDestroy", __FILE__, __LINE__);
  ierr = VecDestroy(d_b);
  if(ierr)
    throw UintahPetscError(ierr, "VecDestroy", __FILE__, __LINE__);
  ierr = VecDestroy(d_x);
  if(ierr)
    throw UintahPetscError(ierr, "VecDestroy", __FILE__, __LINE__);
  ierr = MatDestroy(A);
  if(ierr)
    throw UintahPetscError(ierr, "MatDestroy", __FILE__, __LINE__);
}

//______________________________________________________________________
//
// Shutdown PETSc
void PetscSolver::finalizeSolver()
{
// The following is to enable PETSc memory logging
//  int ierrd = PetscTrDump(NULL);
//  if(ierrd)
//    throw UintahPetscError(ierrd, "PetscTrDump");
  int ierr = PetscFinalize();
  if(ierr){
    throw UintahPetscError(ierr, "PetscFinalize", __FILE__, __LINE__);
  }
}

