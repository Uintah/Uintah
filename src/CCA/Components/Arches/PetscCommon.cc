/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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
#include <CCA/Components/Arches/PetscCommon.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/UintahPetscError.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Thread/Time.h>

using namespace std;
using namespace Uintah;
namespace Uintah {

//______________________________________________________________________
bool PetscLinearSolve(Mat& A, 
                      Vec& B, Vec& X, Vec& U,
                      const string pcType,
                      const string solverType,
                      const int overlap,
                      const int fill,
                      const double residual,
                      const int maxIter,
                      const ProcessorGroup* myworld)
{
  double solve_start = Time::currentSeconds();
  KSP solver;
  PC preConditioner;

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
  ierr = VecAssemblyBegin(B);
  if(ierr){
    throw UintahPetscError(ierr, "VecAssemblyBegin", __FILE__, __LINE__);
  }
  
  ierr = VecAssemblyEnd(B);
  if(ierr){
    throw UintahPetscError(ierr, "VecAssemblyEnd", __FILE__, __LINE__);
  }
  //__________________________________
  //            X
  ierr = VecAssemblyBegin(X);
  if(ierr){
    throw UintahPetscError(ierr, "VecAssemblyBegin", __FILE__, __LINE__);
  }
  
  ierr = VecAssemblyEnd(X);
  if(ierr){
    throw UintahPetscError(ierr, "VecAssemblyEnd", __FILE__, __LINE__);
  }
  
  //__________________________________
  //Debugging output Matrix & rhs.
#if 0
  char matrix_file[100],RHS_file[100], X_file[100];
  
  sprintf(RHS_file,   "RHS.proc_%d_iter",   myworld->myrank());
  sprintf(X_file,     "X_proc_%d_iter",     myworld->myrank());
  sprintf(matrix_file,"matrix_proc_%d",myworld->myrank());
  
  PetscViewer RHS_view;
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,RHS_file,&RHS_view);
  VecView(B,RHS_view);
  PetscViewerDestroy(RHS_view);
  
  PetscViewer X_view;
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,X_file,&X_view);
  VecView(X,X_view);
  PetscViewerDestroy(X_view);
  
  PetscViewer matrix_view;
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,matrix_file,&matrix_view);
  MatView(A, matrix_view);
  PetscViewerDestroy(matrix_view);
#endif
  
  //__________________________________
  // compute the initial error
  double neg_one = -1.0;
  double sum_b;
  double init_norm;
  ierr = VecSum(B, &sum_b);
  Vec u_tmp;
  
  ierr = VecDuplicate(X,&u_tmp);
  if(ierr){
    throw UintahPetscError(ierr, "VecDuplicate", __FILE__, __LINE__);
  }
  
  ierr = MatMult(A, X, u_tmp);
  if(ierr){
    throw UintahPetscError(ierr, "MatMult", __FILE__, __LINE__);
  }
  
#if (PETSC_VERSION_MAJOR == 2 && PETSC_VERSION_MINOR == 2)
  ierr = VecAXPY(&neg_one, B, u_tmp);
#else
  ierr = VecAXPY(u_tmp,neg_one,B);
#endif
  if(ierr){ 
    throw UintahPetscError(ierr, "VecAXPY", __FILE__, __LINE__);
  }
  
  ierr  = VecNorm(u_tmp,NORM_2,&init_norm);
  if(ierr){
    throw UintahPetscError(ierr, "VecNorm", __FILE__, __LINE__);
  }
  
#if ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR >= 2))
  ierr = VecDestroy(&u_tmp);
#else // v3.1
  ierr = VecDestroy(u_tmp);
#endif
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
  
  
  ierr = KSPGetPC(solver, &preConditioner);
  if(ierr)
    throw UintahPetscError(ierr, "KSPGetPC", __FILE__, __LINE__);
  
  if (pcType == "jacobi") {
    ierr = PCSetType(preConditioner, PCJACOBI);
    if(ierr)
      throw UintahPetscError(ierr, "PCSetType", __FILE__, __LINE__);
  }
  else if (pcType == "asm") {
    ierr = PCSetType(preConditioner, PCASM);
    if(ierr)
      throw UintahPetscError(ierr, "PCSetType", __FILE__, __LINE__);
    
    ierr = PCASMSetOverlap(preConditioner, overlap);
    if(ierr)
      throw UintahPetscError(ierr, "PCASMSetOverlap", __FILE__, __LINE__);
  }
  else if (pcType == "ilu") {
    ierr = PCSetType(preConditioner, PCILU);
    if(ierr)
      throw UintahPetscError(ierr, "PCSetType", __FILE__, __LINE__);
#if (PETSC_VERSION_MAJOR == 2)
  #if (PETSC_VERSION_MINOR == 3 && PETSC_VERSION_SUBMINOR >= 1) // 2.3.1
    ierr = PCFactorSetFill(preConditioner, fill);
    if(ierr)
      throw UintahPetscError(ierr, "PCFactorSetFill", __FILE__, __LINE__);
  #else
    ierr = PCILUSetFill(preConditioner, fill);
    if(ierr)
      throw UintahPetscError(ierr, "PCILUSetFill", __FILE__, __LINE__);
  #endif
#else //3.*.*
    ierr = PCFactorSetFill(preConditioner, fill);
    if(ierr)
      throw UintahPetscError(ierr, "PCFactorSetFill", __FILE__, __LINE__);
#endif
  }
  else {
    ierr = PCSetType(preConditioner, PCBJACOBI);
    if(ierr)
      throw UintahPetscError(ierr, "PCSetType", __FILE__, __LINE__);
  }
  if (solverType == "cg") {
    ierr = KSPSetType(solver, KSPCG);
    if(ierr)
      throw UintahPetscError(ierr, "KSPSetType", __FILE__, __LINE__);
  }
  else {
    ierr = KSPSetType(solver, KSPGMRES);
    if(ierr)
      throw UintahPetscError(ierr, "KSPSetType", __FILE__, __LINE__);
  }
  
  ierr = KSPSetTolerances(solver, 1.0e-50, residual, PETSC_DEFAULT, maxIter);
  if(ierr)
    throw UintahPetscError(ierr, "KSPSetTolerances", __FILE__, __LINE__);


  ierr = KSPSetInitialGuessNonzero(solver, PETSC_TRUE);
  if(ierr)
    throw UintahPetscError(ierr, "KSPSetInitialGuessNonzero", __FILE__, __LINE__);
    
  ierr = KSPSetFromOptions(solver);
  if(ierr)
    throw UintahPetscError(ierr, "KSPSetFromOptions", __FILE__, __LINE__);  
    

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
     Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  int its;
  ierr = KSPSolve(solver,B,X);
  if(ierr)
    throw UintahPetscError(ierr, "KSPSolve", __FILE__, __LINE__);

  ierr = KSPGetIterationNumber(solver,&its);
  if (ierr)
    throw UintahPetscError(ierr, "KSPGetIterationNumber", __FILE__, __LINE__);

  int me = myworld->myrank();

  ierr = VecNorm(X,NORM_1,&norm);
  if(ierr)
    throw UintahPetscError(ierr, "VecNorm", __FILE__,  __LINE__);

  // check the error
  ierr = MatMult(A, X, U);
  if(ierr)
    throw UintahPetscError(ierr, "MatMult", __FILE__, __LINE__);
#if (PETSC_VERSION_MAJOR==2 && PETSC_VERSION_MINOR == 2)
  ierr = VecAXPY(&neg_one, B, U);
#else
  ierr = VecAXPY(U,neg_one,B);
#endif
  if(ierr)
    throw UintahPetscError(ierr, "VecAXPY", __FILE__, __LINE__);
  ierr  = VecNorm(U,NORM_2,&norm);
  if(ierr)
    throw UintahPetscError(ierr, "VecNorm", __FILE__, __LINE__);
    
  if(me == 0) {
    cerr << "KSPSolve: Norm of error: " << norm << ", iterations: " << its << ", solver time: " << Time::currentSeconds()-solve_start << " seconds\n";
    cerr << "Init Norm: " << init_norm << " Error reduced by: " << norm/(init_norm+1.0e-20) << endl;
    cerr << "Sum of RHS vector: " << sum_b << endl;
  }
  
#if ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR >= 2))
  ierr =  KSPDestroy(&solver);
#else // v3.1
  ierr =  KSPDestroy(solver);
#endif
  if (ierr)
    throw UintahPetscError(ierr, "KSPDestroy", __FILE__, __LINE__);

  if ((norm/(init_norm+1.0e-20) < 1.0)&& (norm < 2.0)){
    return true;
  }else{
    return false;
  }
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
void PetscLocalToGlobalMapping(const PatchSet* perproc_patches,
                               const PatchSubset* mypatches,
                               vector<int>& numCells,
                               int& totalCells,
                               map<const Patch*, int>& petscGlobalStart,
                               map<const Patch*, Array3<int> >& petscLocalToGlobal,
                               const ProcessorGroup* myworld)
{  
  //loop through patches and compute the the petscGlobalStart for each patch
  for(int s=0;s<perproc_patches->size();s++){
    int mytotal = 0;
    const PatchSubset* patchsub = perproc_patches->getSubset(s);
    
    for(int ps=0;ps<patchsub->size();ps++){
      const Patch* patch = patchsub->get(ps);
      
      IntVector plowIndex  = patch->getFortranCellLowIndex();
      IntVector phighIndex = patch->getFortranCellHighIndex()+IntVector(1,1,1);

      long nc = (phighIndex[0]-plowIndex[0])*
                (phighIndex[1]-plowIndex[1])*
                (phighIndex[2]-plowIndex[2]);
                
      petscGlobalStart[patch]=totalCells;
      
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
      int petscglobalIndex = petscGlobalStart[neighbor];
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
    petscLocalToGlobal[patch].copyPointer(l2g);
  }
}



//______________________________________________________________________
// Shutdown PETSc
void finalizePetscSolver()
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


//______________________________________________________________________
//   Free work space.  All PETSc objects should be destroyed when they are no longer needed
void
destroyPetscObjects(Mat A, Vec X, Vec B, Vec U) 
{
  int ierr;

#if ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR >= 2))
  PetscClassId id;
  if (U) {
    PetscObjectGetClassId((PetscObject)U, &id);
  }

  PetscBool flg = (id ? PETSC_TRUE : PETSC_FALSE);

  if( flg ){
    ierr = VecDestroy(&U);
    if(ierr)
      throw UintahPetscError(ierr, "destroyPetscObjects::VecDestroy", __FILE__, __LINE__);
  }
  ierr = VecDestroy(&B);
  if(ierr)
    throw UintahPetscError(ierr, "destroyPetscObjects::VecDestroy", __FILE__, __LINE__);

  ierr = VecDestroy(&X);
  if(ierr)
    throw UintahPetscError(ierr, "destroyPetscObjects::VecDestroy", __FILE__, __LINE__);

  ierr = MatDestroy(&A);
  if(ierr)
    throw UintahPetscError(ierr, "destroyPetscObjects::MatDestroy", __FILE__, __LINE__);
#else // v3.1
  PetscTruth flg;
  VecValid(U, &flg);
  
  if( flg ){
    ierr = VecDestroy(U);
    if(ierr)
      throw UintahPetscError(ierr, "destroyPetscObjects::VecDestroy", __FILE__, __LINE__);
  }
  ierr = VecDestroy(B);
  if(ierr)
    throw UintahPetscError(ierr, "destroyPetscObjects::VecDestroy", __FILE__, __LINE__);
  
  ierr = VecDestroy(X);
  if(ierr)
    throw UintahPetscError(ierr, "destroyPetscObjects::VecDestroy", __FILE__, __LINE__);
  
  ierr = MatDestroy(A);
  if(ierr)
    throw UintahPetscError(ierr, "destroyPetscObjects::MatDestroy", __FILE__, __LINE__);
#endif
}

} // uintah namespace

