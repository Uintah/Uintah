/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//#define PETSC_USE_LOG

#include <sci_defs/mpi_defs.h>
#include <sci_defs/petsc_defs.h>

#include <TauProfilerForSCIRun.h>
#include <CCA/Components/MPM/PetscSolver.h>
#include <Core/Exceptions/UintahPetscError.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>

#include <vector>
#include <iostream>

// If I'm not mistaken, this #define replaces the CHKERRQ() from PETSc itself...
#undef CHKERRQ
#define CHKERRQ(x) if(x) throw PetscError(x, __FILE__, __FILE__, __LINE__);

using namespace Uintah;
using namespace std;

#undef LOG
//#define LOG
#undef DEBUG_PETSC

//#define USE_SPOOLES
#undef  USE_SPOOLES

//#define USE_SUPERLU
#undef  USE_SUPERLU

//#define OUTPUT_A_B  // output A matrix and B vector to files.
#undef OUTPUT_A_B

MPMPetscSolver::MPMPetscSolver()
{
  d_A = 0;
  d_B = 0;
  d_diagonal = 0;
  d_x = 0;
  d_t = 0;
  d_flux = 0;
  d_iteration=0;
}

MPMPetscSolver::~MPMPetscSolver()
{
}


void MPMPetscSolver::initialize()
{
  // store in a vector so we can customize the settings easily
  vector<char*> args;

  // Null argument ("") is needed as it normaly stores the command
  args.push_back(const_cast<char*>(""));
  
#ifdef DEBUG_PETSC
  args.push_back("ImpMPM::problemSetup");
  args.push_back("-on_error_attach_debugger");
#endif
#ifdef LOG
  args.push_back("-log_summary");
  args.push_back("-log_info");
  args.push_back("-info");
#if 0
  args.push_back("-log_exclude_actions");
  args.push_back("-log_exclude_objects");
  args.push_back("-log_info");
  args.push_back("-trmalloc");
  args.push_back("-trdump");
  args.push_back("-trmalloc_log");
  args.push_back("-log_summary");
#endif
#endif

  int argc = args.size();
  char** argv = 0;

  // copy the vector to argv, as I think petsc wants to store it around -- bjw
  if (argc > 0) {
    argv = scinew char*[argc];
    for (int i = 0; i < argc; i++) {
      argv[i] = args[i];
    }
  }
  PetscInitialize(&argc,&argv, PETSC_NULL, PETSC_NULL);
#ifdef USE_SPOOLES
  PetscOptionsSetValue("-mat_spooles_ordering","BestOfNDandMS");
  PetscOptionsSetValue("-mat_spooles_symmetryflag","0");
#endif
#if 0
  PetscOptionsSetValue("-options_table", PETSC_NULL);
  PetscOptionsSetValue("-mat_superlu_dist_iterrefine", "TRUE");
  PetscOptionsSetValue("-mat_superlu_dist_statprint", PETSC_NULL);
  PetscOptionsSetValue("-log_summary", PETSC_NULL);
  PetscOptionsSetValue("-log_info", PETSC_NULL);
  PetscOptionsSetValue("-trmalloc", PETSC_NULL);
  PetscOptionsSetValue("-trmalloc_log", PETSC_NULL);
  PetscOptionsSetValue("-trdump", PETSC_NULL);
#endif
  PetscPopSignalHandler();

  if(argc>0)
    delete [] argv;
}
/**************************************************************
 * Creates a mapping from nodal coordinates, IntVector(x,y,z), 
 * to global matrix coordinates.  The matrix is laid out as 
 * follows:
 *
 * Proc 0 patches
 *    patch 0 nodes
 *    patch 1 nodes
 *    ...
 * Proc 1 patches
 *    patch 0 nodes
 *    patch 1 nodes
 *    ...
 * ...
 *
 * Thus the entrance at node xyz provides the global index into the
 * matrix for that nodes entry.  And each processor owns a 
 * consecutive block of those rows.  In order to translate a 
 * nodal position to the processors local position (needed when using 
 * a local array) the global index
 * of the processors first patch must be subtracted from the global
 * index of the node in question.  This will provide a zero-based index 
 * into each processors data.
 *************************************************************/
void 
MPMPetscSolver::createLocalToGlobalMapping(const ProcessorGroup* d_myworld,
                                           const PatchSet* perproc_patches,
                                           const PatchSubset* patches,
                                           const int DOFsPerNode,
                                           const int n8or27)
{
  TAU_PROFILE("MPMPetscSolver::createLocalToGlobalMapping", " ", TAU_USER);
  int numProcessors = d_myworld->size();
  d_numNodes.resize(numProcessors, 0);
  d_startIndex.resize(numProcessors);
  d_totalNodes = 0;
  //compute the total number of nodes and the global offset for each patch
  for (int p = 0; p < perproc_patches->size(); p++) {
    d_startIndex[p] = d_totalNodes;
    int mytotal = 0;
    const PatchSubset* patchsub = perproc_patches->getSubset(p);
    for (int ps = 0; ps<patchsub->size(); ps++) {
      const Patch* patch = patchsub->get(ps);
      IntVector plowIndex(0,0,0),phighIndex(0,0,0);
      if(n8or27==8){
        plowIndex = patch->getNodeLowIndex();
        phighIndex = patch->getNodeHighIndex();
      } else if(n8or27==27){
        plowIndex = patch->getExtraNodeLowIndex();
        phighIndex = patch->getExtraNodeHighIndex();
      }

      long nn = (phighIndex[0]-plowIndex[0])*
                (phighIndex[1]-plowIndex[1])*
                (phighIndex[2]-plowIndex[2])*DOFsPerNode;

      d_petscGlobalStart[patch]=d_totalNodes;
      d_totalNodes+=nn;
      mytotal+=nn;
    }
    d_numNodes[p] = mytotal;
  }

  for(int p=0;p<patches->size();p++){
    const Patch* patch=patches->get(p);
    IntVector lowIndex,highIndex;
    if(n8or27==8){
        lowIndex = patch->getNodeLowIndex();
        highIndex = patch->getNodeHighIndex() + IntVector(1,1,1);
    } else if(n8or27==27){
        lowIndex = patch->getExtraNodeLowIndex();
        highIndex = patch->getExtraNodeHighIndex() + IntVector(1,1,1);
    }
    Array3<int> l2g(lowIndex, highIndex);
    l2g.initialize(-1234);
    long totalNodes=0;
    const Level* level = patch->getLevel();

    Patch::selectType neighbors;
    level->selectPatches(lowIndex, highIndex, neighbors);
    //For each neighbor and myself
    for(int i=0;i<neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      IntVector plow(0,0,0),phigh(0,0,0);
      if(n8or27==8){
        plow = neighbor->getNodeLowIndex();
        phigh = neighbor->getNodeHighIndex();
      } else if(n8or27==27){
        plow = neighbor->getExtraNodeLowIndex();
        phigh = neighbor->getExtraNodeHighIndex();
      }
      //intersect my patch with my neighbor patch
      IntVector low = Max(lowIndex, plow);
      IntVector high= Min(highIndex, phigh);
      if( ( high.x() < low.x() ) || ( high.y() < low.y() ) 
                                 || ( high.z() < low.z() ) )
         throw InternalError("Patch doesn't overlap?", __FILE__, __LINE__);
     
      //global start for this neighbor
      int petscglobalIndex = d_petscGlobalStart[neighbor];
      IntVector dnodes = phigh-plow;
      IntVector start = low-plow;

#if 0     
      petscglobalIndex += start.z()*dnodes.x()*dnodes.y()*DOFsPerNode
                       + start.y()*dnodes.x()*(DOFsPerNode-1) + start.x();
#endif

      //compute the starting index by computing the starting node index and multiplying it by the degrees of freedom per node
      petscglobalIndex += (start.z()*dnodes.x()*dnodes.y()+ start.y()*dnodes.x()+ start.x())*DOFsPerNode; 

      for (int colZ = low.z(); colZ < high.z(); colZ ++) {
        int idx_slab = petscglobalIndex;
        petscglobalIndex += dnodes.x()*dnodes.y()*DOFsPerNode;
        
        for (int colY = low.y(); colY < high.y(); colY ++) {
          int idx = idx_slab;
          idx_slab += dnodes.x()*DOFsPerNode;
          for (int colX = low.x(); colX < high.x(); colX ++) {
            l2g[IntVector(colX, colY, colZ)] = idx;
            idx += DOFsPerNode;
          }
        }
      }
      IntVector d = high-low;
      totalNodes+=d.x()*d.y()*d.z()*DOFsPerNode;
    }
    d_petscLocalToGlobal[patch].copyPointer(l2g);
  }
  d_DOFsPerNode=DOFsPerNode;
}

void MPMPetscSolver::solve(vector<double>& guess)
{
  TAU_PROFILE("MPMPetscSolver::solve", " ", TAU_USER);
  PC          precond;           
  KSP         solver;
#if 0
  if(d_DOFsPerNode<3){
    PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_DENSE);
    MatView(d_A,PETSC_VIEWER_STDOUT_WORLD);
  }
#endif

  KSPCreate(PETSC_COMM_WORLD,&solver);
  KSPSetOperators(solver,d_A,d_A,DIFFERENT_NONZERO_PATTERN);
  KSPGetPC(solver,&precond);

#if defined(USE_SPOOLES) || defined(USE_SUPERLU)
  KSPSetType(solver,KSPPREONLY);
  KSPSetFromOptions(solver);
  PCSetType(precond,PCLU);
#else
  KSPSetType(solver,KSPCG);
  PCSetType(precond,PCJACOBI);
#endif

  KSPSetTolerances(solver,PETSC_DEFAULT,PETSC_DEFAULT,
                          PETSC_DEFAULT,PETSC_DEFAULT);

  if (!guess.empty()) {
    KSPSetInitialGuessNonzero(solver,PETSC_TRUE);
    for (int i = 0; i < (int) guess.size(); i++) {
      VecSetValues(d_x,1,&i,&guess[i],INSERT_VALUES);
    }

  }
  TAU_PROFILE_TIMER(solve, "Petsc:KPSolve()", "", TAU_USER);
  TAU_PROFILE_START(solve);
  KSPSolve(solver,d_B,d_x);
  TAU_PROFILE_STOP(solve);
#ifdef LOG
  KSPView(solver,PETSC_VIEWER_STDOUT_WORLD);
  int its;
  KSPGetIterationNumber(solver,&its);
  PetscPrintf(PETSC_COMM_WORLD,"Iterations %d\n",its);
  VecView(d_x,PETSC_VIEWER_STDOUT_WORLD);
#endif

#if ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR >= 2))
  KSPDestroy(&solver);
#else
  KSPDestroy(solver);
#endif
}
void MPMPetscSolver::createMatrix(const ProcessorGroup* d_myworld,
                                  const map<int,int>& dof_diag)
{
  TAU_PROFILE("MPMPetscSolver::createMatrix", " ", TAU_USER);
  int me = d_myworld->myrank();
  int numlrows = d_numNodes[me];
  
  int numlcolumns = numlrows;
  int globalrows = (int)d_totalNodes;
  int globalcolumns = (int)d_totalNodes; 

  int *diag, *onnz;
  diag = scinew int[numlrows];
  onnz = scinew int[numlrows];
  for (int i = 0; i < numlrows; i++)
    diag[i] = 1;

  map<int,int>::const_iterator itr;
  for (itr=dof_diag.begin(); itr != dof_diag.end(); itr++) {
    ASSERTRANGE(itr->first,0,numlrows);
    ASSERT(itr->second>0);
    diag[itr->first] = itr->second;
  }

#if ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR >= 2))
  PetscBool exists;
#else
  PetscTruth exists;
#endif

#if 0
  cerr << "me = " << me << endl;
  cerr << "numlrows = " << numlrows << endl;
  cerr << "numlcolumns = " << numlcolumns << endl;
  cerr << "globalrows = " << globalrows << endl;
  cerr << "globalcolumns = " << globalcolumns << endl;
  for (int i = 0; i < numlrows; i++) 
    cerr << "diag[" << i << "] = " << diag[i] << endl;
#endif

#if ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR >= 2))
  PetscClassId id;
  if (d_A) {
    PetscObjectGetClassId((PetscObject)d_A,&id);
  }
  if (id) {
    exists = PETSC_TRUE;
  } else {
    exists = PETSC_FALSE;
  }
#elif ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR == 1))
  PetscCookie cookie = 0;
  if (d_A) {
    PetscObjectGetCookie((PetscObject)d_A,&cookie);
  }
  if (cookie) {
    exists = PETSC_TRUE;
  } else {
    exists = PETSC_FALSE;
  }

#else
  PetscObjectExists((PetscObject)d_A, &exists);
#endif

#if 0
    // This one works
    MatCreateMPIAIJ(PETSC_COMM_WORLD, numlrows, numlcolumns, globalrows,
                    globalcolumns, PETSC_DEFAULT, diag, 
                    PETSC_DEFAULT,PETSC_NULL, &d_A);
#endif

    // This one is much faster
    int ONNZ_MAX=57;
    int DIAG_MAX=81;
    if(d_DOFsPerNode==1){
      ONNZ_MAX=19;
      DIAG_MAX=27;
    }

    if (numlcolumns < ONNZ_MAX)
      ONNZ_MAX = numlcolumns;

    if (numlcolumns < DIAG_MAX)
      DIAG_MAX = numlcolumns;

    for (int i = 0; i < numlrows; i++){
      ASSERT(diag[i]>0);
      onnz[i]=ONNZ_MAX;
      if(diag[i]==1){
         onnz[i]=0;
      }
      diag[i]=min(diag[i],DIAG_MAX);
    }

#if defined(USE_SPOOLES) || defined(USE_SUPERLU)
    PetscErrorCode ierr;
    ierr = MatCreate(PETSC_COMM_WORLD, &d_A);CHKERRQ(ierr);
    ierr = MatSetSizes(d_A,numlrows,numlcolumns,globalrows,globalcolumns);
#ifdef USE_SPOOLES
    ierr = MatSetType(d_A,MATAIJSPOOLES);
#else
    ierr = MatSetType(d_A,MATSUPERLU_DIST);CHKERRQ(ierr);
#endif
    ierr = MatMPIAIJSetPreallocation(d_A,PETSC_DEFAULT,diag,PETSC_DEFAULT,onnz);
#else
#if ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR == 3))
    MatCreateAIJ(PETSC_COMM_WORLD, numlrows, numlcolumns, globalrows,
                    globalcolumns, PETSC_DEFAULT, diag,
                    PETSC_DEFAULT, onnz, &d_A);
#else
    MatCreateMPIAIJ(PETSC_COMM_WORLD, numlrows, numlcolumns, globalrows,
                    globalcolumns, PETSC_DEFAULT, diag, 
                    PETSC_DEFAULT, onnz, &d_A);
#endif
#endif
  
    //allocate the diagonal
    int low,high;
    MatGetOwnershipRange(d_A,&low,&high);
    for(int i=low;i<high;i++) {
      MatSetValue(d_A,i,i,0,ADD_VALUES);
    }
    flushMatrix();
//    MatType type;
//    MatGetType(d_A, &type);
//    cout << "MatType = " << type << endl;

    //set the initial stash size.
    //for now set it to be 1M
    //it should be counted and set dynamically
    //the stash is used by nodes that neighbor my patches on the + faces.
    MatStashSetInitialSize(d_A,1000000,0);
    if(d_DOFsPerNode>=1){
#if (PETSC_VERSION_MAJOR==3)
      MatSetOption(d_A, MAT_USE_INODES, PETSC_TRUE);
#else
      MatSetOption(d_A, MAT_USE_INODES);
#endif
    }
    
#if (PETSC_VERSION_MAJOR==3)
#if (PETSC_VERSION_MINOR >= 1)
    MatSetOption(d_A,MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
#else
    MatSetOption(d_A, MAT_KEEP_ZEROED_ROWS, PETSC_TRUE);
#endif
    MatSetOption(d_A,MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
#else
    MatSetOption(d_A, MAT_KEEP_ZEROED_ROWS);
    MatSetOption(d_A,MAT_IGNORE_ZERO_ENTRIES);
#endif

    // Create vectors.  Note that we form 1 vector from scratch and
    // then duplicate as needed.
#ifdef USE_SPOOLES
    VecCreate(PETSC_COMM_WORLD,&d_B);
    VecSetSizes(d_B,numlrows,globalrows);
    ierr = VecSetFromOptions(d_B);
#else
    VecCreateMPI(PETSC_COMM_WORLD,numlrows, globalrows,&d_B);
#endif
    VecDuplicate(d_B,&d_diagonal);
    VecDuplicate(d_B,&d_x);
    VecDuplicate(d_B,&d_t);
    VecDuplicate(d_B,&d_flux);

  delete[] diag;
  delete[] onnz;
}

void MPMPetscSolver::destroyMatrix(bool recursion)
{
  TAU_PROFILE("MPMPetscSolver::destroyMatrix", " ", TAU_USER);
  if (recursion) {
    MatZeroEntries(d_A);
    PetscScalar zero = 0.;
#if (PETSC_VERSION_MAJOR == 2 && PETSC_VERSION_MINOR == 2)
      VecSet(&zero,d_B);
      VecSet(&zero,d_diagonal);
      VecSet(&zero,d_x);
      VecSet(&zero,d_t);
      VecSet(&zero,d_flux);
#else 
      VecSet(d_B,zero);
      VecSet(d_diagonal,zero);
      VecSet(d_x,zero);
      VecSet(d_t,zero);
      VecSet(d_flux,zero);
#endif
  } else {
#if ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR >= 2))
  PetscBool exists;
#else
  PetscTruth exists;
#endif

#if ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR >= 2))
  PetscClassId id;
  if (d_A) {
    PetscObjectGetClassId((PetscObject)d_A,&id);
  }
  if (id) {
    exists = PETSC_TRUE;
  }
  else {
    exists = PETSC_FALSE;
  }
#elif ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR == 1))
  PetscCookie cookie = 0;
  if (d_A) {
    PetscObjectGetCookie((PetscObject)d_A,&cookie);
  }
  if (cookie) {
    exists = PETSC_TRUE;
  } else {
    exists = PETSC_FALSE;
  }
#else
    PetscObjectExists((PetscObject)d_A,&exists);
#endif
    if (exists == PETSC_TRUE) {
#if ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR >= 2))
      MatDestroy(&d_A);
      VecDestroy(&d_B);
      VecDestroy(&d_diagonal);
      VecDestroy(&d_x);
      VecDestroy(&d_t);
      VecDestroy(&d_flux);
#else
      MatDestroy(d_A);
      VecDestroy(d_B);
      VecDestroy(d_diagonal);
      VecDestroy(d_x);
      VecDestroy(d_t);
      VecDestroy(d_flux);
#endif
    }
  }
  if (recursion == false) {
    d_DOF.clear();
    d_DOFFlux.clear();
    d_DOFZero.clear();
  }
}

void
MPMPetscSolver::flushMatrix()
{
  MatAssemblyBegin(d_A,MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(d_A,MAT_FLUSH_ASSEMBLY);
}

void
MPMPetscSolver::fillVector(int i,double v,bool add)
{
  PetscScalar value = v;
  if (add) {
    VecSetValues(d_B,1,&i,&value,ADD_VALUES);
  } 
  else {
    VecSetValues(d_B,1,&i,&value,INSERT_VALUES);
  }
}

void
MPMPetscSolver::fillTemporaryVector(int i,double v)
{
  PetscScalar value = v;
  VecSetValues(d_t,1,&i,&value,INSERT_VALUES);
}

void
MPMPetscSolver::fillFluxVector(int i,double v)
{
  PetscScalar value = v;
  VecSetValues(d_flux,1,&i,&value,INSERT_VALUES);
}

void
MPMPetscSolver::assembleVector()
{
  VecAssemblyBegin(d_B);
  VecAssemblyEnd(d_B);
}

void
MPMPetscSolver::assembleTemporaryVector()
{
  VecAssemblyBegin(d_t);
  VecAssemblyEnd(d_t);
}


void
MPMPetscSolver::assembleFluxVector()
{
  VecAssemblyBegin(d_flux);
  VecAssemblyEnd(d_flux);
}

void
MPMPetscSolver::applyBCSToRHS()
{
  int ierr = MatMultAdd(d_A,d_t,d_B,d_B);
  if (ierr) {
    throw UintahPetscError(ierr, "MatMultAdd", __FILE__, __LINE__);
  }
}

void
MPMPetscSolver::copyL2G(Array3<int>& mapping,const Patch* patch)
{
  mapping.copy(d_petscLocalToGlobal[patch]);
}

void
MPMPetscSolver::removeFixedDOF()
{
  TAU_PROFILE("MPMPetscSolver::removeFixedDOF", " ", TAU_USER);
  flushMatrix();
  IS is;
  int* indices;
  int in=0;
  indices = scinew int[d_DOF.size()];
  for (set<int>::iterator iter = d_DOF.begin(); iter != d_DOF.end();
       iter++) {
    indices[in++] = *iter;

    // Take care of the d_B side
    PetscScalar v = 0.;
    const int index = *iter;
      
    VecSetValues(d_B,1,&index,&v,INSERT_VALUES);
    MatSetValue(d_A,index,index,1.,INSERT_VALUES);
  }

  finalizeMatrix();

#if 0
  MatTranspose(d_A,PETSC_NULL);
  MatZeroRows(d_A,is,&one);
  MatTranspose(d_A,PETSC_NULL);
#endif
  
  PetscScalar one = 1.0;

  // Make sure the nodes that are outside of the material have values 
  // assigned and solved for.  The solutions will be 0.
  int low=0,high=0;
  MatGetOwnershipRange(d_A,&low,&high);
  int size=high-low;

  MatGetDiagonal(d_A,d_diagonal);
  PetscScalar* diag;
  VecGetArray(d_diagonal,&diag);
  
  for (int j = 0; j < size; j++) {
    if (compare(diag[j],0.)) {
      VecSetValues(d_diagonal,1,&j,&one,INSERT_VALUES);
      PetscScalar v = 0.;
      VecSetValues(d_B,1,&j,&v,INSERT_VALUES);
    }
  }
  VecRestoreArray(d_diagonal,&diag);

  VecAssemblyBegin(d_B);
  VecAssemblyEnd(d_B);
  VecAssemblyBegin(d_diagonal);
  VecAssemblyEnd(d_diagonal);
  MatDiagonalSet(d_A,d_diagonal,INSERT_VALUES);
  
#if ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR >= 2))
  ISCreateGeneral(PETSC_COMM_SELF, d_DOF.size(), indices, PETSC_COPY_VALUES, &is);
#else
  ISCreateGeneral(PETSC_COMM_SELF, d_DOF.size(), indices, &is);
#endif
  delete[] indices;

#if (PETSC_VERSION_MAJOR == 2 && PETSC_VERSION_MINOR == 2)
  MatZeroRows(d_A,is,&one);
#elif ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR >= 2))
  MatZeroRowsIS(d_A, is, one, 0, 0);
#else
  MatZeroRowsIS(d_A,is,one);
#endif

#if ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR >= 2))
  ISDestroy(&is);
#else
  ISDestroy(is);
#endif


  //__________________________________
  //  debugging
#ifdef OUTPUT_A_B
  char matfile[100],vecfile[100];
  
  PetscViewer matview, vecview;
  sprintf(vecfile,"output/vector.%d.%d",Parallel::getMPISize(),d_iteration);
  sprintf(matfile,"output/matrix.%d.%d",Parallel::getMPISize(),d_iteration);
  
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,vecfile,&vecview);
  VecView(d_B,vecview);
  PetscViewerDestroy(vecview);
  
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,matfile,&matview);
  MatView(d_A,matview);
  PetscViewerDestroy(matview);

  d_iteration++;
#endif
}

void MPMPetscSolver::removeFixedDOFHeat()
{
  TAU_PROFILE("MPMPetscSolver::removeFixedDOFHEAT", " ", TAU_USER);

  //do matrix modifications first 
 
  for (set<int>::iterator iter = d_DOFZero.begin(); iter != d_DOFZero.end();
       iter++) {
    int j = *iter;

    PetscScalar v_zero = 0.;
    //    VecSetValues(d_diagonal,1,&j,&v_one,INSERT_VALUES);
    VecSetValues(d_B,1,&j,&v_zero,INSERT_VALUES);
    MatSetValue(d_A,j,j,1.,INSERT_VALUES);

  }
  
  // Zero the rows/columns that contain the node numbers with BCs.

  int* indices = scinew int[d_DOF.size()];  
  int in = 0;
  for (set<int>::iterator iter = d_DOF.begin(); iter != d_DOF.end(); 
       iter++) {
    indices[in++] = *iter;
  }

  if( d_DOF.size() !=0)
  {
    // zeroing out the columns
    for (set<int>::iterator iter = d_DOF.begin(); iter != d_DOF.end(); 
       iter++) {
      const int index = *iter;
      vector<int>& neighbors = d_DOFNeighbors[index];

      for (vector<int>::iterator n = neighbors.begin(); n != neighbors.end();
           n++) {
        int ierr;
        // zero out the columns
        ierr = MatSetValue(d_A,*n,index,0,INSERT_VALUES);
        if (ierr)
          cout << "MatSetValue error for " << index << "," << *n << endl;
      }
    }
  }
  
  finalizeMatrix();

  if (d_DOF.size() != 0) {
    cout << "Zeroing out rows" << endl;
  }
  IS is;
#if ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR >= 2))
  ISCreateGeneral(PETSC_COMM_SELF, d_DOF.size(), indices, PETSC_COPY_VALUES, &is);
#else
  ISCreateGeneral(PETSC_COMM_SELF, d_DOF.size(), indices, &is);
#endif

  PetscScalar one = 1.0;
#if (PETSC_VERSION_MAJOR == 2 && PETSC_VERSION_MINOR == 2)
  MatZeroRows(d_A,is,&one);
#elif ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR >= 2))
  MatZeroRowsIS(d_A, is, one, 0, 0);
#else
  MatZeroRowsIS(d_A, is, one);
#endif

#if ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR >= 2))
  ISDestroy(&is);
#else
  ISDestroy(is);
#endif

  int* indices_flux = scinew int[d_DOFFlux.size()];
  in = 0;
  for (set<int>::iterator iter = d_DOFFlux.begin(); iter != d_DOFFlux.end();
       iter++) {
    indices_flux[in++] = *iter;
  }


  //do vector modifications

  PetscScalar* y = scinew PetscScalar[d_DOF.size()];
  PetscScalar* y_flux = scinew PetscScalar[d_DOFFlux.size()];

  assembleFluxVector();
#if ( PETSC_VERSION_MAJOR==2 && PETSC_VERSION_MINOR == 2)
  PetscInt nlocal_t,nlocal_flux;
  PetscScalar minus_one = -1.;
  VecScale(&minus_one,d_t);
  PetscScalar* d_t_tmp;
  PetscScalar* d_flux_tmp;
  VecGetArray(d_t,&d_t_tmp);
  VecGetArray(d_flux,&d_flux_tmp);
  VecGetLocalSize(d_t,&nlocal_t);
  VecGetLocalSize(d_flux,&nlocal_flux);
  PetscInt low_t,high_t;
  VecGetOwnershipRange(d_t,&low_t,&high_t);
  PetscInt low_flux,high_flux;
  VecGetOwnershipRange(d_flux,&low_flux,&high_flux);

  for (int i = 0; i < (int) d_DOF.size();i++) {
    int offset = indices[i] - low_t;
    y[i] = d_t_tmp[offset];
  }
  for (int i = 0; i < (int) d_DOFFlux.size();i++) {
    int offset = indices_flux[i] - low_flux;
    y_flux[i] = d_flux_tmp[offset];
  }
  VecRestoreArray(d_t,&d_t_tmp);
  VecRestoreArray(d_flux,&d_flux_tmp);

#else
  VecScale(d_t,-1.);
  VecGetValues(d_t,d_DOF.size(),indices,y);
  VecGetValues(d_flux,d_DOFFlux.size(),indices_flux,y_flux);
#endif
  
  VecSetValues(d_B,d_DOF.size(),indices,y,INSERT_VALUES);
  assembleVector();
  VecSetValues(d_B,d_DOFFlux.size(),indices_flux,y_flux,ADD_VALUES);

  delete[] y;
  delete[] y_flux;

  assembleFluxVector();
  assembleVector();

  delete[] indices;
  delete[] indices_flux;


#if 0
  MatView(d_A,PETSC_VIEWER_STDOUT_WORLD);
  VecView(d_B,PETSC_VIEWER_STDOUT_WORLD);
#endif

  //__________________________________
  //  debugging
#ifdef OUTPUT_A_B
  char matfile[100],vecfile[100];
  
  PetscViewer matview, vecview;
  sprintf(vecfile,"output/HeatVector.%d.%d",Parallel::getMPISize(),d_iteration);
  sprintf(matfile,"output/HeatMatrix.%d.%d",Parallel::getMPISize(),d_iteration);
  
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,vecfile,&vecview);
  VecView(d_B,vecview);
  PetscViewerDestroy(vecview);
  
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,matfile,&matview);
  MatView(d_A,matview);
  PetscViewerDestroy(matview);

  d_iteration++;
#endif

}

void MPMPetscSolver::finalizeMatrix()
{
  MatAssemblyBegin(d_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(d_A,MAT_FINAL_ASSEMBLY);
}

int MPMPetscSolver::getSolution(vector<double>& xPetsc)
{
  int nlocal,ierr,begin,end;
  double* x;
  VecGetLocalSize(d_x,&nlocal);
  VecGetOwnershipRange(d_x,&begin,&end);
  ierr = VecGetArray(d_x,&x);
  if (ierr)
    cerr << "VecGetArray failed" << endl;
  for (int ii = 0; ii < nlocal; ii++) {
    xPetsc.push_back(x[ii]);
  }
  VecRestoreArray(d_x,&x);
  return begin;
}

int MPMPetscSolver::getRHS(vector<double>& QPetsc)
{
  int nlocal,ierr,begin,end;
  double* q;
  VecGetLocalSize(d_B,&nlocal);
  VecGetOwnershipRange(d_B,&begin,&end);
  ierr = VecGetArray(d_B,&q);
  if (ierr)
    cerr << "VecGetArray failed" << endl;
  for (int ii = 0; ii < nlocal; ii++) {
    QPetsc.push_back(q[ii]);
  }
  VecRestoreArray(d_B,&q);
  return begin;
}
