#define PETSC_USE_LOG

#include <sci_defs/mpi_defs.h>
#include <sci_defs/petsc_defs.h>

#include <Packages/Uintah/CCA/Components/MPM/PetscSolver.h>
#include <Packages/Uintah/Core/Exceptions/PetscError.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>

#include <vector>
#include <iostream>
#undef CHKERRQ
#define CHKERRQ(x) if(x) throw PetscError(x, __FILE__, __FILE__, __LINE__);

using namespace Uintah;
using namespace SCIRun;
using namespace std;

//#define LOG
#undef LOG
#undef DEBUG_PETSC

//#define USE_SPOOLES
#undef  USE_SPOOLES

//#define USE_SUPERLU
#undef  USE_SUPERLU

MPMPetscSolver::MPMPetscSolver()
{
  d_A = 0;
  d_B = 0;
  d_diagonal = 0;
  d_x = 0;
  d_t = 0;
  d_flux = 0;
}

MPMPetscSolver::~MPMPetscSolver()
{
    PetscFinalize();
}


void MPMPetscSolver::initialize()
{
  // store in a vector so we can customize the settings easily
  vector<char*> args;
#  ifdef DEBUG_PETSC
  args.push_back("ImpMPM::problemSetup");
  args.push_back("-on_error_attach_debugger");
#  endif
#  ifdef LOG
  args.push_back("-log_summary");
  args.push_back("-log_info");
#    if 0
  args.push_back("-log_exclude_actions");
  args.push_back("-log_exclude_objects");
  args.push_back("-log_info");
  args.push_back("-trmalloc");
  args.push_back("-trdump");
  args.push_back("-trmalloc_log");
  args.push_back("-log_summary");
#    endif
#  endif

  int argc = args.size();
  char** argv = 0;

  // copy the vector to argv, as I think petsc wants to store it around -- bjw
  if (argc > 0) {
    argv = new char*[argc];
    for (int i = 0; i < argc; i++) {
      argv[i] = args[i];
    }
  }
  PetscInitialize(&argc,&argv, PETSC_NULL, PETSC_NULL);
#ifdef USE_SPOOLES
  PetscOptionsSetValue("-mat_spooles_ordering","BestOfNDandMS");
  PetscOptionsSetValue("-mat_spooles_symmetryflag","0");
#endif
  PetscOptionsSetValue("-options_table", PETSC_NULL);
//  PetscOptionsSetValue("-log_summary", PETSC_NULL);
//  PetscOptionsSetValue("-log_info", PETSC_NULL);
//  PetscOptionsSetValue("-trmalloc", PETSC_NULL);
//  PetscOptionsSetValue("-trmalloc_log", PETSC_NULL);
//  PetscOptionsSetValue("-trdump", PETSC_NULL);
}

void 
MPMPetscSolver::createLocalToGlobalMapping(const ProcessorGroup* d_myworld,
					   const PatchSet* perproc_patches,
					   const PatchSubset* patches,
                                           const int DOFsPerNode)
{
  int numProcessors = d_myworld->size();
  d_numNodes.resize(numProcessors, 0);
  d_startIndex.resize(numProcessors);
  d_totalNodes = 0;

   for (int p = 0; p < perproc_patches->size(); p++) {
    d_startIndex[p] = d_totalNodes;
    int mytotal = 0;
    const PatchSubset* patchsub = perproc_patches->getSubset(p);
    for (int ps = 0; ps<patchsub->size(); ps++) {
      const Patch* patch = patchsub->get(ps);
      IntVector plowIndex = patch->getInteriorNodeLowIndex();
      IntVector phighIndex = patch->getInteriorNodeHighIndex();

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
    IntVector lowIndex = patch->getInteriorNodeLowIndex();
    IntVector highIndex = patch->getInteriorNodeHighIndex() + IntVector(1,1,1);
    Array3<int> l2g(lowIndex, highIndex);
    l2g.initialize(-1234);
    long totalNodes=0;
    const Level* level = patch->getLevel();
    Patch::selectType neighbors;
    level->selectPatches(lowIndex, highIndex, neighbors);
    for(int i=0;i<neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      IntVector plow = neighbor->getInteriorNodeLowIndex();
      IntVector phigh = neighbor->getInteriorNodeHighIndex();
      IntVector low = Max(lowIndex, plow);
      IntVector high= Min(highIndex, phigh);
      if( ( high.x() < low.x() ) || ( high.y() < low.y() ) 
                                 || ( high.z() < low.z() ) )
         throw InternalError("Patch doesn't overlap?", __FILE__, __LINE__);
      
      int petscglobalIndex = d_petscGlobalStart[neighbor];
      IntVector dnodes = phigh-plow;
      IntVector start = low-plow;
      petscglobalIndex += start.z()*dnodes.x()*dnodes.y()*DOFsPerNode
                        + start.y()*dnodes.x()*(DOFsPerNode-1) + start.x();
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
  KSPSolve(solver,d_B,d_x);
#ifdef LOG
  KSPView(solver,PETSC_VIEWER_STDOUT_WORLD);
  int its;
  KSPGetIterationNumber(solver,&its);
  PetscPrintf(PETSC_COMM_WORLD,"Iterations %d\n",its);
  VecView(d_x,PETSC_VIEWER_STDOUT_WORLD);
#endif
  KSPDestroy(solver);
}

void MPMPetscSolver::createMatrix(const ProcessorGroup* d_myworld,
				  const map<int,int>& dof_diag)
{
  int me = d_myworld->myrank();
  int numlrows = d_numNodes[me];

  int numlcolumns = numlrows;
  int globalrows = (int)d_totalNodes;
  int globalcolumns = (int)d_totalNodes; 

  int *diag, *onnz;
  diag = new int[numlrows];
  onnz = new int[numlrows];
  for (int i = 0; i < numlrows; i++) 
    diag[i] = 1;

  map<int,int>::const_iterator itr;
  for (itr=dof_diag.begin(); itr != dof_diag.end(); itr++) {
    diag[itr->first] = itr->second;
  }

#if 0
  cerr << "me = " << me << endl;
  cerr << "numlrows = " << numlrows << endl;
  cerr << "numlcolumns = " << numlcolumns << endl;
  cerr << "globalrows = " << globalrows << endl;
  cerr << "globalcolumns = " << globalcolumns << endl;
  for (int i = 0; i < numlrows; i++) 
    cerr << "diag[" << i << "] = " << diag[i] << endl;
#endif

  PetscTruth exists;
  PetscObjectExists((PetscObject)d_A,&exists);
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
    MatCreateMPIAIJ(PETSC_COMM_WORLD, numlrows, numlcolumns, globalrows,
                    globalcolumns, PETSC_DEFAULT, diag, 
                    PETSC_DEFAULT, onnz, &d_A);
#endif

//    MatType type;
//    MatGetType(d_A, &type);
//    cout << "MatType = " << type << endl;

    if(d_DOFsPerNode>=1){
      MatSetOption(d_A, MAT_USE_INODES);
    }

    MatSetOption(d_A, MAT_KEEP_ZEROED_ROWS);
    MatSetOption(d_A,MAT_IGNORE_ZERO_ENTRIES);

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
}


void MPMPetscSolver::destroyMatrix(bool recursion)
{
  if (recursion) {
    MatZeroEntries(d_A);
    PetscScalar zero = 0.;
#if (PETSC_VERSION_MINOR == 2)
      VecSet(&zero,d_B);
      VecSet(&zero,d_diagonal);
      VecSet(&zero,d_x);
      VecSet(&zero,d_t);
      VecSet(&zero,d_flux);
#endif
#if (PETSC_VERSION_MINOR == 3)
      VecSet(d_B,zero);
      VecSet(d_diagonal,zero);
      VecSet(d_x,zero);
      VecSet(d_t,zero);
      VecSet(d_flux,zero);
#endif
  } else {
    PetscTruth exists;
    PetscObjectExists((PetscObject)d_A,&exists);
    if (exists == PETSC_TRUE) {
      MatDestroy(d_A);
      VecDestroy(d_B);
      VecDestroy(d_diagonal);
      VecDestroy(d_x);
      VecDestroy(d_t);
      VecDestroy(d_flux);
    }
  }
  if (recursion == false) {
    d_DOF.clear();
    d_DOFFlux.clear();
    d_DOFZero.clear();
  }
}

void MPMPetscSolver::flushMatrix()
{
  MatAssemblyBegin(d_A,MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(d_A,MAT_FLUSH_ASSEMBLY);
}

void MPMPetscSolver::fillVector(int i,double v,bool add)
{
  PetscScalar value = v;
  if (add)
    VecSetValues(d_B,1,&i,&value,ADD_VALUES);
  else
    VecSetValues(d_B,1,&i,&value,INSERT_VALUES);
}

void MPMPetscSolver::fillTemporaryVector(int i,double v)
{
  PetscScalar value = v;
  VecSetValues(d_t,1,&i,&value,INSERT_VALUES);
}


void MPMPetscSolver::fillFluxVector(int i,double v)
{
  PetscScalar value = v;
  VecSetValues(d_flux,1,&i,&value,INSERT_VALUES);
}

void MPMPetscSolver::assembleVector()
{
  VecAssemblyBegin(d_B);
  VecAssemblyEnd(d_B);
  VecAssemblyBegin(d_x);
  VecAssemblyEnd(d_x);
}

void MPMPetscSolver::assembleTemporaryVector()
{
  VecAssemblyBegin(d_t);
  VecAssemblyEnd(d_t);
}


void MPMPetscSolver::assembleFluxVector()
{
  VecAssemblyBegin(d_flux);
  VecAssemblyEnd(d_flux);
}

void MPMPetscSolver::applyBCSToRHS()
{
  int ierr = MatMultAdd(d_A,d_t,d_B,d_B);
  if (ierr)
    throw PetscError(ierr, "MatMultAdd", __FILE__, __LINE__);

}

void MPMPetscSolver::copyL2G(Array3<int>& mapping,const Patch* patch)
{
  mapping.copy(d_petscLocalToGlobal[patch]);
}


void MPMPetscSolver::removeFixedDOF(int num_nodes)
{
  IS is;
  int* indices;
  int in = 0;

  indices = new int[d_DOF.size()];
  for (set<int>::iterator iter = d_DOF.begin(); iter != d_DOF.end(); 
       iter++) {
    indices[in++] = *iter;

    // Take care of the d_B side
    PetscScalar v = 0.;
    const int index = *iter;

    VecSetValues(d_B,1,&index,&v,INSERT_VALUES);
    MatSetValue(d_A,index,index,1.,INSERT_VALUES);
  }

  MatAssemblyBegin(d_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(d_A,MAT_FINAL_ASSEMBLY);

  ISCreateGeneral(PETSC_COMM_SELF,d_DOF.size(),indices,&is);
  delete[] indices;
  

  PetscScalar one = 1.0;
#if (PETSC_VERSION_MINOR == 2)
  MatZeroRows(d_A,is,&one);
#endif
#if (PETSC_VERSION_MINOR == 3)
  MatZeroRowsIS(d_A,is,one);
#endif
  ISDestroy(is);
#if 0
  MatTranspose(d_A,PETSC_NULL);
  MatZeroRows(d_A,is,&one);
  MatTranspose(d_A,PETSC_NULL);
#endif

  // Make sure the nodes that are outside of the material have values 
  // assigned and solved for.  The solutions will be 0.
  MatGetDiagonal(d_A,d_diagonal);
  PetscScalar* diag;
  VecGetArray(d_diagonal,&diag);
  for (int j = 0; j < num_nodes; j++) {
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
  MatAssemblyBegin(d_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(d_A,MAT_FINAL_ASSEMBLY);
}

void MPMPetscSolver::removeFixedDOFHeat(int num_nodes)
{
  finalizeMatrix();
  assembleFluxVector();

  for (set<int>::iterator iter = d_DOFZero.begin(); iter != d_DOFZero.end();
       iter++) {
    int j = *iter;

    PetscScalar v_zero = 0.;
    //    VecSetValues(d_diagonal,1,&j,&v_one,INSERT_VALUES);
    MatSetValue(d_A,j,j,1.,INSERT_VALUES);
    VecSetValues(d_B,1,&j,&v_zero,INSERT_VALUES);

  }
  
  // Zero the rows/columns that contain the node numbers with BCs.

  int* indices = new int[d_DOF.size()];  
  int in = 0;
  for (set<int>::iterator iter = d_DOF.begin(); iter != d_DOF.end(); 
       iter++) {
    indices[in++] = *iter;
  }


  finalizeMatrix();

  if (d_DOF.size() != 0) {
    cout << "Zeroing out rows" << endl;
    IS is;
    ISCreateGeneral(PETSC_COMM_SELF,d_DOF.size(),indices,&is);
       
    PetscScalar one = 1.0;
#if (PETSC_VERSION_MINOR == 2)
    MatZeroRows(d_A,is,&one);
#endif
#if (PETSC_VERSION_MINOR == 3)
    MatZeroRowsIS(d_A,is,one);
#endif
    ISDestroy(is);
  }

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

  int* indices_flux = new int[d_DOFFlux.size()];
  in = 0;
  for (set<int>::iterator iter = d_DOFFlux.begin(); iter != d_DOFFlux.end(); 
       iter++) {
    indices_flux[in++] = *iter;
  }
  

  PetscScalar* y = new PetscScalar[d_DOF.size()];
  PetscScalar* y_flux = new PetscScalar[d_DOFFlux.size()];
#if (PETSC_VERSION_MINOR == 3)
  VecScale(d_t,-1.);
  VecGetValues(d_t,d_DOF.size(),indices,y);
  VecGetValues(d_flux,d_DOFFlux.size(),indices_flux,y_flux);
  VecSetValues(d_B,d_DOF.size(),indices,y,INSERT_VALUES);
  VecSetValues(d_B,d_DOFFlux.size(),indices_flux,y_flux,ADD_VALUES);
#endif
#if (PETSC_VERSION_MINOR == 2)
  PetscScalar minus_one=-1.;
  VecScale(&minus_one,d_t);
  PetscScalar* d_t_tmp;
  PetscScalar* d_flux_tmp;
  VecGetArray(d_t,&d_t_tmp);
  VecGetArray(d_flux,&d_flux_tmp);
  for (int i = 0; i < (int)d_DOF.size();i++) {
	y[i] = d_t_tmp[indices[i]];
  }
  for (int i = 0; i < (int)d_DOFFlux.size();i++) {
	y_flux[i] = d_flux_tmp[indices_flux[i]];
  }
  VecRestoreArray(d_t,&d_t_tmp);
  VecRestoreArray(d_flux,&d_flux_tmp);
  VecSetValues(d_B,d_DOF.size(),indices,y,INSERT_VALUES);
  VecSetValues(d_B,d_DOFFlux.size(),indices_flux,y_flux,ADD_VALUES);
#endif

  delete[] y;
  delete[] y_flux;

  assembleVector();
  finalizeMatrix();

  delete[] indices;
  delete[] indices_flux;


#if 0
  MatView(d_A,PETSC_VIEWER_STDOUT_WORLD);
  VecView(d_B,PETSC_VIEWER_STDOUT_WORLD);
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
