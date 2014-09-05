#define PETSC_USE_LOG

#include <sci_defs/mpi_defs.h>
#include <sci_defs/petsc_defs.h>

#include <Packages/Uintah/CCA/Components/MPM/PetscSolver.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>

#include <vector>
#include <iostream>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

//#define LOG
#undef LOG
#undef DEBUG_PETSC

MPMPetscSolver::MPMPetscSolver()
{
  d_A = 0;
  d_B = 0;
  d_diagonal = 0;
  d_x = 0;
  d_t = 0;
}

MPMPetscSolver::~MPMPetscSolver()
{
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

void MPMPetscSolver::solve()
{
  PC          precond;           
  KSP         solver;
#  if 0
  if(d_DOFsPerNode<3){
  PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_DENSE);
  MatView(d_A,PETSC_VIEWER_STDOUT_WORLD);
  }
#  endif
  KSPCreate(PETSC_COMM_WORLD,&solver);
  KSPSetOperators(solver,d_A,d_A,DIFFERENT_NONZERO_PATTERN);
  KSPGetPC(solver,&precond);
  KSPSetType(solver,KSPCG);
  PCSetType(precond,PCJACOBI);
  KSPSetTolerances(solver,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
  

#  ifdef debug
  VecView(d_B,PETSC_VIEWER_STDOUT_WORLD);
#  endif
  KSPSolve(solver,d_B,d_x);
#  ifdef LOG
  KSPView(solver,PETSC_VIEWER_STDOUT_WORLD);
  int its;
  KSPGetIterationNumber(solver,&its);
  PetscPrintf(PETSC_COMM_WORLD,"Iterations %d\n",its);
#  endif
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
#if 1
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

    MatCreateMPIAIJ(PETSC_COMM_WORLD, numlrows, numlcolumns, globalrows,
                    globalcolumns, PETSC_DEFAULT, diag, 
                    PETSC_DEFAULT, onnz, &d_A);

    if(d_DOFsPerNode==3){
      MatSetOption(d_A, MAT_USE_INODES);
    }
#endif
    MatSetOption(d_A, MAT_KEEP_ZEROED_ROWS);

    // Create vectors.  Note that we form 1 vector from scratch and
    // then duplicate as needed.

    VecCreateMPI(PETSC_COMM_WORLD,numlrows, globalrows,&d_B);
    VecDuplicate(d_B,&d_diagonal);
    VecDuplicate(d_B,&d_x);
    VecDuplicate(d_B,&d_t);

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
#endif
#if (PETSC_VERSION_MINOR == 3)
      VecSet(d_B,zero);
      VecSet(d_diagonal,zero);
      VecSet(d_x,zero);
      VecSet(d_t,zero);
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
    }
  }
  if (recursion == false)
    d_DOF.clear();
}

void MPMPetscSolver::flushMatrix()
{
  MatAssemblyBegin(d_A,MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(d_A,MAT_FLUSH_ASSEMBLY);
}

void MPMPetscSolver::fillVector(int i,double v)
{
  PetscScalar value = v;
  VecSetValues(d_B,1,&i,&value,INSERT_VALUES);
}

void MPMPetscSolver::fillTemporaryVector(int i,double v)
{
  PetscScalar value = v;
  VecSetValues(d_t,1,&i,&value,INSERT_VALUES);
}

void MPMPetscSolver::assembleVector()
{
  VecAssemblyBegin(d_B);
  VecAssemblyEnd(d_B);
}

void MPMPetscSolver::assembleTemporaryVector()
{
  VecAssemblyBegin(d_t);
  VecAssemblyEnd(d_t);
}

void MPMPetscSolver::applyBCSToRHS()
{
  MatMultAdd(d_A,d_t,d_B,d_B);

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
