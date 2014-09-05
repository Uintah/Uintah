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

#define LOG
#undef LOG

MPMPetscSolver::MPMPetscSolver()
{
#ifdef HAVE_PETSC
  d_A = 0;
  d_B = 0;
  d_diagonal = 0;
  d_x = 0;
  d_t = 0;
#endif
}

MPMPetscSolver::~MPMPetscSolver()
{
}


void MPMPetscSolver::initialize()
{
#ifdef HAVE_PETSC
#ifdef LOG
  int argc = 9;
#else
  int argc = 2;
#endif
  char** argv;
  argv = new char*[argc];
  argv[0] = "ImpMPM::problemSetup";
  //argv[1] = "-on_error_attach_debugger";
  //argv[1] = "-start_in_debugger";
  argv[1] = "-no_signal_handler";
#ifdef LOG
  argv[2] = "-log_exclude_actions";
  argv[3] = "-log_exclude_objects";
  argv[4] = "-log_info";
  argv[5] = "-trmalloc";
  argv[6] = "-trdump";
  argv[7] = "-trmalloc_log";
  argv[8] = "-log_summary";
  //argv[2] = "-log_summary";
#endif

  PetscInitialize(&argc,&argv, PETSC_NULL, PETSC_NULL);
#endif
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
      IntVector plowIndex = patch->getNodeLowIndex();
      IntVector phighIndex = patch->getNodeHighIndex();

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
    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex() + IntVector(1,1,1);
    Array3<int> l2g(lowIndex, highIndex);
    l2g.initialize(-1234);
    long totalNodes=0;
    const Level* level = patch->getLevel();
    Patch::selectType neighbors;
    level->selectPatches(lowIndex, highIndex, neighbors);
    for(int i=0;i<neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      IntVector plow = neighbor->getNodeLowIndex();
      IntVector phigh = neighbor->getNodeHighIndex();
      IntVector low = Max(lowIndex, plow);
      IntVector high= Min(highIndex, phigh);
      if( ( high.x() < low.x() ) || ( high.y() < low.y() ) 
                                 || ( high.z() < low.z() ) )
         throw InternalError("Patch doesn't overlap?");
      
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
#ifdef HAVE_PETSC
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
  KSPSetType(solver,KSPCG);
  PCSetType(precond,PCJACOBI);
  KSPSetTolerances(solver,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
  

#ifdef debug
  VecView(d_B,PETSC_VIEWER_STDOUT_WORLD);
#endif
  KSPSolve(solver,d_B,d_x);
#ifdef LOG
  KSPView(solver,PETSC_VIEWER_STDOUT_WORLD);
  int its;
  KSPGetIterationNumber(solver,&its);
  PetscPrintf(PETSC_COMM_WORLD,"Iterations %d\n",its);
#endif
  KSPDestroy(solver);
#endif
}

void MPMPetscSolver::createMatrix(const ProcessorGroup* d_myworld,
				  const map<int,int>& dof_diag)
{
  int me = d_myworld->myrank();
  int numlrows = d_numNodes[me];

#ifdef HAVE_PETSC
  int numlcolumns = numlrows;
  int globalrows = (int)d_totalNodes;
  int globalcolumns = (int)d_totalNodes; 
#endif
#if 0 
  cerr << "me = " << me << endl;
  cerr << "numlrows = " << numlrows << endl;
  cerr << "numlcolumns = " << numlcolumns << endl;
  cerr << "globalrows = " << globalrows << endl;
  cerr << "globalcolumns = " << globalcolumns << endl;
#endif
  int *diag;
  diag = new int[numlrows];
  for (int i = 0; i < numlrows; i++) 
    diag[i] = 1;

  map<int,int>::const_iterator itr;
  for (itr=dof_diag.begin(); itr != dof_diag.end(); itr++) {
    //    cerr << "diag_before[" << itr->first << "]=" << itr->second << endl;
    diag[itr->first] = itr->second;
  }

#if 0
  for (int i = 0; i < numlrows; i++) 
    cerr << "diag[" << i << "] = " << diag[i] << endl;
#endif

#ifdef HAVE_PETSC
  PetscTruth exists;
  PetscObjectExists((PetscObject)d_A,&exists);
  //if (exists == PETSC_FALSE) {
#if 0
    MatCreateMPIAIJ(PETSC_COMM_WORLD, numlrows, numlcolumns, globalrows,
		    globalcolumns, PETSC_DEFAULT, PETSC_NULL, PETSC_DEFAULT,
		    PETSC_NULL, &d_A);
#endif
#if 1
    // This one works
    MatCreateMPIAIJ(PETSC_COMM_WORLD, numlrows, numlcolumns, globalrows,
		    globalcolumns, PETSC_DEFAULT, diag, 
		    PETSC_DEFAULT,PETSC_NULL, &d_A);
    MatSetOption(d_A,MAT_KEEP_ZEROED_ROWS);
#endif
   /* 
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
  */
    VecCreateMPI(PETSC_COMM_WORLD,numlrows, globalrows,&d_B);
    VecDuplicate(d_B,&d_diagonal);
    VecDuplicate(d_B,&d_x);
    VecDuplicate(d_B,&d_t);
  //}
#endif

  delete[] diag;
}


void MPMPetscSolver::destroyMatrix(bool recursion)
{
#ifdef HAVE_PETSC
  if (recursion) {
    MatZeroEntries(d_A);
    PetscScalar zero = 0.;
    VecSet(&zero,d_B);
    VecSet(&zero,d_diagonal);
    VecSet(&zero,d_x);
    VecSet(&zero,d_t);
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
#endif
  if (recursion == false)
    d_DOF.clear();
}

void MPMPetscSolver::flushMatrix()
{
#ifdef HAVE_PETSC
  MatAssemblyBegin(d_A,MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(d_A,MAT_FLUSH_ASSEMBLY);
#endif
}

void MPMPetscSolver::fillVector(int i,double v)
{
#ifdef HAVE_PETSC
  PetscScalar value = v;
  VecSetValues(d_B,1,&i,&value,INSERT_VALUES);
#endif
}

void MPMPetscSolver::fillTemporaryVector(int i,double v)
{
#ifdef HAVE_PETSC
  PetscScalar value = v;
  VecSetValues(d_t,1,&i,&value,INSERT_VALUES);
#endif
}

void MPMPetscSolver::assembleVector()
{
#ifdef HAVE_PETSC
  VecAssemblyBegin(d_B);
  VecAssemblyEnd(d_B);
#endif
}

void MPMPetscSolver::assembleTemporaryVector()
{
#ifdef HAVE_PETSC
  VecAssemblyBegin(d_t);
  VecAssemblyEnd(d_t);
#endif
}

void MPMPetscSolver::applyBCSToRHS()
{
#ifdef HAVE_PETSC
  MatMultAdd(d_A,d_t,d_B,d_B);
#endif
}

void MPMPetscSolver::copyL2G(Array3<int>& mapping,const Patch* patch)
{
  mapping.copy(d_petscLocalToGlobal[patch]);
}

void MPMPetscSolver::removeFixedDOF(int num_nodes)
{
#ifdef HAVE_PETSC
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
  MatZeroRows(d_A,is,&one);
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
#endif
}

void MPMPetscSolver::finalizeMatrix()
{
#ifdef HAVE_PETSC
  MatAssemblyBegin(d_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(d_A,MAT_FINAL_ASSEMBLY);
#endif
}

int MPMPetscSolver::getSolution(vector<double>& xPetsc)
{
#ifdef HAVE_PETSC
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
#endif
}

int MPMPetscSolver::getRHS(vector<double>& QPetsc)
{
#ifdef HAVE_PETSC
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
#endif
}
