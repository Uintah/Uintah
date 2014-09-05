//----- PetscSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/debug.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/RadLinearSolver.h>
#include <Core/Containers/Array1.h>
#include <Core/Thread/Time.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesVariables.h>
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
#undef CHKERRQ
#define CHKERRQ(x) if(x) throw PetscError(x, __FILE__, __FILE__, __LINE__);
#include <vector>

using namespace std;
using namespace Uintah;
using namespace SCIRun;


// ****************************************************************************
// Default constructor for RadLinearSolver
// ****************************************************************************
RadLinearSolver::RadLinearSolver(const ProcessorGroup* myworld)
   : d_myworld(myworld)
{
}

// ****************************************************************************
// Destructor
// ****************************************************************************
RadLinearSolver::~RadLinearSolver()
{
// Comment out the following till better place to finalize PETSC is found
//  finalizeSolver();
}

// ****************************************************************************
// Problem setup
// ****************************************************************************
void 
RadLinearSolver::problemSetup(const ProblemSpecP& params)
{
  if (params) {
    ProblemSpecP db = params->findBlock("LinearSolver");
    if (db) {
      /*
      if (db->findBlock("shsolver"))
	db->require("shsolver", d_shsolver);
      else
	d_shsolver = false;
      */
      if (db->findBlock("max_iter"))
	db->require("max_iter", d_maxSweeps);
      else
	d_maxSweeps = 75;
      if (db->findBlock("ksptype"))
	db->require("ksptype",d_kspType);
      else
	d_kspType = "gmres";
      if (db->findBlock("res_tol"))
	db->require("res_tol",d_tolerance);
      else
	d_tolerance = 1.e-08;
      if (db->findBlock("pctype"))
	db->require("pctype", d_pcType);
      else
	d_pcType = "blockjacobi";
      if (d_pcType == "asm")
	db->require("overlap",d_overlap);
      if (d_pcType == "ilu")
	db->require("fill",d_fill);
    }
    else {
      d_maxSweeps = 75;
      d_kspType = "gmres";
      d_pcType = "blockjacobi";
      d_tolerance = 1.0e-08;
    }
  }
  else  {
    d_maxSweeps = 75;
    d_kspType = "gmres";
    d_pcType = "blockjacobi";
    d_tolerance = 1.0e-08;
  }
  int argc = 4;
  char** argv;
  argv = new char*[argc];
  argv[0] = "RadLinearSolver::problemSetup";
  argv[1] = "-no_signal_handler";
  argv[2] = "-log_exclude_actions";
  argv[3] = "-log_exclude_objects";
  int ierr = PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  if(ierr)
    throw PetscError(ierr, "PetscInitialize", __FILE__, __LINE__);
}


void 
RadLinearSolver::matrixCreate(const PatchSet* allpatches,
			  const PatchSubset* mypatches)
{
  // for global index get a petsc index that
  // make it a data memeber
  int numProcessors = d_myworld->size();
  ASSERTEQ(numProcessors, allpatches->size());

  // number of patches for each processor
  vector<int> numCells(numProcessors, 0);
  vector<int> startIndex(numProcessors);
  int totalCells = 0;
  for(int s=0;s<allpatches->size();s++){
    startIndex[s]=totalCells;
    int mytotal = 0;
    const PatchSubset* patches = allpatches->getSubset(s);
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      IntVector plowIndex = patch->getCellFORTLowIndex();
      IntVector phighIndex = patch->getCellFORTHighIndex()+IntVector(1,1,1);
  
      long nc = (phighIndex[0]-plowIndex[0])*
	(phighIndex[1]-plowIndex[1])*
	(phighIndex[2]-plowIndex[2]);
      d_petscGlobalStart[patch]=totalCells;
      totalCells+=nc;
      mytotal+=nc;
    }
    numCells[s] = mytotal;
  }

  for(int p=0;p<mypatches->size();p++){
    const Patch* patch=mypatches->get(p);
    IntVector lowIndex = patch->getGhostCellLowIndex(Arches::ONEGHOSTCELL);
    IntVector highIndex = patch->getGhostCellHighIndex(Arches::ONEGHOSTCELL);
    Array3<int> l2g(lowIndex, highIndex);
    l2g.initialize(-1234);
    long totalCells=0;
    const Level* level = patch->getLevel();
    Patch::selectType neighbors;
    level->selectPatches(lowIndex, highIndex, neighbors);
    for(int i=0;i<neighbors.size();i++){
      const Patch* neighbor = neighbors[i];

      IntVector plow = neighbor->getCellFORTLowIndex();
      IntVector phigh = neighbor->getCellFORTHighIndex()+IntVector(1,1,1);
      IntVector low = Max(lowIndex, plow);
      IntVector high= Min(highIndex, phigh);

      if( ( high.x() < low.x() ) || ( high.y() < low.y() ) 
	  || ( high.z() < low.z() ) )
	throw InternalError("Patch doesn't overlap?", __FILE__, __LINE__);
      
      int petscglobalIndex = d_petscGlobalStart[neighbor];
      IntVector dcells = phigh-plow;
      IntVector start = low-plow;
      petscglobalIndex += start.z()*dcells.x()*dcells.y()
	+start.y()*dcells.x()+start.x();
      for (int colZ = low.z(); colZ < high.z(); colZ ++) {
	int idx_slab = petscglobalIndex;
	petscglobalIndex += dcells.x()*dcells.y();
	
	for (int colY = low.y(); colY < high.y(); colY ++) {
	  int idx = idx_slab;
	  idx_slab += dcells.x();
	  for (int colX = low.x(); colX < high.x(); colX ++) {
	    l2g[IntVector(colX, colY, colZ)] = idx++;
	  }
	}
      }
      IntVector d = high-low;
      totalCells+=d.x()*d.y()*d.z();
    }
    d_petscLocalToGlobal[patch].copyPointer(l2g);
  }
  int me = d_myworld->myrank();
  numlrows = numCells[me];
  numlcolumns = numlrows;
  globalrows = (int)totalCells;
  globalcolumns = (int)totalCells;
  /*
  if(d_shsolver){

  d_nz = 7;
  o_nz = 6;
  }
  else {
  */
  d_nz = 4;
  o_nz = 3;
  //  }

}

// ****************************************************************************
// Fill linear parallel matrix
// ****************************************************************************
void 
RadLinearSolver::setMatrix(const ProcessorGroup* ,
			   const Patch* patch,
			   ArchesVariables* vars,
			   bool plusX, bool plusY, bool plusZ,
			   CCVariable<double>& SU,
			   CCVariable<double>& AB,
			   CCVariable<double>& AS,
			   CCVariable<double>& AW,
			   CCVariable<double>& AP,
			   CCVariable<double>& AE,
			   CCVariable<double>& AN,
			   CCVariable<double>& AT)

{
  int ierr;
  ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD, numlrows, numlcolumns, globalrows,
			     globalcolumns, d_nz, PETSC_NULL, o_nz, PETSC_NULL, &A);
  if(ierr)
    throw PetscError(ierr, "MatCreateMPIAIJ", __FILE__, __LINE__);

  /* 
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
  */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,numlrows, globalrows,&d_x);
  if(ierr)
    throw PetscError(ierr, "VecCreateMPI", __FILE__, __LINE__);
  ierr = VecSetFromOptions(d_x);
  if(ierr)
    throw PetscError(ierr, "VecSetFromOptions", __FILE__, __LINE__);
  ierr = VecDuplicate(d_x,&d_b);
  if(ierr)
    throw PetscError(ierr, "VecDuplicate(d_b)", __FILE__, __LINE__);
  ierr = VecDuplicate(d_x,&d_u);
  if(ierr)
    throw PetscError(ierr, "VecDuplicate(d_u)", __FILE__, __LINE__);

  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
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

  int col[4];
  double value[4];
  /*
  if(d_shsolver){
  int col[7];
  double value[7];
  }
  */
  // fill matrix for internal patches
  // make sure that sizeof(d_petscIndex) is the last patch, i.e., appears last in the
  // petsc matrix
  IntVector lowIndex = patch->getGhostCellLowIndex(Arches::ONEGHOSTCELL);
  IntVector highIndex = patch->getGhostCellHighIndex(Arches::ONEGHOSTCELL);

  Array3<int> l2g(lowIndex, highIndex);
  l2g.copy(d_petscLocalToGlobal[patch]);
  MatZeroEntries(A);
  double vecvalueb, vecvaluex;
  int facX = 1;
  if (plusX)
    facX = -1;
  int facY = 1;
  if (plusY)
    facY = -1;
  int facZ = 1;
  if (plusZ)
    facZ = -1;
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	int ii = colX+facX;
	int jj = colY+facY;
	int kk = colZ+facZ;
	/*
	if(d_shsolver){
	col[0] = l2g[IntVector(colX,colY,colZ-1)];  //ab
	col[1] = l2g[IntVector(colX, colY-1, colZ)]; // as
	col[2] = l2g[IntVector(colX-1, colY, colZ)]; // aw
	col[3] = l2g[IntVector(colX, colY, colZ)]; //ap
	col[4] = l2g[IntVector(colX+1, colY, colZ)]; // ae
	col[5] = l2g[IntVector(colX, colY+1, colZ)]; // an
	col[6] = l2g[IntVector(colX, colY, colZ+1)]; // at
	}
	else {
	*/
	col[0] = l2g[IntVector(colX,colY,kk)];  //ab
	col[1] = l2g[IntVector(colX, jj, colZ)]; // as
	col[2] = l2g[IntVector(ii, colY, colZ)]; // aw
	col[3] = l2g[IntVector(colX, colY, colZ)]; //ap
	//	}

	//#ifdef ARCHES_PETSC_DEBUG
	/*
	if(d_shsolver){
	value[0] = -AB[IntVector(colX,colY,colZ)];
	value[1] = -AS[IntVector(colX,colY,colZ)];
	value[2] = -AW[IntVector(colX,colY,colZ)];
	value[3] = AP[IntVector(colX,colY,colZ)];
	value[4] = -AE[IntVector(colX,colY,colZ)];
	value[5] = -AN[IntVector(colX,colY,colZ)];
	value[6] = -AT[IntVector(colX,colY,colZ)];
	}
	else{
	*/
	value[0] = -AB[IntVector(colX,colY,colZ)];
	value[1] = -AS[IntVector(colX,colY,colZ)];
	value[2] = -AW[IntVector(colX,colY,colZ)];
	value[3] = AP[IntVector(colX,colY,colZ)];
	//	}

	int row = col[3];
	/*
	if(d_shsolver){
	ierr = MatSetValues(A,1,&row,7,col,value,INSERT_VALUES);
	}
	else{
	*/
	ierr = MatSetValues(A,1,&row,4,col,value,INSERT_VALUES);
	//	}

	if(ierr)
	  throw PetscError(ierr, "MatSetValues", __FILE__, __LINE__);
	vecvalueb = SU[IntVector(colX,colY,colZ)];
	vecvaluex = vars->cenint[IntVector(colX, colY, colZ)];
	ierr = VecSetValue(d_b, row, vecvalueb, INSERT_VALUES);
	if(ierr)
	  throw PetscError(ierr, "VecSetValue", __FILE__, __LINE__);
	ierr = VecSetValue(d_x, row, vecvaluex, INSERT_VALUES);
	if(ierr)
	  throw PetscError(ierr, "VecSetValue", __FILE__, __LINE__);

      }
    }
  }
}


bool
RadLinearSolver::radLinearSolve()
{
  double solve_start = Time::currentSeconds();
  KSP solver;
  PC peqnpc; // pressure eqn pc
 
  int ierr;
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  if(ierr)
    throw PetscError(ierr, "MatAssemblyBegin", __FILE__, __LINE__);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
  if(ierr)
    throw PetscError(ierr, "MatAssemblyEnd", __FILE__, __LINE__);
  ierr = VecAssemblyBegin(d_b);
  if(ierr)
    throw PetscError(ierr, "VecAssemblyBegin", __FILE__, __LINE__);
  ierr = VecAssemblyEnd(d_b);
  if(ierr)
    throw PetscError(ierr, "VecAssemblyEnd", __FILE__, __LINE__);
  ierr = VecAssemblyBegin(d_x);
  if(ierr)
    throw PetscError(ierr, "VecAssemblyBegin", __FILE__, __LINE__);
  ierr = VecAssemblyEnd(d_x);
  if(ierr)
    throw PetscError(ierr, "VecAssemblyEnd", __FILE__, __LINE__);
  // compute the initial error
  double neg_one = -1.0;
  double init_norm;
  double sum_b;
  ierr = VecSum(d_b, &sum_b);
  Vec u_tmp;
  ierr = VecDuplicate(d_x,&u_tmp);
  if(ierr)
    throw PetscError(ierr, "VecDuplicate", __FILE__, __LINE__);
  ierr = MatMult(A, d_x, u_tmp);
  if(ierr)
    throw PetscError(ierr, "MatMult", __FILE__, __LINE__);
#if (PETSC_VERSION_MINOR == 2)
  ierr = VecAXPY(&neg_one, d_b, u_tmp);
#endif
#if (PETSC_VERSION_MINOR == 3)
  ierr = VecAXPY(u_tmp,neg_one,d_b);
#endif
  if(ierr)
    throw PetscError(ierr, "VecAXPY", __FILE__, __LINE__);
  ierr  = VecNorm(u_tmp,NORM_2,&init_norm);
#if 0
  cerr << "initnorm" << init_norm << endl;
#endif
  if(ierr)
    throw PetscError(ierr, "VecNorm", __FILE__, __LINE__);
  ierr = VecDestroy(u_tmp);
  if(ierr)
    throw PetscError(ierr, "VecDestroy", __FILE__, __LINE__);
  /* debugging - steve */
  double norm;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = KSPCreate(PETSC_COMM_WORLD,&solver);

  if(ierr)
    throw PetscError(ierr, "KSPCreate", __FILE__, __LINE__);
  ierr = KSPSetOperators(solver,A,A,DIFFERENT_NONZERO_PATTERN);
  if(ierr)
    throw PetscError(ierr, "KSPSetOperators", __FILE__, __LINE__);

  ierr = KSPGetPC(solver, &peqnpc);
  if(ierr)
    throw PetscError(ierr, "KSPGetPC", __FILE__, __LINE__);
  if (d_pcType == "jacobi") {
    ierr = PCSetType(peqnpc, PCJACOBI);
    if(ierr)
      throw PetscError(ierr, "PCSetType", __FILE__, __LINE__);
  }
  else if (d_pcType == "asm") {
    ierr = PCSetType(peqnpc, PCASM);
    if(ierr)
      throw PetscError(ierr, "PCSetType", __FILE__, __LINE__);
    ierr = PCASMSetOverlap(peqnpc, d_overlap);
    if(ierr)
      throw PetscError(ierr, "PCASMSetOverlap", __FILE__, __LINE__);
  }
  else if (d_pcType == "ilu") {
    ierr = PCSetType(peqnpc, PCILU);
    if(ierr)
      throw PetscError(ierr, "PCSetType", __FILE__, __LINE__);
#if (PETSC_VERSION_MINOR == 3 && PETSC_VERSION_SUBMINOR >= 1) // 2.3.1
    ierr = PCFactorSetFill(peqnpc, d_fill);
    if(ierr)
      throw PetscError(ierr, "PCFactorSetFill", __FILE__, __LINE__);
#else
    ierr = PCILUSetFill(peqnpc, d_fill);
    if(ierr)
      throw PetscError(ierr, "PCILUSetFill", __FILE__, __LINE__);
#endif
  }
  else {
    ierr = PCSetType(peqnpc, PCBJACOBI);
    if(ierr)
      throw PetscError(ierr, "PCSetType", __FILE__, __LINE__);
  }
  if (d_kspType == "cg") {
    ierr = KSPSetType(solver, KSPCG);
    if(ierr)
      throw PetscError(ierr, "KSPSetType", __FILE__, __LINE__);
  }
  else {
    ierr = KSPSetType(solver, KSPGMRES);
    if(ierr)
      throw PetscError(ierr, "KSPSetType", __FILE__, __LINE__);
  }
  ierr = KSPSetTolerances(solver,1.0e-50, d_tolerance, PETSC_DEFAULT, PETSC_DEFAULT);
  if(ierr)
    throw PetscError(ierr, "KSPSetTolerances", __FILE__, __LINE__);

  ierr = KSPSetInitialGuessNonzero(solver, PETSC_TRUE);
  if(ierr)
    throw PetscError(ierr, "KSPSetInitialGuessNonzero", __FILE__, __LINE__);
  
  ierr = KSPSetFromOptions(solver);
  if(ierr)
    throw PetscError(ierr, "KSPSetFromOptions", __FILE__, __LINE__);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  int its;
  ierr = KSPSolve(solver,d_b,d_x);
  if(ierr)
    throw PetscError(ierr, "KSPSolve", __FILE__, __LINE__);
  ierr = KSPGetIterationNumber(solver,&its);
  if (ierr)
    throw PetscError(ierr, "KSPGetIterationNumber", __FILE__, __LINE__);
  int me = d_myworld->myrank();

  ierr = VecNorm(d_x,NORM_1,&norm);
  if(ierr)
    throw PetscError(ierr, "VecNorm", __FILE__, __LINE__);

  // check the error
  ierr = MatMult(A, d_x, d_u);
  if(ierr)
    throw PetscError(ierr, "MatMult", __FILE__, __LINE__);
#if (PETSC_VERSION_MINOR == 2)
  ierr = VecAXPY(&neg_one, d_b, d_u);
#endif
#if (PETSC_VERSION_MINOR == 3)
  ierr = VecAXPY(d_u,neg_one, d_b);
#endif
  if(ierr)
    throw PetscError(ierr, "VecAXPY", __FILE__, __LINE__);
  ierr  = VecNorm(d_u,NORM_2,&norm);
  if(ierr)
    throw PetscError(ierr, "VecNorm", __FILE__, __LINE__);

  if(me == 0) {
     cerr << "KSPSolve: Norm of error: " << norm << ", iterations: " << its << ", time: " << Time::currentSeconds()-solve_start << " seconds\n";
     cerr << "Init Norm: " << init_norm << " Error reduced by: " << norm/(init_norm+1.0e-20) << endl;
     cerr << "Sum of RHS vector: " << sum_b << endl;
  }
#if 1
  ierr = KSPDestroy(solver);
  if(ierr)
    throw PetscError(ierr, "KSPDestroy", __FILE__, __LINE__);
#endif
#if 0
  ierr = VecDestroy(d_u);
  if(ierr)
    throw PetscError(ierr, "VecDestroy", __FILE__, __LINE__);
  ierr = VecDestroy(d_b);
  if(ierr)
    throw PetscError(ierr, "VecDestroy", __FILE__, __LINE__);
  ierr = VecDestroy(d_x);
  if(ierr)
    throw PetscError(ierr, "VecDestroy", __FILE__, __LINE__);
  ierr = MatDestroy(A);
  if(ierr)
    throw PetscError(ierr, "MatDestroy", __FILE__, __LINE__);

#endif
  if ((norm/(init_norm+1.0e-20) < 1.0)&& (norm < 2.0))
    return true;
  else
    return false;
}


void
RadLinearSolver::copyRadSoln(const Patch* patch, ArchesVariables* vars)
{
  // copy solution vector back into the array
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  double* xvec;
  int ierr;
  ierr = VecGetArray(d_x, &xvec);
  if(ierr)
    throw PetscError(ierr, "VecGetArray", __FILE__, __LINE__);
  Array3<int> l2g = d_petscLocalToGlobal[patch];
  int rowinit = l2g[IntVector(idxLo.x(), idxLo.y(), idxLo.z())]; 
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	int row = l2g[IntVector(colX, colY, colZ)]-rowinit;
	vars->cenint[IntVector(colX, colY, colZ)] = xvec[row];
      }
    }
  }
#if 0
  cerr << "Print computed pressure" << endl;
  vars->pressure.print(cerr);
#endif
  ierr = VecRestoreArray(d_x, &xvec);
  if(ierr)
    throw PetscError(ierr, "VecRestoreArray", __FILE__, __LINE__);
}
  
void
RadLinearSolver::destroyMatrix() 
{
  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
#if 1
  int ierr;
  ierr = VecDestroy(d_u);
  if(ierr)
    throw PetscError(ierr, "VecDestroy", __FILE__, __LINE__);
  ierr = VecDestroy(d_b);
  if(ierr)
    throw PetscError(ierr, "VecDestroy", __FILE__, __LINE__);
  ierr = VecDestroy(d_x);
  if(ierr)
    throw PetscError(ierr, "VecDestroy", __FILE__, __LINE__);
  ierr = MatDestroy(A);
  if(ierr)
    throw PetscError(ierr, "MatDestroy", __FILE__, __LINE__);
#endif
}


// Shutdown PETSc
void RadLinearSolver::finalizeSolver()
{
// The following is to enable PETSc memory logging
//  int ierrd = PetscTrDump(NULL);
//  if(ierrd)
//    throw PetscError(ierrd, "PetscTrDump");
  int ierr = PetscFinalize();
  if(ierr)
    throw PetscError(ierr, "PetscFinalize", __FILE__, __LINE__);
}













