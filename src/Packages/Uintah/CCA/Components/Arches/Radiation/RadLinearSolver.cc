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
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#undef CHKERRQ
#define CHKERRQ(x) if(x) throw PetscError(x, __FILE__);
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
      if (db->findBlock("underelax"))
	db->require("underrelax", d_underrelax);
      else
	d_underrelax = 1.0;
      
      if (db->findBlock("max_iter"))
	db->require("max_iter", d_maxSweeps);
      else
	d_maxSweeps = 75;
      if (db->findBlock("ksptype"))
	db->require("ksptype",d_kspType);
      else
	d_kspType = "gmres";
      if (db->findBlock("tolerance"))
	db->require("tolerance",d_tolerance);
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
      d_underrelax = 1.0;
      d_maxSweeps = 75;
      d_kspType = "gmres";
      d_pcType = "blockjacobi";
      d_tolerance = 1.0e-08;
    }
  }
  else  {
    d_underrelax = 1.0;
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
    throw PetscError(ierr, "PetscInitialize");
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
#ifdef ARCHES_PETSC_DEBUG
  cerr << "totalCells = " << totalCells << '\n';
#endif

  for(int p=0;p<mypatches->size();p++){
    const Patch* patch=mypatches->get(p);
    IntVector lowIndex = patch->getGhostCellLowIndex(Arches::ONEGHOSTCELL);
    IntVector highIndex = patch->getGhostCellHighIndex(Arches::ONEGHOSTCELL);
    Array3<int> l2g(lowIndex, highIndex);
    l2g.initialize(-1234);
    long totalCells=0;
    const Level* level = patch->getLevel();
    Level::selectType neighbors;
    level->selectPatches(lowIndex, highIndex, neighbors);
    for(int i=0;i<neighbors.size();i++){
      const Patch* neighbor = neighbors[i];

      IntVector plow = neighbor->getCellFORTLowIndex();
      IntVector phigh = neighbor->getCellFORTHighIndex()+IntVector(1,1,1);
      IntVector low = Max(lowIndex, plow);
      IntVector high= Min(highIndex, phigh);

      if( ( high.x() < low.x() ) || ( high.y() < low.y() ) 
	  || ( high.z() < low.z() ) )
	throw InternalError("Patch doesn't overlap?");
      
      int petscglobalIndex = d_petscGlobalStart[neighbor];
      IntVector dcells = phigh-plow;
      IntVector start = low-plow;
      petscglobalIndex += start.z()*dcells.x()*dcells.y()
	+start.y()*dcells.x()+start.x();
      //#ifdef ARCHES_PETSC_DEBUG
#if 0
      cerr << "Looking at patch: " << neighbor->getID() << '\n';
      cerr << "low=" << low << '\n';
      cerr << "high=" << high << '\n';
      cerr << "start at: " << d_petscGlobalStart[neighbor] << '\n';
      cerr << "globalIndex = " << petscglobalIndex << '\n';
#endif
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
    //#ifdef ARCHES_PETSC_DEBUG
#if 0
    {	
      IntVector l = l2g.getWindow()->getLowIndex();
      IntVector h = l2g.getWindow()->getHighIndex();
      for(int z=l.z();z<h.z();z++){
	for(int y=l.y();y<h.y();y++){
	  for(int x=l.x();x<h.x();x++){
	    IntVector idx(x,y,z);
	    cerr << "l2g" << idx << "=" << l2g[idx] << '\n';
	  }
	}
      }
    }
#endif
  }
  int me = d_myworld->myrank();
  numlrows = numCells[me];
  numlcolumns = numlrows;
  globalrows = (int)totalCells;
  globalcolumns = (int)totalCells;
  d_nz = 4;
  o_nz = 3;
  // #ifdef ARCHES_PETSC_DEBUG
#if 0
  cerr << "matrixCreate: local size: " << numlrows << ", " << numlcolumns << ", global size: " << globalrows << ", " << globalcolumns << "\n";
#endif
#if 0
  int ierr;
  ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD, numlrows, numlcolumns, globalrows,
			     globalcolumns, d_nz, PETSC_NULL, o_nz, PETSC_NULL, &A);
  if(ierr)
    throw PetscError(ierr, "MatCreateMPIAIJ");

  /* 
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
  */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,numlrows, globalrows,&d_x);
  if(ierr)
    throw PetscError(ierr, "VecCreateMPI");
  ierr = VecSetFromOptions(d_x);
  if(ierr)
    throw PetscError(ierr, "VecSetFromOptions");
  ierr = VecDuplicate(d_x,&d_b);
  if(ierr)
    throw PetscError(ierr, "VecDuplicate(d_b)");
  ierr = VecDuplicate(d_x,&d_u);
  if(ierr)
    throw PetscError(ierr, "VecDuplicate(d_u)");
#endif
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
			   CCVariable<double>& AP)
{
  int ierr;
  ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD, numlrows, numlcolumns, globalrows,
			     globalcolumns, d_nz, PETSC_NULL, o_nz, PETSC_NULL, &A);
  if(ierr)
    throw PetscError(ierr, "MatCreateMPIAIJ");

  /* 
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
  */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,numlrows, globalrows,&d_x);
  if(ierr)
    throw PetscError(ierr, "VecCreateMPI");
  ierr = VecSetFromOptions(d_x);
  if(ierr)
    throw PetscError(ierr, "VecSetFromOptions");
  ierr = VecDuplicate(d_x,&d_b);
  if(ierr)
    throw PetscError(ierr, "VecDuplicate(d_b)");
  ierr = VecDuplicate(d_x,&d_u);
  if(ierr)
    throw PetscError(ierr, "VecDuplicate(d_u)");

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
	col[0] = l2g[IntVector(colX,colY,kk)];  //ab
	col[1] = l2g[IntVector(colX, jj, colZ)]; // as
	col[2] = l2g[IntVector(ii, colY, colZ)]; // aw
	col[3] = l2g[IntVector(colX, colY, colZ)]; //ap
	//#ifdef ARCHES_PETSC_DEBUG
	value[0] = -AB[IntVector(colX,colY,colZ)];
	value[1] = -AS[IntVector(colX,colY,colZ)];
	value[2] = -AW[IntVector(colX,colY,colZ)];
	value[3] = AP[IntVector(colX,colY,colZ)];
	int row = col[3];
	ierr = MatSetValues(A,1,&row,4,col,value,INSERT_VALUES);
	if(ierr)
	  throw PetscError(ierr, "MatSetValues");
	vecvalueb = SU[IntVector(colX,colY,colZ)];
	vecvaluex = vars->cenint[IntVector(colX, colY, colZ)];
	ierr = VecSetValue(d_b, row, vecvalueb, INSERT_VALUES);
	if(ierr)
	  throw PetscError(ierr, "VecSetValue");
	ierr = VecSetValue(d_x, row, vecvaluex, INSERT_VALUES);
	if(ierr)
	  throw PetscError(ierr, "VecSetValue");

#ifdef ARCHES_PETSC_DEBUG
	cerr << "ierr=" << ierr << '\n';
#endif
      }
    }
  }
#ifdef ARCHES_PETSC_DEBUG
  cerr << "assemblign rhs\n";
#endif
  // assemble right hand side and solution vector
#if 0
  int numCells = (idxHi.x()-idxLo.x()+1)*(idxHi.y()-idxLo.y()+1)*
    (idxHi.z()-idxLo.z()+1);
  vector<double> vecb(numCells);
  vector<double> vecx(numCells);
  vector<int> indexes(numCells);
  int count = 0;
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	vecb[count] = SU[IntVector(colX,colY,colZ)];
	vecx[count] = vars->cenint[IntVector(colX, colY, colZ)];
	indexes[count] = l2g[IntVector(colX, colY, colZ)];	  
	count++;
      }
    }
  }

  ierr = VecSetValues(d_b, numCells, &indexes[0], &vecb[0], INSERT_VALUES);
  if(ierr)
    throw PetscError(ierr, "VecSetValue");
  ierr = VecSetValues(d_x, numCells, &indexes[0], &vecx[0], INSERT_VALUES);
  if(ierr)
    throw PetscError(ierr, "VecSetValue");
#endif
#if 0
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
	for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	  vecvalueb = SU[IntVector(colX,colY,colZ)];
	  vecvaluex = vars->cenint[IntVector(colX, colY, colZ)];
	  int row = l2g[IntVector(colX, colY, colZ)];	  
	  ierr = VecSetValue(d_b, row, vecvalueb, INSERT_VALUES);
	  if(ierr)
	    throw PetscError(ierr, "VecSetValue");
	  ierr = VecSetValue(d_x, row, vecvaluex, INSERT_VALUES);
	  if(ierr)
	    throw PetscError(ierr, "VecSetValue");
	}
      }
    }
#endif
    //#ifdef ARCHES_PETSC_DEBUG
#if 0
    cerr << " all done\n";
#endif
}


bool
RadLinearSolver::radLinearSolve()
{
  double solve_start = Time::currentSeconds();
  KSP ksp;
  PC peqnpc; // pressure eqn pc
 
  int ierr;
#ifdef ARCHES_PETSC_DEBUG
  cerr << "Doing mat/vec assembly\n";
#endif
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  if(ierr)
    throw PetscError(ierr, "MatAssemblyBegin");
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
  if(ierr)
    throw PetscError(ierr, "MatAssemblyEnd");
  ierr = VecAssemblyBegin(d_b);
  if(ierr)
    throw PetscError(ierr, "VecAssemblyBegin");
  ierr = VecAssemblyEnd(d_b);
  if(ierr)
    throw PetscError(ierr, "VecAssemblyEnd");
  ierr = VecAssemblyBegin(d_x);
  if(ierr)
    throw PetscError(ierr, "VecAssemblyBegin");
  ierr = VecAssemblyEnd(d_x);
  if(ierr)
    throw PetscError(ierr, "VecAssemblyEnd");
  // compute the initial error
  double neg_one = -1.0;
  double init_norm;
  double sum_b;
  ierr = VecSum(d_b, &sum_b);
  Vec u_tmp;
  ierr = VecDuplicate(d_x,&u_tmp);
  if(ierr)
    throw PetscError(ierr, "VecDuplicate");
  ierr = MatMult(A, d_x, u_tmp);
  if(ierr)
    throw PetscError(ierr, "MatMult");
  ierr = VecAXPY(&neg_one, d_b, u_tmp);
  if(ierr)
    throw PetscError(ierr, "VecAXPY");
  ierr  = VecNorm(u_tmp,NORM_2,&init_norm);
#if 0
  cerr << "initnorm" << init_norm << endl;
#endif
  if(ierr)
    throw PetscError(ierr, "VecNorm");
  ierr = VecDestroy(u_tmp);
  if(ierr)
    throw PetscError(ierr, "VecDestroy");
  /* debugging - steve */
  double norm;
#if 0
  // #ifdef ARCHES_PETSC_DEBUG
  ierr = PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_DEFAULT);
  ierr = MatNorm(A,NORM_1,&norm);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"matrix A norm = %g\n",norm);
  //  ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);
  ierr = VecNorm(d_x,NORM_1,&norm);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"vector x norm = %g\n",norm);
  ierr = VecView(d_x, PETSC_VIEWER_STDOUT_WORLD);
  ierr = VecNorm(d_b,NORM_1,&norm);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"vector b norm = %g\n",norm);
  ierr = VecView(d_b, PETSC_VIEWER_STDOUT_WORLD);
#endif

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles);

  if(ierr)
    throw PetscError(ierr, "SLESCreate");
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);
  if(ierr)
    throw PetscError(ierr, "SLESSetOperators");
  ierr = SLESGetKSP(sles,&ksp);
  if(ierr)
    throw PetscError(ierr, "SLESGetKSP");
  ierr = SLESGetPC(sles, &peqnpc);
  if(ierr)
    throw PetscError(ierr, "SLESGetPC");
  if (d_pcType == "jacobi") {
    ierr = PCSetType(peqnpc, PCJACOBI);
    if(ierr)
      throw PetscError(ierr, "PCSetType");
  }
  else if (d_pcType == "asm") {
    ierr = PCSetType(peqnpc, PCASM);
    if(ierr)
      throw PetscError(ierr, "PCSetType");
    ierr = PCASMSetOverlap(peqnpc, d_overlap);
    if(ierr)
      throw PetscError(ierr, "PCASMSetOverlap");
  }
  else if (d_pcType == "ilu") {
    ierr = PCSetType(peqnpc, PCILU);
    if(ierr)
      throw PetscError(ierr, "PCSetType");
    ierr = PCILUSetFill(peqnpc, d_fill);
    if(ierr)
      throw PetscError(ierr, "PCILUSetFill");
  }
  else {
    ierr = PCSetType(peqnpc, PCBJACOBI);
    if(ierr)
      throw PetscError(ierr, "PCSetType");
  }
  if (d_kspType == "cg") {
    ierr = KSPSetType(ksp, KSPCG);
    if(ierr)
      throw PetscError(ierr, "KSPSetType");
  }
  else {
    ierr = KSPSetType(ksp, KSPGMRES);
    if(ierr)
      throw PetscError(ierr, "KSPSetType");
  }
  ierr = KSPSetTolerances(ksp, d_tolerance, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
  if(ierr)
    throw PetscError(ierr, "KSPSetTolerances");

  ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
  if(ierr)
    throw PetscError(ierr, "KSPSetInitialGuessNonzero");
  
  ierr = SLESSetFromOptions(sles);
  if(ierr)
    throw PetscError(ierr, "SLESSetFromOptions");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  int its;
  ierr = SLESSolve(sles,d_b,d_x,&its);
  if(ierr)
    throw PetscError(ierr, "SLESSolve");
  int me = d_myworld->myrank();

  ierr = VecNorm(d_x,NORM_1,&norm);
  if(ierr)
    throw PetscError(ierr, "VecNorm");
#ifdef ARCHES_PETSC_DEBUG
  ierr = VecView(d_x, VIEWER_STDOUT_WORLD);
#endif

  // check the error
  ierr = MatMult(A, d_x, d_u);
  if(ierr)
    throw PetscError(ierr, "MatMult");
  ierr = VecAXPY(&neg_one, d_b, d_u);
  if(ierr)
    throw PetscError(ierr, "VecAXPY");
  ierr  = VecNorm(d_u,NORM_2,&norm);
  if(ierr)
    throw PetscError(ierr, "VecNorm");

  if(me == 0) {
     cerr << "SLESSolve: Norm of error: " << norm << ", iterations: " << its << ", time: " << Time::currentSeconds()-solve_start << " seconds\n";
     cerr << "Init Norm: " << init_norm << " Error reduced by: " << norm/init_norm << endl;
     cerr << "Sum of RHS vector: " << sum_b << endl;
  }
#if 1
  ierr = SLESDestroy(sles);
  if(ierr)
    throw PetscError(ierr, "SLESDestroy");
#endif
#if 0
  ierr = VecDestroy(d_u);
  if(ierr)
    throw PetscError(ierr, "VecDestroy");
  ierr = VecDestroy(d_b);
  if(ierr)
    throw PetscError(ierr, "VecDestroy");
  ierr = VecDestroy(d_x);
  if(ierr)
    throw PetscError(ierr, "VecDestroy");
  ierr = MatDestroy(A);
  if(ierr)
    throw PetscError(ierr, "MatDestroy");

#endif
  if ((norm/init_norm < 1.0)&& (norm < 2.0))
    return true;
  else
    return true;
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
    throw PetscError(ierr, "VecGetArray");
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
    throw PetscError(ierr, "VecRestoreArray");
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
    throw PetscError(ierr, "VecDestroy");
  ierr = VecDestroy(d_b);
  if(ierr)
    throw PetscError(ierr, "VecDestroy");
  ierr = VecDestroy(d_x);
  if(ierr)
    throw PetscError(ierr, "VecDestroy");
  ierr = MatDestroy(A);
  if(ierr)
    throw PetscError(ierr, "MatDestroy");
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
    throw PetscError(ierr, "PetscFinalize");
}













