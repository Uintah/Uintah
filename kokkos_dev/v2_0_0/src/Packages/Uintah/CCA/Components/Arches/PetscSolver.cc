//----- PetscSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/debug.h>
#include <Packages/Uintah/CCA/Components/Arches/PetscSolver.h>
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
#undef CHKERRQ
#define CHKERRQ(x) if(x) throw PetscError(x, __FILE__);

using namespace std;
using namespace Uintah;
using namespace SCIRun;

#include <Packages/Uintah/CCA/Components/Arches/fortran/rescal_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/underelax_fort.h>

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
  ProblemSpecP db = params->findBlock("LinearSolver");
  db->require("max_iter", d_maxSweeps);
  //db->require("res_tol", d_residual);
  db->getWithDefault("res_tol", d_residual, 1.0e-7);
  db->require("underrelax", d_underrelax);
  db->require("pctype", d_pcType);
  if (d_pcType == "asm")
    db->require("overlap",d_overlap);
  if (d_pcType == "ilu")
    db->require("fill",d_fill);
  db->require("ksptype",d_kspType);
//  int argc = 2;
  int argc = 4;
  char** argv;
  argv = new char*[argc];
  argv[0] = "PetscSolver::problemSetup";
  //argv[1] = "-on_error_attach_debugger";
  argv[1] = "-no_signal_handler";
  argv[2] = "-log_exclude_actions";
  argv[3] = "-log_exclude_objects";
  int ierr = PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  if(ierr)
    throw PetscError(ierr, "PetscInitialize");
//  ierr = PetscOptionsSetValue("-log_exclude_actions", "");
//  if(ierr)
//    throw PetscError(ierr, "PetscExcludeActions");
//  ierr = PetscOptionsSetValue("-log_exclude_objects", "");
//  if(ierr)
//    throw PetscError(ierr, "PetscExcludeObjects");
}


// ****************************************************************************
// Actual compute of pressure residual
// ****************************************************************************
void 
PetscSolver::computePressResidual(const ProcessorGroup*,
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
// Actual calculation of order of magnitude term for pressure equation
// ****************************************************************************
void 
PetscSolver::computePressOrderOfMagnitude(const ProcessorGroup* ,
				const Patch* ,
				DataWarehouseP& ,
				DataWarehouseP& , ArchesVariables* )
{

//&vars->truncPress

}

void 
PetscSolver::matrixCreate(const PatchSet* allpatches,
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
#ifdef ARCHES_PETSC_DEBUG
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
#ifdef ARCHES_PETSC_DEBUG
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
  int numlrows = numCells[me];
  int numlcolumns = numlrows;
  int globalrows = (int)totalCells;
  int globalcolumns = (int)totalCells;
  int d_nz = 7;
  int o_nz = 6;
#ifdef ARCHES_PETSC_DEBUG
  cerr << "matrixCreate: local size: " << numlrows << ", " << numlcolumns << ", global size: " << globalrows << ", " << globalcolumns << "\n";
#endif
  int ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD, numlrows, numlcolumns, globalrows,
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
}

// ****************************************************************************
// Actual compute of pressure underrelaxation
// ****************************************************************************
void 
PetscSolver::computePressUnderrelax(const ProcessorGroup*,
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
PetscSolver::setPressMatrix(const ProcessorGroup* ,
			    const Patch* patch,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars,
			    const ArchesLabel*)
{
  double solve_start = Time::currentSeconds();

#ifdef ARCHES_PETSC_DEBUG
   cerr << "in setPressMatrix on patch: " << patch->getID() << '\n';
#endif
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
  int ierr;
  int col[7];
  double value[7];
  // fill matrix for internal patches
  // make sure that sizeof(d_petscIndex) is the last patch, i.e., appears last in the
  // petsc matrix
  IntVector lowIndex = patch->getGhostCellLowIndex(Arches::ONEGHOSTCELL);
  IntVector highIndex = patch->getGhostCellHighIndex(Arches::ONEGHOSTCELL);

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
#ifdef ARCHES_PETSC_DEBUG
	  cerr << "filling in row: " << col[3] << '\n';
#endif
	  value[0] = -constvars->pressCoeff[Arches::AB][IntVector(colX,colY,colZ)];
	  value[1] = -constvars->pressCoeff[Arches::AS][IntVector(colX,colY,colZ)];
	  value[2] = -constvars->pressCoeff[Arches::AW][IntVector(colX,colY,colZ)];
	  value[3] = constvars->pressCoeff[Arches::AP][IntVector(colX,colY,colZ)];
	  value[4] = -constvars->pressCoeff[Arches::AE][IntVector(colX,colY,colZ)];
	  value[5] = -constvars->pressCoeff[Arches::AN][IntVector(colX,colY,colZ)];
	  value[6] = -constvars->pressCoeff[Arches::AT][IntVector(colX,colY,colZ)];
#ifdef ARCHES_PETSC_DEBUG
	  for(int i=0;i<7;i++)
	     cerr << "A[" << col[3] << "][" << col[i] << "]=" << value[i] << '\n';
#endif
	  int row = col[3];
	  ierr = MatSetValues(A,1,&row,7,col,value,INSERT_VALUES);
	  if(ierr)
	    throw PetscError(ierr, "MatSetValues");
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
  double vecvalueb, vecvaluex;
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
	for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	  vecvalueb = constvars->pressNonlinearSrc[IntVector(colX,colY,colZ)];
	  vecvaluex = vars->pressure[IntVector(colX, colY, colZ)];
	  int row = l2g[IntVector(colX, colY, colZ)];	  
//	  VecSetValue(d_b, row, vecvalueb, INSERT_VALUES);
	  ierr = VecSetValue(d_b, row, vecvalueb, INSERT_VALUES);
	  if(ierr)
	    throw PetscError(ierr, "VecSetValue");
//	  VecSetValue(d_x, row, vecvaluex, INSERT_VALUES);
	  ierr = VecSetValue(d_x, row, vecvaluex, INSERT_VALUES);
	  if(ierr)
	    throw PetscError(ierr, "VecSetValue");
	}
      }
    }
    int me = d_myworld->myrank();
    if(me == 0) {
     cerr << "Time in PETSC Assemble: " << Time::currentSeconds()-solve_start << " seconds\n";
    }
#ifdef ARCHES_PETSC_DEBUG
    cerr << " all done\n";
#endif
}


bool
PetscSolver::pressLinearSolve()
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
  if(ierr)
    throw PetscError(ierr, "VecNorm");
  ierr = VecDestroy(u_tmp);
  if(ierr)
    throw PetscError(ierr, "VecDestroy");
  /* debugging - steve */
  double norm;
#if 0
  // #ifdef ARCHES_PETSC_DEBUG
  ierr = ViewerSetFormat(VIEWER_STDOUT_WORLD, VIEWER_FORMAT_ASCII_DEFAULT, 0);
  ierr = MatNorm(A,NORM_1,&norm);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"matrix A norm = %g\n",norm);
  ierr = MatView(A, VIEWER_STDOUT_WORLD);
  ierr = VecNorm(d_x,NORM_1,&norm);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"vector x norm = %g\n",norm);
  ierr = VecView(d_x, VIEWER_STDOUT_WORLD);
  ierr = VecNorm(d_b,NORM_1,&norm);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"vector b norm = %g\n",norm);
  ierr = VecView(d_b, VIEWER_STDOUT_WORLD);
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
  ierr = KSPSetTolerances(ksp, d_residual, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
  if(ierr)
    throw PetscError(ierr, "KSPSetTolerances");

  // set null space for preconditioner
  // change for a newer version
#ifdef NULL_MATRIX
  PCNullSpace nullsp;
  ierr = PCNullSpaceCreate(PETSC_COMM_WORLD, 1, 0, PETSC_NULL, &nullsp); 
  ierr = PCNullSpaceAttach(peqnpc, nullsp);
  ierr = PCNullSpaceDestroy(nullsp);
#endif
  ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
  // ierr = KSPSetInitialGuessNonzero(ksp);
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
     cerr << "SLESSolve: Norm of error: " << norm << ", iterations: " << its << ", solver time: " << Time::currentSeconds()-solve_start << " seconds\n";
     cerr << "Init Norm: " << init_norm << " Error reduced by: " << norm/(init_norm+1.0e-20) << endl;
     cerr << "Sum of RHS vector: " << sum_b << endl;
  }
  if ((norm/(init_norm+1.0e-20) < 1.0)&& (norm < 2.0))
    return true;
  else
    return false;
}


void
PetscSolver::copyPressSoln(const Patch* patch, ArchesVariables* vars)
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
    throw PetscError(ierr, "VecRestoreArray");
}
  
void
PetscSolver::destroyMatrix() 
{
  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  int ierr;
  ierr = SLESDestroy(sles);
  if(ierr)
    throw PetscError(ierr, "SLESDestroy");
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
}

// ****************************************************************************
// Actual linear solve for pressure
// ****************************************************************************
void 
PetscSolver::pressLisolve(const ProcessorGroup*,
			 const Patch*,
			 DataWarehouseP&,
			 DataWarehouseP&,
			 ArchesVariables*,
			 const ArchesLabel*)
{

}


// Shutdown PETSc
void PetscSolver::finalizeSolver()
{
// The following is to enable PETSc memory logging
//  int ierrd = PetscTrDump(NULL);
//  if(ierrd)
//    throw PetscError(ierrd, "PetscTrDump");
  int ierr = PetscFinalize();
  if(ierr)
    throw PetscError(ierr, "PetscFinalize");
}

//****************************************************************************
// Actual compute of Velocity residual
//****************************************************************************

void 
PetscSolver::computeVelResidual(const ProcessorGroup* ,
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
PetscSolver::computeVelOrderOfMagnitude(const ProcessorGroup* ,
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
PetscSolver::computeVelUnderrelax(const ProcessorGroup* ,
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
PetscSolver::velocityLisolve(const ProcessorGroup*,
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
#ifdef ARCHES_PETSC_DEBUG
    cerr << "After u Velocity solve " << velIter << " " << velResid << endl;
    cerr << "After u Velocity solve " << nlResid << " " << trunc_conv <<  endl;
#endif
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
PetscSolver::computeScalarResidual(const ProcessorGroup* ,
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
PetscSolver::computeScalarOrderOfMagnitude(const ProcessorGroup* ,
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
PetscSolver::computeScalarUnderrelax(const ProcessorGroup* ,
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
PetscSolver::scalarLisolve(const ProcessorGroup*,
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
PetscSolver::computeEnthalpyUnderrelax(const ProcessorGroup* ,
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
PetscSolver::enthalpyLisolve(const ProcessorGroup*,
			     const Patch*,
			     double,
			     ArchesVariables*,
			     ArchesConstVariables*,
			     CellInformation*)
{
}
