//----- PetscSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/PetscSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/PressureSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Components/Arches/StencilMatrix.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesFort.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesVariables.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Core/Containers/Array1.h>
#include <Core/Thread/Time.h>

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
}

// ****************************************************************************
// Problem setup
// ****************************************************************************
void 
PetscSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("LinearSolver");
  db->require("max_iter", d_maxSweeps);
  db->require("res_tol", d_residual);
  db->require("underrelax", d_underrelax);
  int argc = 2;
  char** argv;
  argv = new char*[2];
  argv[0] = "PetscSolver::problemSetup";
  //argv[1] = "-on_error_attach_debugger";
  argv[1] = "-no_signal_handler";
  int ierr = PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  CHKERRQ(ierr);
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
  IntVector domLo = vars->pressure.getFortLowIndex();
  IntVector domHi = vars->pressure.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call

  FORT_COMPUTERESID(domLo.get_pointer(), domHi.get_pointer(),
		    idxLo.get_pointer(), idxHi.get_pointer(),
		    vars->pressure.getPointer(),
		    vars->residualPressure.getPointer(),
		    vars->pressCoeff[Arches::AE].getPointer(), 
		    vars->pressCoeff[Arches::AW].getPointer(), 
		    vars->pressCoeff[Arches::AN].getPointer(), 
		    vars->pressCoeff[Arches::AS].getPointer(), 
		    vars->pressCoeff[Arches::AT].getPointer(), 
		    vars->pressCoeff[Arches::AB].getPointer(), 
		    vars->pressCoeff[Arches::AP].getPointer(), 
		    vars->pressNonlinearSrc.getPointer(),
		    &vars->residPress, &vars->truncPress);

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
PetscSolver::matrixCreate(const LevelP& level, LoadBalancer* lb)
{
  // for global index get a petsc index that
  // make it a data memeber
  int numProcessors = d_myworld->size();
  // number of patches for each processor
  vector<int> numCells(numProcessors, 0);
  long totalCells = 0;
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    IntVector plowIndex = patch->getCellFORTLowIndex();
    IntVector phighIndex = patch->getCellFORTHighIndex()+IntVector(1,1,1);
  
    int proc = lb->getPatchwiseProcessorAssignment(patch, d_myworld);
    int nc = (phighIndex[0]-plowIndex[0])*
             (phighIndex[1]-plowIndex[1])*
             (phighIndex[2]-plowIndex[2]);
    numCells[proc] += nc;
    totalCells += nc;
  }
#ifdef ARCHES_PETSC_DEBUG
  cerr << "totalCells = " << totalCells << '\n';
#endif
  vector<int> startIndex(numProcessors);
  startIndex[0]=0;
  for(int i=1;i<numProcessors;i++)
     startIndex[i]=startIndex[i-1]+numCells[i-1];

  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
     const Patch* patch=*iter;
     int proc = lb->getPatchwiseProcessorAssignment(patch, d_myworld);
     int globalIndex = startIndex[proc];
     d_petscGlobalStart[patch]=globalIndex;

    IntVector plowIndex = patch->getCellFORTLowIndex();
    IntVector phighIndex = patch->getCellFORTHighIndex()+IntVector(1,1,1);
     int nc = (phighIndex[0]-plowIndex[0])*
	      (phighIndex[1]-plowIndex[1])*
	      (phighIndex[2]-plowIndex[2]);
     startIndex[proc]+=nc;
  }

  int me = d_myworld->myrank();
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
     const Patch* patch=*iter;
     int proc = lb->getPatchwiseProcessorAssignment(patch, d_myworld);
     if(proc == me){
	IntVector lowIndex = patch->getGhostCellLowIndex(1);
	IntVector highIndex = patch->getGhostCellHighIndex(1);
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
	                      +start.y()*dcells.x()
	                      +start.x();
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
	d_petscLocalToGlobal[patch]=l2g;
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
#if 0
	IntVector dn = highIndex-lowIndex;
	long wantcells = dn.x()*dn.y()*dn.z();
	ASSERTEQ(wantcells, totalCells);
#endif
     }
  }
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
  CHKERRA(ierr);  
  /* 
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
  */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,numlrows, globalrows,&d_x);CHKERRA(ierr);
  ierr = VecSetFromOptions(d_x);CHKERRA(ierr);
  ierr = VecDuplicate(d_x,&d_b);CHKERRA(ierr);
  ierr = VecDuplicate(d_x,&d_u);CHKERRA(ierr);


}

// ****************************************************************************
// Actual compute of pressure underrelaxation
// ****************************************************************************
void 
PetscSolver::computePressUnderrelax(const ProcessorGroup*,
				   const Patch* patch,
				    ArchesVariables* vars)
{
  // Get the patch bounds and the variable bounds
  IntVector domLo = vars->pressure.getFortLowIndex();
  IntVector domHi = vars->pressure.getFortHighIndex();
  IntVector domLong = vars->pressCoeff[Arches::AP].getFortLowIndex();
  IntVector domHing = vars->pressCoeff[Arches::AP].getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call
  FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		 domLong.get_pointer(), domHing.get_pointer(),
		 idxLo.get_pointer(), idxHi.get_pointer(),
		 vars->pressure.getPointer(),
		 vars->pressCoeff[Arches::AP].getPointer(), 
		 vars->pressNonlinearSrc.getPointer(), 
		 &d_underrelax);

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
			    const ArchesLabel* lab)
{
#ifdef ARCHES_PETSC_DEBUG
   cerr << "in setPressMatrix on patch: " << patch->getID() << '\n';
#endif
  // Get the patch bounds and the variable bounds
  IntVector domLo = vars->pressure.getFortLowIndex();
  IntVector domHi = vars->pressure.getFortHighIndex();
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
  int row;
  int col[7];
  double value[7];
  //  int globalIndex = d_petscGlobalStart[patch];
  int nnx = idxHi[0]-idxLo[0]+1;
  int nny = idxHi[1]-idxLo[1]+1;
  int nnz = idxHi[2]-idxLo[2]+1;
  //  int totalrows = nnx*nny*nnz;
  // fill matrix for internal patches
  // make sure that sizeof(d_petscIndex) is the last patch, i.e., appears last in the
  // petsc matrix

  Array3<int> l2g = d_petscLocalToGlobal[patch];

#if 0
  if ((patchNumber != 0)&&(patchNumber != sizeof(d_petscIndex)-1)) {
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
	  value[0] = -vars->pressCoeff[Arches::AB][IntVector(colX,colY,colZ)];
	  value[1] = -vars->pressCoeff[Arches::AS][IntVector(colX,colY,colZ)];
	  value[2] = -vars->pressCoeff[Arches::AW][IntVector(colX,colY,colZ)];
	  value[3] = vars->pressCoeff[Arches::AP][IntVector(colX,colY,colZ)];
	  value[4] = -vars->pressCoeff[Arches::AE][IntVector(colX,colY,colZ)];
	  value[5] = -vars->pressCoeff[Arches::AN][IntVector(colX,colY,colZ)];
	  value[6] = -vars->pressCoeff[Arches::AT][IntVector(colX,colY,colZ)];
#ifdef ARCHES_PETSC_DEBUG
	  for(int i=0;i<7;i++)
	     cerr << "A[" << col[3] << "][" << col[i] << "]=" << value[i] << '\n';
#endif
	  int row = col[3];
	  ierr = MatSetValues(A,1,&row,7,col,value,INSERT_VALUES);   CHKERRA(ierr);
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
	  vecvalueb = vars->pressNonlinearSrc[IntVector(colX,colY,colZ)];
	  vecvaluex = vars->pressure[IntVector(colX, colY, colZ)];
	  int row = l2g[IntVector(colX, colY, colZ)];
	  VecSetValue(d_b, row, vecvalueb, INSERT_VALUES);
	  VecSetValue(d_x, row, vecvaluex, INSERT_VALUES);
	}
      }
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
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = VecAssemblyBegin(d_b);CHKERRA(ierr);
  ierr = VecAssemblyEnd(d_b);CHKERRA(ierr);
  ierr = VecAssemblyBegin(d_x);CHKERRA(ierr);
  ierr = VecAssemblyEnd(d_x);CHKERRA(ierr);
  // compute the initial error
  double neg_one = -1.0;
  double init_norm;
  double sum_b;
  ierr = VecSum(d_b, &sum_b);
  Vec u_tmp;
  ierr = VecDuplicate(d_x,&u_tmp);CHKERRA(ierr);
  ierr = MatMult(A, d_x, u_tmp);CHKERRA(ierr);
  ierr = VecAXPY(&neg_one, d_b, u_tmp); CHKERRA(ierr);
  ierr  = VecNorm(u_tmp,NORM_2,&init_norm);CHKERRA(ierr);
  ierr = VecDestroy(u_tmp);CHKERRA(ierr);
  /* debugging - steve */
  double norm;
#ifdef ARCHES_PETSC_DEBUG
  ierr = ViewerSetFormat(VIEWER_STDOUT_WORLD, VIEWER_FORMAT_ASCII_DEFAULT, 0); CHKERRQ(ierr);
  ierr = MatNorm(A,NORM_1,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"matrix A norm = %g\n",norm);CHKERRQ(ierr);
  ierr = MatView(A, VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  ierr = VecNorm(d_x,NORM_1,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"vector x norm = %g\n",norm);CHKERRQ(ierr);
  ierr = VecView(d_x, VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  ierr = VecNorm(d_b,NORM_1,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"vector b norm = %g\n",norm);CHKERRQ(ierr);
  ierr = VecView(d_b, VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
#endif
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = SLESGetKSP(sles,&ksp);CHKERRA(ierr);
  ierr = SLESGetPC(sles, &peqnpc); CHKERRA(ierr);
  ierr = PCSetType(peqnpc, PCJACOBI); CHKERRA(ierr);
  // set null space for preconditioner
  // change for a newer version
#ifdef NULL_MATRIX
  PCNullSpace nullsp;
  ierr = PCNullSpaceCreate(PETSC_COMM_WORLD, 1, 0, PETSC_NULL, &nullsp); 
  CHKERRA(ierr);
  ierr = PCNullSpaceAttach(peqnpc, nullsp); CHKERRA(ierr);
  ierr = PCNullSpaceDestroy(nullsp); CHKERRA(ierr);
#endif
  ierr = KSPSetInitialGuessNonzero(ksp);CHKERRA(ierr);
  
  ierr = SLESSetFromOptions(sles);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  int its;
  ierr = SLESSolve(sles,d_b,d_x,&its);CHKERRA(ierr);
  int me = d_myworld->myrank();

  ierr = VecNorm(d_x,NORM_1,&norm);CHKERRQ(ierr);
  double tsolve = Time::currentSeconds()-solve_start;
#ifdef ARCHES_PETSC_DEBUG
  ierr = VecView(d_x, VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
#endif

  // check the error
  ierr = MatMult(A, d_x, d_u);CHKERRA(ierr);
  ierr = VecAXPY(&neg_one, d_b, d_u); CHKERRA(ierr);
  ierr  = VecNorm(d_u,NORM_2,&norm);CHKERRA(ierr);
  if(me == 0) {
     cerr << "SLESSolve: Norm of error: " << norm << ", iterations: " << its << ", time: " << Time::currentSeconds()-solve_start << " seconds\n";
     cerr << "Init Norm: " << init_norm << " Error reduced by: " << norm/init_norm << endl;
     cerr << "Sum of RHS vector: " << sum_b << endl;
  }
  if (norm/init_norm < 1.0)
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
  ierr = VecGetArray(d_x, &xvec); CHKERRQ(ierr);
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
  ierr = VecRestoreArray(d_x, &xvec); CHKERRQ(ierr);
}
  
void
PetscSolver::destroyMatrix() 
{
  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  int ierr;
  ierr = SLESDestroy(sles);CHKERRA(ierr); 
  ierr = VecDestroy(d_u);CHKERRA(ierr);
  ierr = VecDestroy(d_b);CHKERRA(ierr);
  ierr = VecDestroy(d_x);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);
}

// ****************************************************************************
// Actual linear solve for pressure
// ****************************************************************************
void 
PetscSolver::pressLisolve(const ProcessorGroup* pc,
			 const Patch* patch,
			 DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw,
			 ArchesVariables* vars,
			 const ArchesLabel* lab)
{
 
  // Get the patch bounds and the variable bounds
  IntVector domLo = vars->pressure.getFortLowIndex();
  IntVector domHi = vars->pressure.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  Vec x,b,u; // approx solution, RHS, exact solution
  Mat A; // linear system matrix
  SLES sles; // linear solver context
  KSP ksp;
  int ierr;
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
  int numrows = (idxHi[0]-idxLo[0]+1)*(idxHi[1]-idxLo[1]+1)*
                (idxHi[2]-idxLo[2]+1);
  int nnx = idxHi[0]-idxLo[0]+1;
  int nny = idxHi[1]-idxLo[1]+1;
  int nnz = idxHi[2]-idxLo[2]+1;
  int numcolumns = numrows;
  int numnonzero = 7; // 7 point stencil
  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD, numrows, numcolumns, numnonzero,
			 PETSC_NULL, &A);
  CHKERRA(ierr);
  int row;
  int col[7];
  double value[7];
  // fill matrix where all 7 components are present
  for (row = nnx*nny; row < (nnz-1)*nnx*nny; row++) {
    col[0] = row-nnx*nny;
    col[1] = row-nnx;
    col[2] = row - 1;
    col[3] = row;
    col[4] = row + 1;
    col[5] = row+nnx;
    col[6] = row+nnx*nny;
    int ii = ((row%(nnx*nny))%nnx);
    int jj = ((row - ii)%(nnx*nny))/nnx;
    int kk = (row-ii-nnx*jj)/(nnx*nny);
    // all the off-diagnol components follow different sign convention
    value[0] = -vars->pressCoeff[Arches::AB][IntVector(ii,jj,kk)];
    value[1] = -vars->pressCoeff[Arches::AS][IntVector(ii,jj,kk)];
    value[2] = -vars->pressCoeff[Arches::AW][IntVector(ii,jj,kk)];
    value[3] = vars->pressCoeff[Arches::AP][IntVector(ii,jj,kk)];
    value[4] = -vars->pressCoeff[Arches::AE][IntVector(ii,jj,kk)];
    value[5] = -vars->pressCoeff[Arches::AN][IntVector(ii,jj,kk)];
    value[6] = -vars->pressCoeff[Arches::AT][IntVector(ii,jj,kk)];
    ierr = MatSetValues(A,1,&row,7,col,value,INSERT_VALUES);
#ifdef ARCHES_PETSC_DEBUG
    cerr << "set values: row=" << row << ", col=";
    for(int i=0;i<7;i++)
       cerr << col[i] << " ";
    cerr << ", value=";
    for(int i=0;i<7;i++)
       cerr << value[i] << " ";
    cerr << '\n';
#endif
    CHKERRA(ierr);
  }
  for (row = nnx; row < nny*nnx; row++) {
    col[0] = row-nnx;
    col[1] = row-1;
    col[2] = row;
    col[3] = row+1;
    col[4] = row+nnx;
    col[5] = row+nnx*nny;
    int ii = ((row%(nnx*nny))%nnx);
    int jj = ((row - ii)%(nnx*nny))/nnx;
    int kk = (row-ii-nnx*jj)/(nnx*nny);
    value[0] = -vars->pressCoeff[Arches::AS][IntVector(ii,jj,kk)];
    value[1] = -vars->pressCoeff[Arches::AW][IntVector(ii,jj,kk)];
    value[2] = vars->pressCoeff[Arches::AP][IntVector(ii,jj,kk)];
    value[3] = -vars->pressCoeff[Arches::AE][IntVector(ii,jj,kk)];
    value[4] = -vars->pressCoeff[Arches::AN][IntVector(ii,jj,kk)];
    value[5] = -vars->pressCoeff[Arches::AT][IntVector(ii,jj,kk)];
    ierr = MatSetValues(A,1,&row,6,col,value,INSERT_VALUES);
#ifdef ARCHES_PETSC_DEBUG
    cerr << "2. set values: row=" << row << ", col=";
    for(int i=0;i<6;i++)
       cerr << col[i] << " ";
    cerr << ", value=";
    for(int i=0;i<6;i++)
       cerr << value[i] << " ";
    cerr << '\n';
#endif
    CHKERRA(ierr);
  }
  for (row = nnx*nny*(nnz-1); row < (nnz-1)*nnx*nny+nnx*(nny-1); row++){
    col[0] = row-nnx*nny;
    col[1] = row-nnx;
    col[2] = row-1;
    col[3] = row;
    col[4] = row+1;
    col[5] = row+nnx;
    int ii = ((row%(nnx*nny))%nnx);
    int jj = ((row - ii)%(nnx*nny))/nnx;
    int kk = (row-ii-nnx*jj)/(nnx*nny);
    value[0] = -vars->pressCoeff[Arches::AB][IntVector(ii,jj,kk)];
    value[1] = -vars->pressCoeff[Arches::AS][IntVector(ii,jj,kk)];
    value[2] = -vars->pressCoeff[Arches::AW][IntVector(ii,jj,kk)];
    value[3] = vars->pressCoeff[Arches::AP][IntVector(ii,jj,kk)];
    value[4] = -vars->pressCoeff[Arches::AE][IntVector(ii,jj,kk)];
    value[5] = -vars->pressCoeff[Arches::AN][IntVector(ii,jj,kk)];
    ierr = MatSetValues(A,1,&row,6,col,value,INSERT_VALUES);
#ifdef ARCHES_PETSC_DEBUG
    cerr << "set values: row=" << row << ", col=";
    for(int i=0;i<6;i++)
       cerr << col[i] << " ";
    cerr << ", value=";
    for(int i=0;i<6;i++)
       cerr << value[i] << " ";
    cerr << '\n';
#endif
    CHKERRA(ierr);
  }
  for (row = 1; row < nnx; row++) {
    col[0] = row-1;
    col[1] = row;
    col[2] = row+1;
    col[3] = row+nnx;
    col[4] = row+nnx*nny;
    int ii = ((row%(nnx*nny))%nnx);
    int jj = ((row - ii)%(nnx*nny))/nnx;
    int kk = (row-ii-nnx*jj)/(nnx*nny);
    value[0] = -vars->pressCoeff[Arches::AW][IntVector(ii,jj,kk)];
    value[1] = vars->pressCoeff[Arches::AP][IntVector(ii,jj,kk)];
    value[2] = -vars->pressCoeff[Arches::AE][IntVector(ii,jj,kk)];
    value[3] = -vars->pressCoeff[Arches::AN][IntVector(ii,jj,kk)];
    value[4] = -vars->pressCoeff[Arches::AT][IntVector(ii,jj,kk)];
    ierr = MatSetValues(A,1,&row,5,col,value,INSERT_VALUES);
#ifdef ARCHES_PETSC_DEBUG
    cerr << "set values: row=" << row << ", col=";
    for(int i=0;i<5;i++)
       cerr << col[i] << " ";
    cerr << ", value=";
    for(int i=0;i<5;i++)
       cerr << value[i] << " ";
    cerr << '\n';
#endif
    CHKERRA(ierr);
  }
  for (row = nnx*nny*(nnz-1)+nnx*(nny-1); 
       row < nnx*nny*(nnz-1)+nnx*(nny-1)+ nnx-1; row++) {
    col[0] = row-nnx*nny;
    col[1] = row-nnx;
    col[2] = row-1;
    col[3] = row;
    col[4] = row+1;
    int ii = ((row%(nnx*nny))%nnx);
    int jj = ((row - ii)%(nnx*nny))/nnx;
    int kk = (row-ii-nnx*jj)/(nnx*nny);
    value[0] = -vars->pressCoeff[Arches::AB][IntVector(ii,jj,kk)];
    value[1] = -vars->pressCoeff[Arches::AS][IntVector(ii,jj,kk)];
    value[2] = -vars->pressCoeff[Arches::AW][IntVector(ii,jj,kk)];
    value[3] = vars->pressCoeff[Arches::AP][IntVector(ii,jj,kk)];
    value[4] = -vars->pressCoeff[Arches::AE][IntVector(ii,jj,kk)];
    ierr = MatSetValues(A,1,&row,5,col,value,INSERT_VALUES);
#ifdef ARCHES_PETSC_DEBUG
    cerr << "set values: row=" << row << ", col=";
    for(int i=0;i<5;i++)
       cerr << col[i] << " ";
    cerr << ", value=";
    for(int i=0;i<5;i++)
       cerr << value[i] << " ";
    cerr << '\n';
#endif
    CHKERRA(ierr);
  }
  row = 0;
  col[0] = row;
  col[1] = row+1;
  col[2] = row+nnx;
  col[3] = row+nnx*nny;
  int ii = 0;
  int jj = 0;
  int kk = 0;
  value[0] = vars->pressCoeff[Arches::AP][IntVector(ii,jj,kk)];
  value[1] = -vars->pressCoeff[Arches::AE][IntVector(ii,jj,kk)];
  value[2] = -vars->pressCoeff[Arches::AN][IntVector(ii,jj,kk)];
  value[3] = -vars->pressCoeff[Arches::AT][IntVector(ii,jj,kk)];
  ierr = MatSetValues(A,1,&row,4,col,value,INSERT_VALUES);
  CHKERRA(ierr);
  row = nnx*nny*nnz-1;
  col[0] = row-nnx*nny;
  col[1] = row-nnx;
  col[2] = row-1;
  col[3] = row;
  ii = nnx-1;
  jj = nny-1;
  kk = nnz-1;
  value[0] = -vars->pressCoeff[Arches::AB][IntVector(ii,jj,kk)];
  value[1] = -vars->pressCoeff[Arches::AS][IntVector(ii,jj,kk)];
  value[2] = -vars->pressCoeff[Arches::AW][IntVector(ii,jj,kk)];
  value[3] = vars->pressCoeff[Arches::AP][IntVector(ii,jj,kk)];
  ierr = MatSetValues(A,1,&row,4,col,value,INSERT_VALUES);
  CHKERRA(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  /* 
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,numrows,&x);CHKERRA(ierr);
  ierr = VecSetFromOptions(x);CHKERRA(ierr);
  ierr = VecDuplicate(x,&b);CHKERRA(ierr);
  ierr = VecDuplicate(x,&u);CHKERRA(ierr);
  // assemble right hand side and solution vector
  double vecvalueb, vecvaluex;
  for(row = 0; row < numrows; row++) {
    int ii = ((row%(nnx*nny))%nnx);
    int jj = ((row - ii)%(nnx*nny))/nnx;
    int kk = (row-ii-nnx*jj)/(nnx*nny);
    vecvalueb = vars->pressNonlinearSrc[IntVector(ii,jj,kk)];
#ifdef ARCHES_PETSC_DEBUG
    cerr << "vecvalueb=" << vecvalueb << '\n';
#endif
    vecvaluex = vars->pressure[IntVector(ii,jj,kk)];
    VecSetValue(b, row, vecvalueb, INSERT_VALUES);
    VecSetValue(x, row, vecvaluex, INSERT_VALUES);
  }
  ierr = VecAssemblyBegin(b);CHKERRA(ierr);
  ierr = VecAssemblyEnd(b);CHKERRA(ierr);
  ierr = VecAssemblyBegin(x);CHKERRA(ierr);
  ierr = VecAssemblyEnd(x);CHKERRA(ierr);

  /* debugging - steve */
  ierr = ViewerSetFormat(VIEWER_STDOUT_WORLD, VIEWER_FORMAT_ASCII_DEFAULT, 0); CHKERRQ(ierr);
  double norm;
  ierr = MatNorm(A,NORM_1,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"matrix A norm = %g\n",norm);CHKERRQ(ierr);
  ierr = MatView(A, VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  ierr = VecNorm(x,NORM_1,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"vector x norm = %g\n",norm);CHKERRQ(ierr);
  ierr = VecView(x, VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  ierr = VecNorm(b,NORM_1,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"vector b norm = %g\n",norm);CHKERRQ(ierr);
  ierr = VecView(b, VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = SLESGetKSP(sles,&ksp);CHKERRA(ierr);
  ierr = KSPSetInitialGuessNonzero(ksp);CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  int its;
  ierr = SLESSolve(sles,b,x,&its);CHKERRA(ierr);

  ierr = VecNorm(x,NORM_1,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"AFTER SOLVE vector x norm = %g\n",norm);CHKERRQ(ierr);
  ierr = VecView(x, VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  // check the error
  double neg_one = -1.0;
  ierr = MatMult(A, x, u);CHKERRA(ierr);
  ierr = VecAXPY(&neg_one, b, u); CHKERRA(ierr);
  ierr  = VecNorm(u,NORM_2,&norm);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %A, Iterations %d\n",norm,its);CHKERRA(ierr);
  // copy solution vector back into the array
  double* xvec;
  ierr = VecGetArray(x, &xvec); CHKERRQ(ierr);
  for(row = 0; row < numrows; row++) {
    int ii = ((row%(nnx*nny))%nnx);
    int jj = ((row - ii)%(nnx*nny))/nnx;
    int kk = (row-ii-nnx*jj)/(nnx*nny);
    vars->pressure[IntVector(ii,jj,kk)] = xvec[row];
#ifdef ARCHES_PETSC_DEBUG
    cerr << "press" << IntVector(ii,jj,kk) << "=" << xvec[row] << '\n';
#endif
  }
  ierr = VecRestoreArray(x, &xvec); CHKERRQ(ierr);
  
  
/* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = SLESDestroy(sles);CHKERRA(ierr); 
  ierr = VecDestroy(u);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);


 

#ifdef ARCHES_PRES_DEBUG
    cerr << "Iter # = " << pressIter << " Max Iters = " << d_maxSweeps 
	 << " Press. Resid = " << pressResid << " d_residual = " << d_residual
	 << " nlResid = " << nlResid << endl;
#endif
  // while((pressIter < d_maxSweeps)&&((pressResid > d_residual*nlResid)||
  //			      (pressResid > trunc_conv)));
#ifdef ARCHES_PRES_DEBUG
  cerr << "After pressure solve " << pressIter << " " << pressResid << endl;
  cerr << "After pressure solve " << nlResid << " " << trunc_conv <<  endl;
  cerr << " After Pressure solve : " << endl;
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
#endif

  

}


// Shutdown PETSc
void PetscSolver::finalizeSolver()
{
  int ierr = PetscFinalize(); CHKERRQ(ierr);
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

    FORT_COMPUTERESID(domLo.get_pointer(), domHi.get_pointer(),
		      idxLo.get_pointer(), idxHi.get_pointer(),
		      vars->uVelocity.getPointer(),
		      vars->residualUVelocity.getPointer(),
		      vars->uVelocityCoeff[Arches::AE].getPointer(), 
		      vars->uVelocityCoeff[Arches::AW].getPointer(), 
		      vars->uVelocityCoeff[Arches::AN].getPointer(), 
		      vars->uVelocityCoeff[Arches::AS].getPointer(), 
		      vars->uVelocityCoeff[Arches::AT].getPointer(), 
		      vars->uVelocityCoeff[Arches::AB].getPointer(), 
		      vars->uVelocityCoeff[Arches::AP].getPointer(), 
		      vars->uVelNonlinearSrc.getPointer(),
		      &vars->residUVel, &vars->truncUVel);

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

    FORT_COMPUTERESID(domLo.get_pointer(), domHi.get_pointer(),
		      idxLo.get_pointer(), idxHi.get_pointer(),
		      vars->vVelocity.getPointer(),
		      vars->residualVVelocity.getPointer(),
		      vars->vVelocityCoeff[Arches::AE].getPointer(), 
		      vars->vVelocityCoeff[Arches::AW].getPointer(), 
		      vars->vVelocityCoeff[Arches::AN].getPointer(), 
		      vars->vVelocityCoeff[Arches::AS].getPointer(), 
		      vars->vVelocityCoeff[Arches::AT].getPointer(), 
		      vars->vVelocityCoeff[Arches::AB].getPointer(), 
		      vars->vVelocityCoeff[Arches::AP].getPointer(), 
		      vars->vVelNonlinearSrc.getPointer(),
		      &vars->residVVel, &vars->truncVVel);

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

    FORT_COMPUTERESID(domLo.get_pointer(), domHi.get_pointer(),
		      idxLo.get_pointer(), idxHi.get_pointer(),
		      vars->wVelocity.getPointer(),
		      vars->residualWVelocity.getPointer(),
		      vars->wVelocityCoeff[Arches::AE].getPointer(), 
		      vars->wVelocityCoeff[Arches::AW].getPointer(), 
		      vars->wVelocityCoeff[Arches::AN].getPointer(), 
		      vars->wVelocityCoeff[Arches::AS].getPointer(), 
		      vars->wVelocityCoeff[Arches::AT].getPointer(), 
		      vars->wVelocityCoeff[Arches::AB].getPointer(), 
		      vars->wVelocityCoeff[Arches::AP].getPointer(), 
		      vars->wVelNonlinearSrc.getPointer(),
		      &vars->residWVel, &vars->truncWVel);

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
				  int index, ArchesVariables* vars)
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
    domLo = vars->uVelocity.getFortLowIndex();
    domHi = vars->uVelocity.getFortHighIndex();
    domLong = vars->uVelocityCoeff[Arches::AP].getFortLowIndex();
    domHing = vars->uVelocityCoeff[Arches::AP].getFortHighIndex();
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();
    FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		   domLong.get_pointer(), domHing.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   vars->uVelocity.getPointer(),
		   vars->uVelocityCoeff[Arches::AP].getPointer(), 
		   vars->uVelNonlinearSrc.getPointer(),
		   &d_underrelax);

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
    domLo = vars->vVelocity.getFortLowIndex();
    domHi = vars->vVelocity.getFortHighIndex();
    domLong = vars->vVelocityCoeff[Arches::AP].getFortLowIndex();
    domHing = vars->vVelocityCoeff[Arches::AP].getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
    FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		   domLong.get_pointer(), domHing.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   vars->vVelocity.getPointer(),
		   vars->vVelocityCoeff[Arches::AP].getPointer(), 
		   vars->vVelNonlinearSrc.getPointer(),
		   &d_underrelax);

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
    domLo = vars->wVelocity.getFortLowIndex();
    domHi = vars->wVelocity.getFortHighIndex();
    domLong = vars->wVelocityCoeff[Arches::AP].getFortLowIndex();
    domHing = vars->wVelocityCoeff[Arches::AP].getFortHighIndex();
    idxLo = patch->getSFCZFORTLowIndex();
    idxHi = patch->getSFCZFORTHighIndex();
    FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		   domLong.get_pointer(), domHing.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   vars->wVelocity.getPointer(),
		   vars->wVelocityCoeff[Arches::AP].getPointer(), 
		   vars->wVelNonlinearSrc.getPointer(),
		   &d_underrelax);

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
PetscSolver::velocityLisolve(const ProcessorGroup* pc,
			     const Patch* patch,
			     int index, double delta_t,
			     ArchesVariables* vars,
			     CellInformation* cellinfo,
			     const ArchesLabel* lab)
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
    FORT_EXPLICIT(domLo.get_pointer(), domHi.get_pointer(),
		  idxLo.get_pointer(), idxHi.get_pointer(),
		  vars->uVelocity.getPointer(),
		  vars->old_uVelocity.getPointer(),
		  vars->uVelocityCoeff[Arches::AE].getPointer(), 
		  vars->uVelocityCoeff[Arches::AW].getPointer(), 
		  vars->uVelocityCoeff[Arches::AN].getPointer(), 
		  vars->uVelocityCoeff[Arches::AS].getPointer(), 
		  vars->uVelocityCoeff[Arches::AT].getPointer(), 
		  vars->uVelocityCoeff[Arches::AB].getPointer(), 
		  vars->uVelocityCoeff[Arches::AP].getPointer(), 
		  vars->uVelNonlinearSrc.getPointer(),
		  domLoDen.get_pointer(), domHiDen.get_pointer(),
		  vars->old_density.getPointer(), 
		  cellinfo->sewu.get_objs(), cellinfo->sns.get_objs(),
		  cellinfo->stb.get_objs(), &delta_t);

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
    FORT_EXPLICIT(domLo.get_pointer(), domHi.get_pointer(),
		  idxLo.get_pointer(), idxHi.get_pointer(),
		  vars->vVelocity.getPointer(),
		  vars->old_vVelocity.getPointer(),
		  vars->vVelocityCoeff[Arches::AE].getPointer(), 
		  vars->vVelocityCoeff[Arches::AW].getPointer(), 
		  vars->vVelocityCoeff[Arches::AN].getPointer(), 
		  vars->vVelocityCoeff[Arches::AS].getPointer(), 
		  vars->vVelocityCoeff[Arches::AT].getPointer(), 
		  vars->vVelocityCoeff[Arches::AB].getPointer(), 
		  vars->vVelocityCoeff[Arches::AP].getPointer(), 
		  vars->vVelNonlinearSrc.getPointer(),
		  domLoDen.get_pointer(), domHiDen.get_pointer(),
		  vars->old_density.getPointer(), 
		  cellinfo->sew.get_objs(), cellinfo->snsv.get_objs(),
		  cellinfo->stb.get_objs(), &delta_t);

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
    FORT_EXPLICIT(domLo.get_pointer(), domHi.get_pointer(),
		  idxLo.get_pointer(), idxHi.get_pointer(),
		  vars->wVelocity.getPointer(),
		  vars->old_wVelocity.getPointer(),
		  vars->wVelocityCoeff[Arches::AE].getPointer(), 
		  vars->wVelocityCoeff[Arches::AW].getPointer(), 
		  vars->wVelocityCoeff[Arches::AN].getPointer(), 
		  vars->wVelocityCoeff[Arches::AS].getPointer(), 
		  vars->wVelocityCoeff[Arches::AT].getPointer(), 
		  vars->wVelocityCoeff[Arches::AB].getPointer(), 
		  vars->wVelocityCoeff[Arches::AP].getPointer(), 
		  vars->wVelNonlinearSrc.getPointer(),
		  domLoDen.get_pointer(), domHiDen.get_pointer(),
		  vars->old_density.getPointer(), 
		  cellinfo->sew.get_objs(), cellinfo->sns.get_objs(),
		  cellinfo->stbw.get_objs(), &delta_t);

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
				  int index,
				  ArchesVariables* vars)
{
  // Get the patch bounds and the variable bounds
  IntVector domLo = vars->scalar.getFortLowIndex();
  IntVector domHi = vars->scalar.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call

  FORT_COMPUTERESID(domLo.get_pointer(), domHi.get_pointer(),
		    idxLo.get_pointer(), idxHi.get_pointer(),
		    vars->scalar.getPointer(),
		    vars->residualScalar.getPointer(),
		    vars->scalarCoeff[Arches::AE].getPointer(), 
		    vars->scalarCoeff[Arches::AW].getPointer(), 
		    vars->scalarCoeff[Arches::AN].getPointer(), 
		    vars->scalarCoeff[Arches::AS].getPointer(), 
		    vars->scalarCoeff[Arches::AT].getPointer(), 
		    vars->scalarCoeff[Arches::AB].getPointer(), 
		    vars->scalarCoeff[Arches::AP].getPointer(), 
		    vars->scalarNonlinearSrc.getPointer(),
		    &vars->residScalar, &vars->truncScalar);
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
				    int index,
				    ArchesVariables* vars)
{
  // Get the patch bounds and the variable bounds
  IntVector domLo = vars->scalar.getFortLowIndex();
  IntVector domHi = vars->scalar.getFortHighIndex();
  IntVector domLong = vars->scalarCoeff[Arches::AP].getFortLowIndex();
  IntVector domHing = vars->scalarCoeff[Arches::AP].getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call
  FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		 domLong.get_pointer(), domHing.get_pointer(),
		 idxLo.get_pointer(), idxHi.get_pointer(),
		 vars->scalar.getPointer(),
		 vars->scalarCoeff[Arches::AP].getPointer(), 
		 vars->scalarNonlinearSrc.getPointer(),
		 &d_underrelax);
}

//****************************************************************************
// Scalar Solve
//****************************************************************************
void 
PetscSolver::scalarLisolve(const ProcessorGroup* pc,
			  const Patch* patch,
			  int index, double delta_t,
			  ArchesVariables* vars,
			  CellInformation* cellinfo,
			  const ArchesLabel* lab)
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
     FORT_EXPLICIT(domLo.get_pointer(), domHi.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   vars->scalar.getPointer(), vars->old_scalar.getPointer(),
		   vars->scalarCoeff[Arches::AE].getPointer(), 
		   vars->scalarCoeff[Arches::AW].getPointer(), 
		   vars->scalarCoeff[Arches::AN].getPointer(), 
		   vars->scalarCoeff[Arches::AS].getPointer(), 
		   vars->scalarCoeff[Arches::AT].getPointer(), 
		   vars->scalarCoeff[Arches::AB].getPointer(), 
		   vars->scalarCoeff[Arches::AP].getPointer(), 
		   vars->scalarNonlinearSrc.getPointer(),
		   domLoDen.get_pointer(), domHiDen.get_pointer(),
		   vars->old_density.getPointer(), 
		   cellinfo->sew.get_objs(), cellinfo->sns.get_objs(),
		   cellinfo->stb.get_objs(), &delta_t);
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
