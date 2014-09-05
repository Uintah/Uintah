//----- Filter.cc ----------------------------------------------

#include <TauProfilerForSCIRun.h>
#include <Packages/Uintah/CCA/Components/Arches/Filter.h>
#include <Core/Containers/Array1.h>
#include <Core/Thread/Time.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/StencilMatrix.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
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
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#undef CHKERRQ
#define CHKERRQ(x) if(x) throw PetscError(x, __FILE__);

using namespace std;
using namespace Uintah;
using namespace SCIRun;

// ****************************************************************************
// Default constructor for Filter
// ****************************************************************************
Filter::Filter(const ArchesLabel* label,
	       BoundaryCondition* bndryCondition,
	       const ProcessorGroup* myworld) :
  d_myworld(myworld), d_lab(label), d_boundaryCondition(bndryCondition)
{
  d_perproc_patches= 0;
  d_matrixInitialize = false;
}

// ****************************************************************************
// Destructor
// ****************************************************************************
Filter::~Filter()
{
  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;
  destroyMatrix();
}

// ****************************************************************************
// Problem setup
// ****************************************************************************
void 
Filter::problemSetup(const ProblemSpecP& params)
{
  int argc = 4;
  char** argv;
  argv = new char*[argc];
  argv[0] = "Filter::problemSetup";
  //argv[1] = "-on_error_attach_debugger";
  argv[1] = "-no_signal_handler";
  argv[2] = "-log_exclude_actions";
  argv[3] = "-log_exclude_objects";
  int ierr = PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  if(ierr)
    throw PetscError(ierr, "PetscInitialize");

}

void
Filter::sched_buildFilterMatrix(const LevelP& level,
				SchedulerP& sched)
{
  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;

  LoadBalancer* lb = sched->getLoadBalancer();
  d_perproc_patches = lb->createPerProcessorPatchSet(level, d_myworld);
  d_perproc_patches->addReference();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();

  Task* tsk = scinew Task("Filter::BuildFilterMatrix",
			  this,
			  &Filter::buildFilterMatrix);
  // Requires
  // coefficient for the variable for which solve is invoked

  sched->addTask(tsk, d_perproc_patches, matls);
}


void 
Filter::buildFilterMatrix (const ProcessorGroup* ,
			   const PatchSubset* patches,
			   const MaterialSubset*,
			   DataWarehouse*,
			   DataWarehouse* )
{
  // initializeMatrix...
  if (!d_matrixInitialize)
    matrixCreate(d_perproc_patches, patches);
}


void 
Filter::matrixCreate(const PatchSet* allpatches,
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

      // #ifdef notincludeBdry
#if 1
      IntVector plowIndex = patch->getCellFORTLowIndex();
      IntVector phighIndex = patch->getCellFORTHighIndex()+IntVector(1,1,1);
#else
      IntVector plowIndex = patch->getLowIndex();
      IntVector phighIndex = patch->getHighIndex();
#endif
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
      // #ifdef notincludeBdry
#if 1
      IntVector plow = neighbor->getCellFORTLowIndex();
      IntVector phigh = neighbor->getCellFORTHighIndex()+IntVector(1,1,1);
#else
      IntVector plow = neighbor->getLowIndex();
      IntVector phigh = neighbor->getHighIndex();
#endif
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
    // #ifdef ARCHES_PETSC_DEBUG
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
  int numlrows = numCells[me];
  int numlcolumns = numlrows;
  int globalrows = (int)totalCells;
  int globalcolumns = (int)totalCells;
  // for box filter of size 2 matrix width is 27
  d_nz = 27; // defined in Filter.h
  o_nz = 26;
  // #ifdef ARCHES_PETSC_DEBUG
#if 0
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
}


// ****************************************************************************
// Fill linear parallel matrix
// ****************************************************************************
void 
Filter::setFilterMatrix(const ProcessorGroup* ,
			const Patch* patch,
			CellInformation* cellinfo,
			constCCVariable<int>& cellType )
{
#ifdef ARCHES_PETSC_DEBUG
   cerr << "in setFilterMatrix on patch: " << patch->getID() << '\n';
#endif
  // Get the patch bounds and the variable bounds
   // #ifdef notincludeBdry
   if (!d_matrixInitialize) {
#if 1
     IntVector idxLo = patch->getCellFORTLowIndex();
     IntVector idxHi = patch->getCellFORTHighIndex();
#else
     IntVector idxLo = patch->getLowIndex();
     IntVector idxHi = patch->getHighIndex()-IntVector(1,1,1);
#endif
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
         Compute the matrix that defines the filter function Ax
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
     int col[27];
     double value[27];
     // fill matrix for internal patches
     // make sure that sizeof(d_petscIndex) is the last patch, i.e., appears last in the
     // petsc matrix
     IntVector lowIndex = patch->getGhostCellLowIndex(Arches::ONEGHOSTCELL);
     IntVector highIndex = patch->getGhostCellHighIndex(Arches::ONEGHOSTCELL);
     
     Array3<int> l2g(lowIndex, highIndex);
     l2g.copy(d_petscLocalToGlobal[patch]);
     int flowID = d_boundaryCondition->flowCellType();
     for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
       for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
	 for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	   IntVector currCell(colX, colY, colZ);
	   int bndry_count=0;
	   if  (!(cellType[IntVector(colX+1, colY, colZ)] == flowID))
		   bndry_count++;
	   if  (!(cellType[IntVector(colX-1, colY, colZ)] == flowID))
		   bndry_count++;
	   if  (!(cellType[IntVector(colX, colY+1, colZ)] == flowID))
		   bndry_count++;
	   if  (!(cellType[IntVector(colX, colY-1, colZ)] == flowID))
		   bndry_count++;
	   if  (!(cellType[IntVector(colX, colY, colZ+1)] == flowID))
		   bndry_count++;
	   if  (!(cellType[IntVector(colX, colY, colZ-1)] == flowID))
		   bndry_count++;
	   bool corner = (bndry_count==3);
	   int count = 0;
	   double totalVol = 0.0;
	   for (int kk = -1; kk <= 1; kk ++) {
	     for (int jj = -1; jj <= 1; jj ++) {
	       for (int ii = -1; ii <= 1; ii ++) {
		 IntVector filterCell = IntVector(colX+ii,colY+jj,colZ+kk);
		 double vol = cellinfo->sew[colX+ii]*cellinfo->sns[colY+jj]*
		   cellinfo->stb[colZ+kk];
		 if (!(corner)) vol *= (1.0-0.5*abs(ii))*
		   (1.0-0.5*abs(jj))*(1.0-0.5*abs(kk));
		 col[count] = l2g[filterCell];  //ab
#if 1
		 // on the boundary
		 if (cellType[currCell] != flowID)
		   if (filterCell == currCell) {
		     totalVol = vol;
		     value[count] = vol;
		   }
		   else
		     value[count] = 0;
		 else if ((col[count] != -1234)&&
		     (cellType[filterCell] == flowID)) {
		   totalVol += vol;
		   value[count] = vol;
		 }
		 else 
		   value[count] = 0;
#else
		 if (col[count] != -1234) // not on the boundary
		   totalVol += vol;
		 value[count] = vol;
#endif
		 count++;
	       }
	     }
	   }
	   for (int ii = 0; ii < d_nz; ii++)
	     value[ii] /= totalVol;
	   int row = l2g[IntVector(colX,colY,colZ)];
	   
	   ierr = MatSetValues(A,1,&row,d_nz,col,value,INSERT_VALUES);
	   if(ierr)
	     throw PetscError(ierr, "MatSetValues");
	 }
       }
     }
     d_matrixInitialize = true; 
   }
}


bool
Filter::applyFilter(const ProcessorGroup* ,
		    const Patch* patch,
		    Array3<double>& var,
		    Array3<double>& filterVar)
{
  TAU_PROFILE("applyFilter", "[Filter::applyFilter]" , TAU_USER);
  // assemble x vector
  int ierr;
  // fill matrix for internal patches
  // make sure that sizeof(d_petscIndex) is the last patch, i.e., appears last in the
  // petsc matrix
  IntVector lowIndex = patch->getGhostCellLowIndex(Arches::ONEGHOSTCELL);
  IntVector highIndex = patch->getGhostCellHighIndex(Arches::ONEGHOSTCELL);

  Array3<int> l2g(lowIndex, highIndex);
  l2g.copy(d_petscLocalToGlobal[patch]);
  // #ifdef notincludeBdry
#if 1
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
#else
  IntVector idxLo = patch->getLowIndex();
  IntVector idxHi = patch->getHighIndex()-IntVector(1,1,1);
#endif
  double vecvaluex;
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	vecvaluex = var[IntVector(colX, colY, colZ)];
	int row = l2g[IntVector(colX, colY, colZ)];	  
	ierr = VecSetValue(d_x, row, vecvaluex, INSERT_VALUES);
	if(ierr)
	  throw PetscError(ierr, "VecSetValue");
      }
    }
  }

#ifdef ARCHES_PETSC_DEBUG
  cerr << "Doing mat/vec assembly\n";
#endif
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  if(ierr)
    throw PetscError(ierr, "MatAssemblyBegin");
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
#if 0
  ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);
#endif

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
  ierr = MatMult(A, d_x, d_b);
  if(ierr)
    throw PetscError(ierr, "MatMult");
  // copy vector b in the filterVar array
#if 0
  ierr = VecView(d_x, PETSC_VIEWER_STDOUT_WORLD);
  ierr = VecView(d_b, PETSC_VIEWER_STDOUT_WORLD);
#endif
  double* xvec;
  ierr = VecGetArray(d_b, &xvec);
  if(ierr)
    throw PetscError(ierr, "VecGetArray");
  int rowinit = l2g[IntVector(idxLo.x(), idxLo.y(), idxLo.z())]; 
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	int row = l2g[IntVector(colX, colY, colZ)]-rowinit;
	filterVar[IntVector(colX, colY, colZ)] = xvec[row];
      }
    }
  }
#if 0
  cerr << "In the filter class" << endl;
  var.print(cerr);
  filterVar.print(cerr);
#endif
  ierr = VecRestoreArray(d_b, &xvec);
  if(ierr)
    throw PetscError(ierr, "VecRestoreArray");

  return true;
}

bool
Filter::applyFilter(const ProcessorGroup* ,
		    const Patch* patch,
		    constCCVariable<double>& var,
		    Array3<double>& filterVar)
{
  // assemble x vector
  int ierr;
  // fill matrix for internal patches
  // make sure that sizeof(d_petscIndex) is the last patch, i.e., appears last in the
  // petsc matrix
  IntVector lowIndex = patch->getGhostCellLowIndex(Arches::ONEGHOSTCELL);
  IntVector highIndex = patch->getGhostCellHighIndex(Arches::ONEGHOSTCELL);

  Array3<int> l2g(lowIndex, highIndex);
  l2g.copy(d_petscLocalToGlobal[patch]);
  // #ifdef notincludeBdry
#if 1
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
#else
  IntVector idxLo = patch->getLowIndex();
  IntVector idxHi = patch->getHighIndex()-IntVector(1,1,1);
#endif
  double vecvaluex;
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	vecvaluex = var[IntVector(colX, colY, colZ)];
	int row = l2g[IntVector(colX, colY, colZ)];	  
	ierr = VecSetValue(d_x, row, vecvaluex, INSERT_VALUES);
	if(ierr)
	  throw PetscError(ierr, "VecSetValue");
      }
    }
  }

#ifdef ARCHES_PETSC_DEBUG
  cerr << "Doing mat/vec assembly\n";
#endif
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  if(ierr)
    throw PetscError(ierr, "MatAssemblyBegin");
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
#if 0
  cerr << "In the filter class for matview" << endl;
  ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);
#endif

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
  ierr = MatMult(A, d_x, d_b);
  if(ierr)
    throw PetscError(ierr, "MatMult");
  // copy vector b in the filterVar array
#if 0
  cerr << "In the filter class" << endl;
  ierr = VecView(d_x, PETSC_VIEWER_STDOUT_WORLD);
  ierr = VecView(d_b, PETSC_VIEWER_STDOUT_WORLD);
#endif
  double* xvec;
  ierr = VecGetArray(d_b, &xvec);
  if(ierr)
    throw PetscError(ierr, "VecGetArray");
  int rowinit = l2g[IntVector(idxLo.x(), idxLo.y(), idxLo.z())]; 
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	int row = l2g[IntVector(colX, colY, colZ)]-rowinit;
	filterVar[IntVector(colX, colY, colZ)] = xvec[row];
      }
    }
  }
#if 0
  cerr << "In the filter class after filter operation" << endl;
  var.print(cerr);
  filterVar.print(cerr);
#endif
  ierr = VecRestoreArray(d_b, &xvec);
  if(ierr)
    throw PetscError(ierr, "VecRestoreArray");

  return true;
}


void
Filter::destroyMatrix() 
{
  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  int ierr;
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



