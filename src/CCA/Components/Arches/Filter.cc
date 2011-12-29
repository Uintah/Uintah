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


//----- Filter.cc ----------------------------------------------

#include <TauProfilerForSCIRun.h>
#include <CCA/Components/Arches/Filter.h>
#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/CellInformationP.h>
#include <CCA/Components/Arches/CellInformation.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/UintahPetscError.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Grid/SimulationState.h>


using namespace std;
using namespace Uintah;

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
  d_matrix_vectors_created = false;
}

// ****************************************************************************
// Destructor
// ****************************************************************************
Filter::~Filter()
{
  if (d_perproc_patches && d_perproc_patches->removeReference()){
    delete d_perproc_patches;
  }
  if (d_matrix_vectors_created){
    destroyMatrix();
  }
}

// ****************************************************************************
// Problem setup
// ****************************************************************************
void 
Filter::problemSetup(const ProblemSpecP& params)
{
  int argc = 4;
  char** argv;
  argv = scinew char*[argc];
  argv[0] = const_cast<char*>("Filter::problemSetup");
  //argv[1] = "-on_error_attach_debugger";
  argv[1] = const_cast<char*>("-no_signal_handler");
  argv[2] = const_cast<char*>("-log_exclude_actions");
  argv[3] = const_cast<char*>("-log_exclude_objects");
  int ierr = PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  if(ierr)
    throw UintahPetscError(ierr, "PetscInitialize", __FILE__, __LINE__);
  delete[] argv;
}
//______________________________________________________________________
//
void
Filter::sched_buildFilterMatrix(const LevelP& level,
                                SchedulerP& sched)
{
  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;

  LoadBalancer* lb = sched->getLoadBalancer();
  d_perproc_patches = lb->getPerProcessorPatchSet(level);
  d_perproc_patches->addReference();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();

  Task* tsk = scinew Task("Filter::BuildFilterMatrix",this,
                          &Filter::buildFilterMatrix);


  sched->addTask(tsk, d_perproc_patches, matls);
}

//______________________________________________________________________
//
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

//______________________________________________________________________
//
void 
Filter::matrixCreate(const PatchSet* allpatches,
                     const PatchSubset* mypatches)
{
  int numProcs = d_myworld->size();
  ASSERTEQ(numProcs, allpatches->size());

  vector<int> numCells(numProcs, 0);
  vector<int> startIndex(numProcs);
  int totalCells = 0;
  
  PetscLocalToGlobalMapping(allpatches, mypatches, numCells, totalCells,
                            d_petscGlobalStart, d_petscLocalToGlobal, d_myworld);
                            
  int me            = d_myworld->myrank();
  int numlrows      = numCells[me];
  int numlcolumns   = numlrows;
  int globalrows    = totalCells;
  int globalcolumns = totalCells;
  // for box filter of size 2 matrix width is 27
  d_nz = 27; // defined in Filter.h
  o_nz = 26;
  
  
  //__________________________________
  //  create the Petsc matrix A
  proc0cout << "Creating the patch matrix... \n Note: if sus crashes here, try increasing your resolution.\n"<<endl;
  int ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD, numlrows, numlcolumns, globalrows,
                             globalcolumns, d_nz, PETSC_NULL, o_nz, PETSC_NULL, &A);
  if(ierr)
    throw UintahPetscError(ierr, "MatCreateMPIAIJ", __FILE__, __LINE__);

  //__________________________________
  //  Create Petsc vectors.  Note that we form 1 vector from scratch and
  //  then duplicate as needed.
  ierr = VecCreateMPI(PETSC_COMM_WORLD,numlrows, globalrows,&d_x);
  if(ierr)
    throw UintahPetscError(ierr, "VecCreateMPI", __FILE__, __LINE__);
    
  ierr = VecSetFromOptions(d_x);
  if(ierr)
    throw UintahPetscError(ierr, "VecSetFromOptions", __FILE__, __LINE__);
  
  ierr = VecDuplicate(d_x,&d_b);
  if(ierr)
    throw UintahPetscError(ierr, "VecDuplicate(d_b)", __FILE__, __LINE__);

  d_matrix_vectors_created = true;
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
  // Get the patch bounds and the variable bounds
   if (!d_matrixInitialize) {
     // #ifdef notincludeBdry
#if 1
     IntVector idxLo = patch->getFortranCellLowIndex();
     IntVector idxHi = patch->getFortranCellHighIndex();
#else
     IntVector idxLo = patch->getExtraCellLowIndex();
     IntVector idxHi = patch->getExtraCellHighIndex()-IntVector(1,1,1);
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
     IntVector lowIndex  = patch->getExtraCellLowIndex(Arches::ONEGHOSTCELL);
     IntVector highIndex = patch->getExtraCellHighIndex(Arches::ONEGHOSTCELL);

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
                 double vol = cellinfo->sew[colX+ii] * cellinfo->sns[colY+jj] * cellinfo->stb[colZ+kk];
                
                 if (!(corner)){
                  vol *= (1.0 - 0.5 * abs(ii)) * (1.0 - 0.5 * abs(jj)) * (1.0 - 0.5 * abs(kk));
                 }
                 col[count] = l2g[filterCell];  //ab
#if 1
                 // on the boundary
                 if (cellType[currCell] != flowID){
                   if (filterCell == currCell) {
                     totalVol = vol;
                     value[count] = vol;
                   }else{
                     value[count] = 0;
                   }
                 }else if ((col[count] != -1234) && (cellType[filterCell] == flowID)) {
                   totalVol += vol;
                   value[count] = vol;
                 }
                 else{
                   value[count] = 0;
                 }
#else           
                 if (col[count] != -1234){ // not on the boundary
                   totalVol += vol;
                 }
                 
                 value[count] = vol;
#endif
                 count++;
               }
             }
           }
           
           for (int ii = 0; ii < d_nz; ii++){
             value[ii] /= totalVol;
           }
           
           int row = l2g[IntVector(colX,colY,colZ)];

#if SCI_ASSERTION_LEVEL > 0
           for(int i=0;i<d_nz;i++){
             ASSERT(!isnan(value[i]));
           }
#endif           
           ierr = MatSetValues(A,1,&row,d_nz,col,value,INSERT_VALUES);
           if(ierr)
             throw UintahPetscError(ierr, "MatSetValues", __FILE__, __LINE__);
         }
       }
     }
     d_matrixInitialize = true; 
   }
}
//______________________________________________________________________
//
void
Filter::destroyMatrix() 
{
  destroyPetscObjects(A, d_x, d_b, PETSC_NULL);
}
