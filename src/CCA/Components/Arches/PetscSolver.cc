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


//----- PetscSolver.cc ----------------------------------------------

#include <CCA/Components/Arches/PetscSolver.h>
#include <CCA/Components/Arches/PetscCommon.h>
#include <Core/Thread/Time.h>
#include <CCA/Components/Arches/Arches.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/UintahPetscError.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>

using namespace std;
using namespace Uintah;


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
  finalizePetscSolver();
}

// ****************************************************************************
// Problem setup
// ****************************************************************************
void 
PetscSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("Parameters");
  
  if(!db) {
    ostringstream warn;
    warn << "INPUT FILE ERROR: ARCHES:PressureSolver: missing <parameters> tag \n";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__); 
  }
  
  //__________________________________
  //bulletproofing
  string test = "none";
  string test2 = "none";
  db->get("ksptype",test);
  db->get("pctype", test2);
  
  if (test != "none" || test2 != "none"){
    ostringstream warn;
    warn << "INPUT FILE ERROR: ARCHES: using a depreciated linear solver option \n"
         << "change  <ksptype>   to    <solver> \n"
         << "change  <pctype>    to    <preconditioner> \n"<< endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);    
  }
  
  db->require("preconditioner",      d_pcType);
  db->require("solver",              d_solverType);
  db->require("maxiterations",       d_maxSweeps);
  db->getWithDefault("tolerance",    d_residual, 1.0e-7);
  
  if (d_pcType == "asm"){
    db->require("overlap",d_overlap);
  }
  
  if (d_pcType == "ilu"){
    db->require("fill",d_fill);
  }
   
  //__________________________________
  //bulletproofing
  if(d_solverType != "cg" ){
    ostringstream warn;
    warn << "INPUT FILE ERROR: ARCHES: unknown linear solve type ("<<d_solverType<<") \n"
         << "Valid PETSC Option:  cg"<< endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  
  if(d_pcType != "asm"   && d_pcType != "ilu" && d_pcType != "jacobi"){
    ostringstream warn;
    warn << "INPUT FILE ERROR: ARCHES: unknown PETSC preconditioner type ("<<d_pcType<<") \n"
         << "Valid Options:  asm, ilu, jacobi, none"<< endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  
  int argc = 4;
  char** argv;
  argv = scinew char*[argc];
  argv[0] = const_cast<char*>("PetscSolver::problemSetup");
  //argv[1] = "-on_error_attach_debugger";
  argv[1] = const_cast<char*>("-no_signal_handler");
  argv[2] = const_cast<char*>("-log_exclude_actions");
  argv[3] = const_cast<char*>("-log_exclude_objects");
  
  
  int ierr = PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  if(ierr){
    throw UintahPetscError(ierr, "PetscInitialize", __FILE__, __LINE__);
  }
  delete argv;
    
//  ierr = PetscOptionsSetValue("-log_exclude_actions", "");
//  if(ierr)
//    throw UintahPetscError(ierr, "PetscExcludeActions");
//  ierr = PetscOptionsSetValue("-log_exclude_objects", "");
//  if(ierr)
//    throw UintahPetscError(ierr, "PetscExcludeObjects");
}



//______________________________________________________________________*/
void 
PetscSolver::matrixCreate(const PatchSet* perproc_patches,
                          const PatchSubset* mypatches)
{
  int numProcs = d_myworld->size();
  ASSERTEQ(numProcs, perproc_patches->size());

  vector<int> numCells(numProcs, 0);
  vector<int> startIndex(numProcs);
  int totalCells = 0;

  PetscLocalToGlobalMapping(perproc_patches, mypatches, numCells, totalCells,
                            d_petscGlobalStart, d_petscLocalToGlobal, d_myworld);


  int me = d_myworld->myrank();
  int numlrows      = numCells[me];
  int numlcolumns   = numlrows;
  int globalrows    = (int)totalCells;
  int globalcolumns = (int)totalCells;
  int d_nz = 7;
  int o_nz = 6;
  
  
  //__________________________________
  //  create the Petsc matrix A
  int ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD, numlrows, numlcolumns, globalrows,
                             globalcolumns, d_nz, PETSC_NULL, o_nz, PETSC_NULL, &A);
  if(ierr){
    throw UintahPetscError(ierr, "MatCreateMPIAIJ", __FILE__, __LINE__);
  }
  
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
  
  ierr = VecDuplicate(d_x,&d_u);
  if(ierr)
    throw UintahPetscError(ierr, "VecDuplicate(d_u)", __FILE__, __LINE__);
}

// ****************************************************************************
// Fill linear parallel matrix
// ****************************************************************************
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
     Compute the matrix and right-hand-side vector that define
     the linear system, Ax = b.
      
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
void 
PetscSolver::setMatrix(const ProcessorGroup* ,
                       const Patch* patch,
                       CCVariable<Stencil7>& coeff)
{
  int ierr = 0;
  int col[7];
  double value[7];
  // fill matrix for internal patches
  // make sure that sizeof(d_petscIndex) is the last patch, i.e., appears last in the
  // petsc matrix
  IntVector lowIndex  = patch->getExtraCellLowIndex(Arches::ONEGHOSTCELL);
  IntVector highIndex = patch->getExtraCellHighIndex(Arches::ONEGHOSTCELL);
  
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();
  
  Array3<int> l2g(lowIndex, highIndex);
  l2g.copy(d_petscLocalToGlobal[patch]);

#if 0
  //if ((patchNumber != 0)&&(patchNumber != sizeof(d_petscIndex)-1)) 
#endif
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
      
        IntVector c(colX, colY, colZ);
      
        col[0] = l2g[IntVector(colX,  colY,   colZ-1)]; // ab
        col[1] = l2g[IntVector(colX,  colY-1, colZ)];   // as
        col[2] = l2g[IntVector(colX-1,colY,   colZ)];   // aw
        col[3] = l2g[IntVector(colX,  colY,   colZ)];   // ap
        col[4] = l2g[IntVector(colX+1,colY,   colZ)];   // ae
        col[5] = l2g[IntVector(colX,  colY+1, colZ)];   // an
        col[6] = l2g[IntVector(colX,  colY,   colZ+1)]; // at
        
        value[0] = coeff[c].b;
        value[1] = coeff[c].s;
        value[2] = coeff[c].w;
        value[3] = coeff[c].p;
        value[4] = coeff[c].e;
        value[5] = coeff[c].n;
        value[6] = coeff[c].t;
        int row = col[3];
        ierr = MatSetValues(A,1,&row,7,col,value,INSERT_VALUES);
        if(ierr)
          throw UintahPetscError(ierr, "MatSetValues", __FILE__, __LINE__);
      }
    }
  }

  
  
  
}
// ****************************************************************************
// Fill linear parallel RHS
// ****************************************************************************
void 
PetscSolver::setRHS_X(const ProcessorGroup* ,
                      const Patch* patch,
                      constCCVariable<double>& guess,
                      constCCVariable<double>& rhs, 
                      bool construct_A )
{
  //double solve_start = Time::currentSeconds();
  int ierr;
  //int col[7];
  //double value[7];
  // fill matrix for internal patches
  // make sure that sizeof(d_petscIndex) is the last patch, i.e., appears last in the
  // petsc matrix
  IntVector lowIndex  = patch->getExtraCellLowIndex(Arches::ONEGHOSTCELL);
  IntVector highIndex = patch->getExtraCellHighIndex(Arches::ONEGHOSTCELL);
  
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();
  
  Array3<int> l2g(lowIndex, highIndex);
  l2g.copy(d_petscLocalToGlobal[patch]);

  // assemble right hand side and solution vector
  double vecvalueb, vecvaluex;
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector c(colX, colY, colZ);
        
        vecvalueb = rhs[c];
        vecvaluex = guess[c];
        int row = l2g[c];   
        
        ierr = VecSetValue(d_b, row, vecvalueb, INSERT_VALUES);
        
        if(ierr)
          throw UintahPetscError(ierr, "VecSetValue", __FILE__, __LINE__);

        ierr = VecSetValue(d_x, row, vecvaluex, INSERT_VALUES);
        
        if(ierr)
          throw UintahPetscError(ierr, "VecSetValue", __FILE__, __LINE__);
      }
    }
  }
}

//______________________________________________________________________
//
bool
PetscSolver::pressLinearSolve()
{
  bool test;
  test = PetscLinearSolve(A, d_b, d_x, d_u,
                          d_pcType, d_solverType, d_overlap,
                          d_fill, d_residual, d_maxSweeps, d_myworld);
  return test;
}

//______________________________________________________________________
//
void
PetscSolver::copyPressSoln(const Patch* patch, ArchesVariables* vars)
{
  PetscToUintah_Vector(patch, vars->pressure, d_x, d_petscLocalToGlobal);
}

//______________________________________________________________________
//   Free work space.  All PETSc objects should be destroyed when they are no longer needed
void
PetscSolver::destroyMatrix() 
{
  destroyPetscObjects(A, d_x, d_b, d_u);
}
//______________________________________________________________________
//
void
PetscSolver::print(const string& desc, const int timestep, const int step)
{
  char A_file[100],B_file[100], X_file[100];
  
  PetscViewer matview, vecview;
  sprintf(B_file,"output/b.%s.%i.%i",desc.c_str(), timestep, step);
  sprintf(X_file,"output/x.%s.%i.%i",desc.c_str(), timestep, step);
  sprintf(A_file,"output/A.%s.%i.%i",desc.c_str(), timestep, step);
  
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,B_file,&vecview);
  VecView(d_b,vecview);
  PetscViewerDestroy(vecview);
  
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,X_file,&vecview);
  VecView(d_x,vecview);
  PetscViewerDestroy(vecview);
  
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,A_file,&matview);
  MatView(A,matview);
  PetscViewerDestroy(matview);
}

