/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//----- PetscSolver.cc ----------------------------------------------

#include <CCA/Components/Arches/Radiation/RadPetscSolver.h>
#include <Core/Thread/Time.h>
#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/Discretization.h>
#include <CCA/Components/Arches/PetscCommon.h>
#include <CCA/Components/Arches/Source.h>
#include <CCA/Components/Arches/StencilMatrix.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/UintahPetscError.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <vector>

using namespace std;
using namespace Uintah;


// ****************************************************************************
// Default constructor for RadPetscSolver
// ****************************************************************************
RadPetscSolver::RadPetscSolver(const ProcessorGroup* myworld)
   : d_myworld(myworld)
{
}

// ****************************************************************************
// Destructor
// ****************************************************************************
RadPetscSolver::~RadPetscSolver()
{
  //finalizePetscSolver();
}

// ****************************************************************************
// Problem setup
// ****************************************************************************
void 
RadPetscSolver::problemSetup(const ProblemSpecP& params)
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
      if (db->findBlock("max_iter")){
        db->require("max_iter", d_maxSweeps);
      }else{
        d_maxSweeps = 75;
      }
      
      if (db->findBlock("ksptype")){
        db->require("ksptype",d_kspType);
      }else{
        d_kspType = "gmres";
      }
      
      if (db->findBlock("res_tol")){
        db->require("res_tol",d_tolerance);
      }else{
        d_tolerance = 1.e-08;
      }
           
      if (db->findBlock("pctype")){
        db->require("pctype", d_pcType);
      }else{
        d_pcType = "blockjacobi";
      }
      
      if (d_pcType == "asm"){
        db->require("overlap",d_overlap);
      }
        
      if (d_pcType == "ilu"){
        db->require("fill",d_fill);
      }
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
  argv = scinew  char*[argc];
  argv[0] = const_cast<char*>("RadPetscSolver::problemSetup");
  argv[1] = const_cast<char*>("-no_signal_handler");
  argv[2] = const_cast<char*>("-log_exclude_actions");
  argv[3] = const_cast<char*>("-log_exclude_objects");
  int ierr = PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  if(ierr)
    throw UintahPetscError(ierr, "PetscInitialize", __FILE__, __LINE__);
  
  delete argv;
}


//______________________________________________________________________
// This function creates the Petsc local to global mapping and some global
// vector/matrix size parameters
void 
RadPetscSolver::matrixCreate(const PatchSet* allpatches,
                              const PatchSubset* mypatches)
{

  //cout << d_myworld->myrank() <<"    RadPetscSolver::matrixCreate " << *mypatches << endl;
  // for global index get a petsc index that
  // make it a data memeber
  int numProcs = d_myworld->size();
  ASSERTEQ(numProcs, allpatches->size());

  vector<int> numCells(numProcs, 0);
  vector<int> startIndex(numProcs);
  int totalCells = 0;
  
  PetscLocalToGlobalMapping(allpatches, mypatches, numCells, totalCells,
                            d_petscGlobalStart, d_petscLocalToGlobal, d_myworld);

  int me          = d_myworld->myrank();
  d_numlrows      = numCells[me];
  d_numlcolumns   = d_numlrows;
  d_globalrows    = (int)totalCells;
  d_globalcolumns = (int)totalCells;

  d_nz = 4;
  o_nz = 3;
}

// ****************************************************************************
// Fill linear parallel matrix
// ****************************************************************************
void 
RadPetscSolver::setMatrix(const ProcessorGroup* ,
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
  //cout << d_myworld->myrank() <<"    RadPetscSolver::setMatrix " << patch->getGridIndex() <<  endl;

  //__________________________________
  //  create the Petsc matrix A and vectors X, B and U.  This routine is called
  // multiple times per radiation solve.
  int ierr;
#if ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR == 3))
  ierr = MatCreateAIJ(PETSC_COMM_WORLD, d_numlrows, d_numlcolumns, d_globalrows,
                         d_globalcolumns, d_nz, PETSC_NULL, o_nz, PETSC_NULL, &A);
#else
  ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD, d_numlrows, d_numlcolumns, d_globalrows,
                         d_globalcolumns, d_nz, PETSC_NULL, o_nz, PETSC_NULL, &A);
#endif
  if(ierr)
    throw UintahPetscError(ierr, "MatCreateMPIAIJ", __FILE__, __LINE__);

  //__________________________________
  //  Create Petsc vectors.  Note that we form 1 vector from scratch and
  //  then duplicate as needed.
  ierr = VecCreateMPI(PETSC_COMM_WORLD,d_numlrows, d_globalrows,&d_x);
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
  IntVector lowIndex  = patch->getExtraCellLowIndex(Arches::ONEGHOSTCELL);
  IntVector highIndex = patch->getExtraCellHighIndex(Arches::ONEGHOSTCELL);
  
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();

  Array3<int> l2g(lowIndex, highIndex);
  l2g.copy(d_petscLocalToGlobal[patch]);
//  MatZeroEntries(A);
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
        IntVector c(colX, colY, colZ);
        
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
        //        }

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
        value[0] = -AB[c];
        value[1] = -AS[c];
        value[2] = -AW[c];
        value[3] =  AP[c];
        //        }

        int row = col[3];
        /*
        if(d_shsolver){
        ierr = MatSetValues(A,1,&row,7,col,value,INSERT_VALUES);
        }
        else{
        */
        int ierr = MatSetValues(A,1,&row,4,col,value,INSERT_VALUES);
        //        }

        if(ierr)
          throw UintahPetscError(ierr, "MatSetValues", __FILE__, __LINE__);
        
        //__________________________________
        //  Fill in B and X  
        vecvalueb = SU[c];
        vecvaluex = vars->cenint[c];
        
        ierr = VecSetValue(d_b, row, vecvalueb, INSERT_VALUES);
        if(ierr){
          throw UintahPetscError(ierr, "VecSetValue", __FILE__, __LINE__);
        }
        
        ierr = VecSetValue(d_x, row, vecvaluex, INSERT_VALUES);
        if(ierr){
          throw UintahPetscError(ierr, "VecSetValue", __FILE__, __LINE__);
        }
      }
    }
  }
}

//______________________________________________________________________
//
bool
RadPetscSolver::radLinearSolve()
{
  //cout << d_myworld->myrank() <<"    RadPetscSolver::radLinearSolve "<<  endl;
  bool test;
  test = PetscLinearSolve(A, d_b, d_x, d_u,
                          d_pcType, d_kspType, d_overlap,
                          d_fill, d_tolerance, d_maxSweeps, d_myworld);
  return test;
}

//______________________________________________________________________
//
void
RadPetscSolver::copyRadSoln(const Patch* patch, ArchesVariables* vars)
{
  PetscToUintah_Vector(patch, vars->cenint, d_x, d_petscLocalToGlobal);
}
//______________________________________________________________________
//  Destroy Petsc objects
void
RadPetscSolver::destroyMatrix() 
{
  destroyPetscObjects(A, d_x, d_b, d_u);
}
