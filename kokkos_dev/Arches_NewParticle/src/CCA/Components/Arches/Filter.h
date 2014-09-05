/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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



#ifndef Uintah_Components_Arches_Filter_h
#define Uintah_Components_Arches_Filter_h


#include <CCA/Components/Arches/CellInformationP.h>
#include <CCA/Components/Arches/PetscCommon.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/SchedulerP.h>
#include <Core/Containers/Array1.h>
#include <Core/Exceptions/UintahPetscError.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/Array3.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <sci_defs/petsc_defs.h>


#ifdef HAVE_PETSC
#undef PETSC_USE_LOG
extern "C" {
#  include <petscmat.h>
}
#endif

namespace Uintah {

class ProcessorGroup;
class ArchesLabel;
class BoundaryCondition;

/**************************************
CLASS
   Filter
   
   Class Filter uses petsc's matmult operation
   solver

GENERAL INFORMATION
   Filter.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)
   
   Creation Date:   July 10, 2002
   
   C-SAFE 
   
   Copyright U of U 2002

KEYWORDS


DESCRIPTION
   Class Filter uses petsc matmult operation for applying box filter

WARNING
   none

****************************************/
class Filter {

public:

  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  // Construct an instance of a Filter.
  Filter(const ArchesLabel* label,
         BoundaryCondition* bndryCondition,
         const ProcessorGroup* myworld);

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Destructor
  ~Filter();

  // GROUP: Problem Setup:
  ////////////////////////////////////////////////////////////////////////
  // Problem setup
  void problemSetup(const ProblemSpecP& params);

  bool isInitialized() {
    return d_matrixInitialize;
  }
  void sched_buildFilterMatrix(const LevelP& level,
                               SchedulerP& sched);

  void buildFilterMatrix(const ProcessorGroup* pg,
                         const PatchSubset* patches,
                         const MaterialSubset*,
                         DataWarehouse*,
                         DataWarehouse* new_dw);

  // constructs filter matrix will be different for different types of filters

  void matrixCreate(const PatchSet* allpatches,
                    const PatchSubset* mypatches);
                    
  void setFilterMatrix(const ProcessorGroup* pc, 
                       const Patch* patch,
                       CellInformation* cellinfo, 
                       constCCVariable<int>& cellType);            
                       
  void destroyMatrix();
  
//______________________________________________________________________
//
template<class T>
bool applyFilter(const ProcessorGroup* ,
                 const Patch* patch,               
                 T& var,                           
                 Array3<double>& filterVar)        
{
//  TAU_PROFILE("applyFilter", "[Filter::applyFilter]" , TAU_USER);
  // assemble x vector
  int ierr;
  // fill matrix for internal patches
  // make sure that sizeof(d_petscIndex) is the last patch, i.e., appears last in the
  // petsc matrix
  int oneGhostCell = 1;
  IntVector lowIndex  = patch->getExtraCellLowIndex(oneGhostCell);
  IntVector highIndex = patch->getExtraCellHighIndex(oneGhostCell);

  Array3<int> l2g(lowIndex, highIndex);
  l2g.copy(d_petscLocalToGlobal[patch]);

  // #ifdef notincludeBdry
#if 1
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();
#else
  IntVector idxLo = patch->getExtraCellLowIndex();
  IntVector idxHi = patch->getExtraCellHighIndex()-IntVector(1,1,1);
#endif
  IntVector inputLo = idxLo;
  IntVector inputHi = idxHi;

  double vecvaluex;
  for (int colZ = inputLo.z(); colZ <= inputHi.z(); colZ ++) {
    for (int colY = inputLo.y(); colY <= inputHi.y(); colY ++) {
      for (int colX = inputLo.x(); colX <= inputHi.x(); colX ++) {
        
        vecvaluex = var[IntVector(colX, colY, colZ)];
        int row = l2g[IntVector(colX, colY, colZ)];         
        
        ASSERT(!isnan(vecvaluex));
        ierr = VecSetValue(d_x, row, vecvaluex, INSERT_VALUES);
        if(ierr)
          throw UintahPetscError(ierr, "VecSetValue", __FILE__, __LINE__);
      }
    }
  }

  //__________________________________
  //  Matrix A
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  if(ierr)
    throw UintahPetscError(ierr, "MatAssemblyBegin", __FILE__, __LINE__);
  
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
  if(ierr)
    throw UintahPetscError(ierr, "MatAssemblyEnd", __FILE__, __LINE__);

  //ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);


  //__________________________________
  //  Vector B
  ierr = VecAssemblyBegin(d_b);
  if(ierr)
    throw UintahPetscError(ierr, "VecAssemblyBegin", __FILE__, __LINE__);
  
  ierr = VecAssemblyEnd(d_b);
  if(ierr)
    throw UintahPetscError(ierr, "VecAssemblyEnd", __FILE__, __LINE__);
  
  //__________________________________
  //  Vector X
  ierr = VecAssemblyBegin(d_x);
  if(ierr)
    throw UintahPetscError(ierr, "VecAssemblyBegin", __FILE__, __LINE__);
  
  ierr = VecAssemblyEnd(d_x);
  if(ierr)
    throw UintahPetscError(ierr, "VecAssemblyEnd", __FILE__, __LINE__);
  
  //__________________________________
  //  Solve 
  ierr = MatMult(A, d_x, d_b);
  if(ierr)
    throw UintahPetscError(ierr, "MatMult", __FILE__, __LINE__);
  
  
  //__________________________________
  // copy vector b in the filterVar array
#if 0
  ierr = VecView(d_x, PETSC_VIEWER_STDOUT_WORLD);
  ierr = VecView(d_b, PETSC_VIEWER_STDOUT_WORLD);
#endif


  Uintah::PetscToUintah_Vector<Array3<double> >(patch, filterVar, d_b, d_petscLocalToGlobal);

  return true;
}
//______________________________________________________________________
protected:

private:
  const ProcessorGroup* d_myworld;
  const PatchSet* d_perproc_patches;
  const ArchesLabel* d_lab;
  BoundaryCondition* d_boundaryCondition;

  bool d_matrixInitialize;
  bool d_matrix_vectors_created;
#ifdef HAVE_PETSC
  map<const Patch*, int> d_petscGlobalStart;
  map<const Patch*, Array3<int> > d_petscLocalToGlobal;
  Mat A;
  Vec d_x, d_b;
  int d_nz, o_nz; // number of non zero values in a row
#endif
}; // End class Filter.h

} // End namespace Uintah

#endif  
  





