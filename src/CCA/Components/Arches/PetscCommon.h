#ifndef Uintah_Components_Arches_PetscCommon_h
#define Uintah_Components_Arches_PetscCommon_h


#include <CCA/Components/Arches/ArchesConstVariables.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <Core/Exceptions/UintahPetscError.h>
#include <Core/Grid/Patch.h>
#include <sci_defs/petsc_defs.h>


#ifdef HAVE_PETSC
extern "C" {
#include "petscksp.h"
}
#endif

namespace Uintah { 
  class ProcessorGroup;
  
  void finalizePetscSolver();  
  
#ifdef HAVE_PETSC  
  bool PetscLinearSolve(Mat& A, 
                        Vec& B, Vec& X, Vec& U,
                        const string pcType,
                        const string solverType,
                        const int overlap,
                        const int fill,
                        const double residual,
                        const int maxIter,
                        const ProcessorGroup* myworld);

  void destroyPetscObjects(Mat A, Vec X, Vec B, Vec U);
                             
  void PetscLocalToGlobalMapping(const PatchSet* perproc_patches,
                                 const PatchSubset* mypatches,
                                 vector<int>& numCells,
                                 int& totalCells,
                                 map<const Patch*, int>& petscGlobalStart,
                                 map<const Patch*, Array3<int> >& petscLocalToGlobal,
                                 const ProcessorGroup* myworld);
                                 
  //______________________________________________________________________
  //  Copy Petsc solution vector back into the Uintah CCVariable<double>array
  template<class T>
  void  PetscToUintah_Vector(const Patch* patch, 
                              T& var, 
                              Vec X, 
                              map<const Patch*, Array3<int> > petscLocalToGlobal)
  {
    IntVector idxLo = patch->getFortranCellLowIndex();
    IntVector idxHi = patch->getFortranCellHighIndex();
    double* xvec;
    int ierr;
    PetscInt begin, end;

    //get the ownership range so we know where the local indicing on this processor begins
    VecGetOwnershipRange(X, &begin, &end);

    ierr = VecGetArray(X, &xvec);
    if(ierr){
      throw UintahPetscError(ierr, "VecGetArray", __FILE__, __LINE__);
    }

    Array3<int> l2g = petscLocalToGlobal[patch];

    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {

          IntVector c(colX, colY, colZ);

          //subtract the begining index from the global index to get to the local array index
          int row = l2g[c] - begin;
          ASSERTRANGE(l2g[c],begin,end);
          var[c] = xvec[row];
        }
      }
    }

    ierr = VecRestoreArray(X, &xvec);
    if(ierr){
      throw UintahPetscError(ierr, "VecRestoreArray", __FILE__, __LINE__);
    }
  }
#endif
} // End namespace Uintah

#endif
