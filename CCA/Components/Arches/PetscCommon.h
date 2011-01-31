#ifndef Uintah_Components_Arches_PetscCommon_h
#define Uintah_Components_Arches_PetscCommon_h
#include <sci_defs/petsc_defs.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>
#include <Core/Grid/Patch.h>

#ifdef HAVE_PETSC
extern "C" {
#include "petscksp.h"
}
#endif

namespace Uintah { 
  class ProcessorGroup;
  
  
bool PetscLinearSolve(Mat& A, 
                      Vec& B, Vec& X, Vec& U,
                      const string pcType,
                      const string solverType,
                      const int overlap,
                      const int fill,
                      const double residual,
                      const ProcessorGroup* myworld);
  
  void finalizePetscSolver();


  void destroyPetscObjects(Mat A, Vec X, Vec B, Vec U);
  
  void  PetscToUintah_Vector(const Patch* patch, 
                             CCVariable<double>& var, 
                             Vec X,
                             map<const Patch*, Array3<int> > petscLocalToGlobal);
                             
  void PetscLocalToGlobalMapping(const PatchSet* perproc_patches,
                                 const PatchSubset* mypatches,
                                 vector<int>& numCells,
                                 int& totalCells,
                                 map<const Patch*, int>& petscGlobalStart,
                                 map<const Patch*, Array3<int> >& petscLocalToGlobal,
                                 const ProcessorGroup* myworld);
} // End namespace Uintah

#endif
