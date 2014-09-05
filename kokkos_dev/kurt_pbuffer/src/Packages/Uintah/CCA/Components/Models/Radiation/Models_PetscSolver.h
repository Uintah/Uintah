
#ifndef Uintah_Components_Models_PetscSolver_h
#define Uintah_Components_Models_PetscSolver_h

#include <sci_defs/petsc_defs.h>

#include <Packages/Uintah/CCA/Components/Models/Radiation/Models_RadiationSolver.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/RadiationVariables.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>

#include <Core/Containers/Array1.h>

#ifdef HAVE_PETSC
extern "C" {
#include "petscksp.h"
}
#endif

namespace Uintah {

class ProcessorGroup;

using namespace SCIRun;

/**************************************
CLASS
   Models_PetscSolver
   
   Class Models_PetscSolver PETSc
   solver

GENERAL INFORMATION
   Models_PetscSolver.h - declaration of the class
   
   Author: Seshadri Kumar (skumar@crsim.utah.edu)
   
   Creation Date:   April 11, 2005
   
   C-SAFE 
   
   Copyright University of Utah 2005

KEYWORDS


DESCRIPTION
   Class Models_PetscSolver is a gmres linear solver

WARNING
   none

****************************************/

class Models_PetscSolver: public Models_RadiationSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of a Models_PetscSolver.
      Models_PetscSolver(const ProcessorGroup* myworld);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual Destructor
      virtual ~Models_PetscSolver();

      // GROUP: Problem Setup:
      ////////////////////////////////////////////////////////////////////////
      // Problem setup
      void problemSetup(const ProblemSpecP& params);

      // to close petsc 
      void finalizeSolver();

      void matrixCreate(const PatchSet* allpatches,
				const PatchSubset* mypatches);
      void setMatrix(const ProcessorGroup* pc, const Patch* patch,
		     RadiationVariables* vars,
		     bool xplus, bool yplus, bool zplus,
		     CCVariable<double>& SU,
		     CCVariable<double>& AB,
		     CCVariable<double>& AS,
		     CCVariable<double>& AW,
		     CCVariable<double>& AP,
		     CCVariable<double>& AE,
		     CCVariable<double>& AN,
		     CCVariable<double>& AT);

      bool radLinearSolve();

      virtual void copyRadSoln(const Patch* patch, RadiationVariables* vars);
      virtual void destroyMatrix();
protected:

private:
      int numlrows;
      int numlcolumns;
      int globalrows;
      int globalcolumns;
      int d_nz;
      int o_nz;
      string d_pcType;
      string d_kspType;
      int d_overlap;
      int d_fill;
      int d_maxSweeps;
      bool d_shsolver;
      double d_tolerance; // convergence tolerence
      double d_underrelax;
      double d_initResid;
      double d_residual;
   const ProcessorGroup* d_myworld;
#ifdef HAVE_PETSC
   map<const Patch*, int> d_petscGlobalStart;
   map<const Patch*, Array3<int> > d_petscLocalToGlobal;
   Mat A;
   Vec d_x, d_b, d_u;
#endif
}; // End class Models_PetscSolver.h

} // End namespace Uintah

#endif  
  


