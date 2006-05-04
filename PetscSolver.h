
#ifndef Uintah_Components_Arches_PetscSolver_h
#define Uintah_Components_Arches_PetscSolver_h

#include <sci_defs/petsc_defs.h>

#include <Packages/Uintah/CCA/Components/Arches/LinearSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesVariables.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesConstVariables.h>
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
   PetscSolver
   
   Class PetscSolver uses gmres solver
   solver

GENERAL INFORMATION
   PetscSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class PetscSolver is a gmres linear solver

WARNING
   none

****************************************/

class PetscSolver: public LinearSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of a PetscSolver.
      PetscSolver(const ProcessorGroup* myworld);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual Destructor
      virtual ~PetscSolver();

      // GROUP: Problem Setup:
      ////////////////////////////////////////////////////////////////////////
      // Problem setup
      void problemSetup(const ProblemSpecP& params);

       // to close petsc 
      void finalizeSolver();

   virtual void matrixCreate(const PatchSet* allpatches,
			     const PatchSubset* mypatches);
   virtual void setPressMatrix(const ProcessorGroup* pc, const Patch* patch,
			       ArchesVariables* vars,
			       ArchesConstVariables* constvars,
			       const ArchesLabel* lab);
   

   virtual bool pressLinearSolve();
   virtual void copyPressSoln(const Patch* patch, ArchesVariables* vars);
   virtual void destroyMatrix();
protected:

private:
      string d_pcType;
      string d_kspType;
      int d_overlap;
      int d_fill;
      int d_maxSweeps;
      double d_convgTol; // convergence tolerence
      double d_residual;
   const ProcessorGroup* d_myworld;
#ifdef HAVE_PETSC
   map<const Patch*, int> d_petscGlobalStart;
   map<const Patch*, Array3<int> > d_petscLocalToGlobal;
   Mat A;
   Vec d_x, d_b, d_u;
#endif
}; // End class PetscSolver.h

} // End namespace Uintah

#endif  
  
