
#ifndef Uintah_Components_Arches_RadLinearSolver_h
#define Uintah_Components_Arches_RadLinearSolver_h

#include <sci_defs.h>

#include <Packages/Uintah/CCA/Components/Arches/LinearSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesVariables.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>

#include <Core/Containers/Array1.h>

#ifdef HAVE_PETSC
extern "C" {
#include "petscsles.h"
}
#endif

namespace Uintah {

class ProcessorGroup;

using namespace SCIRun;

/**************************************
CLASS
   RadLinearSolver
   
   Class RadLinearSolver uses gmres solver
   solver

GENERAL INFORMATION
   RadLinearSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class RadLinearSolver is a gmres linear solver

WARNING
   none

****************************************/

class RadLinearSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of a RadLinearSolver.
      RadLinearSolver(const ProcessorGroup* myworld);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual Destructor
      virtual ~RadLinearSolver();

      // GROUP: Problem Setup:
      ////////////////////////////////////////////////////////////////////////
      // Problem setup
      void problemSetup(const ProblemSpecP& params);

      // to close petsc 
      void finalizeSolver();

      void matrixCreate(const PatchSet* allpatches,
				const PatchSubset* mypatches);
      void setMatrix(const ProcessorGroup* pc, const Patch* patch,
		     ArchesVariables* vars,
		     bool xplus, bool yplus, bool zplus,
		     CCVariable<double>& SU,
			   CCVariable<double>& AB,
			   CCVariable<double>& AS,
			   CCVariable<double>& AW,
		     CCVariable<double>& AP);

      bool radLinearSolve();

      virtual void copyRadSoln(const Patch* patch, ArchesVariables* vars);
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
   SLES sles;
#endif
}; // End class RadLinearSolver.h

} // End namespace Uintah

#endif  
  


