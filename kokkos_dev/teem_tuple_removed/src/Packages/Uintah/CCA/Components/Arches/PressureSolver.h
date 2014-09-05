//----- PressureSolver.h -----------------------------------------------

#ifndef Uintah_Components_Arches_PressureSolver_h
#define Uintah_Components_Arches_PressureSolver_h

#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Core/Geometry/IntVector.h>

namespace Uintah {

  class MPMArchesLabel;
  class ArchesLabel;
class ProcessorGroup;
class ArchesVariables;
class TurbulenceModel;
class PhysicalConstants;
class Discretization;
class Source;
class BoundaryCondition;
class LinearSolver;
class TimeIntegratorLabel;

using namespace SCIRun;

/**************************************

CLASS
   PressureSolver
   
   Class PressureSolver linearizes and solves pressure
   equation on a grid hierarchy


GENERAL INFORMATION
   PressureSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class PressureSolver linearizes and solves pressure
   equation on a grid hierarchy


WARNING
   none

************************************************************************/

class PressureSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of the Pressure solver.
      // PRECONDITIONS
      // POSTCONDITIONS
      //   A linear level solver is partially constructed.  
      PressureSolver(const ArchesLabel* label,
		     const MPMArchesLabel* MAlb,
		     TurbulenceModel* turb_model, 
		     BoundaryCondition* bndry_cond,
		     PhysicalConstants* physConst,
		     const ProcessorGroup* myworld);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Destructor
      ~PressureSolver();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      // Set up the problem specification database
      void problemSetup(const ProblemSpecP& params);

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      // Schedule Solve of linearized pressure equation
      void solve(const LevelP& level, SchedulerP&,
		 const TimeIntegratorLabel* timelabels);
   
      ///////////////////////////////////////////////////////////////////////
      // Schedule the build of the linearized eqn
      void sched_buildLinearMatrix(SchedulerP&, const PatchSet* patches,
				   const MaterialSet* matls,
		 		   const TimeIntegratorLabel* timelabels);
 
      void sched_pressureLinearSolve(const LevelP& level, SchedulerP& sched,
		 		     const TimeIntegratorLabel* timelabels);

      ///////////////////////////////////////////////////////////////////////
      // Schedule the addition of the hydrostatic term to the relative pressure

      void sched_addHydrostaticTermtoPressure(SchedulerP& sched,
					      const PatchSet* patches,
					      const MaterialSet* matls);

      inline void setPressureCorrectionFlag(bool pressure_correction) {
	d_pressure_correction = pressure_correction;
      }
protected:

private:

      // GROUP: Constructors (Private):
      ////////////////////////////////////////////////////////////////////////
      // Default : Construct an empty instance of the Pressure solver.
      PressureSolver(const ProcessorGroup* myworld);

      // GROUP: Action Methods (private) :
      ///////////////////////////////////////////////////////////////////////
      // Actually Build the linear matrix
      //    [in] 
      //        add documentation here
      void buildLinearMatrix(const ProcessorGroup* pc,
			     const PatchSubset* patches,
			     const MaterialSubset* matls,
			     DataWarehouse* new_dw,
			     DataWarehouse* matrix_dw,
		 	     const TimeIntegratorLabel* timelabels);

      void pressureLinearSolve_all(const ProcessorGroup* pc,
				   const PatchSubset* patches,
                                   const MaterialSubset* matls,
				   DataWarehouse* new_dw,
				   DataWarehouse* matrix_dw,
		 		   const TimeIntegratorLabel* timelabels);

      void pressureLinearSolve(const ProcessorGroup* pc,
			       const Patch* patch,
			       const int matlIndex,
			       DataWarehouse* new_dw,
			       DataWarehouse* matrix_dw,
			       ArchesVariables& pressureVars,
		 	       const TimeIntegratorLabel* timelabels);

      ////////////////////////////////////////////////////////////////
      // addition of hydrostatic term to relative pressure

      void addHydrostaticTermtoPressure(const ProcessorGroup* pc,
					const PatchSubset* patches,
					const MaterialSubset* matls,
					DataWarehouse* old_dw,
					DataWarehouse* new_dw);
      
      ///////////////////////////////////////////////////////////////////////
      // Actually do normPressure
      //    [in] 
      //        add documentation here
      void normPressure(const ProcessorGroup* pc,
			const Patch* patch,
			ArchesVariables* vars);

      void updatePressure(const ProcessorGroup* pc,
			  const Patch* patch,
			  ArchesVariables* vars);

 private:

      // const VarLabel* (required)
      const ArchesLabel* d_lab;
      // for multimaterial
      const MPMArchesLabel* d_MAlab;
      
      // computes coefficients
      Discretization* d_discretize;
      // computes sources
      Source* d_source;
      // linear solver
      LinearSolver* d_linearSolver;
      // turbulence model
      TurbulenceModel* d_turbModel;
      // boundary condition
      BoundaryCondition* d_boundaryCondition;
      // physical constants
      PhysicalConstants* d_physicalConsts;

      // Maximum number of iterations to take before stopping/giving up.
      int d_maxIterations;
      // underrealaxation parameter, read from an input database
      double d_underrelax;
      //reference points for the solvers
      IntVector d_pressRef;
      const Patch* d_pressRefPatch;
      int d_pressRefProc;
      const PatchSet* d_perproc_patches;

      const ProcessorGroup* d_myworld;
#ifdef multimaterialform
      // set the values in problem setup
      MultiMaterialInterface* d_mmInterface;
      MultiMaterialSGSModel* d_mmSGSModel;
#endif
      bool d_pressure_correction;
      bool d_norm_pres;
}; // End class PressureSolver

} // End namespace Uintah

#endif

