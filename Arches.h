//----- Arches.h -----------------------------------------------

#ifndef Uintah_Component_Arches_Arches_h
#define Uintah_Component_Arches_Arches_h

/**************************************

CLASS
   Arches
   
   Short description...

GENERAL INFORMATION

   Arches.h

   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   Department of Chemical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 University of Utah

KEYWORDS
   Arches

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

#include <sci_defs.h>

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Components/SimulationController/SimulationController.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/Handle.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>

// Divergence constraint instead of drhodt in pressure equation
//#define divergenceconstraint

// Exact Initialization for first time step in
// MPMArches problem, to eliminate problem of
// sudden appearance of mass in second step
//
// #define ExactMPMArchesInitialize

// Choices of scheme for convection of velocity
// So far only central differencing is implemented for filtered convection terms

//#define filter_convection_terms

// What to do with negative velocity correction
// The velocity on the outlet will not be negative in any case as requested
// by Rajesh

//#define discard_negative_velocity_correction

#ifdef HAVE_PETSC
  #define PetscFilter
#endif

// Filtering of drhodt is now an input parameter for Properties

namespace Uintah {

  class VarLabel;
  class PhysicalConstants;
  class NonlinearSolver;
  class Properties;
  class TurbulenceModel;
  class BoundaryCondition;
  class MPMArchesLabel;
  class ArchesLabel;
#ifdef PetscFilter
  class Filter;
#endif

class Arches : public UintahParallelComponent, public SimulationInterface {

public:

      // GROUP: Static Variables:
      ////////////////////////////////////////////////////////////////////////
      // Number of dimensions in the problem
      static const int NDIM;

      // GROUP: Constants:
      ////////////////////////////////////////////////////////////////////////
      enum d_eqnType { PRESSURE, MOMENTUM, SCALAR };
      enum d_dirName { NODIR, XDIR, YDIR, ZDIR };
      enum d_stencilName { AP, AE, AW, AN, AS, AT, AB };
      enum d_numGhostCells {ZEROGHOSTCELLS , ONEGHOSTCELL, TWOGHOSTCELLS,
			    THREEGHOSTCELLS, FOURGHOSTCELLS, FIVEGHOSTCELLS };

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Arches constructor
      Arches(const ProcessorGroup* myworld);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual destructor 
      virtual ~Arches();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      // Set up the problem specification database
      virtual void problemSetup(const ProblemSpecP& params, 
				GridP& grid,
				SimulationStateP&);

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      // Schedule initialization
      virtual void scheduleInitialize(const LevelP& level,
				      SchedulerP&);
	 
      ///////////////////////////////////////////////////////////////////////
      // Schedule parameter initialization
      virtual void sched_paramInit(const LevelP& level,
				   SchedulerP&);
      
      ///////////////////////////////////////////////////////////////////////
      // Schedule Compute if Stable time step
      virtual void scheduleComputeStableTimestep(const LevelP& level,
						 SchedulerP&);

      ///////////////////////////////////////////////////////////////////////
      // Schedule time advance
      virtual void scheduleTimeAdvance( const LevelP& level, 
					SchedulerP&, int step, int nsteps );

      ///////////////////////////////////////////////////////////////////////
       // Function to return boolean for recompiling taskgraph

	virtual bool need_recompile(double time, double dt,
			    const GridP& grid);

      // GROUP: Access Functions :
      ///////////////////////////////////////////////////////////////////////
	// Boolean to see whether or not Enthalpy is solved for
      inline bool checkSolveEnthalpy() const{
	return d_calcEnthalpy;
      }

      // for multimaterial
      void setMPMArchesLabel(const MPMArchesLabel* MAlb)
	{
	  d_MAlab = MAlb;
	}

      const ArchesLabel* getArchesLabel()
	{
	  return d_lab;
	}
      NonlinearSolver* getNonlinearSolver()
	{
	  return d_nlSolver;
	}

      BoundaryCondition* getBoundaryCondition()
	{
	  return d_boundaryCondition;
	}

      TurbulenceModel* getTurbulenceModel()
	{
	  return d_turbModel;
	}


protected:

private:

      // GROUP: Constructors (Private):
      ////////////////////////////////////////////////////////////////////////
      // Default Arches constructor
      Arches();

      ////////////////////////////////////////////////////////////////////////
      // Arches copy constructor
      Arches(const Arches&);

      // GROUP: Overloaded Operators (Private):
      ////////////////////////////////////////////////////////////////////////
      // Arches assignment constructor
      Arches& operator=(const Arches&);

      // GROUP: Action Methods (Private):
      ////////////////////////////////////////////////////////////////////////
      // Arches assignment constructor
      void paramInit(const ProcessorGroup*,
		     const PatchSubset* patches,
		     const MaterialSubset*,
		     DataWarehouse* ,
		     DataWarehouse* new_dw );

      void computeStableTimeStep(const ProcessorGroup* ,
				 const PatchSubset* patches,
				 const MaterialSubset* matls,
				 DataWarehouse* ,
				 DataWarehouse* new_dw);

private:

      double d_deltaT;
      int d_nofScalars;
      int d_nofScalarStats;
      bool d_variableTimeStep;
      bool d_reactingFlow;
      bool d_calcReactingScalar;
      bool d_calcEnthalpy;
      PhysicalConstants* d_physicalConsts;
      NonlinearSolver* d_nlSolver;
      // properties...solves density, temperature and species concentrations
      Properties* d_props;
      // Turbulence Model
      TurbulenceModel* d_turbModel;
      // Boundary conditions
      BoundaryCondition* d_boundaryCondition;
      SimulationStateP d_sharedState;
      // Variable labels that are used by the simulation controller
      ArchesLabel* d_lab;
      // for multimaterial
      const MPMArchesLabel* d_MAlab;
#ifdef PetscFilter
      Filter* d_filter;
#endif

      int nofTimeSteps;
#ifdef multimaterialform
      MultiMaterialInterface* d_mmInterface;
#endif

    string d_timeIntegratorType;

    bool d_recompile;

}; // end class Arches

} // End namespace Uintah

#endif

