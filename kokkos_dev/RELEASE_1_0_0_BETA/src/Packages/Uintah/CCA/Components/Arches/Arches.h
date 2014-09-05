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

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/CCA/Ports/CFDInterface.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/MPMArches/MPMArchesLabel.h>

namespace Uintah {

class VarLabel;
class PhysicalConstants;
class NonlinearSolver;
class Properties;
class TurbulenceModel;
class BoundaryCondition;

class Arches : public UintahParallelComponent, public CFDInterface {

public:

      // GROUP: Static Variables:
      ////////////////////////////////////////////////////////////////////////
      // Number of dimensions in the problem
      static const int NDIM = 3;

      // GROUP: Constants:
      ////////////////////////////////////////////////////////////////////////
      enum d_eqnType { PRESSURE, MOMENTUM, SCALAR };
      enum d_dirName { NODIR, XDIR, YDIR, ZDIR };
      enum d_stencilName { AP, AE, AW, AN, AS, AT, AB };

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
				      SchedulerP&,
				      DataWarehouseP&);
	 
      ///////////////////////////////////////////////////////////////////////
      // Schedule parameter initialization
      virtual void sched_paramInit(const LevelP& level,
				   SchedulerP&,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw);
      
      ///////////////////////////////////////////////////////////////////////
      // Schedule Compute if Stable time step
      virtual void scheduleComputeStableTimestep(const LevelP& level,
						 SchedulerP&,
						 DataWarehouseP&);

      ///////////////////////////////////////////////////////////////////////
      // Schedule time advance
      virtual void scheduleTimeAdvance(double t, double dt,
				       const LevelP& level, 
				       SchedulerP&,
				       DataWarehouseP&, 
				       DataWarehouseP&);


      // for multimaterial
      void setMPMArchesLabel(const MPMArchesLabel* MAlb)
	{
	  d_MAlab = MAlb;
	}

      const ArchesLabel* getArchesLabel()
	{
	  return d_lab;
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
		     const Patch* patch,
		     DataWarehouseP& old_dw,
		     DataWarehouseP& );

private:

      double d_deltaT;
      int d_nofScalars;

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
      const ArchesLabel* d_lab;
      // for multimaterial
      const MPMArchesLabel* d_MAlab;
#ifdef multimaterialform
      MultiMaterialInterface* d_mmInterface;
#endif

}; // end class Arches

} // End namespace Uintah

#endif

