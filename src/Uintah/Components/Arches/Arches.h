
#ifndef Uintah_Component_Arches_Arches_h
#define Uintah_Component_Arches_Arches_h

/*
 * Placeholder - nothing here yet
 */

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Parallel/ProcessorContext.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Interface/CFDInterface.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Region.h>
#include <Uintah/Grid/VarTypes.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SimulationStateP.h>

#include <SCICore/Geometry/Vector.h>

namespace Uintah {

class VarLabel;

namespace ArchesSpace {

class PhysicalConstants;
class NonlinearSolver;
class Properties;
class TurbulenceModel;
class BoundaryCondition;

/**************************************

CLASS
   Arches
   
   Short description...

GENERAL INFORMATION

   Arches.h

   Author
   Department
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Arches

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class Arches : public UintahParallelComponent, public CFDInterface {
public:
    static const int NDIM = 3;

    Arches( int MpiRank, int MpiProcesses );
    virtual ~Arches();

    virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
			      SimulationStateP&);

    virtual void problemInit(const LevelP& level,
			     SchedulerP& sched, DataWarehouseP& dw,
			     bool restrt);
    virtual void scheduleInitialize(const LevelP& level,
				    SchedulerP&,
				    DataWarehouseP&);
	 
    virtual void scheduleComputeStableTimestep(const LevelP& level,
					       SchedulerP&,
					       DataWarehouseP&);
    virtual void scheduleTimeAdvance(double t, double dt,
				     const LevelP& level, SchedulerP&,
				     const DataWarehouseP&, DataWarehouseP&);
    void sched_paramInit(const LevelP& level,
			 SchedulerP& sched, DataWarehouseP& dw);

private:
    Arches(const Arches&);
    Arches& operator=(const Arches&);
    void paramInit(const ProcessorContext*,
		   const Region* region,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& );
    double d_deltaT;
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
    // 
    //    const VarLabel* d_deltLabel;
    const VarLabel* d_densityLabel;
    const VarLabel* d_pressureLabel;
    const VarLabel* d_scalarLabel;
    const VarLabel* d_velocityLabel;
    const VarLabel* d_viscosityLabel;

};

} // end namespace ArchesSpace
} // end namespace Uintah

//
// $Log$
// Revision 1.21  2000/05/30 17:06:11  dav
// added Cocoon doc template.  fixed non compilation problem.
//
// Revision 1.20  2000/05/30 15:44:58  rawat
// modified computeStableTimestep
//
// Revision 1.19  2000/05/20 22:54:14  bbanerje
// Again, adding the first set of changes to get the scheduler to add tasks.
//
// Revision 1.18  2000/05/09 22:56:22  sparker
// Changed name of namespace
//
// Revision 1.17  2000/04/26 06:48:00  sparker
// Streamlined namespaces
//
// Revision 1.16  2000/04/24 21:04:20  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.15  2000/04/20 18:56:10  sparker
// Updates to MPM
//
// Revision 1.14  2000/04/19 20:59:11  dav
// adding MPI support
//
// Revision 1.13  2000/04/19 05:25:56  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.12  2000/04/13 06:50:50  sparker
// More implementation to get this to work
//
// Revision 1.11  2000/04/11 19:55:52  rawat
// modified nonlinear solver for initialization
//
// Revision 1.10  2000/04/07 23:02:16  sparker
// Fixed arches compile
//
// Revision 1.9  2000/04/07 18:30:12  rawat
// Added problem initialization function in Arches.cc
//
// Revision 1.8  2000/03/23 20:05:13  jas
// Changed the location of ProblemSpec from Grid to Interface in the include
// file path.
//
// Revision 1.7  2000/03/21 21:27:03  dav
// namespace fixs
//
//

#endif

