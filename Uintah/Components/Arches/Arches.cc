/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/Arches.h>
#include <Uintah/Components/Arches/PicardNonlinearSolver.h>
#include <Uintah/Components/Arches/PhysicalConstants.h>
#include <Uintah/Components/Arches/SmagorinskyModel.h>
#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Components/Arches/Properties.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Grid/SoleVariable.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Task.h>
#include <iostream>
using std::cerr;
using std::endl;

using Uintah::Components::Arches;
using Uintah::Exceptions::InvalidValue;
using namespace Uintah::Grid;

namespace Uintah {
namespace Components {

Arches::Arches( int MpiRank, int MpiProcesses ) :
  UintahParallelComponent( MpiRank, MpiProcesses )
{
}

Arches::~Arches()
{

}

void Arches::problemSetup(const ProblemSpecP& params, GridP&,
			  const SimulationStateP&)
{
  ProblemSpecP db = params->findBlock("CFD")->findBlock("Arches");

  db->require("grow_dt", d_deltaT);
  // physical constants
  d_physicalConsts = new PhysicalConstants();
  d_physicalConsts->problemSetup(params);
  // read properties, boundary and turbulence model
  d_props = new Properties();
  d_props->problemSetup(db);
  string turbModel;
  db->require("turbulence_model", turbModel);
  if (turbModel == "Smagorinsky") 
    d_turbModel = new SmagorinskyModel(d_physicalConsts);
  else 
    throw InvalidValue("Turbulence Model not supported" + turbModel);
  d_turbModel->problemSetup(db);
  d_boundaryCondition = new BoundaryCondition(d_turbModel);
  d_boundaryCondition->problemSetup(db);
  string nlSolver;
  db->require("nonlinear_solver", nlSolver);
  if(nlSolver == "picard")
    d_nlSolver = new PicardNonlinearSolver(d_props, d_boundaryCondition,
					   d_turbModel, d_physicalConsts);
  else
    throw InvalidValue("Nonlinear solver not supported: "+nlSolver);

  //d_nlSolver->problemSetup(db, dw); /* 2 params ? */
  d_nlSolver->problemSetup(db);
}

void Arches::problemInit(const LevelP& level,
			 SchedulerP& sched, DataWarehouseP& dw,
			 bool restrt )
{
#if 0
  // initializes variables
  if (!restrt) {
    sched_paramInit(level, sched, dw);
    // initialize velocity, scalars and properties at the boundary
    d_boundaryCondition->sched_setProfile(level, sched, dw);
  }
  d_properties->sched_computeProperties(level, sched, dw, dw);
  d_turbModel->sched_computeTurbSubmodel(level, sched, dw, dw);
  d_boundaryCondition->sched_pressureBC(level, sched, dw, dw);
#endif
}

void Arches::scheduleInitialize(const LevelP& level,
				   SchedulerP& sched,
				   DataWarehouseP& dw)
{
   cerr << "SerialMPM::scheduleInitialize not done\n";
}

void Arches::scheduleComputeStableTimestep(const LevelP& level,
					   SchedulerP& sched,
					   DataWarehouseP& dw)
{
#ifdef WONT_COMPILE_YET
  dw->put(SoleVariable<double>(d_deltaT), "delt"); 
#endif
}

void Arches::scheduleTimeAdvance(double time, double dt,
	      const LevelP& level, SchedulerP& sched,
	      const DataWarehouseP& old_dw, DataWarehouseP& new_dw)
{
#ifdef WONT_COMPILE_YET
  int error_code = d_nlSolver->nonlinearSolve(time, dt, level, 
					      sched, old_dw, new_dw);
  if (!error_code) {
#if 0
    old_dw = new_dw;
#endif
  }
  else {
    cerr << "Nonlinear Solver didn't converge" << endl;
  }
#endif
}

void Arches::sched_paramInit(const LevelP& level,
			     SchedulerP& sched, DataWarehouseP& dw)
{
#ifdef WONT_COMPILE_YET
    for(Level::const_regionIterator iter=level->regionsBegin();
      iter != level->regionsEnd(); iter++){
    const Region* region=*iter;
    {
      Task* tsk = new Task("Arches::Initialization",
			   region, dw, this,
			   Arches::paramInit);
      tsk->computes(dw, "velocity", region, 0,
		    FCVariable<Vector>::getTypeDescription());
      tsk->computes(dw, "pressure", region, 0,
		    CCVariable<double>::getTypeDescription());
      tsk->computes(dw, "scalars", region, 0,
		    CCVariable<Vector>::getTypeDescription());
      tsk->computes(dw, "density", region, 0,
		    CCVariable<double>::getTypeDescription());
      tsk->computes(dw, "viscosity", region, 0,
		    CCVariable<double>::getTypeDescription());
      sched->addTask(tsk);
    }
  }
#endif
}

void Arches::paramInit(const ProcessorContext*,
		       const Region* region,
		       const DataWarehouseP& old_dw)
{
#ifdef WONT_COMPILE_YET
  FCVariable<Vector> velocity;
  // ....but will only compute for computational domain
  old_dw->allocate(velocity,"velocity",region, 1);
  CCVariable<double> pressure;
  old_dw->allocate(pressure, "pressure", region, 1);
  CCVariable<Vector> scalar;
  old_dw->allocate(scalar, "scalar", region, 1);
  CCVariable<double> density;
  old_dw->allocate(density, "density", region, 1);
  CCVariable<double> viscosity;
  old_dw->allocate(viscosity, "viscosity", region, 1);
  Array3Index lowIndex = region->getLowIndex();
  Array3Index highIndex = region->getHighIndex();

  FORT_INIT(velocity, pressure, scalar, density, viscosity,
	    lowIndex, highIndex);
  old_dw->put(velocity, "velocity", region, 0);
  old_dw->put(pressure, "pressure", region, 0);
  old_dw->put(scalar, "scalar", region, 0);
  old_dw->put(density, "density", region, 0);
  old_dw->put(viscosity, "viscosity", region, 0);
#endif
}
  



} // end namespace Components
} // end namespace Uintah

//
// $Log$
// Revision 1.21  2000/04/20 18:56:10  sparker
// Updates to MPM
//
// Revision 1.20  2000/04/19 20:59:11  dav
// adding MPI support
//
// Revision 1.19  2000/04/19 05:25:56  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.18  2000/04/13 06:50:50  sparker
// More implementation to get this to work
//
// Revision 1.17  2000/04/12 22:58:29  sparker
// Resolved conflicts
// Making it compile
//
// Revision 1.16  2000/04/12 22:46:42  dav
// changes to make it compile
//
// Revision 1.15  2000/04/11 19:55:51  rawat
// modified nonlinear solver for initialization
//
// Revision 1.14  2000/04/11 07:10:35  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.13  2000/04/10 23:11:23  rawat
// Added sub-grid scale turbulence models
//
// Revision 1.12  2000/04/09 00:53:50  rawat
// added PhysicalConstants and Properties classes
//
// Revision 1.11  2000/04/07 18:30:12  rawat
// Added problem initialization function in Arches.cc
//
// Revision 1.10  2000/03/31 17:35:05  moulding
// A call to d_nlSolver->problemSetup() was attempted with 2 parameters, causing
// a compile error (too many parameters).  I changed it to one parameter.
// The change is at line 40.
//
// Revision 1.9  2000/03/29 21:18:16  rawat
// modified boundarycondition.cc for inlet bcs
//
// Revision 1.8  2000/03/23 20:05:13  jas
// Changed the location of ProblemSpec from Grid to Interface in the include
// file path.
//
// Revision 1.7  2000/03/21 23:14:47  rawat
// Implemented more of problem setup routines
//
// Revision 1.6  2000/03/21 21:27:03  dav
// namespace fixs
//
//

