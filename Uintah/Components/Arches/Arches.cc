/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/Arches.h>
#include <Uintah/Components/Arches/PicardNonlinearSolver.h>
#if 0
#include <Uintah/Components/Arches/SmagorinskyModel.h>
#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Components/Arches/Properties.h>
#endif
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Grid/SoleVariable.h>

using Uintah::Components::Arches;
using namespace Uintah::Grid;

namespace Uintah {
namespace Components {

Arches::Arches()
{
}

Arches::~Arches()
{

}

void Arches::problemSetup(const ProblemSpecP& params, GridP&,
			  DataWarehouseP& dw)
{
#if 0
  ProblemSpecP db = params->findBlock("Arches");

  db->require("grow_dt", d_deltaT);
  // physical constants
  d_physicalConsts = new PhysicalConstants();
  d_physicalConsts->problemSetup(db);
  // read properties, boundary and turbulence model
  d_props = new Properties();
  d_props->problemSetup(db, dw);
  string turbModel;
  db->require("turbulence_model", turbModel);
  if (turbModel == "Smagorinsky") 
    d_turbModel = new SmagorinskyModel();
  else 
    throw InvalidValue("Turbulence Model not supported" + turbModel, db);
  d_turbModel->problemSetup(db, dw);
  d_boundaryCondition = new BoundaryCondition(d_turbModel);
  d_boundaryCondition->problemSetup(db, dw);
  string nlSolver;
  db->require("nonlinear_solver", nlSolver);
  if(nlSolver == "picard")
    d_nlSolver = new PicardNonlinearSolver(d_props, d_boundaryCondition,
					   d_turbModel);
  else
    throw InvalidValue("Nonlinear solver not supported: "+nlSolver, db);

  //d_nlSolver->problemSetup(db, dw); /* 2 params ? */
  d_nlSolver->problemSetup(db);
#else
  NOT_FINISHED("Arches::probemSetup");
#endif
}
#if 0
void Arches::problemInit(const LevelP& level,
			 SchedulerP& sched, DataWarehouseP& dw,
			 bool restrt )
{
  // initializes variables
  if (!restrt) {
    sched_paramInit(level, sched, dw);
    // initialize velocity, scalars and properties at the boundary
    d_boundaryCondition->sched_setProfile(level, sched, dw);
  }
  d_properties->sched_computeProperties(level, sched, dw, dw);
  d_turbModel->sched_computeTurbSubmodel(level, sched, dw, dw);
  d_boundaryCondition->sched_pressureBC(level, sched, dw, dw);
}
#endif
void Arches::computeStableTimestep(const LevelP& level,
				   SchedulerP& sched, DataWarehouseP& dw)
{
  dw->put(SoleVariable<double>(d_deltaT), "delt"); 
}

void Arches::timeStep(double time, double dt,
	      const LevelP& level, SchedulerP& sched,
	      const DataWarehouseP& old_dw, DataWarehouseP& new_dw)
{
  int error_code = d_nlSolver->nonlinearSolve(level, sched, old_dw, new_dw);
   /*
    * if the solver works then put thecomputed values in
    * the database.
    */
   // not sure if this is the correct way to do it
   // we need temp pressure, velocity and scalar vars
   //   new_dw->put(CCVariable<double>(pressure), "pressure");

}

} // end namespace Components
} // end namespace Uintah

//
// $Log$
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

