/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/Arches.h>
#include <Uintah/Components/Arches/PicardNonlinearSolver.h>
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
  ProblemSpecP db = params->findBlock("Arches");

  db->require("grow_dt", d_deltaT);

  string nlSolver;
  db->require("nonlinear_solver", nlSolver);
  if(nlSolver == "picard")
    d_nlSolver = new PicardNonlinearSolver();
  else
    throw InvalidValue("Nonlinear solver not supported: "+nlSolver, db);

  //d_nlSolver->problemSetup(db, dw); /* 2 params ? */
  d_nlSolver->problemSetup(db);
}

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
// Revision 1.10  2000/03/31 17:35:05  moulding
// A call to d_nlSolver->problemSetup() was attempted with 2 parameters, causing
// a compile error (too many parameters).  I changed it to one parameter.
// The change is at line 40.
//
// Revision 1.9  2000/03/29 21:18:16  rawat
// modified boundaryconidtion.cc for inlet bcs
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

