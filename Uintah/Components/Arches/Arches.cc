
#include <Uintah/Components/Arches/Arches.h>
#include <SCICore/Util/NotFinished.h>

Arches::Arches()
{
}

Arches::~Arches()
{
}

void Arches::problemSetup(const ProblemSpecP& params, GridP& grid,
		  DataWarehouseP& dw, DatabaseP& db)
{
  int nlSolver;
  if (db->keyExists("grow_dt")) {
    d_deltaT = db->getDouble("grow_dt");
  } else {
    cerr << "grow_dt not in input database" << endl;
  }
  if (db->keyExists("nonlinear_solver")) {
    nlSolver = db->getInteger("nonlinear_solver");
  } else {
    cerr << "nonlinear_solver not in input database" << endl;
  }
  if (nlSolver == 0) {
    d_nlSolver = new PicardNonlinearSolver();
  }
  else {
    cerr << "incorrect nonlinear_solver option" << endl;
  }
  if (db->keyExists("Picard Solver")) {
    DatabaseP nlSolverDB = db->getDatabase("Picard Solver");
  } else {
    cerr << "Picard Solver not in input database" << endl;
  }
  d_nlSolver->problemSetup(nlSolverDB);


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
  int error_code = d_solver->nonlinearSolve(level, sched, old_dw, new_dw);

   /*
    * if the solver works then put thecomputed values in
    * the database.
    */
   // not sure if this is the correct way to do it
   // we need temp pressure, velocity and scalar vars
   //   new_dw->put(CCVariable<double>(pressure), "pressure");

   
}


/* void Arches::timeStep(double time, double dt,
	      const LevelP& level, SchedulerP& sched,
	      const DataWarehouseP& old_dw, DataWarehouseP& new_dw)
{
    for(Level::const_regionIterator iter=level->regionsBegin();
	iter != level->regionsEnd(); iter++){
	const Region* region=*iter;
	{
	  //copies old db to new_db and then uses non-linear
	  //solver to compute new values
	  Task* tsk = new Task("Arches::timeStep",region, old_dw, new_dw,
		       this, Arches::advanceTimeStep);
	  tsk->requires(old_dw, "pressure", region, 0,
			CCVariable<double>::getTypeDescription());
	  tsk->requires(old_dw, "velocity", region, 0,
			FCVariable<Vector>::getTypeDescription());
	  tsk->requires(old_dw, "scalar", region, 0,
			CCVariable<Vector>::getTypeDescription());
	  tsk->computes(new_dw "pressure", region, 0,
			CCVariable<double>::getTypeDescription());
	  tsk->computes(new_dw, "velocity", region, 0,
			FCVariable<Vector>::getTypeDescription());
	  tsk->computes(new_dw, "scalar", region, 0,
			CCVariable<Vector>::getTypeDescription());
	  sched->addTask(tsk);
	}

    }
}

*/

/*
void Arches::advanceTimeStep(const ProcessorContext* pc,
			     const Region* region,
			     const DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw)
{
 /* This calls the nonlinear solver.  It returns 0 for success,
    * and 1 for an error.
    */
   int error_code = d_solver->nonlinearSolve(pc, region, old_dw, new_dw);

   /*
    * if the solver works then put thecomputed values in
    * the database.
    */
   // not sure if this is the correct way to do it
   // we need temp pressure, velocity and scalar vars
   //   new_dw->put(CCVariable<double>(pressure), "pressure");

   
}



*/
