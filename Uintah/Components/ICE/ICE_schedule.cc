
#include <Uintah/Components/ICE/ICE.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Interface/Scheduler.h>
using Uintah::ICESpace::ICE;


/* ---------------------------------------------------------------------
GENERAL INFORMATION
 Function:  ICE::scheduleInitialize--
 Filename:  ICE_schedule.cc
 Purpose:   Schedule the initialization of data warehouse variable
            need by ICE 

History: 
Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     John Schmidt   06/23/00                              
_____________________________________________________________________*/ 
void ICE::scheduleInitialize(
    const LevelP&   level,
    SchedulerP&     sched,
    DataWarehouseP& dw)
{
  Level::const_patchIterator iter;

  for(iter=level->patchesBegin(); iter != level->patchesEnd(); iter++)
  {
    const Patch* patch=*iter;
    {
      Task* t = scinew Task("ICE::actuallyInitialize", patch, dw, dw,
			    this, &ICE::actuallyInitialize);
      t->computes(dw, vel_CCLabel,      0,patch);
      t->computes(dw, press_CCLabel,    0,patch);
      t->computes(dw, press_CCLabel_1,    0,patch);
      t->computes(dw, rho_CCLabel,      0,patch);
      t->computes(dw, temp_CCLabel,     0,patch);
      t->computes(dw, cv_CCLabel,       0,patch);

      t->computes(dw, vel_FCLabel,      0,patch);
      t->computes(dw, press_FCLabel,    0,patch);
      t->computes(dw, tau_FCLabel,      0,patch);

      sched->addTask(t);
    }
  }
  cerr << "ICE::scheduleInitialize not done\n";
}


/* ---------------------------------------------------------------------
GENERAL INFORMATION
 Function:  ICE::scheduleComputeStableTimestep--
 Filename:  ICE_schedule.cc
 Purpose:   Schedule a task to compute the time step for ICE

History: 
Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     John Schmidt   06/23/00                              
_____________________________________________________________________*/
void ICE::scheduleComputeStableTimestep(
    const LevelP&   level,
    SchedulerP&     sched,
    DataWarehouseP& dw)
{

    for(Level::const_patchIterator iter=level->patchesBegin();
	iter != level->patchesEnd(); iter++)
    {
	const Patch* patch=*iter;
	Task* t = scinew Task(   "ICE::computeStableTimestep", 
                                patch, 
                                dw, 
                                dw,
			           this, 
                                &ICE::actuallyComputeStableTimestep);
                           
	 t->requires(dw,     vel_CCLabel,    0,patch, Ghost::None);
//      t->requires(dw, "params", ProblemSpec::getTypeDescription());
	 t->computes(dw,      delTLabel);
	 t->usesMPI(false);
	 t->usesThreads(false);
//      t->whatis the cost model?();
	 sched->addTask(t);
    }
}



/* ---------------------------------------------------------------------
GENERAL INFORMATION
 Function:  ICE::scheduleTimeAdvance--
 Filename:  ICE_schedule.cc
 Purpose:   Schedule tasks for each of the major steps in ICE

History: 
Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     John Schmidt   06/23/00                              
_____________________________________________________________________*/
void ICE::scheduleTimeAdvance(
    double /*t*/, 
    double /*delt*/,
    const LevelP&   level, 
    SchedulerP&     sched,
    DataWarehouseP& old_dw, 
    DataWarehouseP& new_dw)
{

    for(Level::const_patchIterator iter=level->patchesBegin();
	iter != level->patchesEnd(); iter++)
    {
	const Patch* patch=*iter;
	{
	//--Top of main loop

	Task* t = scinew Task("ICE::Top_of_main_loop", 
                    patch,      old_dw,         new_dw,
		      this,       &ICE::actually_Top_of_main_loop);
	t->requires( old_dw, vel_CCLabel,     0,patch, Ghost::None);
//  	t->requires( old_dw, "params",ProblemSpec::getTypeDescription());
//	t->computes( new_dw, vel_CCLabel,     0,patch);
	t->usesMPI(false);
	t->usesThreads(false);
//     t->whatis the cost model?();
	sched->addTask(t);
	}	

	{
	//--Step 1

	Task* t = scinew Task("ICE::step1", 
                    patch,      old_dw,         new_dw,
		      this,       &ICE::actuallyStep1);
	t->requires( old_dw, press_CCLabel,   0,patch, Ghost::None);
//     t->requires( old_dw, press_CCLabel_1, 0,patch, Ghost::None);
       t->requires( old_dw, rho_CCLabel,     0,patch, Ghost::None);
	t->requires( old_dw, temp_CCLabel,    0,patch, Ghost::None);
	t->requires( old_dw, cv_CCLabel,      0,patch, Ghost::None);
//  	t->requires( old_dw, "params", ProblemSpec::getTypeDescription());
//	t->computes( old_dw, press_CCLabel_1, 0,patch);
	t->usesMPI(false);
	t->usesThreads(false);
//     t->whatis the cost model?();
	sched->addTask(t);
	}

	{
	//--Step 2

	Task* t = scinew Task("ICE::step2", 
                    patch,      old_dw,         new_dw,
		      this,       &ICE::actuallyStep2);
	t->requires( old_dw, vel_CCLabel,     0,patch, Ghost::None);
//  	t->requires( old_dw, "params", ProblemSpec::getTypeDescription());
//	t->computes( new_dw, vel_CCLabel,     0,patch);
	t->usesMPI(false);
	t->usesThreads(false);
//     t->whatis the cost model?();
	sched->addTask(t);
	}

	{
	//--Step 3

	Task* t = scinew Task("ICE::step3", 
                    patch,      old_dw,         new_dw,
		      this,       &ICE::actuallyStep3);
                    
	t->requires(old_dw, vel_CCLabel,     0,patch, Ghost::None);
//  	t->requires(old_dw, "params", ProblemSpec::getTypeDescription());
//	t->computes(new_dw, vel_CCLabel,     0,patch);
	t->usesMPI(false);
	t->usesThreads(false);
//     t->whatis the cost model?();
	sched->addTask(t);
	}

	{
	//--Step 4

	Task* t = scinew Task("ICE::step4", 
                    patch,      old_dw,         new_dw,
		      this,       &ICE::actuallyStep4);
	t->requires(old_dw, vel_CCLabel,     0,patch, Ghost::None);
//  	t->requires(old_dw, "params", ProblemSpec::getTypeDescription());
//	t->computes(new_dw, vel_CCLabel,     0,patch);
	t->usesMPI(false);
	t->usesThreads(false);
//     t->whatis the cost model?();
	sched->addTask(t);
	}
	
	{
	//--Step 5

	Task* t = scinew Task("ICE::step5", 
                    patch,      old_dw,         new_dw,
		      this,       &ICE::actuallyStep5);
	t->requires(old_dw, vel_CCLabel,     0,patch, Ghost::None);
//  	t->requires(old_dw, "params", ProblemSpec::getTypeDescription());
//	t->computes(new_dw, vel_CCLabel,     0,patch);
	t->usesMPI(false);
	t->usesThreads(false);
//     t->whatis the cost model?();
	sched->addTask(t);
	}

	{
	//--Step 6

	Task* t = scinew Task("ICE::step6and7", 
                    patch,      old_dw,         new_dw,
		      this,       &ICE::actuallyStep6and7);
	t->requires(old_dw, vel_CCLabel,     0,patch, Ghost::None);
//  	t->requires(old_dw, "params", ProblemSpec::getTypeDescription());
//	t->computes(new_dw, vel_CCLabel,     0,patch);
	t->usesMPI(false);
	t->usesThreads(false);
//     t->whatis the cost model?();
	sched->addTask(t);
	}
        //--Bottom of main loop
	{
	Task* t = scinew Task("ICE::actually_Bottom_of_main_loop", 
                    patch,      old_dw,         new_dw,
		      this,       &ICE::actually_Bottom_of_main_loop);
	t->requires(old_dw, vel_CCLabel,     0,patch, Ghost::None);
//  	t->requires(old_dw, "params", ProblemSpec::getTypeDescription());
  	t->computes(new_dw, vel_CCLabel,     0,patch);
	t->computes(new_dw, temp_CCLabel,    0,patch);
	t->computes(new_dw, cv_CCLabel,      0,patch);
	t->usesMPI(false);
	t->usesThreads(false);
//     t->whatis the cost model?();
	sched->addTask(t);
	}
    }

    this->cheat_t   =t;
    this->cheat_delt=delt;
}
