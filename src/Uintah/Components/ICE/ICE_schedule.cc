
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
      Task* t = scinew Task("ICE::actuallyInitialize", 
                patch,      dw,                 dw,
                this,       &ICE::actuallyInitialize);
        t->computes( dw,    press_CCLabel,      0,patch);
        t->computes( dw,    rho_CCLabel,        0,patch);
        t->computes( dw,    temp_CCLabel,       0,patch);
        t->computes( dw,    vel_CCLabel,        0,patch);

      sched->addTask(t);
    }
  }
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
                    patch,      dw,             dw,
                    this,       &ICE::actuallyComputeStableTimestep);
//       t->requires(dw,    "params",           ProblemSpec::getTypeDescription());                          
         t->requires(dw,    vel_CCLabel,        0,patch, Ghost::None);
         t->computes(dw,    delTLabel);
         t->usesMPI(false);
         t->usesThreads(false);
//       t->whatis the cost model?();
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
       /*__________________________________
       *       T  O  P 
       *___________________________________*/
        Task* t = scinew Task("ICE::Top_of_main_loop", 
                    patch,      old_dw,         new_dw,
                    this,       &ICE::actually_Top_of_main_loop);
                    
//      t->requires( old_dw,    "params",       ProblemSpec::getTypeDescription());
#if switch_UCF_stepTop_of_main_loopOnOff 
        t->requires( old_dw,    press_CCLabel,  0,patch, Ghost::None);
        t->requires( old_dw,    rho_CCLabel,    0,patch, Ghost::None);
        t->requires( old_dw,    temp_CCLabel,   0,patch, Ghost::None);
        t->requires( old_dw,    vel_CCLabel,    0,patch, Ghost::None);


/* currently broken*/        
        t->computes( new_dw,    press_CCLabel_0, 0,patch);
        t->computes( new_dw,    rho_CCLabel_0,  0,patch);
        t->computes( new_dw,    temp_CCLabel_0, 0,patch);
        t->computes( new_dw,    vel_CCLabel_0,  0,patch);
#endif
        t->usesMPI(false);
        t->usesThreads(false);
//      t->whatis the cost model?();
        sched->addTask(t);
        }       
       

        /*__________________________________
        *      S  T  E  P     1 
        *___________________________________*/
        {
        Task* t = scinew Task("ICE::step1", 
                    patch,      old_dw,         new_dw,
                    this,       &ICE::actuallyStep1);
                    
//      t->requires( old_dw,    "params",       ProblemSpec::getTypeDescription());
        t->requires( old_dw,    press_CCLabel,0,patch, Ghost::None);
        t->requires( old_dw,    rho_CCLabel,  0,patch, Ghost::None);
        t->requires( old_dw,    temp_CCLabel, 0,patch, Ghost::None);
        t->requires( old_dw,    vel_CCLabel,  0,patch, Ghost::None);
 
        t->computes( new_dw,    press_CCLabel_1,0,patch);
        t->computes( new_dw,    rho_CCLabel_1,  0,patch);
        t->computes( new_dw,    temp_CCLabel_1, 0,patch);
        t->computes( new_dw,    vel_CCLabel_1,  0,patch);
        t->usesMPI(false);
        t->usesThreads(false);
//     t->whatis the cost model?();
        sched->addTask(t);
        }

        /*__________________________________
        *      S  T  E  P     2
        *___________________________________*/
        {
        Task* t = scinew Task("ICE::step2", 
                    patch,      new_dw,         new_dw,
                    this,       &ICE::actuallyStep2);
                    
//      t->requires( old_dw, "params",        ProblemSpec::getTypeDescription());
        t->requires( new_dw,    press_CCLabel_1, 0,patch, Ghost::None);
        t->requires( new_dw,    rho_CCLabel_1,   0,patch, Ghost::None);
        t->requires( new_dw,    temp_CCLabel_1,  0,patch, Ghost::None);
        t->requires( new_dw,    vel_CCLabel_1,   0,patch, Ghost::None);
       
        t->computes( new_dw,    press_CCLabel_2, 0,patch);
        t->computes( new_dw,    rho_CCLabel_2,   0,patch);
        t->computes( new_dw,    temp_CCLabel_2,  0,patch);
        t->computes( new_dw,    vel_CCLabel_2,   0,patch);
        t->usesMPI(false);
        t->usesThreads(false);
//      t->whatis the cost model?();
        sched->addTask(t);
        }
       
        /*__________________________________
        *      S  T  E  P     3
        *___________________________________*/
        {
        Task* t = scinew Task("ICE::step3", 
                    patch,      new_dw,         new_dw,
                    this,       &ICE::actuallyStep3);
                    
//      t->requires(  old_dw,   "params",       ProblemSpec::getTypeDescription());
        t->requires( new_dw,    press_CCLabel_2, 0,patch, Ghost::None);
        t->requires( new_dw,    rho_CCLabel_2,   0,patch, Ghost::None);
        t->requires( new_dw,    temp_CCLabel_2,  0,patch, Ghost::None);
        t->requires( new_dw,    vel_CCLabel_2,   0,patch, Ghost::None);
       
        t->computes( new_dw,    press_CCLabel_3, 0,patch);
        t->computes( new_dw,    rho_CCLabel_3,   0,patch);
        t->computes( new_dw,    temp_CCLabel_3,  0,patch);
        t->computes( new_dw,    vel_CCLabel_3,   0,patch);
        t->usesMPI(false);
        t->usesThreads(false);
//      t->whatis the cost model?();
        sched->addTask(t);
        }

        /*__________________________________
        *      S  T  E  P     4
        *___________________________________*/
        {
        Task* t = scinew Task("ICE::step4", 
                    patch,      new_dw,         new_dw,
                    this,       &ICE::actuallyStep4);
                    
//      t->requires(old_dw,    "params",        ProblemSpec::getTypeDescription());
        t->requires( new_dw,    press_CCLabel_3, 0,patch, Ghost::None);
        t->requires( new_dw,    rho_CCLabel_3,   0,patch, Ghost::None);
        t->requires( new_dw,    temp_CCLabel_3,  0,patch, Ghost::None);
        t->requires( new_dw,    vel_CCLabel_3,   0,patch, Ghost::None);
       
        t->computes( new_dw,    press_CCLabel_4, 0,patch);
        t->computes( new_dw,    rho_CCLabel_4,   0,patch);
        t->computes( new_dw,    temp_CCLabel_4,  0,patch);
        t->computes( new_dw,    vel_CCLabel_4,   0,patch);
        t->usesMPI(false);
        t->usesThreads(false);
//     t->whatis the cost model?();
        sched->addTask(t);
        }
        
       
        /*__________________________________
        *      S  T  E  P     5
        *___________________________________*/
        {
        Task* t = scinew Task("ICE::step5", 
                    patch,      new_dw,         new_dw,
                    this,       &ICE::actuallyStep5);
                    
//      t->requires(old_dw,    "params",        ProblemSpec::getTypeDescription());
        t->requires( new_dw,    press_CCLabel_4, 0,patch, Ghost::None);
        t->requires( new_dw,    rho_CCLabel_4,   0,patch, Ghost::None);
        t->requires( new_dw,    temp_CCLabel_4,  0,patch, Ghost::None);
        t->requires( new_dw,    vel_CCLabel_4,   0,patch, Ghost::None);
       
        t->computes( new_dw,    press_CCLabel_5, 0,patch);
        t->computes( new_dw,    rho_CCLabel_5,   0,patch);
        t->computes( new_dw,    temp_CCLabel_5,  0,patch);
        t->computes( new_dw,    vel_CCLabel_5,   0,patch);
        t->usesMPI(false);
        t->usesThreads(false);
//      t->whatis the cost model?();
        sched->addTask(t);
        }
       
        /*__________________________________
        *      S  T  E  P     6 & 7
        *___________________________________*/
        {
        Task* t = scinew Task("ICE::step6and7", 
                    patch,      new_dw,         new_dw,
                    this,       &ICE::actuallyStep6and7);
                    
//      t->requires(old_dw,    "params",        ProblemSpec::getTypeDescription());
        t->requires( new_dw,    press_CCLabel_5, 0,patch, Ghost::None);
        t->requires( new_dw,    rho_CCLabel_5,   0,patch, Ghost::None);
        t->requires( new_dw,    temp_CCLabel_5,  0,patch, Ghost::None);
        t->requires( new_dw,    vel_CCLabel_5,   0,patch, Ghost::None);
       
        t->computes( new_dw,    press_CCLabel_6_7,0,patch);
        t->computes( new_dw,    rho_CCLabel_6_7,  0,patch);
        t->computes( new_dw,    temp_CCLabel_6_7, 0,patch);
        t->computes( new_dw,    vel_CCLabel_6_7,  0,patch);
        t->usesMPI(false);
        t->usesThreads(false);
//     t->whatis the cost model?();
        sched->addTask(t);
        }
      
       /*__________________________________
       *    B  O  T  T  O  M
       *___________________________________*/
       {
        Task* t = scinew Task("ICE::actually_Bottom_of_main_loop", 
                    patch,      new_dw,         new_dw,
                    this,       &ICE::actually_Bottom_of_main_loop);
                    
//      t->requires(old_dw,     "params",       ProblemSpec::getTypeDescription());
        t->requires( new_dw,    press_CCLabel_6_7,  0,patch, Ghost::None);
        t->requires( new_dw,    rho_CCLabel_6_7,    0,patch, Ghost::None);
        t->requires( new_dw,    temp_CCLabel_6_7,   0,patch, Ghost::None);
        t->requires( new_dw,    vel_CCLabel_6_7,    0,patch, Ghost::None);
        
        t->computes( new_dw,    press_CCLabel,   0,patch);
        t->computes( new_dw,    rho_CCLabel,     0,patch);
        t->computes( new_dw,    temp_CCLabel,    0,patch);
        t->computes( new_dw,    vel_CCLabel,     0,patch);
        t->usesMPI(false);
        t->usesThreads(false);
//      t->whatis the cost model?();
        sched->addTask(t);
        }
        
    }

    this->cheat_t   =t;
    this->cheat_delt=delt;
}
