
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
        for (m = 1; m <= nMaterials; m++)          
        { 
          t->computes( dw,    press_CCLabel,      m,patch);
          t->computes( dw,    rho_CCLabel,        m,patch);
          t->computes( dw,    temp_CCLabel,       m,patch);
          t->computes( dw,    vel_CCLabel,        m,patch);
        }
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
        for (m = 1; m <= nMaterials; m++)          
        { 
           t->requires(dw,    vel_CCLabel,        m,patch, Ghost::None);
        }
        t->computes(dw,    delTLabel);
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
    int m=0;

    for(Level::const_patchIterator iter=level->patchesBegin();
        iter != level->patchesEnd(); iter++)
    {
        const Patch* patch=*iter;
        {
       /*__________________________________
       *       T  O  P 
       *___________________________________*/
        Task* ttop = scinew Task("ICE::Top_of_main_loop", 
                    patch,      old_dw,         new_dw,
                    this,       &ICE::actually_Top_of_main_loop);
                    
//      t->requires( old_dw,    "params",       ProblemSpec::getTypeDescription());
//#if switch_UCF_stepTop_of_main_loopOnOff 
        for (m = 1; m <= nMaterials; m++)          
        { 
          ttop->requires( old_dw,    press_CCLabel,  m,patch, Ghost::None);
          ttop->requires( old_dw,    rho_CCLabel,    m,patch, Ghost::None);
          ttop->requires( old_dw,    temp_CCLabel,   m,patch, Ghost::None);
          ttop->requires( old_dw,    vel_CCLabel,    m,patch, Ghost::None);

          ttop->computes( new_dw,    press_CCLabel_0,m,patch);
          ttop->computes( new_dw,    rho_CCLabel_0,  m,patch);
          ttop->computes( new_dw,    temp_CCLabel_0, m,patch);
          ttop->computes( new_dw,    vel_CCLabel_0,  m,patch);
        }
//#endif
        ttop->usesMPI(false);
        ttop->usesThreads(false);
//      ttop->whatis the cost model?();
        sched->addTask(ttop);
        }       
       

        /*__________________________________
        *      S  T  E  P     1 
        *___________________________________*/
        {
        Task* t1 = scinew Task("ICE::step1", 
                    patch,      new_dw,         new_dw,
                    this,       &ICE::actuallyStep1);
//      t1->requires( old_dw,    "params",       ProblemSpec::getTypeDescription());
        for (m = 1; m <= nMaterials; m++)         
        {                   
            t1->requires( new_dw,    press_CCLabel_0,m,patch, Ghost::None);
            t1->requires( new_dw,    rho_CCLabel_0,  m,patch, Ghost::None);
            t1->requires( new_dw,    temp_CCLabel_0, m,patch, Ghost::None);
            t1->requires( new_dw,    vel_CCLabel_0,  m,patch, Ghost::None);

            t1->computes( new_dw,    press_CCLabel_1,m,patch);
            t1->computes( new_dw,    rho_CCLabel_1,  m,patch);
            t1->computes( new_dw,    temp_CCLabel_1, m,patch);
            t1->computes( new_dw,    vel_CCLabel_1,  m,patch);
        }
        t1->usesMPI(false);
        t1->usesThreads(false);
//      t1->whatis the cost model?();
        sched->addTask(t1);
        }

        /*__________________________________
        *      S  T  E  P     2
        *___________________________________*/
        Task* t2 = scinew Task("ICE::step2", 
                    patch,      new_dw,         new_dw,
                    this,       &ICE::actuallyStep2);
                     
//      t->requires( old_dw, "params",        ProblemSpec::getTypeDescription());
        for (m = 1; m <= nMaterials; m++)          
        {
          t2->requires( new_dw,    press_CCLabel_1, m,patch, Ghost::None);
          t2->requires( new_dw,    rho_CCLabel_1,   m,patch, Ghost::None);
          t2->requires( new_dw,    temp_CCLabel_1,  m,patch, Ghost::None);
          t2->requires( new_dw,    vel_CCLabel_1,   m,patch, Ghost::None);
 
          t2->computes( new_dw,    press_CCLabel_2, m,patch);
          t2->computes( new_dw,    rho_CCLabel_2,   m,patch);
          t2->computes( new_dw,    temp_CCLabel_2,  m,patch);
          t2->computes( new_dw,    vel_CCLabel_2,   m,patch);
        }
        t2->usesMPI(false);
        t2->usesThreads(false);
//      t2->whatis the cost model?();
        sched->addTask(t2);
       
        /*__________________________________
        *      S  T  E  P     3
        *___________________________________*/
        Task* t3 = scinew Task("ICE::step3", 
                    patch,      new_dw,         new_dw,
                    this,       &ICE::actuallyStep3);
                    
//      t3->requires(  old_dw,   "params",       ProblemSpec::getTypeDescription());
        for (m = 1; m <= nMaterials; m++)          
        {
          t3->requires( new_dw,    press_CCLabel_2, m,patch, Ghost::None);
          t3->requires( new_dw,    rho_CCLabel_2,   m,patch, Ghost::None);
          t3->requires( new_dw,    temp_CCLabel_2,  m,patch, Ghost::None);
          t3->requires( new_dw,    vel_CCLabel_2,   m,patch, Ghost::None);
 
          t3->computes( new_dw,    press_CCLabel_3, m,patch);
          t3->computes( new_dw,    rho_CCLabel_3,   m,patch);
          t3->computes( new_dw,    temp_CCLabel_3,  m,patch);
          t3->computes( new_dw,    vel_CCLabel_3,   m,patch);
        }
        t3->usesMPI(false);
        t3->usesThreads(false);
//      t3->whatis the cost model?();
        sched->addTask(t3);


        /*__________________________________
        *      S  T  E  P     4
        *___________________________________*/
        Task* t4 = scinew Task("ICE::step4", 
                    patch,      new_dw,         new_dw,
                    this,       &ICE::actuallyStep4);
                    
//      t->requires(old_dw,    "params",        ProblemSpec::getTypeDescription());
        for (m = 1; m <= nMaterials; m++)          
        {
          t4->requires( new_dw,    press_CCLabel_3, m,patch, Ghost::None);
          t4->requires( new_dw,    rho_CCLabel_3,   m,patch, Ghost::None);
          t4->requires( new_dw,    temp_CCLabel_3,  m,patch, Ghost::None);
          t4->requires( new_dw,    vel_CCLabel_3,   m,patch, Ghost::None);
 
          t4->computes( new_dw,    press_CCLabel_4, m,patch);
          t4->computes( new_dw,    rho_CCLabel_4,   m,patch);
          t4->computes( new_dw,    temp_CCLabel_4,  m,patch);
          t4->computes( new_dw,    vel_CCLabel_4,   m,patch);
        }
        t4->usesMPI(false);
        t4->usesThreads(false);
//      t4->whatis the cost model?();
        sched->addTask(t4);
        
       
        /*__________________________________
        *      S  T  E  P     5
        *___________________________________*/
        Task* t5 = scinew Task("ICE::step5", 
                    patch,      new_dw,         new_dw,
                    this,       &ICE::actuallyStep5);
                    
//      t->requires(old_dw,    "params",        ProblemSpec::getTypeDescription());
        for (m = 1; m <= nMaterials; m++)          
        {
          t5->requires( new_dw,    press_CCLabel_4, m,patch, Ghost::None);
          t5->requires( new_dw,    rho_CCLabel_4,   m,patch, Ghost::None);
          t5->requires( new_dw,    temp_CCLabel_4,  m,patch, Ghost::None);
          t5->requires( new_dw,    vel_CCLabel_4,   m,patch, Ghost::None);
 
          t5->computes( new_dw,    press_CCLabel_5, m,patch);
          t5->computes( new_dw,    rho_CCLabel_5,   m,patch);
          t5->computes( new_dw,    temp_CCLabel_5,  m,patch);
          t5->computes( new_dw,    vel_CCLabel_5,   m,patch);
        }
        t5->usesMPI(false);
        t5->usesThreads(false);
//      t5->whatis the cost model?();
        sched->addTask(t5);
       
        /*__________________________________
        *      S  T  E  P     6 & 7
        *___________________________________*/
        Task* t6_7 = scinew Task("ICE::step6and7", 
                    patch,      new_dw,         new_dw,
                    this,       &ICE::actuallyStep6and7);
                    
//      t->requires(old_dw,    "params",        ProblemSpec::getTypeDescription());
        for (m = 1; m <= nMaterials; m++)          
        {
          t6_7->requires( new_dw,   press_CCLabel_5, m,patch, Ghost::None);
          t6_7->requires( new_dw,   rho_CCLabel_5,   m,patch, Ghost::None);
          t6_7->requires( new_dw,   temp_CCLabel_5,  m,patch, Ghost::None);
          t6_7->requires( new_dw,   vel_CCLabel_5,   m,patch, Ghost::None);

          t6_7->computes( new_dw,   press_CCLabel_6_7,m,patch);
          t6_7->computes( new_dw,   rho_CCLabel_6_7,  m,patch);
          t6_7->computes( new_dw,   temp_CCLabel_6_7, m,patch);
          t6_7->computes( new_dw,   vel_CCLabel_6_7,  m,patch);
        }
        t6_7->usesMPI(false);
        t6_7->usesThreads(false);
//      t6_7->whatis the cost model?();
        sched->addTask(t6_7);
        
      
       /*__________________________________
       *    B  O  T  T  O  M
       *___________________________________*/
        Task* tbot = scinew Task("ICE::actually_Bottom_of_main_loop", 
                    patch,      new_dw,         new_dw,
                    this,       &ICE::actually_Bottom_of_main_loop);
                    
//      t->requires(old_dw,     "params",       ProblemSpec::getTypeDescription());
        for (m = 1; m <= nMaterials; m++)          
        {
          tbot->requires( new_dw,    press_CCLabel_6_7,  m,patch, Ghost::None);
          tbot->requires( new_dw,    rho_CCLabel_6_7,    m,patch, Ghost::None);
          tbot->requires( new_dw,    temp_CCLabel_6_7,   m,patch, Ghost::None);
          tbot->requires( new_dw,    vel_CCLabel_6_7,    m,patch, Ghost::None);

          tbot->computes( new_dw,    press_CCLabel,   m,patch);
          tbot->computes( new_dw,    rho_CCLabel,     m,patch);
          tbot->computes( new_dw,    temp_CCLabel,    m,patch);
          tbot->computes( new_dw,    vel_CCLabel,     m,patch);
        }
        tbot->usesMPI(false);
        tbot->usesThreads(false);
//      tbot->whatis the cost model?();
        sched->addTask(tbot);
        
    }

    this->cheat_t   =t;
    this->cheat_delt=delt;
}
