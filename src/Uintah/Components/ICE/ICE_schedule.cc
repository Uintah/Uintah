
#include <Uintah/Components/ICE/ICE.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Components/ICE/ICEMaterial.h>
#include <Uintah/Grid/SimulationState.h>
using Uintah::ICESpace::ICE;
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleInitialize--
 Purpose~   Schedule the initialization of the dw variables

Programmer         Date       Description                      
----------         ----       -----------                 
John Schmidt      10/04/00                              
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
    Task* t = scinew Task("ICE::actuallyInitialize", patch, dw, dw,this,
			    &ICE::actuallyInitialize);
    t->computes( dw,    d_sharedState->get_delt_label());
     
    for (int m = 0; m < d_sharedState->getNumICEMatls(); m++ ) 
    {
	ICEMaterial*  matl = d_sharedState->getICEMaterial(m);
	int dwindex = matl->getDWIndex();
       
	t->computes( dw, lb->temp_CCLabel,      dwindex, patch);
	t->computes( dw, lb->rho_micro_CCLabel, dwindex, patch);
	t->computes( dw, lb->rho_CCLabel,       dwindex, patch);
	t->computes( dw, lb->cv_CCLabel,        dwindex, patch);
	t->computes( dw, lb->viscosity_CCLabel, dwindex, patch);
	t->computes( dw, lb->vol_frac_CCLabel,  dwindex, patch);
	t->computes( dw, lb->uvel_CCLabel,      dwindex, patch);
	t->computes( dw, lb->vvel_CCLabel,      dwindex, patch);
	t->computes( dw, lb->wvel_CCLabel,      dwindex, patch);
	t->computes( dw, lb->uvel_FCLabel,      dwindex, patch);
	t->computes( dw, lb->vvel_FCLabel,      dwindex, patch);
	t->computes( dw, lb->wvel_FCLabel,      dwindex, patch);
    }

    t->computes(dw, lb->press_CCLabel,0, patch);

    sched->addTask(t);
  }
}
//STOP_DOC
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleComputeStableTimestep--
 Purpose~  Schedule a task to compute the time step for ICE

Programmer         Date       Description                      
----------         ----       -----------                 
John Schmidt    10/04/00                              
_____________________________________________________________________*/
void ICE::scheduleComputeStableTimestep(
    const LevelP&   level,
    SchedulerP&     sched,
    DataWarehouseP& dw)
{
#if 0
  // Compute the stable timestep
  int numMatls = d_sharedState->getNumICEMatls();

  for (Level::const_patchIterator iter = level->patchesBegin();
       iter != level->patchesEnd(); iter++) 
  {
        const Patch* patch = *iter;
  
        Task* task = scinew Task("ICE::actuallyComputeStableTimestep",patch, dw,
			       dw,this, &ICE::actuallyComputeStableTimestep);

      for (int m = 0; m < numMatls; m++) 
      {
        ICEMaterial* matl = d_sharedState->getICEMaterial(m);
        int dwindex = matl->getDWIndex();
        task->requires(dw, lb->uvel_CCLabel,    dwindex,    patch,  Ghost::None);
        task->requires(dw, lb->vvel_CCLabel,    dwindex,    patch,  Ghost::None);
        task->requires(dw, lb->wvel_CCLabel,    dwindex,    patch,  Ghost::None);
      }
      sched->addTask(task);
  }
#endif
}
//STOP_DOC
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleTimeAdvance--
 Purpose~   Schedule tasks for each of the major steps 

Programmer         Date       Description                      
----------         ----       -----------                 
John Schmidt      10/04/00                             
_____________________________________________________________________*/
void ICE::scheduleTimeAdvance(
    double t,   
    double dt,
    const LevelP&   level,
    SchedulerP&     sched,
    DataWarehouseP& old_dw,
    DataWarehouseP& new_dw)
{
  for(Level::const_patchIterator iter=level->patchesBegin();
       iter != level->patchesEnd(); iter++)
  {
    const Patch* patch=*iter;

    // Step 1a  computeSoundSpeed
    scheduleStep1a( patch,  sched,  old_dw, new_dw);
    
    // Step 1b calculate equlibration pressure
    scheduleStep1b( patch,  sched,  old_dw, new_dw);
    
    // Step 1c compute face centered velocities
    scheduleStep1c( patch,  sched,  old_dw, new_dw);
    
    // Step 1d computes momentum exchange on FC velocities
    scheduleStep1d( patch,  sched,  old_dw, new_dw);
    
    // Step 2 computes delPress and the new pressure
    scheduleStep2(  patch,  sched,  old_dw, new_dw);
    
    // Step 3 compute face centered pressure
    scheduleStep3(  patch,  sched,  old_dw, new_dw);
    
    // Step 4a compute sources of momentum
    scheduleStep4a( patch,  sched,  old_dw, new_dw);
    
    // Step 4b compute sources of energy
    scheduleStep4b( patch,  sched,  old_dw, new_dw);
    
    // Step 5a compute lagrangian quantities
    scheduleStep5a( patch,  sched,  old_dw, new_dw);
    
    // Step 5b cell centered momentum exchange
    scheduleStep5b( patch,  sched,  old_dw, new_dw);
    
    // Step 6and7 advect and advance in time
    scheduleStep6and7(patch,sched,  old_dw, new_dw);
  }
}
//STOP_DOC
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleStep1a--
 Purpose~   Compute the speed of sound
 
Programmer         Date       Description                      
----------         ----       -----------                 
John Schmidt      10/04/00                             
_____________________________________________________________________*/
void ICE::scheduleStep1a(
    const Patch*    patch,
    SchedulerP&     sched,
    DataWarehouseP& old_dw,
    DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumICEMatls();
  
  Task* task = scinew Task("ICE::step1a",patch, old_dw, new_dw,this,
			&ICE::actuallyStep1a);
  for (int m = 0; m < numMatls; m++) 
  {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    EquationOfState* eos = matl->getEOS();
    // Compute the speed of sound
    eos->addComputesAndRequiresSS(task,matl,patch,old_dw,new_dw);
  }
  sched->addTask(task);
}
//STOP_DOC
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleStep1b--
 Purpose~   Compute the equilibration pressure
 
Programmer         Date       Description                      
----------         ----       -----------                 
John Schmidt      10/04/00                             
_____________________________________________________________________*/
void ICE::scheduleStep1b(
    const Patch*    patch,
    SchedulerP&     sched,
    DataWarehouseP& old_dw,
    DataWarehouseP& new_dw)
{
  Task* task = scinew Task("ICE::step1b",patch, old_dw, new_dw,this,
			   &ICE::actuallyStep1b);
  
  task->requires(old_dw,lb->press_CCLabel, 0,patch,Ghost::None);
  
  int numMatls=d_sharedState->getNumICEMatls();
  for (int m = 0; m < numMatls; m++) 
  {
    ICEMaterial*  matl = d_sharedState->getICEMaterial(m);
    int dwindex = matl->getDWIndex();
    EquationOfState* eos = matl->getEOS();
    // Compute the rho micro
    eos->addComputesAndRequiresRM(task,matl,patch,old_dw,new_dw);
    task->requires(old_dw,lb->vol_frac_CCLabel,  dwindex,patch,Ghost::None);
    task->requires(old_dw,lb->rho_CCLabel,       dwindex,patch,Ghost::None);
    task->requires(old_dw,lb->rho_micro_CCLabel, dwindex,patch,Ghost::None);
    task->requires(old_dw,lb->temp_CCLabel,      dwindex,patch,Ghost::None);
    task->requires(old_dw,lb->cv_CCLabel,        dwindex,patch,Ghost::None);
    task->requires(new_dw,lb->speedSound_CCLabel,dwindex,patch,Ghost::None);
    task->computes(new_dw,lb->vol_frac_CCLabel,          dwindex, patch);
    task->computes(new_dw,lb->speedSound_equiv_CCLabel,  dwindex, patch);
    task->computes(new_dw,lb->rho_micro_equil_CCLabel,   dwindex, patch);
  }

  task->computes(new_dw,lb->press_CCLabel,0, patch);
  sched->addTask(task);
}
//STOP_DOC
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleStep1c--
 Purpose~   Compute the face-centered velocities

Version   Programmer         Date       Description                      
-------   ----------         ----       -----------                 
  1.0     John Schmidt      10/04/00                             
_____________________________________________________________________*/
void ICE::scheduleStep1c(
    const Patch*    patch,
    SchedulerP&     sched,
    DataWarehouseP& old_dw,
    DataWarehouseP& new_dw)
{
  Task* task = scinew Task("ICE::step1c",patch, old_dw, new_dw,this,
			   &ICE::actuallyStep1c);

  task->requires(new_dw,lb->press_CCLabel,0,patch,Ghost::None);

  int numMatls=d_sharedState->getNumICEMatls();
  for (int m = 0; m < numMatls; m++) 
  {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    int dwindex = matl->getDWIndex();
    task->requires(old_dw,lb->rho_CCLabel,   dwindex,patch,Ghost::None);
    task->requires(old_dw,lb->uvel_CCLabel,  dwindex,patch,Ghost::None);
    task->requires(old_dw,lb->vvel_CCLabel,  dwindex,patch,Ghost::None);
    task->requires(old_dw,lb->wvel_CCLabel,  dwindex,patch,Ghost::None);
    task->requires(new_dw,lb->rho_micro_equil_CCLabel,
		                               dwindex,patch,Ghost::None);
    
    task->computes(new_dw,lb->uvel_FCLabel,  dwindex, patch);
    task->computes(new_dw,lb->vvel_FCLabel,  dwindex, patch);
    task->computes(new_dw,lb->wvel_FCLabel,  dwindex, patch);
  }
  sched->addTask(task);
}
//STOP_DOC
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleStep1d--
 Purpose~   Schedule compute the momentum exchange for the face centered 
            velocities

Programmer         Date       Description                      
----------         ----       -----------                 
John Schmidt      10/04/00                             
_____________________________________________________________________*/
void ICE::scheduleStep1d(
    const Patch*    patch,
    SchedulerP&     sched,
    DataWarehouseP& old_dw,
    DataWarehouseP& new_dw)
{
  Task* task = scinew Task("ICE::step1d",patch, old_dw, new_dw,this,
			   &ICE::actuallyStep1d);
  int numMatls=d_sharedState->getNumICEMatls();
  
  for (int m = 0; m < numMatls; m++) 
  {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    int dwindex = matl->getDWIndex();
    task->requires(new_dw,lb->rho_micro_equil_CCLabel,
		                                  dwindex,patch,Ghost::None);
    task->requires(new_dw,lb->vol_frac_CCLabel, dwindex, patch,Ghost::None);
    task->requires(old_dw,lb->uvel_FCLabel,     dwindex, patch,Ghost::None);
    task->requires(old_dw,lb->vvel_FCLabel,     dwindex, patch,Ghost::None);
    task->requires(old_dw,lb->wvel_FCLabel,     dwindex, patch,Ghost::None);
    
    task->computes(new_dw,lb->uvel_FCMELabel,   dwindex, patch);
    task->computes(new_dw,lb->vvel_FCMELabel,   dwindex, patch);
    task->computes(new_dw,lb->wvel_FCMELabel,   dwindex, patch);
  }
  sched->addTask(task);
}
//STOP_DOC
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleStep2--
 Purpose~   Schedule compute delpress and new press_CC

Programmer         Date       Description                      
----------         ----       -----------                 
John Schmidt      10/04/00                             
_____________________________________________________________________*/
void ICE::scheduleStep2(
    const Patch*    patch,
    SchedulerP&     sched,
    DataWarehouseP& old_dw,
    DataWarehouseP& new_dw)
{
  Task* task = scinew Task("ICE::step2",patch, old_dw, new_dw,this,
			   &ICE::actuallyStep2);
  
  task->requires(new_dw,lb->press_CCLabel, 0,patch,Ghost::None);
  int numMatls=d_sharedState->getNumICEMatls();
  for (int m = 0; m < numMatls; m++) 
  {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    int dwindex = matl->getDWIndex();
    task->requires( new_dw, lb->vol_frac_CCLabel,  dwindex,patch,Ghost::None);
    task->requires( new_dw, lb->uvel_FCMELabel,    dwindex,patch,Ghost::None);
    task->requires( new_dw, lb->vvel_FCMELabel,    dwindex,patch,Ghost::None);
    task->requires( new_dw, lb->wvel_FCMELabel,    dwindex,patch,Ghost::None);
    task->requires( new_dw, lb->speedSound_equiv_CCLabel,
                                                   dwindex,patch,Ghost::None);
    task->requires( new_dw, lb->rho_micro_equil_CCLabel,
		                                     dwindex,patch,Ghost::None);
  }
  task->computes(   new_dw,lb->pressdP_CCLabel,     0,     patch);
  task->computes(   new_dw,lb->delPress_CCLabel,    0,     patch);
  
  sched->addTask(task);
}
//STOP_DOC
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleStep3--
 Purpose~   Schedule compute face centered pressure press_FC

Programmer         Date       Description                      
----------         ----       -----------                 
John Schmidt      10/04/00                             
_____________________________________________________________________*/
void ICE::scheduleStep3(
    const Patch*    patch,
    SchedulerP&     sched,
    DataWarehouseP& old_dw,
    DataWarehouseP& new_dw)
{
  Task* task = scinew Task("ICE::step3",patch, old_dw, new_dw,this,
			   &ICE::actuallyStep3);
  
  task->requires(   new_dw,lb->pressdP_CCLabel, 0,      patch,  Ghost::None);
  int numMatls = d_sharedState->getNumICEMatls();
  for (int m = 0; m < numMatls; m++) 
  {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    int dwindex = matl->getDWIndex();
    task->requires( old_dw, lb->rho_CCLabel,    dwindex, patch, Ghost::None);
  }
  
  task->computes(   new_dw, lb->pressX_FCLabel, 0,      patch);
  task->computes(   new_dw, lb->pressY_FCLabel, 0,      patch);
  task->computes(   new_dw, lb->pressZ_FCLabel, 0,      patch);
  
  sched->addTask(task);
}
//STOP_DOC
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleStep4a--
 Purpose~   Schedule compute sources and sinks of momentum

Programmer         Date       Description                      
----------         ----       -----------                 
John Schmidt      10/04/00                             
_____________________________________________________________________*/
void ICE::scheduleStep4a(
    const Patch*    patch,
    SchedulerP&     sched,
    DataWarehouseP& old_dw,
    DataWarehouseP& new_dw)
{
  Task* task = scinew Task("ICE::step4a",
            patch,      old_dw,         new_dw,     this,
	     &ICE::actuallyStep4a);

  task->requires(new_dw,    lb->pressX_FCLabel,     0,  patch,  Ghost::None);
  task->requires(new_dw,    lb->pressY_FCLabel,     0,  patch,  Ghost::None);
  task->requires(new_dw,    lb->pressZ_FCLabel,     0,  patch,  Ghost::None);
  int numMatls=d_sharedState->getNumICEMatls();
  
  for (int m = 0; m < numMatls; m++) 
  {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    int dwindex = matl->getDWIndex();
    task->requires(old_dw,  lb->rho_CCLabel,        dwindex,patch,Ghost::None);
    task->requires(old_dw,  lb->uvel_CCLabel,       dwindex,patch,Ghost::None);
    task->requires(old_dw,  lb->vvel_CCLabel,       dwindex,patch,Ghost::None);
    task->requires(old_dw,  lb->wvel_CCLabel,       dwindex,patch,Ghost::None);
    task->requires(old_dw,  lb->viscosity_CCLabel,  dwindex,patch,Ghost::None);
    task->requires(new_dw,  lb->vol_frac_CCLabel,   dwindex,patch,Ghost::None);
 
    task->computes(new_dw,  lb->xmom_source_CCLabel,dwindex,patch);
    task->computes(new_dw,  lb->ymom_source_CCLabel,dwindex,patch);
    task->computes(new_dw,  lb->zmom_source_CCLabel,dwindex,patch);
    task->computes(new_dw,  lb->tau_X_FCLabel,      dwindex,patch);
    task->computes(new_dw,  lb->tau_Y_FCLabel,      dwindex,patch);
    task->computes(new_dw,  lb->tau_Z_FCLabel,      dwindex,patch);
  }
  sched->addTask(task);
}
//STOP_DOC
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleStep4b--
 Purpose~   Schedule compute sources and sinks of energy

Programmer         Date       Description                      
----------         ----       -----------                 
John Schmidt      10/04/00                             
_____________________________________________________________________*/
void ICE::scheduleStep4b(
    const Patch*    patch,
    SchedulerP&     sched,
    DataWarehouseP& old_dw,
    DataWarehouseP& new_dw)

{
  Task* task = scinew Task("ICE::step4b",patch, old_dw, new_dw,this,
			   &ICE::actuallyStep4b);
  
  task->requires(new_dw,    lb->press_CCLabel,    0, patch, Ghost::None);
  task->requires(new_dw,    lb->delPress_CCLabel, 0, patch, Ghost::None);
  int numMatls=d_sharedState->getNumICEMatls();
  
  for (int m = 0; m < numMatls; m++) 
  {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    int dwindex = matl->getDWIndex();
    
    task->requires( new_dw, lb->rho_micro_equil_CCLabel,    dwindex, patch,
                                                            Ghost::None);
    task->requires( new_dw, lb->speedSound_equiv_CCLabel,   dwindex, patch,
                                                            Ghost::None);
    task->requires( new_dw, lb->vol_frac_CCLabel,           dwindex, patch,
                                                            Ghost::None);
    
    task->computes (new_dw, lb->int_eng_source_CCLabel,     dwindex, patch);
  }
  sched->addTask(task);
}
//STOP_DOC
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleStep5a--
 Purpose~   Schedule compute lagrangian mass momentum and internal energy
 
Programmer         Date       Description                      
----------         ----       -----------                 
John Schmidt      10/04/00                             
_____________________________________________________________________*/
void ICE::scheduleStep5a(
    const Patch*    patch,
    SchedulerP&     sched,
    DataWarehouseP& old_dw,
    DataWarehouseP& new_dw)
{
  Task* task = scinew Task("ICE::step5a",patch, old_dw, new_dw,this,
			       &ICE::actuallyStep5a);
  int numMatls=d_sharedState->getNumICEMatls();
  for (int m = 0; m < numMatls; m++) 
  {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    int dwindex = matl->getDWIndex();
    task->requires( old_dw, lb->rho_CCLabel,        dwindex,patch,Ghost::None);
    task->requires( old_dw, lb->uvel_CCLabel,       dwindex,patch,Ghost::None);
    task->requires( old_dw, lb->vvel_CCLabel,       dwindex,patch,Ghost::None);
    task->requires( old_dw, lb->wvel_CCLabel,       dwindex,patch,Ghost::None);
    task->requires( old_dw, lb->cv_CCLabel,         dwindex,patch,Ghost::None);
    task->requires( old_dw, lb->temp_CCLabel,       dwindex,patch,Ghost::None);
    task->requires( new_dw, lb->xmom_source_CCLabel,dwindex,patch,Ghost::None);
    task->requires( new_dw, lb->ymom_source_CCLabel,dwindex,patch,Ghost::None);
    task->requires( new_dw, lb->zmom_source_CCLabel,dwindex,patch,Ghost::None);
    task->requires( new_dw, lb->int_eng_source_CCLabel,
		                                      dwindex,patch,Ghost::None);
 
    task->computes( new_dw, lb->xmom_L_CCLabel,     dwindex,patch);
    task->computes( new_dw, lb->ymom_L_CCLabel,     dwindex,patch);
    task->computes( new_dw, lb->zmom_L_CCLabel,     dwindex,patch);
    task->computes( new_dw, lb->int_eng_L_CCLabel,  dwindex,patch);
    task->computes( new_dw, lb->mass_L_CCLabel,     dwindex,patch);
    task->computes( new_dw, lb->rho_L_CCLabel,      dwindex,patch);
  }
  sched->addTask(task);
}
//STOP_DOC
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleStep5b--
 Purpose~   Schedule momentum and energy exchange on the lagrangian quantities

Programmer         Date       Description                      
----------         ----       -----------                 
John Schmidt      10/04/00                             
_____________________________________________________________________*/
void ICE::scheduleStep5b(
    const Patch*    patch,
    SchedulerP&     sched,
    DataWarehouseP& old_dw,
    DataWarehouseP& new_dw)
{
  Task* task = scinew Task("ICE::step5b",patch, old_dw, new_dw,this,
			   &ICE::actuallyStep5b);
  int numMatls=d_sharedState->getNumICEMatls();
  
  for (int m = 0; m < numMatls; m++) 
  {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    int dwindex = matl->getDWIndex();
    task->requires( old_dw, lb->rho_CCLabel,        dwindex,patch,Ghost::None);
    task->requires( new_dw, lb->xmom_L_CCLabel,     dwindex,patch,Ghost::None);
    task->requires( new_dw, lb->ymom_L_CCLabel,     dwindex,patch,Ghost::None);
    task->requires( new_dw, lb->zmom_L_CCLabel,     dwindex,patch,Ghost::None);
    task->requires( new_dw, lb->int_eng_L_CCLabel,  dwindex,patch,Ghost::None);
    task->requires( new_dw, lb->vol_frac_CCLabel,   dwindex,patch,Ghost::None);
    task->requires( old_dw, lb->cv_CCLabel,         dwindex,patch,Ghost::None);
    task->requires( new_dw, lb->rho_micro_equil_CCLabel,
		                                      dwindex,patch,Ghost::None);
    
    task->computes( new_dw, lb->xmom_L_ME_CCLabel,  dwindex,patch);
    task->computes( new_dw, lb->ymom_L_ME_CCLabel,  dwindex,patch);
    task->computes( new_dw, lb->zmom_L_ME_CCLabel,  dwindex,patch);
    task->computes( new_dw, lb->int_eng_L_ME_CCLabel,dwindex,patch);
  }
  sched->addTask(task);
}
//STOP_DOC
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleStep6and7--
 Purpose~   Schedule advance and advect in time for mass, momentum
            and energy.  Note this function puts (*)vel_CC, rho_CC
            and Temp_CC into new dw, not flux variables

Programmer         Date       Description                      
----------         ----       -----------                 
John Schmidt      10/04/00                             
_____________________________________________________________________*/
void ICE::scheduleStep6and7(
    const Patch*    patch,
    SchedulerP&     sched,
    DataWarehouseP& old_dw,
    DataWarehouseP& new_dw)
{
  Task* task = scinew Task("ICE::step6and7",patch, old_dw, new_dw,this,
			   &ICE::actuallyStep6and7);
  int numMatls=d_sharedState->getNumICEMatls();
  for (int m = 0; m < numMatls; m++ ) 
  {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    int dwindex = matl->getDWIndex();
    task->requires(old_dw, lb->cv_CCLabel,dwindex,patch,Ghost::None,0);
    task->requires(old_dw, lb->rho_CCLabel,       dwindex,patch,Ghost::None);
    task->requires(old_dw, lb->uvel_CCLabel,      dwindex,patch,Ghost::None);
    task->requires(old_dw, lb->vvel_CCLabel,      dwindex,patch,Ghost::None);
    task->requires(old_dw, lb->wvel_CCLabel,      dwindex,patch,Ghost::None);
    task->requires(old_dw, lb->temp_CCLabel,      dwindex,patch,Ghost::None);
    task->requires(new_dw, lb->xmom_L_ME_CCLabel,
		                                   dwindex,patch,Ghost::None,0);
    task->requires(new_dw, lb->ymom_L_ME_CCLabel,
		                                   dwindex,patch,Ghost::None,0);
    task->requires(new_dw, lb->zmom_L_ME_CCLabel,
		                                   dwindex,patch,Ghost::None,0);
    task->requires(new_dw, lb->int_eng_L_ME_CCLabel,
		                                   dwindex,patch,Ghost::None,0);

    task->requires(new_dw, lb->speedSound_CCLabel,dwindex,patch,Ghost::None);
    task->computes(new_dw, lb->temp_CCLabel,      dwindex,patch);
    task->computes(new_dw, lb->rho_CCLabel,       dwindex,patch);
    task->computes(new_dw, lb->cv_CCLabel,        dwindex,patch);
    task->computes(new_dw, lb->uvel_CCLabel,      dwindex,patch);
    task->computes(new_dw, lb->vvel_CCLabel,      dwindex,patch);
    task->computes(new_dw, lb->wvel_CCLabel,      dwindex,patch);
  }
  task->computes(new_dw, d_sharedState->get_delt_label());
  sched->addTask(task);
}
//STOP_DOC
