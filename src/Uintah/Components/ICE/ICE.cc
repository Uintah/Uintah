#include <Uintah/Components/ICE/ICE.h>
#include <Uintah/Interface/CFDInterface.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/CCVariable.h>
#include <SCICore/Geometry/Vector.h>
#include <Uintah/Parallel/ProcessorGroup.h>
#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/NodeIterator.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/PerPatch.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SoleVariable.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Exceptions/ParameterNotFound.h>
#include <Uintah/Parallel/ProcessorGroup.h>
#include <Uintah/Components/ICE/ICEMaterial.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Grid/VarTypes.h>
#include <SCICore/Datatypes/DenseMatrix.h>
#include <vector>
#include <Uintah/Grid/BoundCond.h>
#include <Uintah/Grid/PressureBoundCond.h>
#include <Uintah/Grid/VelocityBoundCond.h>
#include <Uintah/Grid/TemperatureBoundCond.h>
#include <Uintah/Grid/DensityBoundCond.h>

using std::vector;
using std::max;
using SCICore::Geometry::Vector;

using namespace Uintah;
using namespace Uintah::ICESpace;
using SCICore::Datatypes::DenseMatrix;

ICE::ICE(const ProcessorGroup* myworld) 
  : UintahParallelComponent(myworld)
{
  lb = new ICELabel();

  IFS_CCLabel = scinew VarLabel("IFS_CC",
                                CCVariable<fflux>::getTypeDescription());
  OFS_CCLabel = scinew VarLabel("OFS_CC",
                                CCVariable<fflux>::getTypeDescription());
  IFE_CCLabel = scinew VarLabel("IFE_CC",
                                CCVariable<eflux>::getTypeDescription());
  OFE_CCLabel = scinew VarLabel("OFE_CC",
                                CCVariable<eflux>::getTypeDescription());
  q_outLabel = scinew VarLabel("q_out",
                                CCVariable<fflux>::getTypeDescription());
  q_out_EFLabel = scinew VarLabel("q_out_EF",
                                CCVariable<eflux>::getTypeDescription());
  q_inLabel = scinew VarLabel("q_in",
                                CCVariable<fflux>::getTypeDescription());
  q_in_EFLabel = scinew VarLabel("q_in_EF",
                                CCVariable<eflux>::getTypeDescription());

}

ICE::~ICE()
{
  delete IFS_CCLabel;
  delete OFS_CCLabel;
  delete IFE_CCLabel;
  delete OFE_CCLabel;
  delete q_outLabel;;
  delete q_out_EFLabel;;
  delete q_inLabel;;
  delete q_in_EFLabel;;
}

void ICE::problemSetup(const ProblemSpecP& prob_spec, GridP& grid,
		       SimulationStateP& sharedState)
{
  d_sharedState = sharedState;
  d_SMALL_NUM = 1.e-12;

    cerr << "In the preprocessor . . ." << endl;
    
  // Search for the MaterialProperties block and then get the MPM section
  
  ProblemSpecP mat_ps =  prob_spec->findBlock("MaterialProperties");
 
  ProblemSpecP ice_mat_ps = mat_ps->findBlock("ICE");  

  for (ProblemSpecP ps = ice_mat_ps->findBlock("material"); ps != 0;
       ps = ps->findNextBlock("material") ) {
    // Extract out the type of EOS and the 
    // associated parameters
     ICEMaterial *mat = scinew ICEMaterial(ps);
     sharedState->registerMaterial(mat);
     sharedState->registerICEMaterial(mat);
  }     

  // Pull out the exchange coefficients

  ProblemSpecP exch_ps = ice_mat_ps->findBlock("exchange_coefficients");

  exch_ps->require("momentum",d_K_mom);
  exch_ps->require("heat",d_K_heat);

  cout << "K_mom = " << d_K_mom << endl;
  cout << "K_heat = " << d_K_heat << endl;

  cout << "Number of ICE materials: " << d_sharedState->getNumICEMatls()<< endl;

  ProblemSpecP ic_ps = prob_spec->findBlock("InitialConditions");
  ProblemSpecP ice_ic_ps = ic_ps->findBlock("ICE");
  ice_ic_ps->require("pressure",d_pressure);   
}

void ICE::scheduleInitialize(const LevelP& level, SchedulerP& sched, 
			     DataWarehouseP& dw)
{

  Level::const_patchIterator iter;

  for(iter=level->patchesBegin(); iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    Task* t = scinew Task("ICE::actuallyInitialize", patch, dw, dw,this,
			    &ICE::actuallyInitialize);
    t->computes(dw, d_sharedState->get_delt_label());
    for (int m = 0; m < d_sharedState->getNumICEMatls(); m++ ) {
	ICEMaterial*  matl = d_sharedState->getICEMaterial(m);
	int dwindex = matl->getDWIndex();
	t->computes(dw, lb->temp_CCLabel,      dwindex, patch);
	t->computes(dw, lb->rho_micro_CCLabel, dwindex, patch);
	t->computes(dw, lb->rho_CCLabel,       dwindex, patch);
	t->computes(dw, lb->cv_CCLabel,        dwindex, patch);
	t->computes(dw, lb->viscosity_CCLabel, dwindex, patch);
	t->computes(dw, lb->vol_frac_CCLabel,  dwindex, patch);
	t->computes(dw, lb->uvel_CCLabel,      dwindex, patch);
	t->computes(dw, lb->vvel_CCLabel,      dwindex, patch);
	t->computes(dw, lb->wvel_CCLabel,      dwindex, patch);
	t->computes(dw, lb->uvel_FCLabel,      dwindex, patch);
	t->computes(dw, lb->vvel_FCLabel,      dwindex, patch);
	t->computes(dw, lb->wvel_FCLabel,      dwindex, patch);
    }

    t->computes(dw, lb->press_CCLabel,0, patch);

    sched->addTask(t);
  }

}

void ICE::scheduleComputeStableTimestep(const LevelP& level,
					SchedulerP& sched,
					DataWarehouseP& dw)
{

}


void ICE::scheduleTimeAdvance(double t, double dt,
			      const LevelP& level,
			      SchedulerP& sched,
			      DataWarehouseP& old_dw,
			      DataWarehouseP& new_dw)
{
  for(Level::const_patchIterator iter=level->patchesBegin();
       iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;

    // Step 1a  computeSoundSpeed
    scheduleStep1a(patch,sched,old_dw,new_dw);
    // Step 1b calculate equlibration pressure
    scheduleStep1b(patch,sched,old_dw,new_dw);
    // Step 1c compute face centered velocities
    scheduleStep1c(patch,sched,old_dw,new_dw);
    // Step 1d computes momentum exchange on FC velocities
    scheduleStep1d(patch,sched,old_dw,new_dw);
    // Step 2 computes delPress and the new pressure
    scheduleStep2(patch,sched,old_dw,new_dw);
    // Step 3 compute face centered pressure
    scheduleStep3(patch,sched,old_dw,new_dw);
    // Step 4a compute sources of momentum
    scheduleStep4a(patch,sched,old_dw,new_dw);
    // Step 4b compute sources of energy
    scheduleStep4b(patch,sched,old_dw,new_dw);
    // Step 5a compute lagrangian quantities
    scheduleStep5a(patch,sched,old_dw,new_dw);
    // Step 5b cell centered momentum exchange
    scheduleStep5b(patch,sched,old_dw,new_dw);
    // Step 6and7 advect and advance in time
    scheduleStep6and7(patch,sched,old_dw,new_dw);
  }

}

void ICE::scheduleStep1a(const Patch* patch,
			 SchedulerP& sched,
			 DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw)
{
  // ComputeSoundSpeed
  int numMatls = d_sharedState->getNumICEMatls();
  
  Task* task = scinew Task("ICE::step1a",patch, old_dw, new_dw,this,
			&ICE::actuallyStep1a);
  for (int m = 0; m < numMatls; m++) {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    EquationOfState* eos = matl->getEOS();
    // Compute the speed of sound
    eos->addComputesAndRequiresSS(task,matl,patch,old_dw,new_dw);
  }
  sched->addTask(task);
}

void ICE::scheduleStep1b(const Patch* patch,
			 SchedulerP& sched,
			 DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw)
{
  Task* task = scinew Task("ICE::step1b",patch, old_dw, new_dw,this,
			   &ICE::actuallyStep1b);
  
  task->requires(old_dw,lb->press_CCLabel, 0,patch,Ghost::None);
  
  int numMatls=d_sharedState->getNumICEMatls();
  for (int m = 0; m < numMatls; m++) {
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
void ICE::scheduleStep1c(const Patch* patch,
			 SchedulerP& sched,
			 DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw)
{
  Task* task = scinew Task("ICE::step1c",patch, old_dw, new_dw,this,
			   &ICE::actuallyStep1c);

  task->requires(new_dw,lb->press_CCLabel,0,patch,Ghost::None);

  int numMatls=d_sharedState->getNumICEMatls();
  for (int m = 0; m < numMatls; m++) {
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

void ICE::scheduleStep1d(const Patch* patch,
			 SchedulerP& sched,
			 DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw)
{
  Task* task = scinew Task("ICE::step1d",patch, old_dw, new_dw,this,
			   &ICE::actuallyStep1d);
  int numMatls=d_sharedState->getNumICEMatls();
  for (int m = 0; m < numMatls; m++) {
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

void ICE::scheduleStep2(const Patch* patch,
			SchedulerP& sched,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{
  Task* task = scinew Task("ICE::step2",patch, old_dw, new_dw,this,
			   &ICE::actuallyStep2);
  
  task->requires(new_dw,lb->press_CCLabel, 0,patch,Ghost::None);
  int numMatls=d_sharedState->getNumICEMatls();
  for (int m = 0; m < numMatls; m++) {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    int dwindex = matl->getDWIndex();
    task->requires(new_dw,lb->vol_frac_CCLabel, dwindex,patch,Ghost::None);
    task->requires(new_dw,lb->uvel_FCMELabel, dwindex,patch,Ghost::None);
    task->requires(new_dw,lb->vvel_FCMELabel, dwindex,patch,Ghost::None);
    task->requires(new_dw,lb->wvel_FCMELabel, dwindex,patch,Ghost::None);
    task->requires(new_dw,lb->speedSound_equiv_CCLabel,
		dwindex,patch,Ghost::None);
    task->requires(new_dw,lb->rho_micro_equil_CCLabel,
		dwindex,patch,Ghost::None);
    
    task->computes(new_dw,lb->div_velfc_CCLabel,dwindex,patch);
  }
  
  task->computes(new_dw,lb->pressdP_CCLabel,  0, patch);
  task->computes(new_dw,lb->delPress_CCLabel, 0, patch);
  
  sched->addTask(task);
}

void ICE::scheduleStep3(const Patch* patch,
			SchedulerP& sched,
			DataWarehouseP& old_dw,
			DataWarehouseP& new_dw)
{
  Task* task = scinew Task("ICE::step3",patch, old_dw, new_dw,this,
			   &ICE::actuallyStep3);
  
  task->requires(new_dw,lb->pressdP_CCLabel,0,patch,Ghost::None);
  int numMatls=d_sharedState->getNumICEMatls();
  for (int m = 0; m < numMatls; m++) {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    int dwindex = matl->getDWIndex();
    task->requires(old_dw,lb->rho_CCLabel, dwindex,patch,Ghost::None);
  }
  
  task->computes(new_dw,lb->pressX_FCLabel, 0, patch);
  task->computes(new_dw,lb->pressY_FCLabel, 0, patch);
  task->computes(new_dw,lb->pressZ_FCLabel, 0, patch);
  
  sched->addTask(task);
}

void ICE::scheduleStep4a(const Patch* patch,
			 SchedulerP& sched,
			 DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw)
{
  Task* task = scinew Task("ICE::step4a",patch, old_dw, new_dw,this,
			   &ICE::actuallyStep4a);

  task->requires(new_dw,lb->pressX_FCLabel, 0,patch,Ghost::None);
  task->requires(new_dw,lb->pressY_FCLabel, 0,patch,Ghost::None);
  task->requires(new_dw,lb->pressZ_FCLabel, 0,patch,Ghost::None);
  int numMatls=d_sharedState->getNumICEMatls();
  for (int m = 0; m < numMatls; m++) {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    int dwindex = matl->getDWIndex();
    task->requires(old_dw,lb->rho_CCLabel,       dwindex,patch,Ghost::None);
    task->requires(old_dw,lb->uvel_CCLabel,      dwindex,patch,Ghost::None);
    task->requires(old_dw,lb->vvel_CCLabel,      dwindex,patch,Ghost::None);
    task->requires(old_dw,lb->wvel_CCLabel,      dwindex,patch,Ghost::None);
    task->requires(old_dw,lb->viscosity_CCLabel, dwindex,patch,Ghost::None);
    task->requires(new_dw,lb->vol_frac_CCLabel,  dwindex,patch,Ghost::None);
    
    task->computes(new_dw,lb->xmom_source_CCLabel, dwindex,patch);
    task->computes(new_dw,lb->ymom_source_CCLabel, dwindex,patch);
    task->computes(new_dw,lb->zmom_source_CCLabel, dwindex,patch);
    task->computes(new_dw,lb->tau_X_FCLabel,       dwindex,patch);
    task->computes(new_dw,lb->tau_Y_FCLabel,       dwindex,patch);
    task->computes(new_dw,lb->tau_Z_FCLabel,       dwindex,patch);
  }
  sched->addTask(task);
}

void ICE::scheduleStep4b(const Patch* patch,
			 SchedulerP& sched,
			 DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw)

{
  Task* task = scinew Task("ICE::step4b",patch, old_dw, new_dw,this,
			   &ICE::actuallyStep4b);
  
  task->requires(new_dw,lb->press_CCLabel,    0,patch,Ghost::None);
  task->requires(new_dw,lb->delPress_CCLabel, 0,patch,Ghost::None);
  int numMatls=d_sharedState->getNumICEMatls();
  for (int m = 0; m < numMatls; m++) {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    int dwindex = matl->getDWIndex();
    task->requires(new_dw,lb->rho_micro_equil_CCLabel,
		   dwindex,patch,Ghost::None);
    task->requires(new_dw,lb->speedSound_equiv_CCLabel,
		   dwindex,patch,Ghost::None);
    task->requires(new_dw,lb->vol_frac_CCLabel, dwindex,patch,Ghost::None);
    
    task->computes(new_dw,lb->int_eng_source_CCLabel, dwindex,patch);
  }
  sched->addTask(task);
}

void ICE::scheduleStep5a(const Patch* patch,
			 SchedulerP& sched,
			 DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw)
{
  Task* task = scinew Task("ICE::step5a",patch, old_dw, new_dw,this,
			       &ICE::actuallyStep5a);
  int numMatls=d_sharedState->getNumICEMatls();
  for (int m = 0; m < numMatls; m++) {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    int dwindex = matl->getDWIndex();
    task->requires(old_dw,lb->rho_CCLabel,        dwindex,patch,Ghost::None);
    task->requires(old_dw,lb->uvel_CCLabel,       dwindex,patch,Ghost::None);
    task->requires(old_dw,lb->vvel_CCLabel,       dwindex,patch,Ghost::None);
    task->requires(old_dw,lb->wvel_CCLabel,       dwindex,patch,Ghost::None);
    task->requires(old_dw,lb->cv_CCLabel,         dwindex,patch,Ghost::None);
    task->requires(old_dw,lb->temp_CCLabel,       dwindex,patch,Ghost::None);
    task->requires(new_dw,lb->xmom_source_CCLabel,dwindex,patch,Ghost::None);
    task->requires(new_dw,lb->ymom_source_CCLabel,dwindex,patch,Ghost::None);
    task->requires(new_dw,lb->zmom_source_CCLabel,dwindex,patch,Ghost::None);
    task->requires(new_dw,lb->int_eng_source_CCLabel,
		   dwindex,patch,Ghost::None);
    
    task->computes(new_dw,lb->xmom_L_CCLabel,     dwindex, patch);
    task->computes(new_dw,lb->ymom_L_CCLabel,     dwindex, patch);
    task->computes(new_dw,lb->zmom_L_CCLabel,     dwindex, patch);
    task->computes(new_dw,lb->int_eng_L_CCLabel,  dwindex, patch);
    task->computes(new_dw,lb->mass_L_CCLabel,     dwindex, patch);
    task->computes(new_dw,lb->rho_L_CCLabel,      dwindex, patch);
  }
  sched->addTask(task);
}

void ICE::scheduleStep5b(const Patch* patch,
			 SchedulerP& sched,
			 DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw)
{
  Task* task = scinew Task("ICE::step5b",patch, old_dw, new_dw,this,
			   &ICE::actuallyStep5b);
  int numMatls=d_sharedState->getNumICEMatls();
  for (int m = 0; m < numMatls; m++) {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    int dwindex = matl->getDWIndex();
    task->requires(old_dw,lb->rho_CCLabel,       dwindex,patch,Ghost::None);
    task->requires(new_dw,lb->xmom_L_CCLabel,    dwindex,patch,Ghost::None);
    task->requires(new_dw,lb->ymom_L_CCLabel,    dwindex,patch,Ghost::None);
    task->requires(new_dw,lb->zmom_L_CCLabel,    dwindex,patch,Ghost::None);
    task->requires(new_dw,lb->int_eng_L_CCLabel, dwindex,patch,Ghost::None);
    task->requires(new_dw,lb->vol_frac_CCLabel,  dwindex,patch,Ghost::None);
    task->requires(old_dw,lb->cv_CCLabel,        dwindex,patch,Ghost::None);
    task->requires(new_dw,lb->rho_micro_equil_CCLabel,
		   dwindex,patch,Ghost::None);
    
    task->computes(new_dw,lb->xmom_L_ME_CCLabel,    dwindex, patch);
    task->computes(new_dw,lb->ymom_L_ME_CCLabel,    dwindex, patch);
    task->computes(new_dw,lb->zmom_L_ME_CCLabel,    dwindex, patch);
    task->computes(new_dw,lb->int_eng_L_ME_CCLabel, dwindex, patch);
  }
  sched->addTask(task);
}

void ICE::scheduleStep6and7(const Patch* patch,
			    SchedulerP& sched,
			    DataWarehouseP& old_dw,
			    DataWarehouseP& new_dw)
{
  Task* task = scinew Task("ICE::step6and7",patch, old_dw, new_dw,this,
			   &ICE::actuallyStep6and7);
  int numMatls=d_sharedState->getNumICEMatls();
  for (int m = 0; m < numMatls; m++ ) {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    int dwindex = matl->getDWIndex();
    task->requires(new_dw, lb->xmom_L_ME_CCLabel,
		   dwindex,patch,Ghost::None,0);
    task->requires(new_dw, lb->ymom_L_ME_CCLabel,
		   dwindex,patch,Ghost::None,0);
    task->requires(new_dw, lb->zmom_L_ME_CCLabel,
		   dwindex,patch,Ghost::None,0);
    task->requires(new_dw, lb->int_eng_L_ME_CCLabel,
		   dwindex,patch,Ghost::None,0);
    
    task->computes(new_dw, lb->temp_CCLabel,dwindex, patch);
    task->computes(new_dw, lb->rho_CCLabel, dwindex, patch);
    task->computes(new_dw, lb->cv_CCLabel,  dwindex, patch);
    task->computes(new_dw, lb->uvel_CCLabel,dwindex, patch);
    task->computes(new_dw, lb->vvel_CCLabel,dwindex, patch);
    task->computes(new_dw, lb->wvel_CCLabel,dwindex, patch);
  }
  task->computes(new_dw, d_sharedState->get_delt_label());
  sched->addTask(task);
}



void ICE::actuallyInitialize(const ProcessorGroup*,
			     const Patch* patch,
			     DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw)
{

  cout << "Doing actually Initialize" << endl;
  double dT = 0.0001;
  new_dw->put(delt_vartype(dT), lb->delTLabel);

  CCVariable<double> rho_micro, temp, cv, rho_CC,press,speedSound;
  CCVariable<double> uvel_CC,vvel_CC,wvel_CC, visc_CC,vol_frac_CC;

  new_dw->allocate(press,lb->press_CCLabel,0,patch);

  SFCXVariable<double> uvel_FC;
  SFCYVariable<double> vvel_FC;
  SFCZVariable<double> wvel_FC;

  cout << "Initial pressure = " << d_pressure << endl;
  // Store the initial pressure
  press.initialize(d_pressure);
  // The application of the pressure bcs will overwrite the extra cell
  // locations.

  // Store the pressure BCs

  setBC(press,"Pressure",patch);


#if 1
  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
    cout << "press["<< *iter<< "]=" << press[*iter] << endl;
  } 
#endif

  for (int m = 0; m < d_sharedState->getNumICEMatls(); m++ ) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int dwindex = ice_matl->getDWIndex();
      new_dw->allocate(rho_micro,  lb->rho_micro_CCLabel,  dwindex,patch);
      new_dw->allocate(rho_CC,     lb->rho_CCLabel,        dwindex,patch);
      new_dw->allocate(temp,       lb->temp_CCLabel,       dwindex,patch);
      new_dw->allocate(cv,         lb->cv_CCLabel,         dwindex,patch);
      new_dw->allocate(speedSound, lb->speedSound_CCLabel, dwindex,patch);
      new_dw->allocate(visc_CC,    lb->viscosity_CCLabel,  dwindex,patch);
      new_dw->allocate(vol_frac_CC,lb->vol_frac_CCLabel,   dwindex,patch);

      new_dw->allocate(uvel_CC,lb->uvel_CCLabel,dwindex,patch);
      new_dw->allocate(vvel_CC,lb->vvel_CCLabel,dwindex,patch);
      new_dw->allocate(wvel_CC,lb->wvel_CCLabel,dwindex,patch);

      new_dw->allocate(uvel_FC,lb->uvel_FCLabel,dwindex,patch);
      new_dw->allocate(vvel_FC,lb->vvel_FCLabel,dwindex,patch);
      new_dw->allocate(wvel_FC,lb->wvel_FCLabel,dwindex,patch);

      // Set the initial conditions:
      ice_matl->initializeCells(rho_micro, rho_CC, temp, cv, speedSound,
				visc_CC, vol_frac_CC, uvel_CC, vvel_CC, 
				wvel_CC,patch, new_dw);

      // Initialize the face centered velocities to 0

      uvel_FC.initialize(0.);
      vvel_FC.initialize(0.);
      wvel_FC.initialize(0.);

      // Set the boundary conditions:
      //    uvel,vvel,wvel,temp,rho_CC
#if 1
      cout << "Before doing the boundary conditions" << endl;
      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	  iter++){
	cout << "rho_CC["<< *iter<< "]=" << rho_CC[*iter] << endl;
	cout << "rho_micro["<< *iter<< "]=" << rho_micro[*iter] << endl;
	cout << "temp["<< *iter<< "]=" << temp[*iter] << endl;
	cout << "uvel_CC["<< *iter<< "]=" << uvel_CC[*iter] << endl;
	cout << "vvel_CC["<< *iter<< "]=" << vvel_CC[*iter] << endl;
	cout << "wvel_CC["<< *iter<< "]=" << wvel_CC[*iter] << endl;
	
      } 
#endif
      setBC(rho_CC,"Density",patch);
      setBC(temp,"Temperature",patch);
      setBC(uvel_CC,"Velocity","x",patch);
      setBC(vvel_CC,"Velocity","y",patch);
      setBC(wvel_CC,"Velocity","z",patch);

#if 1
      cout << "After doing the boundary conditions" << endl;
      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	  iter++){
	cout << "rho_CC["<< *iter<< "]=" << rho_CC[*iter] << endl;
	cout << "rho_micro["<< *iter<< "]=" << rho_micro[*iter] << endl;
	cout << "temp["<< *iter<< "]=" << temp[*iter] << endl;
	cout << "uvel_CC["<< *iter<< "]=" << uvel_CC[*iter] << endl;
	cout << "vvel_CC["<< *iter<< "]=" << vvel_CC[*iter] << endl;
	cout << "wvel_CC["<< *iter<< "]=" << wvel_CC[*iter] << endl;
	
      } 
#endif

      new_dw->put(rho_micro,  lb->rho_micro_CCLabel, dwindex,patch);
      new_dw->put(rho_CC,     lb->rho_CCLabel,       dwindex,patch);
      new_dw->put(vol_frac_CC,lb->vol_frac_CCLabel,  dwindex,patch);
      new_dw->put(temp,       lb->temp_CCLabel,      dwindex,patch);
      new_dw->put(cv,         lb->cv_CCLabel,        dwindex,patch);
      new_dw->put(speedSound, lb->speedSound_CCLabel,dwindex,patch);
      new_dw->put(uvel_CC,    lb->uvel_CCLabel,      dwindex,patch);
      new_dw->put(vvel_CC,    lb->vvel_CCLabel,      dwindex,patch);
      new_dw->put(wvel_CC,    lb->wvel_CCLabel,      dwindex,patch);
      new_dw->put(uvel_FC,    lb->uvel_FCLabel,      dwindex,patch);
      new_dw->put(vvel_FC,    lb->vvel_FCLabel,      dwindex,patch);
      new_dw->put(wvel_FC,    lb->wvel_FCLabel,      dwindex,patch);
      new_dw->put(visc_CC,    lb->viscosity_CCLabel, dwindex,patch);
  }
  new_dw->put(press,lb->press_CCLabel,0,patch);
  
}

void ICE::actuallyComputeStableTimestep(const ProcessorGroup*,
					const Patch* patch,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw)
{
  cout << "Doing actually Compute Stable Timestep " << endl;
}


void ICE::actuallyStep1a(const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{

  cout << "Doing actually step1a -- speed_of_sound_MM" << endl;

  int numMatls = d_sharedState->getNumICEMatls();

  // Compute the speed of sound

  for (int m = 0; m < numMatls; m++) {
    ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
    EquationOfState* eos = ice_matl->getEOS();
    eos->computeSpeedSound(patch,ice_matl,old_dw,new_dw);
  }

}

void ICE::actuallyStep1b(const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{

  cout << "Doing actually step1b -- calc_equilibration_pressure" << endl;

  int numMatls = d_sharedState->getNumICEMatls();

  // Compute the equilibration pressure for all materials

  // Compute initial Rho Micro
  for(int m = 0; m < numMatls; m++){
    ICEMaterial* ice_matl = d_sharedState->getICEMaterial( m );
    ice_matl->getEOS()->computeRhoMicro(patch,ice_matl,old_dw,new_dw);   
  }

  
   // Need to pull out all of the material's data just like in 
   // contact::exMomInterpolated
   // store in a vector<CCVariable<double>>
  
  // Compute the initial volume fraction
  vector<CCVariable<double> > vol_frac(numMatls);
  for (int m = 0; m < numMatls; m++) {
    ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int dwindex = ice_matl->getDWIndex();
      CCVariable<double> rho_micro,rho;    
      new_dw->allocate(vol_frac[m],lb->vol_frac_CCLabel,dwindex,patch);
      old_dw->get(rho,lb->rho_CCLabel,   dwindex, patch,Ghost::None, 0); 
      new_dw->get(rho_micro,lb->rho_micro_CCLabel, dwindex,patch,
		  Ghost::None, 0); 
      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	  iter++){
	vol_frac[m][*iter] = rho[*iter]/rho_micro[*iter];
      }
   }
  
  vector<double> delVol_frac(numMatls),press_eos(numMatls);
  vector<double> dp_drho(numMatls),dp_de(numMatls);
  
  vector<CCVariable<double> > rho_micro(numMatls),rho_micro_equil(numMatls);
  vector<CCVariable<double> > rho(numMatls);
  vector<CCVariable<double> > cv(numMatls);
  vector<CCVariable<double> > Temp(numMatls);
  vector<CCVariable<double> > speedSound(numMatls),speedSound_old(numMatls);
  CCVariable<double> press,press_new;
  
  old_dw->get(press,         lb->press_CCLabel, 0,patch,Ghost::None, 0); 
  new_dw->allocate(press_new,lb->press_CCLabel, 0,patch);

  for (int m = 0; m < numMatls; m++) {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    int dwindex = matl->getDWIndex();
    old_dw->get(cv[m], lb->cv_CCLabel,     dwindex,patch,Ghost::None, 0); 
    old_dw->get(rho[m], lb->rho_CCLabel,   dwindex,patch,Ghost::None, 0); 
    old_dw->get(Temp[m], lb->temp_CCLabel, dwindex,patch,Ghost::None, 0); 
    old_dw->get(speedSound_old[m], lb->speedSound_CCLabel, dwindex,
		  patch,Ghost::None, 0); 

    new_dw->allocate(speedSound[m],lb->speedSound_equiv_CCLabel,dwindex,
		     patch);
    new_dw->allocate(rho_micro_equil[m],lb->rho_micro_equil_CCLabel,dwindex,
		     patch);
    new_dw->get(rho_micro[m],lb->rho_micro_CCLabel,dwindex,patch,
		Ghost::None,0);
  }

  press_new = press;

  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
    double delPress = 0.;
    bool converged = false;
    while( converged == false) {
     double A = 0.;
     double B = 0.;
     double C = 0.;
     
     for (int m = 0; m < numMatls; m++) 
       delVol_frac[m] = 0.;

     for (int m = 0; m < numMatls; m++) {
	ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
	double gamma = ice_matl->getGamma();
	ice_matl->getEOS()->computePressEOS(rho_micro[m][*iter],gamma,
					    cv[m][*iter],
					    Temp[m][*iter],press_eos[m],
					    dp_drho[m],dp_de[m]);
     }
     
     vector<double> Q(numMatls),y(numMatls);     
     for (int m = 0; m < numMatls; m++) {
       ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
	 Q[m] = press_new[*iter] - press_eos[m];
	 y[m] = rho[m][*iter]/(vol_frac[m][*iter]*vol_frac[m][*iter]) 
	   * dp_drho[m];
	 A += vol_frac[m][*iter];
	 B += Q[m]/y[m];
	 C += 1./y[m];
     }
     double vol_frac_not_close_packed = 1.;
     delPress = (A - vol_frac_not_close_packed - B)/C;
     press_new[*iter] += delPress;

     for (int m = 0; m < numMatls; m++) {
	ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
	double gamma = ice_matl->getGamma();
	rho_micro[m][*iter] = ice_matl->getEOS()->
	   computeRhoMicro(press_new[*iter], gamma,cv[m][*iter],
			   Temp[m][*iter]);
     }

     // Compute delVol_frac
     for (int m = 0; m < numMatls; m++) {
       delVol_frac[m] = -(Q[m] + delPress)/y[m];
       vol_frac[m][*iter] += delVol_frac[m];
     }
     
      // compute speed of sound mm
     //  1. compute press eos
     //  2. compute sound speed using dp_drho, dp_de, press_eos;
     for (int m = 0; m < numMatls; m++) {
        ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
	double gamma = ice_matl->getGamma();
	ice_matl->getEOS()->computePressEOS(rho_micro[m][*iter],gamma,
					     cv[m][*iter],
					     Temp[m][*iter],press_eos[m],
					     dp_drho[m],dp_de[m]);
	 
	double temp = dp_drho[m] + dp_de[m] * 
	   (press_eos[m]/(rho_micro[m][*iter]*rho_micro[m][*iter]));
	speedSound[m][*iter] = sqrt(temp);
     }
     
     
     // Check if converged
     
     double test = 0.;
     test = std::max(test,fabs(delPress));
     for (int m = 0; m < numMatls; m++) {
       test = std::max(test,fabs(delVol_frac[m]));
     }
     if (test < d_SMALL_NUM)
       converged = true;
     
    }  // end of converged
  }
  // Update the boundary conditions for the variables:
  // Pressure (press_new)

#if 1
  for (CellIterator iter=patch->getExtraCellIterator(); !iter.done(); iter++) {
    cout << "press_new["<<*iter<<"]="<<press_new[*iter] << endl;
  }
#endif

  setBC(press_new,"Pressure",patch);
 
  
  // Hydrostatic pressure adjustment - subtract off the hydrostatic pressure
  
  Vector dx = patch->dCell();
  Vector gravity = d_sharedState->getGravity();
  IntVector highIndex = patch->getCellHighIndex();
  IntVector lowIndex = patch->getCellLowIndex();
  
  double width = (highIndex.x() - lowIndex.x())*dx.x();
  double height = (highIndex.y() - lowIndex.y())*dx.y();
  double depth = (highIndex.z() - lowIndex.z())*dx.z();
  
  if (gravity.x() != 0.) {
    // x direction
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	iter++){
      IntVector curcell = *iter;
      double press_hydro = 0.;
      for (int m = 0; m < numMatls; m++) {
	press_hydro += rho[m][*iter]* gravity.x()*
	  ((double) (curcell-highIndex).x()*dx.x()- width);
      }
      press_new[*iter] -= press_hydro;
    }
  }
  if (gravity.y() != 0.) {
    // y direction
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	iter++){
      IntVector curcell = *iter;
      double press_hydro = 0.;
      for (int m = 0; m < numMatls; m++) {
	press_hydro += rho[m][*iter]* gravity.y()*
	  ( (double) (curcell-highIndex).y()*dx.y()- height);
      }
      press_new[*iter] -= press_hydro;
    }
  }
  if (gravity.z() != 0.) {
    // z direction
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	iter++){
      IntVector curcell = *iter;
      double press_hydro = 0.;
      for (int m = 0; m < numMatls; m++) {
	press_hydro += rho[m][*iter]* gravity.z()*
	  ((double) (curcell-highIndex).z()*dx.z()- depth);
      }
      press_new[*iter] -= press_hydro;
    }
  }
  
  
  // Store new pressure, speedSound,vol_frac
  for (int m = 0; m < numMatls; m++) {
    ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
    int dwindex = ice_matl->getDWIndex();
    new_dw->put(vol_frac[m],  lb->vol_frac_CCLabel,         dwindex,patch);
    new_dw->put(speedSound[m],lb->speedSound_equiv_CCLabel, dwindex,patch);
    new_dw->put(rho_micro[m], lb->rho_micro_equil_CCLabel,  dwindex,patch);
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      cout << "rho_micro"<<*iter<<"="<<rho_micro[m][*iter] << endl;
    }
  }
  new_dw->put(press_new,lb->press_CCLabel,0,patch);
  
}

void ICE::actuallyStep1c(const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{

  cout << "Doing actually step1c -- compute_face_centered_velocities" << endl;

  int numMatls = d_sharedState->getNumICEMatls();

  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label());
  Vector dx = patch->dCell();
  Vector gravity = d_sharedState->getGravity();

  // Get required variables for this patch
  CCVariable<double> rho_CC, rho_micro_CC;
  CCVariable<double> uvel_CC, vvel_CC, wvel_CC;
  CCVariable<double> press_CC;
  old_dw->get(press_CC,lb->press_CCLabel, 0, patch, Ghost::None, 0);

  // Compute the face centered velocities
  for(int m = 0; m < numMatls; m++){
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial( m );
      int dwindex = ice_matl->getDWIndex();

      old_dw->get(rho_CC,  lb->rho_CCLabel,  dwindex, patch, Ghost::None, 0);
      new_dw->get(rho_micro_CC, lb->rho_micro_equil_CCLabel,
				dwindex, patch, Ghost::None, 0);
      old_dw->get(uvel_CC, lb->uvel_CCLabel,  dwindex, patch, Ghost::None, 0);
      old_dw->get(vvel_CC, lb->vvel_CCLabel,  dwindex, patch, Ghost::None, 0);
      old_dw->get(wvel_CC, lb->wvel_CCLabel,  dwindex, patch, Ghost::None, 0);

      // Create variables for the results
      SFCXVariable<double> uvel_FC;
      SFCYVariable<double> vvel_FC;
      SFCZVariable<double> wvel_FC;
      new_dw->allocate(uvel_FC, lb->uvel_FCLabel, dwindex, patch);
      new_dw->allocate(vvel_FC, lb->vvel_FCLabel, dwindex, patch);
      new_dw->allocate(wvel_FC, lb->wvel_FCLabel, dwindex, patch);

      uvel_FC.initialize(0.);
      vvel_FC.initialize(0.);
      wvel_FC.initialize(0.);
      
      double term1, term2, term3, press_coeff, rho_micro_FC, rho_FC;

#if 1
   // This can't be uncommented until ExtraCells are implemented
      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	  iter++){
	IntVector curcell = *iter;
	cout << "Iterator = " << *iter << endl;

	// Top face
	// Extend the computation into the left and right ExtraCells
	if (curcell.y() < (patch->getCellHighIndex()).y()-1) {
	  IntVector adjcell(curcell.x(),curcell.y()+1,curcell.z()); 
	  cout << "Top face adjacent = " << adjcell << endl;
	  
	  rho_micro_FC = rho_micro_CC[adjcell] + rho_micro_CC[curcell];
	  cout << "rho_micro_CC adjacent = " << rho_micro_CC[adjcell] << 
	    " current = " << rho_micro_CC[curcell] << endl;
	  cout << "Top face rho_micro_FC = " << rho_micro_FC << endl;
	  rho_FC       = rho_CC[adjcell]       + rho_CC[curcell];
	  cout << "Top face rho_FC = " << rho_FC << endl;
	  cout << "vvel_CC adjacent = " << vvel_CC[adjcell] << " current = " 
	       << vvel_CC[curcell] << endl;
	  
	  term1 = (rho_CC[adjcell] * vvel_CC[adjcell] +
		   rho_CC[curcell] * vvel_CC[curcell])/rho_FC;
	  
	  press_coeff = 2.0/(rho_micro_FC);
	  
	  term2 =   delT * press_coeff *
	    (press_CC[adjcell] - press_CC[curcell])/dx.y();
	  term3 =  delT * gravity.y();
	  
	  cout << "Top face term 1 = " << term1 << " term 2 = " << term2 << 
	    " term 3 = " << term3 << endl;
	  
	  // I don't know what this is going to look like yet
	  // but the equations are right I think.
	  //	  uvel_FC[curcell + IntVector(0,1,0)] = 0.0;
	  vvel_FC[curcell + IntVector(0,1,0)] = term1- term2 + term3;
	  //wvel_FC[curcell + IntVector(0,1,0)] = 0.0;

	  cout << "uvel="<< uvel_FC[curcell+IntVector(0,1,0)] << endl;
	  cout << "vvel="<< vvel_FC[curcell+IntVector(0,1,0)] << endl;
	  cout << "wvel="<< wvel_FC[curcell+IntVector(0,1,0)] << endl<<endl;
	}

       // Right face
	if (curcell.x() < (patch->getCellHighIndex()).x()-1) {
	 IntVector adjcell(curcell.x()+1,curcell.y(),curcell.z()); 
	 cout << "Right face adjacent = " << adjcell << endl;
	  
	 rho_micro_FC = rho_micro_CC[adjcell] + rho_micro_CC[curcell];
	 rho_FC       = rho_CC[adjcell]       + rho_CC[curcell];
	 
	 term1 = (rho_CC[adjcell] * uvel_CC[adjcell] +
		  rho_CC[curcell] * uvel_CC[curcell])/rho_FC;
	 
	 press_coeff = 2.0/(rho_micro_FC);
	 
	 term2 =   delT * press_coeff *
	   (press_CC[adjcell] - press_CC[curcell])/dx.x();
	 term3 =  delT * gravity.x();
	 
	 cout << "Right face term 1 = " << term1 << " term 2 = " << term2 << 
	   " term 3 = " << term3 << endl;
	 
	 // I don't know what this is going to look like yet
	 // but the equations are right I think.
	 uvel_FC[curcell + IntVector(1,0,0)] = term1- term2 + term3;
	 //	 vvel_FC[curcell + IntVector(1,0,0)] = 0.0;
	 //wvel_FC[curcell + IntVector(1,0,0)] = 0.0;
	  cout << "uvel="<< uvel_FC[curcell+IntVector(1,0,0)] << endl;
	  cout << "vvel="<< vvel_FC[curcell+IntVector(1,0,0)] << endl;
	  cout << "wvel="<< wvel_FC[curcell+IntVector(1,0,0)] << endl<<endl;
	}

	// Front face
	if (curcell.z() < (patch->getCellHighIndex()).z()-1) {
	  IntVector adjcell(curcell.x(),curcell.y(),curcell.z()+1); 
	  cout << "Front face adjacent = " << adjcell << endl;
	  rho_micro_FC = rho_micro_CC[adjcell] + rho_micro_CC[curcell];
	  rho_FC       = rho_CC[adjcell]       + rho_CC[curcell];
	  
	  term1 = (rho_CC[adjcell] * wvel_CC[adjcell] +
		   rho_CC[curcell] * wvel_CC[curcell])/rho_FC;
	  
	  press_coeff = 2.0/(rho_micro_FC);
	  
	  term2 =   delT * press_coeff *
	    (press_CC[adjcell] - press_CC[curcell])/dx.z();
	  term3 =  delT * gravity.z();
	  
	  cout << "Front face term 1 = " << term1 << " term 2 = " << term2 << 
	    " term 3 = " << term3 << endl;
	  
	  // I don't know what this is going to look like yet
	  // but the equations are right I think.
	  //uvel_FC[curcell + IntVector(0,0,1)] = 0.0;
	  //vvel_FC[curcell + IntVector(0,0,1)] = 0.0;
	  wvel_FC[curcell + IntVector(0,0,1)] = term1- term2 + term3;
	  cout << "uvel="<< uvel_FC[curcell+IntVector(0,0,1)] << endl;
	  cout << "vvel="<< vvel_FC[curcell+IntVector(0,0,1)] << endl;
	  cout << "wvel="<< wvel_FC[curcell+IntVector(0,0,1)] << endl<<endl;
	}
      }
#endif
#if 1
      cout << "Before BC application" << endl << endl;
      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	  iter++) {
	cout << "left face velocity" << *iter << "=" <<uvel_FC[*iter] << endl;
	cout << "right face velocity" << *iter << "=" 
	     <<uvel_FC[*iter+IntVector(1,0,0)] << endl;
	cout << "bottom face velocity" << *iter << "=" <<vvel_FC[*iter] << endl;
	cout << "top face velocity" << *iter << "=" 
	     <<vvel_FC[*iter + IntVector(0,1,0)] << endl;
	cout << "back face velocity" << *iter << "=" <<wvel_FC[*iter] << endl;
	cout << "front face velocity" << *iter << "=" 
	     <<wvel_FC[*iter+IntVector(0,0,1)] << endl << endl;
      }
#endif     
      
      // Put Boundary condition stuff in here
      // Update any neumann boundary conditions
      setBC(uvel_CC,"Velocity","x",patch);
      setBC(vvel_CC,"Velocity","y",patch);
      setBC(wvel_CC,"Velocity","z",patch);
      setBC(uvel_FC,"Velocity","x",patch);
      setBC(vvel_FC,"Velocity","y",patch);
      setBC(wvel_FC,"Velocity","z",patch);


      cout << "Array limits for uvel_FC: " << uvel_FC.getLowIndex() << " " <<
	uvel_FC.getHighIndex() << endl;
      cout << "Array limits for vvel_FC: " << vvel_FC.getLowIndex() << " " <<
	vvel_FC.getHighIndex() << endl;
      cout << "Array limits for wvel_FC: " << wvel_FC.getLowIndex() << " " <<
	wvel_FC.getHighIndex() << endl;
      // Note:  The right face BC is not being applied for some reason
      // Not sure if the limits on the array are correct or not?
#if 1
      cout << "After BC in step 1c application" << endl << endl;
      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	  iter++) {
	cout << "left face velocity" << *iter << "=" <<uvel_FC[*iter] << endl;
	cout << "right face velocity" << *iter << "=" 
	     <<uvel_FC[*iter+IntVector(1,0,0)] << endl;
	cout << "bottom face velocity" << *iter << "=" <<vvel_FC[*iter] << endl;
	cout << "top face velocity" << *iter << "=" 
	     <<vvel_FC[*iter + IntVector(0,1,0)] << endl;
	cout << "back face velocity" << *iter << "=" <<wvel_FC[*iter] << endl;
	cout << "front face velocity" << *iter << "=" 
	     <<wvel_FC[*iter+IntVector(0,0,1)] << endl << endl;
      }
#endif       
      // Put the result in the datawarehouse
      new_dw->put(uvel_FC, lb->uvel_FCLabel, dwindex, patch);
      new_dw->put(vvel_FC, lb->vvel_FCLabel, dwindex, patch);
      new_dw->put(wvel_FC, lb->wvel_FCLabel, dwindex, patch);
  }
}

void ICE::actuallyStep1d(const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{
  cout << "Doing actually step1d -- Add_exchange_contribution_to_FC_vel" << endl;

  int numMatls = d_sharedState->getNumICEMatls();

  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label());
  Vector dx = patch->dCell();
  Vector gravity = d_sharedState->getGravity();

  double temp;

  // Create variables for the required values
  vector<CCVariable<double> > rho_micro_CC(numMatls);
  vector<CCVariable<double> > vol_frac_CC(numMatls);
  vector<SFCXVariable<double> > uvel_FC(numMatls);
  vector<SFCYVariable<double> > vvel_FC(numMatls);
  vector<SFCZVariable<double> > wvel_FC(numMatls);

  // Create variables for the results
  vector<SFCXVariable<double> > uvel_FCME(numMatls);
  vector<SFCYVariable<double> > vvel_FCME(numMatls);
  vector<SFCZVariable<double> > wvel_FCME(numMatls);

  vector<double> b(numMatls);
  DenseMatrix beta(numMatls,numMatls),a(numMatls,numMatls),K(numMatls,numMatls);
  for (int i = 0; i < numMatls; i++ ) {
      K[numMatls-1-i][i] = d_K_mom(i);
  }

  for(int m = 0; m < numMatls; m++){
    ICEMaterial* matl = d_sharedState->getICEMaterial( m );
    int dwindex = matl->getDWIndex();
    new_dw->get(rho_micro_CC[m], lb->rho_micro_equil_CCLabel,
				dwindex, patch, Ghost::None, 0);
    new_dw->get(vol_frac_CC[m],  lb->vol_frac_CCLabel,
				dwindex, patch, Ghost::None, 0);
    new_dw->get(uvel_FC[m], lb->uvel_FCLabel, dwindex, patch, Ghost::None, 0);
    new_dw->get(vvel_FC[m], lb->vvel_FCLabel, dwindex, patch, Ghost::None, 0);
    new_dw->get(wvel_FC[m], lb->wvel_FCLabel, dwindex, patch, Ghost::None, 0);

    new_dw->allocate(uvel_FCME[m], lb->uvel_FCMELabel, dwindex, patch);
    new_dw->allocate(vvel_FCME[m], lb->vvel_FCMELabel, dwindex, patch);
    new_dw->allocate(wvel_FCME[m], lb->wvel_FCMELabel, dwindex, patch);
  }

#if 1
  cout << "At the beginning of step1d" << endl << endl;
  for (int m = 0; m < numMatls; m++) {
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	iter++) {
      cout << "left face velocity" << *iter << "=" <<uvel_FC[m][*iter] << endl;
      cout << "right face velocity" << *iter << "=" 
	   <<uvel_FC[m][*iter+IntVector(1,0,0)] << endl;
      cout << "bottom face velocity" << *iter << "=" <<vvel_FC[m][*iter] << endl;
      cout << "top face velocity" << *iter << "=" 
	   <<vvel_FC[m][*iter + IntVector(0,1,0)] << endl;
      cout << "back face velocity" << *iter << "=" <<wvel_FC[m][*iter] << endl;
      cout << "front face velocity" << *iter << "=" 
	   <<wvel_FC[m][*iter+IntVector(0,0,1)] << endl << endl;
    }
  }
#endif       
  
  for (int m = 0; m < numMatls; m++) {
    uvel_FCME[m] = uvel_FC[m];
    vvel_FCME[m] = vvel_FC[m];
    wvel_FCME[m] = wvel_FC[m];
  }
  
   // This can't be uncommented until ExtraCells are implemented
  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();
      iter++){
    IntVector curcell = *iter;
    cout << "Working on cell " << curcell << endl;

#if 1
   // Top face
    if (curcell.y() < (patch->getCellHighIndex()).y()-1) {
      IntVector adjcell(curcell.x(),curcell.y()+1,curcell.z()); 

      for(int m = 0; m < numMatls; m++){
	for(int n = 0; n < numMatls; n++){
	  temp = (vol_frac_CC[n][adjcell] + vol_frac_CC[n][curcell]) * K[n][m];
	  beta[m][n] = delT * temp/
	    (rho_micro_CC[m][curcell] + rho_micro_CC[m][adjcell]);
	  a[m][n] = -beta[m][n];
	}
      }
      
      for(int m = 0; m < numMatls; m++){
	a[m][m] = 1.;
	for(int n = 0; n < numMatls; n++){
	  a[m][m] +=  beta[m][n];
	}
      }
      
      for(int m = 0; m < numMatls; m++){
	b[m] = 0.0;
	for(int n = 0; n < numMatls; n++){
	  b[m] += beta[m][n] * (vvel_FC[n][*iter] - vvel_FC[m][*iter]);
	}
      }
      
      int itworked = a.solve(b);
      for (int i = 0; i < (int)b.size(); i++) {
	cout << "Top faced b[" << i << "]=" << b[i] << endl;
      }
	
      
      for(int m = 0; m < numMatls; m++){
	vvel_FCME[m][*iter] = vvel_FC[m][*iter] + b[m];
      }
    }

   // Right face
    if (curcell.x() < (patch->getCellHighIndex()).x()-1) {
      IntVector adjcell(curcell.x()+1,curcell.y(),curcell.z()); 
      for(int m = 0; m < numMatls; m++){
	for(int n = 0; n < numMatls; n++){
	  temp = (vol_frac_CC[n][adjcell] + vol_frac_CC[n][curcell]) * K[n][m];
	  beta[m][n] = delT * temp/
	    (rho_micro_CC[m][curcell] + rho_micro_CC[m][adjcell]);
	  a[m][n] = -beta[m][n];
	}
      }
      
      for(int m = 0; m < numMatls; m++){
	a[m][m] = 1.;
	for(int n = 0; n < numMatls; n++){
	  a[m][m] +=  beta[m][n];
	}
      }
      
      for(int m = 0; m < numMatls; m++){
	b[m] = 0.0;
	for(int n = 0; n < numMatls; n++){
	  b[m] += beta[m][n] * (uvel_FC[n][*iter] - uvel_FC[m][*iter]);
	}
      }
      
      int itworked = a.solve(b);
      for (int i = 0; i < (int)b.size(); i++) {
	cout << "Right faced b[" << i << "]=" << b[i] << endl;
      }
	
      
      for(int m = 0; m < numMatls; m++){
	uvel_FCME[m][*iter] = uvel_FC[m][*iter] + b[m];
	cout << "uvel_FC = " << uvel_FC[m][*iter] << " b = " << b[m] <<
	  "uvel_FCME = " << uvel_FCME[m][*iter] << endl;
      }
    }

    // Front face
    if (curcell.z() < (patch->getCellHighIndex()).z()-1) {
      IntVector adjcell(curcell.x(),curcell.y(),curcell.z()+1); 
      for(int m = 0; m < numMatls; m++){
	for(int n = 0; n < numMatls; n++){
	  temp = (vol_frac_CC[n][adjcell] + vol_frac_CC[n][curcell]) * K[n][m];
	  beta[m][n] = delT * temp/
	    (rho_micro_CC[m][curcell] + rho_micro_CC[m][adjcell]);
	  a[m][n] = -beta[m][n];
	}
      }
      
      for(int m = 0; m < numMatls; m++){
	a[m][m] = 1.;
	for(int n = 0; n < numMatls; n++){
	  a[m][m] +=  beta[m][n];
	}
      }
      
      for(int m = 0; m < numMatls; m++){
	b[m] = 0.0;
	for(int n = 0; n < numMatls; n++){
	  b[m] += beta[m][n] * (wvel_FC[n][*iter] - wvel_FC[m][*iter]);
	}
      }
      
      int itworked = a.solve(b);
      for (int i = 0; i < (int)b.size(); i++) {
	cout << "Front faced b[" << i << "]=" << b[i] << endl;
      }      

      for(int m = 0; m < numMatls; m++){
	wvel_FCME[m][*iter] = wvel_FC[m][*iter] + b[m];
      }
    }
#endif
  }


  // Apply grid boundary conditions to the velocity
  // before storing the data
  cout << "Before the BCs in step1d" << endl << endl;
  for (int m = 0; m < numMatls; m++) {
    for (CellIterator iter=patch->getExtraCellIterator(); !iter.done();
	 iter++) {
      cout << "left face velocity" << *iter << "=" <<uvel_FCME[m][*iter] << endl;
      cout << "right face velocity" << *iter << "=" 
	   <<uvel_FCME[m][*iter+IntVector(1,0,0)] << endl;
      cout << "bottom face velocity" << *iter << "=" <<vvel_FCME[m][*iter] << endl;
      cout << "top face velocity" << *iter << "=" 
	   <<vvel_FCME[m][*iter + IntVector(0,1,0)] << endl;
      cout << "back face velocity" << *iter << "=" <<wvel_FCME[m][*iter] << endl;
      cout << "front face velocity" << *iter << "=" 
	   <<wvel_FCME[m][*iter+IntVector(0,0,1)] << endl << endl;
    }
  }

  cout << endl << endl << "Now doing the BC in step1d" << endl << endl;
  for (int m = 0; m < numMatls; m++) {
    setBC(uvel_FCME[m],"Velocity","x",patch);
    setBC(vvel_FCME[m],"Velocity","y",patch);
    setBC(wvel_FCME[m],"Velocity","z",patch);
    for (CellIterator iter=patch->getExtraCellIterator(); !iter.done();
	 iter++) {
      cout << "left face velocity" << *iter << "=" <<uvel_FCME[m][*iter] << endl;
      cout << "right face velocity" << *iter << "=" 
	   <<uvel_FCME[m][*iter+IntVector(1,0,0)] << endl;
      cout << "bottom face velocity" << *iter << "=" <<vvel_FCME[m][*iter] << endl;
      cout << "top face velocity" << *iter << "=" 
	   <<vvel_FCME[m][*iter + IntVector(0,1,0)] << endl;
      cout << "back face velocity" << *iter << "=" <<wvel_FCME[m][*iter] << endl;
      cout << "front face velocity" << *iter << "=" 
	   <<wvel_FCME[m][*iter+IntVector(0,0,1)] << endl << endl;
    }
  }

  
  // Put the result in the datawarehouse
  for(int m = 0; m < numMatls; m++){
    ICEMaterial* matl = d_sharedState->getICEMaterial( m );
    int dwindex = matl->getDWIndex();
    new_dw->put(uvel_FCME[m], lb->uvel_FCMELabel, dwindex, patch);
    new_dw->put(vvel_FCME[m], lb->vvel_FCMELabel, dwindex, patch);
    new_dw->put(wvel_FCME[m], lb->wvel_FCMELabel, dwindex, patch);
  }
}

void ICE::actuallyStep2(const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{
  cout << "Doing actually step2 -- divergence_of_face_centered_velocity_MM" << endl;

  int numMatls = d_sharedState->getNumICEMatls();
  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label());
  Vector dx = patch->dCell();
  double top, bottom, right, left, front, back;
  double vol = dx.x()*dx.y()*dx.z();
  double invvol = 1./vol;

  // Allocate the temporary variables needed for advection
  // These arrays get re-used for each material
  CCVariable<double> q_CC, q_advected;
  CCVariable<fflux> IFS,OFS,q_out,q_in;
  CCVariable<eflux> IFE,OFE,q_out_EF,q_in_EF;

  new_dw->allocate(q_CC,       lb->q_CCLabel,       0, patch);
  new_dw->allocate(q_advected, lb->q_advectedLabel, 0, patch);
  new_dw->allocate(IFS,        IFS_CCLabel,         0, patch);
  new_dw->allocate(OFS,        OFS_CCLabel,         0, patch);
  new_dw->allocate(IFE,        IFE_CCLabel,         0, patch);
  new_dw->allocate(OFE,        OFE_CCLabel,         0, patch);
  new_dw->allocate(q_out,      q_outLabel,          0, patch);
  new_dw->allocate(q_out_EF,   q_out_EFLabel,       0, patch);
  new_dw->allocate(q_in,       q_inLabel,           0, patch);
  new_dw->allocate(q_in_EF,    q_in_EFLabel,        0, patch);

  CCVariable<double> term1, term2, term3;
  new_dw->allocate(term1, lb->term3Label, 0, patch);
  new_dw->allocate(term2, lb->term3Label, 0, patch);
  new_dw->allocate(term3, lb->term3Label, 0, patch);

  term1.initialize(0.);
  term2.initialize(0.);
  term3.initialize(0.);
  
  // Compute the divergence of the face centered velocities
  for(int m = 0; m < numMatls; m++){
    ICEMaterial* matl = d_sharedState->getICEMaterial( m );
      int dwindex = matl->getDWIndex();
      // Get required variables for this patch
      SFCXVariable<double> uvel_FC;
      SFCYVariable<double> vvel_FC;
      SFCZVariable<double> wvel_FC;
      CCVariable<double> vol_frac;
      CCVariable<double> rho_micro_CC;
      CCVariable<double> speedSound;
      new_dw->get(uvel_FC, lb->uvel_FCMELabel,  dwindex, patch, Ghost::None, 0);
      new_dw->get(vvel_FC, lb->vvel_FCMELabel,  dwindex, patch, Ghost::None, 0);
      new_dw->get(wvel_FC, lb->wvel_FCMELabel,  dwindex, patch, Ghost::None, 0);
      new_dw->get(vol_frac,lb->vol_frac_CCLabel,dwindex,patch,Ghost::None, 0);
      new_dw->get(rho_micro_CC, lb->rho_micro_equil_CCLabel,
						 dwindex,patch,Ghost::None, 0);
      new_dw->get(speedSound,lb->speedSound_equiv_CCLabel,
					         dwindex,patch,Ghost::None, 0);

      // Create variables for the divergence of the FC velocity
      CCVariable<double> div_velfc_CC;
      new_dw->allocate(div_velfc_CC, lb->div_velfc_CCLabel, dwindex, patch);
      cout << "dx = " << dx << endl;
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
	cout << "top face velocity = " << vvel_FC[*iter+IntVector(0,1,0)] 
	     << endl;
	cout << "bottom face velocity = " << vvel_FC[*iter+IntVector(0,0,0)] 
	     << endl;
	cout << "left face velocity = " << uvel_FC[*iter+IntVector(1,0,0)] 
	     << endl;
	cout << "right face velocity = " << uvel_FC[*iter+IntVector(0,0,0)] 
	     << endl;
	cout << "front face velocity = " << wvel_FC[*iter+IntVector(0,0,1)] 
	     << endl;
	cout << "back face velocity = " << wvel_FC[*iter+IntVector(0,0,0)] 
	     << endl;
	top      =  dx.x()*dx.z()* vvel_FC[*iter+IntVector(0,1,0)];
	bottom   = -dx.x()*dx.z()* vvel_FC[*iter+IntVector(0,0,0)];
	left     = -dx.y()*dx.z()* uvel_FC[*iter+IntVector(0,0,0)];
	right    =  dx.y()*dx.z()* uvel_FC[*iter+IntVector(1,0,0)];
	front    =  dx.x()*dx.y()* wvel_FC[*iter+IntVector(0,0,1)];
	back     = -dx.x()*dx.y()* wvel_FC[*iter+IntVector(0,0,0)];
	cout << "top = " << top << " bottom = " << bottom << " left = " 
	     << left << " right = " << right << " front = " << front 
	     << " back = " << back << " vol_frac = " << vol_frac[*iter] << 
	  endl;
	  
	div_velfc_CC[*iter] = vol_frac[*iter]*
			     (top + bottom + left + right + front  + back );
      }

      // Advection preprocessing
      //influx_outflux_volume
      influxOutfluxVolume(uvel_FC,vvel_FC,wvel_FC,delT,patch,OFS,OFE,IFS,IFE);

      { // Compute Advection of the volume fraction
        for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();
	    iter++){
          q_CC[*iter] = vol_frac[*iter] * invvol;
        }

        advectQFirst(q_CC,patch,OFS,OFE,IFS,IFE,q_out,q_out_EF,q_in,q_in_EF,
                     q_advected);
      }

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); 
	  iter++){
	term1[*iter] = 0.;
	term2[*iter] -= q_advected[*iter];
	term3[*iter] += vol_frac[*iter] /(rho_micro_CC[*iter] *
					  speedSound[*iter]*speedSound[*iter]);
	cout << "term1 = " << term1[*iter] << " term2 = " << term2[*iter] << " term3 = " << term3[*iter] << endl;
      }

      new_dw->put(div_velfc_CC, lb->div_velfc_CCLabel, dwindex, patch);
  }

  // Compute delPress and the new pressure
  // Create variables for the required variables and the results
  CCVariable<double> pressure;
  CCVariable<double> delPress;
  CCVariable<double> pressdP;
  new_dw->get(pressure,          lb->press_CCLabel,    0, patch,Ghost::None, 0);
  new_dw->allocate(delPress,     lb->delPress_CCLabel, 0, patch);
  new_dw->allocate(pressdP,      lb->pressdP_CCLabel,  0, patch);

  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
    delPress[*iter] = (delT * term1[*iter] - term2[*iter])/(term3[*iter]);
    pressdP[*iter]  = pressure[*iter] + delPress[*iter];
    cout << "delPress = " << delPress[*iter] << " pressdP = " 
	 << pressdP[*iter] << endl;
  }


  // Update the pressure BC

  setBC(pressdP,"Pressure",patch);

  new_dw->put(delPress, lb->delPress_CCLabel, 0, patch);
  new_dw->put(pressdP,  lb->pressdP_CCLabel,  0, patch);

}

void ICE::actuallyStep3(const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{
  cout << "Doing actually step3 -- press_face_MM" << endl;

  int numMatls = d_sharedState->getNumICEMatls();

  double sum_rho, sum_rho_adj, sum_all_rho;

  // Get required variables for this patch
  vector<CCVariable<double> > rho_CC(numMatls);
  CCVariable<double> press_CC;
  new_dw->get(press_CC,lb->pressdP_CCLabel, 0, patch, Ghost::None, 0);


  // Create variables for the results
  SFCXVariable<double> pressX_FC;
  SFCYVariable<double> pressY_FC;
  SFCZVariable<double> pressZ_FC;
  new_dw->allocate(pressX_FC,lb->pressX_FCLabel, 0, patch);
  new_dw->allocate(pressY_FC,lb->pressY_FCLabel, 0, patch);
  new_dw->allocate(pressZ_FC,lb->pressZ_FCLabel, 0, patch);

  // Compute the face centered velocities
  for(int m = 0; m < numMatls; m++){
    ICEMaterial* matl = d_sharedState->getICEMaterial( m );
    int dwindex = matl->getDWIndex();
    old_dw->get(rho_CC[m], lb->rho_CCLabel, dwindex, patch, Ghost::None, 0);
  }

#if 1
   // This can't be uncommented until ExtraCells are implemented
  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
    IntVector curcell = *iter;

   // Top face
    IntVector adjcell(curcell.x(),curcell.y()+1,curcell.z());

    sum_rho=0.0;
    sum_rho_adj  = 0.0;
    for(int m = 0; m < numMatls; m++){
	sum_rho      += (rho_CC[m][curcell] + d_SMALL_NUM);
	sum_rho_adj  += (rho_CC[m][adjcell] + d_SMALL_NUM);
    }
    sum_all_rho  = sum_rho     +  sum_rho_adj;

    pressX_FC[curcell+IntVector(0,1,0)]      =
             (press_CC[curcell] * sum_rho
           +  press_CC[adjcell] * sum_rho_adj)/sum_all_rho;


   // Right face
    adjcell = IntVector(curcell.x()+1,curcell.y(),curcell.z());

    sum_rho=0.0;
    sum_rho_adj  = 0.0;
    for(int m = 0; m < numMatls; m++){
	sum_rho      += (rho_CC[m][curcell] + d_SMALL_NUM);
	sum_rho_adj  += (rho_CC[m][adjcell] + d_SMALL_NUM);
    }
    sum_all_rho  = sum_rho     +  sum_rho_adj;

    pressY_FC[curcell+IntVector(1,0,0)]    =
             (press_CC[curcell] * sum_rho
           +  press_CC[adjcell] * sum_rho_adj)/sum_all_rho;


   // Front face
    adjcell = IntVector(curcell.x(),curcell.y(),curcell.z()+1);

    sum_rho=0.0;
    sum_rho_adj  = 0.0;
    for(int m = 0; m < numMatls; m++){
	sum_rho      += (rho_CC[m][curcell] + d_SMALL_NUM);
	sum_rho_adj  += (rho_CC[m][adjcell] + d_SMALL_NUM);
    }
    sum_all_rho  = sum_rho     +  sum_rho_adj;

    pressZ_FC[curcell+IntVector(0,0,1)]    =
             (press_CC[curcell] * sum_rho
           +  press_CC[adjcell] * sum_rho_adj)/sum_all_rho;

  }
#endif

    /*__________________________________
    * Update the boundary conditions
        update_CC_FC_physical_boundary_conditions(
    *___________________________________*/

    // Update the pressure BC
  setBC(pressX_FC,"Pressure",patch);
  setBC(pressY_FC,"Pressure",patch);
  setBC(pressZ_FC,"Pressure",patch);

  new_dw->put(pressX_FC,lb->pressX_FCLabel, 0, patch);
  new_dw->put(pressY_FC,lb->pressY_FCLabel, 0, patch);
  new_dw->put(pressZ_FC,lb->pressZ_FCLabel, 0, patch);
}


void ICE::actuallyStep4a(const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{
  cout << "Doing actually step4a -- accumulate_momentum_source_sinks_MM" << endl;

  int numMatls = d_sharedState->getNumICEMatls();
  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label());
  Vector dx = patch->dCell();
  Vector gravity = d_sharedState->getGravity();
  double pressure_source, viscous_source, mass, vol=dx.x()*dx.y()*dx.z();

  CCVariable<double> rho_CC;
  CCVariable<double> uvel_CC, vvel_CC, wvel_CC;
  CCVariable<double> visc_CC;
  CCVariable<double> vol_frac;
  SFCXVariable<double> pressX_FC;
  SFCYVariable<double> pressY_FC;
  SFCZVariable<double> pressZ_FC;

  CCVariable<double> xmom_source, ymom_source, zmom_source;
  SFCXVariable<double> tau_X_FC;
  SFCYVariable<double> tau_Y_FC;
  SFCZVariable<double> tau_Z_FC;

  new_dw->get(pressX_FC,lb->pressX_FCLabel, 0, patch,Ghost::None, 0);
  new_dw->get(pressY_FC,lb->pressY_FCLabel, 0, patch,Ghost::None, 0);
  new_dw->get(pressZ_FC,lb->pressZ_FCLabel, 0, patch,Ghost::None, 0);

  for(int m = 0; m < numMatls; m++){
    ICEMaterial* matl = d_sharedState->getICEMaterial( m );
    int dwindex = matl->getDWIndex();
    // Get required variables for this patch
    old_dw->get(rho_CC,  lb->rho_CCLabel,      dwindex,patch,Ghost::None, 0);
    old_dw->get(uvel_CC, lb->uvel_CCLabel,     dwindex,patch,Ghost::None, 0);
    old_dw->get(vvel_CC, lb->vvel_CCLabel,     dwindex,patch,Ghost::None, 0);
    old_dw->get(wvel_CC, lb->wvel_CCLabel,     dwindex,patch,Ghost::None, 0);
    old_dw->get(visc_CC, lb->viscosity_CCLabel,dwindex,patch,Ghost::None, 0);
    new_dw->get(vol_frac,lb->vol_frac_CCLabel, dwindex,patch,Ghost::None, 0);

    // Create variables for the results
    new_dw->allocate(xmom_source, lb->xmom_source_CCLabel, dwindex, patch);
    new_dw->allocate(ymom_source, lb->ymom_source_CCLabel, dwindex, patch);
    new_dw->allocate(zmom_source, lb->zmom_source_CCLabel, dwindex, patch);
    new_dw->allocate(tau_X_FC,    lb->tau_X_FCLabel,       dwindex, patch);
    new_dw->allocate(tau_Y_FC,    lb->tau_Y_FCLabel,       dwindex, patch);
    new_dw->allocate(tau_Z_FC,    lb->tau_Z_FCLabel,       dwindex, patch);

    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
	 mass = rho_CC[*iter] * vol;
         // x-momentum -- needs checking - jas
         pressure_source = pressX_FC[*iter+IntVector(1,0,0)] - 
	   pressX_FC[*iter+IntVector(-1,0,0)];
         viscous_source  = tau_X_FC[*iter+IntVector(1,0,0)] - 
	   tau_X_FC[*iter+IntVector(0,0,0)] + 
	   tau_X_FC[*iter+IntVector(0,1,0)]  - 
	   tau_X_FC[*iter+IntVector(0,0,0)] + 
	   tau_X_FC[*iter+IntVector(0,0,1)] - 
	   tau_X_FC[*iter+IntVector(0,0,0)];
	   xmom_source[*iter]  =   (-pressure_source * dx.y() * dx.z() +
                                   mass * gravity.x()) * delT;
    }

    new_dw->put(xmom_source, lb->xmom_source_CCLabel, dwindex, patch);
    new_dw->put(ymom_source, lb->ymom_source_CCLabel, dwindex, patch);
    new_dw->put(zmom_source, lb->zmom_source_CCLabel, dwindex, patch);
  }
}

void ICE::actuallyStep4b(const ProcessorGroup*,
                   const Patch* patch,
                   DataWarehouseP& old_dw,
                   DataWarehouseP& new_dw)
{
  cout << "Doing actually step4b -- accumulate_energy_source_sinks" << endl;

  int numMatls = d_sharedState->getNumICEMatls();
  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label());
  Vector dx = patch->dCell();
  double A, B, vol=dx.x()*dx.y()*dx.z();

  CCVariable<double> rho_micro_CC;
  CCVariable<double> speedSound;
  CCVariable<double> vol_frac;
  CCVariable<double> press_CC;
  CCVariable<double> delPress;
  CCVariable<double> int_eng_source;

  new_dw->get(press_CC,lb->press_CCLabel,    0, patch,Ghost::None, 0);
  new_dw->get(delPress,lb->delPress_CCLabel, 0, patch,Ghost::None, 0);

  for(int m = 0; m < numMatls; m++){
    ICEMaterial* matl = d_sharedState->getICEMaterial( m );
    int dwindex = matl->getDWIndex();
    // Get required variables for this patch
    new_dw->get(rho_micro_CC,lb->rho_micro_equil_CCLabel,
						dwindex,patch,Ghost::None,0);
    new_dw->get(speedSound, lb->speedSound_equiv_CCLabel,
						dwindex,patch,Ghost::None, 0);
    new_dw->get(vol_frac,lb->vol_frac_CCLabel,dwindex,patch,Ghost::None, 0);

    // Create variables for the results
    new_dw->allocate(int_eng_source,lb->int_eng_source_CCLabel,dwindex,patch);

    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	  iter++){
	A = vol * vol_frac[*iter] * press_CC[*iter];
        B = rho_micro_CC[*iter] * speedSound[*iter]*speedSound[*iter];
        int_eng_source[*iter] = (A/B) * delPress[*iter];
    }

    new_dw->put(int_eng_source,lb->int_eng_source_CCLabel,dwindex,patch);
  }
}

void ICE::actuallyStep5a(const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{
  cout << "Doing actually step5a -- lagrangian_vol_MM" << endl;

  int numMatls = d_sharedState->getNumICEMatls();
  Vector dx = patch->dCell();

  // Compute the Lagrangian quantities
  for(int m = 0; m < numMatls; m++){
    ICEMaterial* matl = d_sharedState->getICEMaterial( m );
    int dwindex = matl->getDWIndex();
    // Get required variables for this patch
    CCVariable<double> rho_CC;
    CCVariable<double> uvel_CC;
    CCVariable<double> vvel_CC;
    CCVariable<double> wvel_CC;
    CCVariable<double> cv_CC;
    CCVariable<double> temp_CC;
    CCVariable<double> xmom_source;
    CCVariable<double> ymom_source;
    CCVariable<double> zmom_source;
    CCVariable<double> int_eng_source;
    old_dw->get(rho_CC,  lb->rho_CCLabel,     dwindex,patch,Ghost::None, 0);
    old_dw->get(uvel_CC, lb->uvel_CCLabel,    dwindex,patch,Ghost::None, 0);
    old_dw->get(vvel_CC, lb->vvel_CCLabel,    dwindex,patch,Ghost::None, 0);
    old_dw->get(wvel_CC, lb->wvel_CCLabel,    dwindex,patch,Ghost::None, 0);
    old_dw->get(cv_CC,   lb->cv_CCLabel,      dwindex,patch,Ghost::None, 0);
    old_dw->get(temp_CC, lb->temp_CCLabel,    dwindex,patch,Ghost::None, 0);
    new_dw->get(xmom_source,    lb->xmom_source_CCLabel,
						dwindex,patch,Ghost::None, 0);
    new_dw->get(ymom_source,    lb->ymom_source_CCLabel,
						dwindex,patch,Ghost::None, 0);
    new_dw->get(zmom_source,    lb->zmom_source_CCLabel,
						dwindex,patch,Ghost::None, 0);
    new_dw->get(int_eng_source, lb->int_eng_source_CCLabel,
						dwindex,patch,Ghost::None, 0);

    // Create variables for the results
    CCVariable<double> xmom_L;
    CCVariable<double> ymom_L;
    CCVariable<double> zmom_L;
    CCVariable<double> int_eng_L;
    CCVariable<double> mass_L;
    CCVariable<double> rho_L;
    new_dw->allocate(xmom_L,    lb->xmom_L_CCLabel,    dwindex,patch);
    new_dw->allocate(ymom_L,    lb->ymom_L_CCLabel,    dwindex,patch);
    new_dw->allocate(zmom_L,    lb->zmom_L_CCLabel,    dwindex,patch);
    new_dw->allocate(int_eng_L, lb->int_eng_L_CCLabel, dwindex,patch);
    new_dw->allocate(mass_L,    lb->mass_L_CCLabel,    dwindex,patch);
    new_dw->allocate(rho_L,     lb->rho_L_CCLabel,     dwindex,patch);

    double vol = dx.x()*dx.y()*dx.z();
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	  iter++){
	double   mass = rho_CC[*iter] * vol;
	mass_L[*iter] = mass; // +  mass_source[*iter];
	rho_L[*iter]  = mass_L[*iter]/vol;
	xmom_L[*iter] = mass * uvel_CC[*iter]
		//- uvel_CC[*iter] * mass_source[*iter]
		+ xmom_source[*iter];
	ymom_L[*iter] = mass * vvel_CC[*iter]
		//- vvel_CC[*iter] * mass_source[*iter]
		+ ymom_source[*iter];
	zmom_L[*iter] = mass * wvel_CC[*iter]
		//- wvel_CC[*iter] * mass_source[*iter]
		+ zmom_source[*iter];
	int_eng_L[*iter] = mass * cv_CC[*iter] * temp_CC[*iter]
		//-cv_CC[*iter] * temp_CC * mass_source[*iter]
		+ int_eng_source[*iter];
    }

    new_dw->put(xmom_L,    lb->xmom_L_CCLabel,    dwindex,patch);
    new_dw->put(ymom_L,    lb->ymom_L_CCLabel,    dwindex,patch);
    new_dw->put(zmom_L,    lb->zmom_L_CCLabel,    dwindex,patch);
    new_dw->put(int_eng_L, lb->int_eng_L_CCLabel, dwindex,patch);
    new_dw->put(mass_L,    lb->mass_L_CCLabel,    dwindex,patch);
  }
}

void ICE::actuallyStep5b(const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{
  cout << "Doing actually step5b -- calc_flux_or_primitive_vars" << endl;

  int numMatls = d_sharedState->getNumICEMatls();

  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label());
  Vector dx = patch->dCell();
  Vector gravity = d_sharedState->getGravity();

  double temp;
  int itworked;

  // Create variables for the required values
  vector<CCVariable<double> > rho_CC(numMatls);
  vector<CCVariable<double> > xmom_L(numMatls);
  vector<CCVariable<double> > ymom_L(numMatls);
  vector<CCVariable<double> > zmom_L(numMatls);
  vector<CCVariable<double> > int_eng_L(numMatls);
  vector<CCVariable<double> > vol_frac_CC(numMatls);
  vector<CCVariable<double> > rho_micro_CC(numMatls);
  vector<CCVariable<double> > cv_CC(numMatls);

  // Create variables for the results
  vector<CCVariable<double> > xmom_L_ME(numMatls);
  vector<CCVariable<double> > ymom_L_ME(numMatls);
  vector<CCVariable<double> > zmom_L_ME(numMatls);
  vector<CCVariable<double> > int_eng_L_ME(numMatls);

  vector<double> b(numMatls);
  vector<double> mass(numMatls);
  DenseMatrix beta(numMatls,numMatls),acopy(numMatls,numMatls);
  DenseMatrix K(numMatls,numMatls),H(numMatls,numMatls),a(numMatls,numMatls);

  for(int m = 0; m < numMatls; m++){
    ICEMaterial* matl = d_sharedState->getICEMaterial( m );
    int dwindex = matl->getDWIndex();
    old_dw->get(rho_CC[m],       lb->rho_CCLabel,
				dwindex, patch, Ghost::None, 0);
    new_dw->get(xmom_L[m],       lb->xmom_L_CCLabel,
				dwindex, patch, Ghost::None, 0);
    new_dw->get(ymom_L[m],       lb->ymom_L_CCLabel,
				dwindex, patch, Ghost::None, 0);
    new_dw->get(zmom_L[m],       lb->zmom_L_CCLabel,
				dwindex, patch, Ghost::None, 0);
    new_dw->get(int_eng_L[m],    lb->int_eng_L_CCLabel,
				dwindex, patch, Ghost::None, 0);
    new_dw->get(vol_frac_CC[m],  lb->vol_frac_CCLabel,
				dwindex, patch, Ghost::None, 0);
    new_dw->get(rho_micro_CC[m], lb->rho_micro_equil_CCLabel,
				dwindex, patch, Ghost::None, 0);
    old_dw->get(cv_CC[m],        lb->cv_CCLabel,
				dwindex, patch, Ghost::None, 0);

    new_dw->allocate(xmom_L_ME[m],   lb->xmom_L_ME_CCLabel,    dwindex, patch);
    new_dw->allocate(ymom_L_ME[m],   lb->ymom_L_ME_CCLabel,    dwindex, patch);
    new_dw->allocate(zmom_L_ME[m],   lb->zmom_L_ME_CCLabel,    dwindex, patch);
    new_dw->allocate(int_eng_L_ME[m],lb->int_eng_L_ME_CCLabel, dwindex, patch);
  }

  double vol = dx.x()*dx.y()*dx.z();
  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
    // Do Momentum Exchange here
    for(int m = 0; m < numMatls; m++){
      temp = rho_micro_CC[m][*iter];
      mass[m] = rho_CC[m][*iter] * vol;
      for(int n = 0; n < numMatls; n++){
	beta[m][n] = delT * vol_frac_CC[n][*iter] * K[n][m]/temp;
	a[m][n] = -beta[m][n];
      }
    }

    for(int m = 0; m < numMatls; m++){
      a[m][m] = 1.;
      for(int n = 0; n < numMatls; n++){
	a[m][m] +=  beta[m][n];
      }
    }

    // x-momentum
    for(int m = 0; m < numMatls; m++){
      b[m] = 0.0;
      for(int n = 0; n < numMatls; n++){
	b[m] += beta[m][n] *
		(xmom_L[n][*iter]/mass[n] - xmom_L[m][*iter]/mass[m]);
      }
    }

   acopy = a;

   itworked = acopy.solve(b);

    for(int m = 0; m < numMatls; m++){
      xmom_L_ME[m][*iter] = xmom_L[m][*iter] + b[m]*mass[m];
    }

    // y-momentum
    for(int m = 0; m < numMatls; m++){
      b[m] = 0.0;
      for(int n = 0; n < numMatls; n++){
	b[m] += beta[m][n] *
		(ymom_L[n][*iter]/mass[n] - ymom_L[m][*iter]/mass[m]);
      }
    }

   acopy = a;

   itworked = acopy.solve(b);

    for(int m = 0; m < numMatls; m++){
      ymom_L_ME[m][*iter] = ymom_L[m][*iter] + b[m]*mass[m];
    }

    // z-momentum
    for(int m = 0; m < numMatls; m++){
      b[m] = 0.0;
      for(int n = 0; n < numMatls; n++){
	b[m] += beta[m][n] *
		(zmom_L[n][*iter]/mass[n] - zmom_L[m][*iter]/mass[m]);
      }
    }

   acopy = a;

   itworked = acopy.solve(b);

    for(int m = 0; m < numMatls; m++){
      zmom_L_ME[m][*iter] = zmom_L[m][*iter] + b[m]*mass[m];
    }

    // Do Energy Exchange here
    for(int m = 0; m < numMatls; m++){
      temp = cv_CC[m][*iter]*rho_micro_CC[m][*iter];
      for(int n = 0; n < numMatls; n++){
	beta[m][n] = delT * vol_frac_CC[n][*iter] * H[n][m]/temp;
	a[m][n] = -beta[m][n];
      }
    }

    for(int m = 0; m < numMatls; m++){
      a[m][m] = 1.;
      for(int n = 0; n < numMatls; n++){
	a[m][m] +=  beta[m][n];
      }
    }

    for(int m = 0; m < numMatls; m++){
      b[m] = 0.0;
      for(int n = 0; n < numMatls; n++){
	b[m] += beta[m][n] *
		(int_eng_L[n][*iter]/(mass[n]*cv_CC[n][*iter]) -
		 int_eng_L[m][*iter]/(mass[m]*cv_CC[m][*iter]));
      }
    }

   itworked = a.solve(b);

    for(int m = 0; m < numMatls; m++){
      int_eng_L_ME[m][*iter] =
		int_eng_L[m][*iter] + b[m]*mass[m]*cv_CC[m][*iter];
    }

  }

  // Update any neumann boundary conditions
  // Update the velocity BC

  for (int m = 0; m < numMatls; m++) {
    setBC(xmom_L_ME[m],"Velocity",patch);
    setBC(ymom_L_ME[m],"Velocity",patch);
    setBC(zmom_L_ME[m],"Velocity",patch);
  }
  

  for(int m = 0; m < numMatls; m++){
     ICEMaterial* matl = d_sharedState->getICEMaterial( m );
     int dwindex = matl->getDWIndex();
     new_dw->put(xmom_L_ME[m],   lb->xmom_L_ME_CCLabel,   dwindex, patch);
     new_dw->put(ymom_L_ME[m],   lb->ymom_L_ME_CCLabel,   dwindex, patch);
     new_dw->put(zmom_L_ME[m],   lb->zmom_L_ME_CCLabel,   dwindex, patch);
     new_dw->put(int_eng_L_ME[m],lb->int_eng_L_ME_CCLabel,dwindex, patch);
  }

}

void ICE::actuallyStep6and7(const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{

  cout << "Doing actually step6 and 7" << endl;
  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label());

  double dT = 0.0001;
  new_dw->put(delt_vartype(dT), lb->delTLabel);
  Vector dx = patch->dCell();
  double vol = dx.x()*dx.y()*dx.z(),mass;
  double invvol = 1.0/vol;

  CCVariable<double> uvel_CC,vvel_CC,wvel_CC,rho_CC,visc_CC,cv,temp;
  CCVariable<double> xmom_L_ME,ymom_L_ME,zmom_L_ME,int_eng_L_ME,mass_L;

  SFCXVariable<double> uvel_FC;
  SFCYVariable<double> vvel_FC;
  SFCZVariable<double> wvel_FC;

  // Allocate the temporary variables needed for advection
  // These arrays get re-used for each material, and for each
  // advected quantity
  CCVariable<double> q_CC, q_advected;
  CCVariable<fflux> IFS,OFS,q_out,q_in;
  CCVariable<eflux> IFE,OFE,q_out_EF,q_in_EF;

  new_dw->allocate(q_CC,       lb->q_CCLabel,       0, patch);
  new_dw->allocate(q_advected, lb->q_advectedLabel, 0, patch);
  new_dw->allocate(IFS,        IFS_CCLabel,         0, patch);
  new_dw->allocate(OFS,        OFS_CCLabel,         0, patch);
  new_dw->allocate(IFE,        IFE_CCLabel,         0, patch);
  new_dw->allocate(OFE,        OFE_CCLabel,         0, patch);
  new_dw->allocate(q_out,      q_outLabel,          0, patch);
  new_dw->allocate(q_out_EF,   q_out_EFLabel,       0, patch);
  new_dw->allocate(q_in,       q_inLabel,           0, patch);
  new_dw->allocate(q_in_EF,    q_in_EFLabel,        0, patch);

  for (int m = 0; m < d_sharedState->getNumICEMatls(); m++ ) {
    ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
    int dwindex = ice_matl->getDWIndex();

    new_dw->get(uvel_FC,lb->uvel_FCMELabel,   dwindex, patch,Ghost::None, 0);
    new_dw->get(vvel_FC,lb->vvel_FCMELabel,   dwindex, patch,Ghost::None, 0);
    new_dw->get(wvel_FC,lb->wvel_FCMELabel,   dwindex, patch,Ghost::None, 0);
    new_dw->get(xmom_L_ME,lb->xmom_L_ME_CCLabel,dwindex,patch,Ghost::None, 0);
    new_dw->get(ymom_L_ME,lb->ymom_L_ME_CCLabel,dwindex, patch,Ghost::None, 0);
    new_dw->get(zmom_L_ME,lb->zmom_L_ME_CCLabel,dwindex, patch,Ghost::None, 0);
    new_dw->get(mass_L,lb->mass_L_CCLabel,   dwindex, patch, Ghost::None, 0);
    new_dw->get(int_eng_L_ME,lb->int_eng_L_ME_CCLabel,
		dwindex, patch, Ghost::None, 0);

    new_dw->allocate(rho_CC, lb->rho_CCLabel,        dwindex,patch);
    new_dw->allocate(temp,   lb->temp_CCLabel,       dwindex,patch);
    new_dw->allocate(cv,     lb->cv_CCLabel,         dwindex,patch);
    new_dw->allocate(uvel_CC,lb->uvel_CCLabel,       dwindex,patch);
    new_dw->allocate(vvel_CC,lb->vvel_CCLabel,       dwindex,patch);
    new_dw->allocate(wvel_CC,lb->wvel_CCLabel,       dwindex,patch);
    new_dw->allocate(visc_CC,lb->viscosity_CCLabel,  dwindex,patch);

    // Advection preprocessing
    //influx_outflux_volume
    influxOutfluxVolume(uvel_FC,vvel_FC,wvel_FC,delT,patch,OFS,OFE,IFS,IFE);

    // outflowVolCentroid goes here if doing second order
    //outflowVolCentroid(uvel_FC,vvel_FC,wvel_FC,delT,dx,
    //		 r_out_x, r_out_y, r_out_z,
    //		 r_out_x_CF, r_out_y_CF, r_out_z_CF);

    { // Advection of the mass (density)
      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	    iter++){
	  q_CC[*iter] = mass_L[*iter] * invvol;
      }

      advectQFirst(q_CC,patch,OFS,OFE,IFS,IFE,q_out,q_out_EF,q_in,q_in_EF,
		     q_advected);

      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	  iter++){
	  rho_CC[*iter] = (mass_L[*iter] + q_advected[*iter])*invvol;
      }
    }


    { // Advection of the x-momentum
      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	  iter++){
	  q_CC[*iter] = xmom_L_ME[*iter] * invvol;
      }

      advectQFirst(q_CC,patch,OFS,OFE,IFS,IFE,q_out,q_out_EF,q_in,q_in_EF,
		     q_advected);

      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	  iter++){
	  mass = rho_CC[*iter] * vol;
	  uvel_CC[*iter] = (xmom_L_ME[*iter] + q_advected[*iter])/mass;
      }
    }



    { // Advection of the y-momentum
      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	  iter++){
	  q_CC[*iter] = ymom_L_ME[*iter] * invvol;
      }

      advectQFirst(q_CC,patch,OFS,OFE,IFS,IFE,q_out,q_out_EF,q_in,q_in_EF,
		     q_advected);

      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	  iter++){
	  mass = rho_CC[*iter] * vol;
	  vvel_CC[*iter] = (ymom_L_ME[*iter] + q_advected[*iter])/mass;
      }
    }

    { // Advection of the z-momentum
      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	  iter++){
	  q_CC[*iter] = zmom_L_ME[*iter] * invvol;
      }

      advectQFirst(q_CC,patch,OFS,OFE,IFS,IFE,q_out,q_out_EF,q_in,q_in_EF,
		     q_advected);

      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	  iter++){
	  mass = rho_CC[*iter] * vol;
	  wvel_CC[*iter] = (zmom_L_ME[*iter] + q_advected[*iter])/mass;
      }
    }

    { // Advection of the internal energy
      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	  iter++){
	  q_CC[*iter] = int_eng_L_ME[*iter] * invvol;
      }

      advectQFirst(q_CC,patch,OFS,OFE,IFS,IFE,q_out,q_out_EF,q_in,q_in_EF,
		     q_advected);

      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
	  iter++){
	  mass = rho_CC[*iter] * vol;
	  temp[*iter] = (int_eng_L_ME[*iter] + q_advected[*iter])/
							(mass*cv[*iter]);
      }
    }

    // Update the BCs
    setBC(rho_CC,"Density",patch);
    setBC(temp,"Temperature",patch);
    setBC(uvel_CC,"Velocity","x",patch);
    setBC(vvel_CC,"Velocity","y",patch);
    setBC(wvel_CC,"Velocity","z",patch);

    new_dw->put(rho_CC, lb->rho_CCLabel,  dwindex,patch);
    new_dw->put(uvel_CC,lb->uvel_CCLabel, dwindex,patch);
    new_dw->put(vvel_CC,lb->vvel_CCLabel, dwindex,patch);
    new_dw->put(wvel_CC,lb->wvel_CCLabel, dwindex,patch);
    new_dw->put(temp,   lb->temp_CCLabel, dwindex,patch);
    
    // These are carried forward variables, they don't change
    new_dw->put(visc_CC,lb->viscosity_CCLabel,dwindex,patch);
    new_dw->put(cv,     lb->cv_CCLabel,       dwindex,patch);
  }
}

void ICE::setBC(CCVariable<double>& variable, const string& kind, 
		const Patch* patch)
{

  Vector dx = patch->dCell();
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    vector<BoundCondBase* > bcs;
    bcs = patch->getBCValues(face);
    
    BoundCondBase* bc_base = 0;
    for (int i = 0; i<(int)bcs.size(); i++ ) {
      if (bcs[i]->getType() == kind) {
	bc_base = bcs[i];
	break;
      }
    }
    
    if (bc_base->getType() == "Pressure") {
      PressureBoundCond* bc = dynamic_cast<PressureBoundCond*>(bc_base);
      if (bc->getKind() == "Dirichlet") 
	variable.fillFace(face,bc->getValue());
      
      if (bc->getKind() == "Neumann") 
	variable.fillFaceFlux(face,bc->getValue(),dx);
	
    }
    if (bc_base->getType() == "Density") {
      DensityBoundCond* bc = dynamic_cast<DensityBoundCond*>(bc_base);
      if (bc->getKind() == "Dirichlet") 
	variable.fillFace(face,bc->getValue());
      
      if (bc->getKind() == "Neumann") 
	variable.fillFaceFlux(face,bc->getValue(),dx);
    }
    if (bc_base->getType() == "Temperature") {
      TemperatureBoundCond* bc = dynamic_cast<TemperatureBoundCond*>(bc_base);
      if (bc->getKind() == "Dirichlet") 
	variable.fillFace(face,bc->getValue());
      
      if (bc->getKind() == "Neumann") 
	variable.fillFaceFlux(face,bc->getValue(),dx);
	
    }
  }

}

void ICE::setBC(CCVariable<double>& variable, const  string& kind, 
		const string& comp, const Patch* patch) 
{
  Vector dx = patch->dCell();
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    vector<BoundCondBase* > bcs;
    bcs = patch->getBCValues(face);
    
    BoundCondBase* bc_base = 0;
    for (int i = 0; i<(int)bcs.size(); i++ ) {
      if (bcs[i]->getType() == kind) {
	bc_base = bcs[i];
	break;
      }
    }
    
    if (bc_base->getType() == "Velocity") {
      VelocityBoundCond* bc = dynamic_cast<VelocityBoundCond*>(bc_base);
      if (bc->getKind() == "Dirichlet") {
	if (comp == "x")
	  variable.fillFace(face,bc->getValue().x());
	if (comp == "y")
	  variable.fillFace(face,bc->getValue().y());
	if (comp == "z")
	  variable.fillFace(face,bc->getValue().z());
      }
      
      if (bc->getKind() == "Neumann") {
	if (comp == "x")
	  variable.fillFaceFlux(face,bc->getValue().x(),dx);
	if (comp == "y")
	  variable.fillFaceFlux(face,bc->getValue().y(),dx);
	if (comp == "z")
	  variable.fillFaceFlux(face,bc->getValue().z(),dx);
      }
    }
  }

}

void ICE::setBC(SFCXVariable<double>& variable, const string& kind, 
		const Patch* patch)
{

  Vector dx = patch->dCell();
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    vector<BoundCondBase* > bcs;
    bcs = patch->getBCValues(face);
    
    BoundCondBase* bc_base = 0;
    for (int i = 0; i<(int)bcs.size(); i++ ) {
      if (bcs[i]->getType() == kind) {
	bc_base = bcs[i];
	break;
      }
    }
    
    if (bc_base->getType() == "Pressure") {
      PressureBoundCond* bc = dynamic_cast<PressureBoundCond*>(bc_base);
      if (bc->getKind() == "Dirichlet") 
	variable.fillFace(face,bc->getValue());
      
      if (bc->getKind() == "Neumann") 
	variable.fillFaceFlux(face,bc->getValue(),dx);
    }
    if (bc_base->getType() == "Density") {
      DensityBoundCond* bc = dynamic_cast<DensityBoundCond*>(bc_base);
      if (bc->getKind() == "Dirichlet") 
	variable.fillFace(face,bc->getValue());
      
      if (bc->getKind() == "Neumann") 
	variable.fillFaceFlux(face,bc->getValue(),dx);
    }
    if (bc_base->getType() == "Temperature") {
      TemperatureBoundCond* bc = dynamic_cast<TemperatureBoundCond*>(bc_base);
      if (bc->getKind() == "Dirichlet") 
	variable.fillFace(face,bc->getValue());
      
      if (bc->getKind() == "Neumann") 
	variable.fillFaceFlux(face,bc->getValue(),dx);
    }
  }

}

void ICE::setBC(SFCXVariable<double>& variable, const  string& kind, 
		const string& comp, const Patch* patch) 
{
  Vector dx = patch->dCell();
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    vector<BoundCondBase* > bcs;
    bcs = patch->getBCValues(face);
    
    BoundCondBase* bc_base = 0;
    for (int i = 0; i<(int)bcs.size(); i++ ) {
      if (bcs[i]->getType() == kind) {
	bc_base = bcs[i];
	break;
      }
    }
    
    if (bc_base->getType() == "Velocity") {
      VelocityBoundCond* bc = dynamic_cast<VelocityBoundCond*>(bc_base);
      if (bc->getKind() == "Dirichlet") {
	if (comp == "x")
	  variable.fillFace(face,bc->getValue().x());
	if (comp == "y")
	  variable.fillFace(face,bc->getValue().y());
	if (comp == "z")
	  variable.fillFace(face,bc->getValue().z());
      }
      
      if (bc->getKind() == "Neumann") {
	if (comp == "x")
	  variable.fillFaceFlux(face,bc->getValue().x(),dx);
	if (comp == "y")
	  variable.fillFaceFlux(face,bc->getValue().y(),dx);
	if (comp == "z")
	  variable.fillFaceFlux(face,bc->getValue().z(),dx);
      }
    }
  }

}

void ICE::setBC(SFCYVariable<double>& variable, const string& kind, 
		const Patch* patch)
{

  Vector dx = patch->dCell();
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    vector<BoundCondBase* > bcs;
    bcs = patch->getBCValues(face);
    
    BoundCondBase* bc_base = 0;
    for (int i = 0; i<(int)bcs.size(); i++ ) {
      if (bcs[i]->getType() == kind) {
	bc_base = bcs[i];
	break;
      }
    }
    
    if (bc_base->getType() == "Pressure") {
      PressureBoundCond* bc = dynamic_cast<PressureBoundCond*>(bc_base);
      if (bc->getKind() == "Dirichlet") 
	variable.fillFace(face,bc->getValue());
      
      if (bc->getKind() == "Neumann") 
	variable.fillFaceFlux(face,bc->getValue(),dx);
    }
    if (bc_base->getType() == "Density") {
      DensityBoundCond* bc = dynamic_cast<DensityBoundCond*>(bc_base);
      if (bc->getKind() == "Dirichlet") 
	variable.fillFace(face,bc->getValue());
      
      if (bc->getKind() == "Neumann") 
	variable.fillFaceFlux(face,bc->getValue(),dx);
    }
    if (bc_base->getType() == "Temperature") {
      TemperatureBoundCond* bc = dynamic_cast<TemperatureBoundCond*>(bc_base);
      if (bc->getKind() == "Dirichlet") 
	variable.fillFace(face,bc->getValue());
      
      if (bc->getKind() == "Neumann") 
	variable.fillFaceFlux(face,bc->getValue(),dx);
    }
  }

}

void ICE::setBC(SFCYVariable<double>& variable, const  string& kind, 
		const string& comp, const Patch* patch) 
{
  Vector dx = patch->dCell();
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    vector<BoundCondBase* > bcs;
    bcs = patch->getBCValues(face);
    
    BoundCondBase* bc_base = 0;
    for (int i = 0; i<(int)bcs.size(); i++ ) {
      if (bcs[i]->getType() == kind) {
	bc_base = bcs[i];
	break;
      }
    }
    
    if (bc_base->getType() == "Velocity") {
      VelocityBoundCond* bc = dynamic_cast<VelocityBoundCond*>(bc_base);
      if (bc->getKind() == "Dirichlet") {
	if (comp == "x")
	  variable.fillFace(face,bc->getValue().x());
	if (comp == "y")
	  variable.fillFace(face,bc->getValue().y());
	if (comp == "z")
	  variable.fillFace(face,bc->getValue().z());
      }
      
      if (bc->getKind() == "Neumann") {
	if (comp == "x")
	  variable.fillFaceFlux(face,bc->getValue().x(),dx);
	if (comp == "y")
	  variable.fillFaceFlux(face,bc->getValue().y(),dx);
	if (comp == "z")
	  variable.fillFaceFlux(face,bc->getValue().z(),dx);
      }
    }
  }

}




void ICE::setBC(SFCZVariable<double>& variable, const string& kind, 
		const Patch* patch)
{

  Vector dx = patch->dCell();
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    vector<BoundCondBase* > bcs;
    bcs = patch->getBCValues(face);
    
    BoundCondBase* bc_base = 0;
    for (int i = 0; i<(int)bcs.size(); i++ ) {
      if (bcs[i]->getType() == kind) {
	bc_base = bcs[i];
	break;
      }
    }
    
    if (bc_base->getType() == "Pressure") {
      PressureBoundCond* bc = dynamic_cast<PressureBoundCond*>(bc_base);
      if (bc->getKind() == "Dirichlet") 
	variable.fillFace(face,bc->getValue());
      
      if (bc->getKind() == "Neumann") 
	variable.fillFaceFlux(face,bc->getValue(),dx);
    }
    if (bc_base->getType() == "Density") {
      DensityBoundCond* bc = dynamic_cast<DensityBoundCond*>(bc_base);
      if (bc->getKind() == "Dirichlet") 
	variable.fillFace(face,bc->getValue());
      
      if (bc->getKind() == "Neumann") 
	variable.fillFaceFlux(face,bc->getValue(),dx);
    }
    if (bc_base->getType() == "Temperature") {
      TemperatureBoundCond* bc = dynamic_cast<TemperatureBoundCond*>(bc_base);
      if (bc->getKind() == "Dirichlet") 
	variable.fillFace(face,bc->getValue());
      
      if (bc->getKind() == "Neumann") 
	variable.fillFaceFlux(face,bc->getValue(),dx);
    }
  }

}

void ICE::setBC(SFCZVariable<double>& variable, const  string& kind, 
		const string& comp, const Patch* patch) 
{
  Vector dx = patch->dCell();
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    vector<BoundCondBase* > bcs;
    bcs = patch->getBCValues(face);
    
    BoundCondBase* bc_base = 0;
    for (int i = 0; i<(int)bcs.size(); i++ ) {
      if (bcs[i]->getType() == kind) {
	bc_base = bcs[i];
	break;
      }
    }
    
    if (bc_base->getType() == "Velocity") {
      VelocityBoundCond* bc = dynamic_cast<VelocityBoundCond*>(bc_base);
      if (bc->getKind() == "Dirichlet") {
	if (comp == "x")
	  variable.fillFace(face,bc->getValue().x());
	if (comp == "y")
	  variable.fillFace(face,bc->getValue().y());
	if (comp == "z")
	  variable.fillFace(face,bc->getValue().z());
      }
      
      if (bc->getKind() == "Neumann") {
	if (comp == "x")
	  variable.fillFaceFlux(face,bc->getValue().x(),dx);
	if (comp == "y")
	  variable.fillFaceFlux(face,bc->getValue().y(),dx);
	if (comp == "z")
	  variable.fillFaceFlux(face,bc->getValue().z(),dx);
      }
    }
  }

}

void ICE::influxOutfluxVolume(const SFCXVariable<double>& uvel_FC,
			      const SFCYVariable<double>& vvel_FC,
			      const SFCZVariable<double>& wvel_FC,
			      const double& delT, const Patch* patch,
			      CCVariable<fflux>& OFS, CCVariable<eflux>& OFE,
			      CCVariable<fflux>& IFS, CCVariable<eflux>& IFE)

{

  Vector dx = patch->dCell();
  double delY_top,delY_bottom,delX_right,delX_left,delZ_front,delZ_back;
  double delX_tmp,delY_tmp,delZ_tmp,totalfluxin;
  double vol = dx.x()*dx.y()*dx.z();

  //Calculate each cells outfluxes first
  //Here the CellIterator must visit ALL cells
  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
    delY_top    = std::max(0.0, (vvel_FC[*iter+IntVector(0,1,0)] * delT));
    delY_bottom = std::max(0.0,-(vvel_FC[*iter+IntVector(0,0,0)] * delT));
    delX_right  = std::max(0.0, (uvel_FC[*iter+IntVector(1,0,0)] * delT));
    delX_left   = std::max(0.0,-(uvel_FC[*iter+IntVector(0,0,0)] * delT));
    delZ_front  = std::max(0.0, (wvel_FC[*iter+IntVector(0,0,1)] * delT));
    delZ_back   = std::max(0.0,-(wvel_FC[*iter+IntVector(0,0,0)] * delT));

    delX_tmp    = dx.x() - delX_right - delX_left;
    delY_tmp    = dx.y() - delY_top   - delY_bottom;
    delZ_tmp    = dx.z() - delZ_front - delZ_back;
    
    // Slabs
    OFS[*iter].d_fflux[TOP]    = delY_top     * delX_tmp * dx.z();
    OFS[*iter].d_fflux[BOTTOM] = delY_bottom  * delX_tmp * dx.z();
    OFS[*iter].d_fflux[RIGHT]  = delX_right   * delY_tmp * dx.z();
    OFS[*iter].d_fflux[LEFT]   = delX_left    * delY_tmp * dx.z();
    OFS[*iter].d_fflux[FRONT]  = delZ_front   * delZ_tmp * dx.y();
    OFS[*iter].d_fflux[BACK]   = delZ_back    * delZ_tmp * dx.y();
    
    // Corners (these are actually edges in 3-d)
    OFE[*iter].d_eflux[TR] = delY_top      * delX_right * dx.z();
    OFE[*iter].d_eflux[TL] = delY_top      * delX_left  * dx.z();
    OFE[*iter].d_eflux[BR] = delY_bottom   * delX_right * dx.z();
    OFE[*iter].d_eflux[BL] = delY_bottom   * delX_left  * dx.z();
    // These need to be filled in for 3-d
    OFE[*iter].d_eflux[TF] = 0.0;
    OFE[*iter].d_eflux[Tb] = 0.0;
    OFE[*iter].d_eflux[BF] = 0.0;
    OFE[*iter].d_eflux[Bb] = 0.0;
    OFE[*iter].d_eflux[FR] = 0.0;
    OFE[*iter].d_eflux[FL] = 0.0;
    OFE[*iter].d_eflux[bR] = 0.0;
    OFE[*iter].d_eflux[bL] = 0.0;
  }
  //Now calculate each cells influxes
  //Here the CellIterator only needs to visit REAL cells
#if 1
      //	CAN'T DO THIS UNTIL EXTRA CELLS ARE IMPLEMENTED
  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
    IntVector curcell = *iter,adjcell;

    // Slabs
    adjcell = IntVector(curcell.x(),curcell.y()+1,curcell.z());
    IFS[*iter].d_fflux[TOP]    = OFS[adjcell].d_fflux[BOTTOM];

    adjcell = IntVector(curcell.x(),curcell.y()-1,curcell.z());
    IFS[*iter].d_fflux[BOTTOM] = OFS[adjcell].d_fflux[TOP];
    
    adjcell = IntVector(curcell.x()+1,curcell.y(),curcell.z());
    IFS[*iter].d_fflux[RIGHT]  = OFS[adjcell].d_fflux[LEFT];

    adjcell = IntVector(curcell.x()-1,curcell.y(),curcell.z());
    IFS[*iter].d_fflux[LEFT]   = OFS[adjcell].d_fflux[RIGHT];

    adjcell = IntVector(curcell.x(),curcell.y(),curcell.z()-1);
    IFS[*iter].d_fflux[FRONT]  = OFS[adjcell].d_fflux[BACK];

    adjcell = IntVector(curcell.x(),curcell.y(),curcell.z()+1);
    IFS[*iter].d_fflux[BACK]   = OFS[adjcell].d_fflux[FRONT];

    // Corners (aka edges)
    adjcell = IntVector(curcell.x()+1,curcell.y()+1,curcell.z());
    IFE[*iter].d_eflux[TR]    = OFE[adjcell].d_eflux[BL];

    adjcell = IntVector(curcell.x()+1,curcell.y()-1,curcell.z());
    IFE[*iter].d_eflux[BR]    = OFE[adjcell].d_eflux[TL];

    adjcell = IntVector(curcell.x()-1,curcell.y()+1,curcell.z());
    IFE[*iter].d_eflux[TL]    = OFE[adjcell].d_eflux[BR];

    adjcell = IntVector(curcell.x()-1,curcell.y()-1,curcell.z());
    IFE[*iter].d_eflux[BL]    = OFE[adjcell].d_eflux[TR];
    
    totalfluxin = IFS[*iter].d_fflux[TOP]   + IFS[*iter].d_fflux[BOTTOM] +
      IFS[*iter].d_fflux[RIGHT] + IFS[*iter].d_fflux[LEFT]   +
      IFS[*iter].d_fflux[FRONT] + IFS[*iter].d_fflux[BACK]   +
      IFE[*iter].d_eflux[TR]    + IFE[*iter].d_eflux[BR]     +
      IFE[*iter].d_eflux[TL]    + IFE[*iter].d_eflux[BL];
#if 0
    cout << "totalfluxin = " << totalfluxin << endl;
    cout << "vol = " << vol << endl;
     ASSERT(totalfluxin < vol);
#endif
  }
#endif

}

void ICE::outflowVolCentroid(const SFCXVariable<double>& uvel_FC,
			     const SFCYVariable<double>& vvel_FC,
			     const SFCZVariable<double>& wvel_FC,
			     const double& delT, const Vector& dx,
			     CCVariable<fflux>& r_out_x,
			     CCVariable<fflux>& r_out_y,
			     CCVariable<fflux>& r_out_z,
			     CCVariable<eflux>& r_out_x_CF,
			     CCVariable<eflux>& r_out_y_CF,
			     CCVariable<eflux>& r_out_z_CF)

{

}

void ICE::advectQFirst(const CCVariable<double>& q_CC,
		       const Patch* patch,
		       const CCVariable<fflux>& OFS,
		       const CCVariable<eflux>& OFE,
		       const CCVariable<fflux>& IFS,
		       const CCVariable<eflux>& IFE,
		       CCVariable<fflux>& q_out,
		       CCVariable<eflux>& q_out_EF,
		       CCVariable<fflux>& q_in,
		       CCVariable<eflux>& q_in_EF,
		       CCVariable<double>& q_advected)

{
  qOutfluxFirst(q_CC, patch, q_out, q_out_EF);
  qInflux(q_out,q_out_EF,patch,q_in,q_in_EF);

  double sum_q_outflux,sum_q_outflux_EF,sum_q_influx,sum_q_influx_EF;

  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
    sum_q_outflux       = 0.0;
    sum_q_outflux_EF    = 0.0;
    sum_q_influx        = 0.0;
    sum_q_influx_EF     = 0.0;

    for(int face = TOP; face <= BACK; face++ ) {
       sum_q_outflux  += q_out[*iter].d_fflux[face] * OFS[*iter].d_fflux[face];
    }

    for(int edge = TR; edge <= bL; edge++ ) {
       sum_q_outflux_EF += q_out_EF[*iter].d_eflux[edge] * OFE[*iter].d_eflux[edge];
    }

    for(int face = TOP; face <= BACK; face++ ) {
       sum_q_influx  += q_in[*iter].d_fflux[face] * IFS[*iter].d_fflux[face];
    }

    for(int edge = TR; edge <= bL; edge++ ) {
       sum_q_influx_EF += q_in_EF[*iter].d_eflux[edge] * IFE[*iter].d_eflux[edge];
    }

    q_advected[*iter] = - sum_q_outflux - sum_q_outflux_EF
			+ sum_q_influx  + sum_q_influx_EF;
  }

}

void ICE::qOutfluxFirst(const CCVariable<double>& q_CC,
		        const Patch* patch,
			CCVariable<fflux>& q_out,
			CCVariable<eflux>& q_out_EF)
{
  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
    for(int face = TOP; face <= BACK; face++ ) {
	q_out[*iter].d_fflux[face] = q_CC[*iter];
    }

    for(int edge = TR; edge <= bL; edge++ ) {
       q_out_EF[*iter].d_eflux[edge] = q_CC[*iter];
    }
  }

}

void ICE::qInflux(const CCVariable<fflux>& q_out,
		  const CCVariable<eflux>& q_out_EF,
		  const Patch* patch,
		  CCVariable<fflux>& q_in,
		  CCVariable<eflux>& q_in_EF)

{

#if 1
   // This can't be uncommented until ExtraCells are implemented
  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
    IntVector curcell = *iter,adjcell;

    // SLABS
    adjcell = IntVector(curcell.x(),curcell.y()+1,curcell.z());
    q_in[*iter].d_fflux[TOP]    = q_out[adjcell].d_fflux[BOTTOM];
    adjcell = IntVector(curcell.x(),curcell.y()-1,curcell.z());
    q_in[*iter].d_fflux[BOTTOM] = q_out[adjcell].d_fflux[TOP];

    adjcell = IntVector(curcell.x()+1,curcell.y(),curcell.z());
    q_in[*iter].d_fflux[RIGHT]  = q_out[adjcell].d_fflux[LEFT];
    adjcell = IntVector(curcell.x()-1,curcell.y(),curcell.z());
    q_in[*iter].d_fflux[LEFT]   = q_out[adjcell].d_fflux[RIGHT];

    adjcell = IntVector(curcell.x(),curcell.y(),curcell.z()+1);
    q_in[*iter].d_fflux[FRONT]  = q_out[adjcell].d_fflux[BACK];
    adjcell = IntVector(curcell.x(),curcell.y(),curcell.z()-1);
    q_in[*iter].d_fflux[BACK]   = q_out[adjcell].d_fflux[FRONT];

    //CORNERS
    adjcell = IntVector(curcell.x()+1,curcell.y()+1,curcell.z());
    q_in_EF[*iter].d_eflux[TR]  = q_out_EF[adjcell].d_eflux[BL];

    adjcell = IntVector(curcell.x()+1,curcell.y()-1,curcell.z());
    q_in_EF[*iter].d_eflux[BR]  = q_out_EF[adjcell].d_eflux[TL];

    adjcell = IntVector(curcell.x()-1,curcell.y()+1,curcell.z());
    q_in_EF[*iter].d_eflux[TL]  = q_out_EF[adjcell].d_eflux[BR];

    adjcell = IntVector(curcell.x()-1,curcell.y()-1,curcell.z());
    q_in_EF[*iter].d_eflux[BL]  = q_out_EF[adjcell].d_eflux[TR];
  }
#endif

}

void ICE::qOutfluxSecond(CCVariable<fflux>& OFS,
			 CCVariable<fflux>& IFS,
			 CCVariable<fflux>& r_out_x,
			 CCVariable<fflux>& r_out_y,
			 CCVariable<fflux>& r_out_z,
			 CCVariable<eflux>& r_out_x_CF,
			 CCVariable<eflux>& r_out_y_CF,
			 CCVariable<eflux>& r_out_z_CF,
			 const Vector& dx)
{

}

#ifdef __sgi
#define IRIX
#pragma set woff 1209
#endif

namespace Uintah {
   namespace ICESpace {


static MPI_Datatype makeMPI_fflux()
{
   ASSERTEQ(sizeof(ICE::fflux), sizeof(double)*6);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 6, 6, MPI_DOUBLE, &mpitype);
   MPI_Type_commit(&mpitype);
   return mpitype;
}

const TypeDescription* fun_getTypeDescription(ICE::fflux*)
{
   static TypeDescription* td = 0;
   if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                               "ICE::fflux", true, &makeMPI_fflux);
   }
   return td;
}

static MPI_Datatype makeMPI_eflux()
{
   ASSERTEQ(sizeof(ICE::eflux), sizeof(double)*12);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 12, 12, MPI_DOUBLE, &mpitype);
   MPI_Type_commit(&mpitype);
   return mpitype;
}

const TypeDescription* fun_getTypeDescription(ICE::eflux*)
{
   static TypeDescription* td = 0;
   if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                               "ICE::eflux", true, &makeMPI_eflux);
   }
   return td;
}


}
}


//
// $Log$
// Revision 1.62  2000/12/05 20:45:49  jas
// Iterate over all the cells in influxOutfluxVolume.  Now step 2 is working.
//
// Revision 1.61  2000/12/05 15:45:29  jas
// Now using SFC{X,Y,Z} data types.  Fixed some small bugs and things appear
// to be working up to the middle of step 2.
//
// Revision 1.60  2000/11/28 03:50:28  jas
// Added {X,Y,Z}FCVariables.  Things still don't work yet!
//
// Revision 1.59  2000/11/22 01:28:05  guilkey
// Changed the way initial conditions are set.  GeometryObjects are created
// to fill the volume of the domain.  Each object has appropriate initial
// conditions associated with it.  ICEMaterial now has an initializeCells
// method, which for now just does what was previously done with the
// initial condition stuct d_ic.  This will be extended to allow regions of
// the domain to be initialized with different materials.  Sorry for the
// lame GeometryObject2, this could be changed to ICEGeometryObject or
// something.
//
// Revision 1.58  2000/11/21 21:52:27  jas
// Simplified scheduleTimeAdvance now is a bunch of functions.  More
// implementation of FC variables.
//
// Revision 1.57  2000/11/15 00:51:54  guilkey
// Changed code to take advantage of the ICEMaterial stuff I committed
// recently in preparation for coupling the two codes.
//
// Revision 1.56  2000/11/14 04:02:11  jas
// Added getExtraCellIterator and things now appear to be working up to
// face centered velocity calculations.
//
// Revision 1.55  2000/11/02 21:33:05  jas
// Added new bc implementation.  Things now work thru step 1b.  Neumann bcs
// are now set correctly.
//
// Revision 1.54  2000/10/31 04:16:17  jas
// Fixed some errors in speed of sound and equilibration pressure calculation.
// Added initial conditions.
//
// Revision 1.53  2000/10/27 23:41:01  jas
// Added more material constants and some debugging output.
//
// Revision 1.52  2000/10/26 23:22:09  jas
// BCs are now implemented.
//
// Revision 1.51  2000/10/26 00:52:54  guilkey
// Work on step4b
//
// Revision 1.50  2000/10/26 00:24:46  guilkey
// Made all pressures belong to material 0.  Implemented step4b.
//
// Revision 1.49  2000/10/25 23:12:17  guilkey
// Fixed step2, reorganized 6and7 just a little bit.
//
// Revision 1.48  2000/10/25 22:22:13  jas
// Change the fflux and eflux struct so that the data members begin with d_.
// This makes g++ happy.
//
// Revision 1.47  2000/10/25 21:15:31  guilkey
// Finished advection
//
// Revision 1.46  2000/10/24 23:07:21  guilkey
// Added code for steps6and7.
//
// Revision 1.45  2000/10/20 23:58:55  guilkey
// Added part of advection code.
//
// Revision 1.44  2000/10/19 02:44:52  guilkey
// Added code for step5b.
//
// Revision 1.43  2000/10/18 21:02:17  guilkey
// Added code for steps 4 and 5.
//
// Revision 1.42  2000/10/18 03:57:22  jas
// Don't print out bc values.
//
// Revision 1.41  2000/10/18 03:43:01  jas
// Implemented pressure boundary conditions during equilibration computation (1b).
//
// Revision 1.40  2000/10/17 23:05:15  guilkey
// Fixed some computes and requires.
//
// Revision 1.39  2000/10/17 20:26:20  jas
// Changed press to press_new.
//
// Revision 1.38  2000/10/17 18:35:20  guilkey
// Added some computes to actuallyInitialize.
//
// Revision 1.37  2000/10/17 04:33:35  jas
// Copied grid bcs into ice for initial testing.
//
// Revision 1.36  2000/10/17 04:13:25  jas
// Implement hydrostatic pressure adjustment as part of step 1b.  Still need
// to implement update bcs.
//
// Revision 1.35  2000/10/16 20:31:00  guilkey
// Step3 added
//
// Revision 1.34  2000/10/16 19:10:34  guilkey
// Combined step1e with step2 and eliminated step1e.
//
// Revision 1.33  2000/10/16 18:32:40  guilkey
// Implemented "step1e" of the ICE algorithm.
//
// Revision 1.32  2000/10/16 17:19:44  guilkey
// Code for ICE::step1d.  Only code for one of the faces is committed
// until things become more concrete.
//
// Revision 1.31  2000/10/14 02:49:46  jas
// Added implementation of compute equilibration pressure.  Still need to do
// the update of BCS and hydrostatic pressure.  Still some issues with
// computes and requires - will compile but won't run.
//
// Revision 1.30  2000/10/13 00:01:11  guilkey
// More work on ICE
//
// Revision 1.29  2000/10/11 00:15:50  jas
// Sketched out the compute equilibration pressure.
//
// Revision 1.28  2000/10/10 20:35:07  jas
// Move some stuff around.
//
// Revision 1.27  2000/10/09 22:37:01  jas
// Cleaned up labels and added more computes and requires for EOS.
//
// Revision 1.25  2000/10/05 04:26:48  guilkey
// Added code for part of the EOS evaluation.
//
// Revision 1.24  2000/10/05 00:16:33  jas
// Starting to work on the speed of sound stuff.
//
// Revision 1.23  2000/10/04 23:38:21  jas
// All of the steps are in place with just dummy functions.  delT is
// hardwired in for the moment so that we can actually do multiple
// time steps with empty functions.
//
// Revision 1.22  2000/10/04 20:15:27  jas
// Start to bring ICE into UCF.
//
