
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

using std::vector;
using std::max;
using SCICore::Geometry::Vector;

using namespace Uintah;
using namespace Uintah::ICESpace;
//using namespace SCICore::Datatypes;
using SCICore::Datatypes::DenseMatrix;


ICE::ICE(const ProcessorGroup* myworld) 
  : UintahParallelComponent(myworld)
{
  lb = new ICELabel();

}

ICE::~ICE()
{
}

void ICE::problemSetup(const ProblemSpecP& prob_spec, GridP& grid,
		       SimulationStateP& sharedState)
{

  d_sharedState = sharedState;

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
  }     

  cout << "Number of materials: " << d_sharedState->getNumMatls() << endl;

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
    for (int m = 0; m < d_sharedState->getNumMatls(); m++ ) {
      Material* matl = d_sharedState->getMaterial(m);
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      if(ice_matl){
	t->computes(dw, lb->temp_CCLabel,ice_matl->getDWIndex(),patch);
	t->computes(dw, lb->rho_micro_CCLabel,ice_matl->getDWIndex(),patch);
	t->computes(dw, lb->rho_CCLabel,ice_matl->getDWIndex(),patch);
	t->computes(dw, lb->cv_CCLabel,ice_matl->getDWIndex(),patch);
	t->computes(dw, lb->press_CCLabel,matl->getDWIndex(), patch);

	t->computes(dw, lb->uvel_CCLabel,ice_matl->getDWIndex(),patch);
	t->computes(dw, lb->vvel_CCLabel,ice_matl->getDWIndex(),patch);
	t->computes(dw, lb->wvel_CCLabel,ice_matl->getDWIndex(),patch);
      }
    }
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
  int numMatls = d_sharedState->getNumMatls();

  for(Level::const_patchIterator iter=level->patchesBegin();
       iter != level->patchesEnd(); iter++){

      const Patch* patch=*iter;

      // Step 1a  computeSoundSpeed
      {
	Task* t = scinew Task("ICE::step1a",patch, old_dw, new_dw,this,
			       &ICE::actuallyStep1a);
	for (int m = 0; m < numMatls; m++) {
            Material* matl = d_sharedState->getMaterial(m);
            ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
            if(ice_matl){
               EquationOfState* eos = ice_matl->getEOS();
	       // Compute the speed of sound
               eos->addComputesAndRequiresSS(t,ice_matl,patch,old_dw,new_dw);
            }
	}
	sched->addTask(t);
      }

      // Step 1b calculate equlibration pressure
      {
	Task* t = scinew Task("ICE::step1b",patch, old_dw, new_dw,this,
			       &ICE::actuallyStep1b);

	for (int m = 0; m < numMatls; m++) {
	  Material* matl = d_sharedState->getMaterial(m);
	  ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
	  if(ice_matl){
	     EquationOfState* eos = ice_matl->getEOS();
	       // Compute the rho micro
               eos->addComputesAndRequiresRM(t,ice_matl,patch,old_dw,new_dw);
	       t->requires(old_dw,lb->vol_frac_CCLabel,
			   matl->getDWIndex(),patch,Ghost::None);
	       t->requires(old_dw,lb->rho_CCLabel,
			   matl->getDWIndex(),patch,Ghost::None);
	       t->requires(old_dw,lb->rho_micro_CCLabel,
			   matl->getDWIndex(),patch,Ghost::None);
	       t->requires(old_dw,lb->temp_CCLabel,
			   matl->getDWIndex(),patch,Ghost::None);
	       t->requires(old_dw,lb->cv_CCLabel,
			   matl->getDWIndex(),patch,Ghost::None);
	       t->requires(new_dw,lb->speedSound_CCLabel,
			   matl->getDWIndex(),patch,Ghost::None);
	       t->requires(old_dw,lb->press_CCLabel,
			   matl->getDWIndex(),patch,Ghost::None);
	       t->computes(new_dw,lb->press_CCLabel,matl->getDWIndex(), patch);
	       t->computes(new_dw,lb->vol_frac_CCLabel,matl->getDWIndex(),
			   patch);
	       t->computes(new_dw,lb->speedSound_CCLabel,matl->getDWIndex(), 
			   patch);
	  }
	}
	sched->addTask(t);
      }

      // Step 1c compute face centered velocities
      {
	Task* t = scinew Task("ICE::step1c",patch, old_dw, new_dw,this,
			       &ICE::actuallyStep1c);

	for (int m = 0; m < numMatls; m++) {
	  Material* matl = d_sharedState->getMaterial(m);
	  ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
	  if(ice_matl){
	    t->requires(old_dw,lb->rho_CCLabel,
			matl->getDWIndex(),patch,Ghost::None);
	    t->requires(old_dw,lb->uvel_CCLabel,
			matl->getDWIndex(),patch,Ghost::None);
	    t->requires(old_dw,lb->vvel_CCLabel,
			matl->getDWIndex(),patch,Ghost::None);
	    t->requires(old_dw,lb->wvel_CCLabel,
			matl->getDWIndex(),patch,Ghost::None);
	    t->requires(new_dw,lb->press_CCLabel,
			matl->getDWIndex(),patch,Ghost::None);

	    t->computes(new_dw,lb->uvel_FCLabel,matl->getDWIndex(), patch);
	    t->computes(new_dw,lb->vvel_FCLabel,matl->getDWIndex(), patch);
	    t->computes(new_dw,lb->wvel_FCLabel,matl->getDWIndex(), patch);
          }
	}
	sched->addTask(t);
      }

      // Step 1d computes momentum exchange on FC velocities
      {
	Task* t = scinew Task("ICE::step1d",patch, old_dw, new_dw,this,
			       &ICE::actuallyStep1d);

	for (int m = 0; m < numMatls; m++) {
	  Material* matl = d_sharedState->getMaterial(m);
	  ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
	  if(ice_matl){
	    t->requires(new_dw,lb->rho_micro_CCLabel,
			matl->getDWIndex(),patch,Ghost::None);
	    t->requires(new_dw,lb->vol_frac_CCLabel,
			matl->getDWIndex(),patch,Ghost::None);
	    t->requires(old_dw,lb->uvel_FCLabel,
			matl->getDWIndex(),patch,Ghost::None);
	    t->requires(old_dw,lb->vvel_FCLabel,
			matl->getDWIndex(),patch,Ghost::None);
	    t->requires(old_dw,lb->wvel_FCLabel,
			matl->getDWIndex(),patch,Ghost::None);

	    t->computes(new_dw,lb->uvel_FCMELabel,matl->getDWIndex(), patch);
	    t->computes(new_dw,lb->vvel_FCMELabel,matl->getDWIndex(), patch);
	    t->computes(new_dw,lb->wvel_FCMELabel,matl->getDWIndex(), patch);
          }
	}
	sched->addTask(t);
      }

      // Step 1e computes momentum exchange on FC velocities
      {
	Task* t = scinew Task("ICE::step1e",patch, old_dw, new_dw,this,
			       &ICE::actuallyStep1e);

	for (int m = 0; m < numMatls; m++) {
	  Material* matl = d_sharedState->getMaterial(m);
	  ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
	  if(ice_matl){
	    t->requires(new_dw,lb->vol_frac_CCLabel,
			matl->getDWIndex(),patch,Ghost::None);
	    t->requires(new_dw,lb->uvel_FCMELabel,
			matl->getDWIndex(),patch,Ghost::None);
	    t->requires(new_dw,lb->vvel_FCMELabel,
			matl->getDWIndex(),patch,Ghost::None);
	    t->requires(new_dw,lb->wvel_FCMELabel,
			matl->getDWIndex(),patch,Ghost::None);

	    t->computes(new_dw,lb->div_velfc_CCLabel,matl->getDWIndex(),patch);
          }
	}
	sched->addTask(t);
      }

      // Step 2
      {
	Task* t = scinew Task("ICE::step2",patch, old_dw, new_dw,this,
			       &ICE::actuallyStep2);
	for (int m = 0; m < numMatls; m++) {
	}
	sched->addTask(t);
      }

      // Step 3
      {
	Task* t = scinew Task("ICE::step3",patch, old_dw, new_dw,this,
			       &ICE::actuallyStep3);
	for (int m = 0; m < numMatls; m++) {
	}
	sched->addTask(t);
      }

      // Step 4
      {
	Task* t = scinew Task("ICE::step4",patch, old_dw, new_dw,this,
			       &ICE::actuallyStep4);
	for (int m = 0; m < numMatls; m++) {
	}
	sched->addTask(t);
      }

      // Step 5
      {
	Task* t = scinew Task("ICE::step5",patch, old_dw, new_dw,this,
			       &ICE::actuallyStep5);
	for (int m = 0; m < numMatls; m++) {
	}
	sched->addTask(t);
      }

      // Step 6and7
      {
	Task* t = scinew Task("ICE::step6and7",patch, old_dw, new_dw,this,
			       &ICE::actuallyStep6and7);
	for (int m = 0; m < d_sharedState->getNumMatls(); m++ ) {
	  Material* matl = d_sharedState->getMaterial(m);
	  ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
	  if(ice_matl){
	    t->computes(new_dw, lb->temp_CCLabel,ice_matl->getDWIndex(),patch);
	    t->computes(new_dw, lb->rho_CCLabel,ice_matl->getDWIndex(),patch);
	    t->computes(new_dw, lb->cv_CCLabel,ice_matl->getDWIndex(),patch);

	    t->computes(new_dw,lb->uvel_CCLabel,matl->getDWIndex(), patch);
	    t->computes(new_dw,lb->vvel_CCLabel,matl->getDWIndex(), patch);
	    t->computes(new_dw,lb->wvel_CCLabel,matl->getDWIndex(), patch);
	  }
	}
	t->computes(new_dw, d_sharedState->get_delt_label());
	sched->addTask(t);
      }
  }

}

void ICE::actuallyInitialize(const ProcessorGroup*,
			     const Patch* patch,
			     DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw)
  

{

  cout << "Doing actually Initialize" << endl;
  double dT = 0.0001;
  new_dw->put(delt_vartype(dT), lb->delTLabel);

  CCVariable<double> rho_micro, temp, cv, rho_CC;
  CCVariable<double> uvel_CC,vvel_CC,wvel_CC;
  for (int m = 0; m < d_sharedState->getNumMatls(); m++ ) {
    Material* matl = d_sharedState->getMaterial(m);
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    if(ice_matl){
      int vfindex = ice_matl->getDWIndex();
      new_dw->allocate(rho_micro,lb->rho_micro_CCLabel,vfindex,patch);
      new_dw->allocate(rho_CC,lb->rho_CCLabel,vfindex,patch);
      new_dw->allocate(temp,lb->temp_CCLabel,vfindex,patch);
      new_dw->allocate(cv,lb->cv_CCLabel,vfindex,patch);

      new_dw->allocate(uvel_CC,lb->uvel_CCLabel,vfindex,patch);
      new_dw->allocate(vvel_CC,lb->vvel_CCLabel,vfindex,patch);
      new_dw->allocate(wvel_CC,lb->wvel_CCLabel,vfindex,patch);

      new_dw->put(rho_micro,lb->rho_micro_CCLabel,vfindex,patch);
      new_dw->put(rho_CC,lb->rho_CCLabel,vfindex,patch);
      new_dw->put(temp,lb->temp_CCLabel,vfindex,patch);
      new_dw->put(cv,lb->cv_CCLabel,vfindex,patch);
      new_dw->put(uvel_CC,lb->uvel_CCLabel,vfindex,patch);
      new_dw->put(vvel_CC,lb->vvel_CCLabel,vfindex,patch);
      new_dw->put(wvel_CC,lb->wvel_CCLabel,vfindex,patch);
    }
  }
  
}

void ICE::actuallyComputeStableTimestep(const ProcessorGroup*,
					const Patch* patch,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw)
{
  cout << "Doing acutally Compute Stable Timestep " << endl;
}


void ICE::actuallyStep1a(const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{

  cout << "Doing actually step1a" << endl;

  int numMatls = d_sharedState->getNumMatls();

  // Compute the speed of sound

  for (int m = 0; m < numMatls; m++) {
    Material* matl = d_sharedState->getMaterial(m);
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    if (ice_matl) {
      EquationOfState* eos = ice_matl->getEOS();
      eos->computeSpeedSound(patch,ice_matl,old_dw,new_dw);
    }
  }

}

void ICE::actuallyStep1b(const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{

  cout << "Doing actually step1b" << endl;

  int numMatls = d_sharedState->getNumMatls();

  // Compute the equilibration pressure for all materials
#if 1
 

  // Compute initial Rho Micro
  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    if(ice_matl){
      ice_matl->getEOS()->computeRhoMicro(patch,ice_matl,old_dw,new_dw);   
    }
  }

  
   // Need to pull out all of the material's data just like in 
  // contact::exMomInterpolated
   // store in a vector<CCVariable<double>>
  
  // Compute the initial volume fraction
  vector<CCVariable<double> > vol_frac(numMatls);
  for (int m = 0; m < numMatls; m++) {
    Material* matl = d_sharedState->getMaterial(m);
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    if (ice_matl) {
      int vfindex = matl->getVFIndex();
      CCVariable<double> rho_micro,rho;    
      new_dw->allocate(vol_frac[m],lb->vol_frac_CCLabel,vfindex,patch);
      old_dw->get(rho,lb->rho_CCLabel, vfindex,patch,Ghost::None, 0); 
      new_dw->get(rho_micro,lb->rho_micro_CCLabel, vfindex,patch,
		  Ghost::None, 0); 
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
	vol_frac[m][*iter] = rho[*iter]/rho_micro[*iter];
      }
      
    }
   }
  
  vector<double> delVol_frac(numMatls),press_eos(numMatls);
  vector<double> dp_drho(numMatls),dp_de(numMatls);
  
  vector<CCVariable<double> > rho_micro(numMatls),rho(numMatls);
  vector<CCVariable<double> > cv(numMatls);
  vector<CCVariable<double> > Temp(numMatls);
  vector<CCVariable<double> > press(numMatls),press_new(numMatls);
  vector<CCVariable<double> > speedSound(numMatls);

  
  for (int m = 0; m < numMatls; m++) {
    Material* matl = d_sharedState->getMaterial(m);
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    if (ice_matl) {
      int vfindex = matl->getVFIndex();
      old_dw->get(cv[m], lb->cv_CCLabel, vfindex,patch,Ghost::None, 0); 
      old_dw->get(rho[m], lb->rho_CCLabel, vfindex,patch,Ghost::None, 0); 
      old_dw->get(Temp[m], lb->temp_CCLabel, vfindex,patch,Ghost::None, 0); 
      old_dw->get(press[m], lb->press_CCLabel, vfindex,patch,Ghost::None, 0); 
      new_dw->allocate(press_new[m],lb->press_CCLabel,vfindex,patch);
      new_dw->allocate(speedSound[m],lb->speedSound_CCLabel,vfindex,patch);
      new_dw->get(rho_micro[m],lb->rho_micro_CCLabel,vfindex,patch,
		  Ghost::None,0);
    }
  }

  bool converged = false;
  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
    double delPress = 0.;
    while( converged == false) {
     double A = 0.;
     double B = 0.;
     double C = 0.;
     
     for (int m = 0; m < numMatls; m++) 
       delVol_frac[m] = 0.;

     for (int m = 0; m < numMatls; m++) {
       Material* matl = d_sharedState->getMaterial(m);
       ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
       if (ice_matl) {
	 double gamma = ice_matl->getGamma();
	 ice_matl->getEOS()->computePressEOS(rho_micro[m][*iter],gamma,
					     cv[m][*iter],
					     Temp[m][*iter],press_eos[m],
					     dp_drho[m],dp_de[m]);
       }
     }
     vector<double> Q(2),y(2);     
     for (int m = 0; m < numMatls; m++) {
       Material* matl = d_sharedState->getMaterial(m);
       ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
       if (ice_matl) {
	 Q[m] = press[m][*iter] - press_eos[m];
	 y[m] = rho[m][*iter]/(vol_frac[m][*iter]*vol_frac[m][*iter]) 
	   * dp_drho[m];
	 A += vol_frac[m][*iter];
	 B += Q[m]/y[m];
	 C += 1./y[m];
       }
     }
     double vol_frac_not_close_packed = 1.;
     delPress = (A - vol_frac_not_close_packed - B)/C;
     for (int m = 0; m < numMatls; m++)
       press[m][*iter] += delPress;

     for (int m = 0; m < numMatls; m++) {
       Material* matl = d_sharedState->getMaterial(m);
       ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
       if (ice_matl) {
	 double gamma = ice_matl->getGamma();
	 rho_micro[m][*iter] = ice_matl->getEOS()->
	   computeRhoMicro(press[m][*iter], gamma,cv[m][*iter],Temp[m][*iter]);
       }
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
       Material* matl = d_sharedState->getMaterial(m);
       ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
       if (ice_matl) {
	 double gamma = ice_matl->getGamma();
	 ice_matl->getEOS()->computePressEOS(rho_micro[m][*iter],gamma,
					     cv[m][*iter],
					     Temp[m][*iter],press_eos[m],
					     dp_drho[m],dp_de[m]);
	 
	 double temp = dp_drho[m] + dp_de[m] * 
	   (press_eos[m]/(rho_micro[m][*iter]*rho_micro[m][*iter]));
	 speedSound[m][*iter] = sqrt(temp);
       }
     }
     
     
     // Check if converged
     
     double test = 0.;
     test = std::max(test,fabs(delPress));
     for (int m = 0; m < numMatls; m++) {
       test = std::max(test,fabs(delVol_frac[m]));
     }
     if (test < 1.e-5)
       converged = true;
     
    }  // end of converged
    
    // Store new pressure, speedSound,vol_frac
    for (int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial(m);
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      if (ice_matl) {
	int vfindex = matl->getVFIndex();
	new_dw->put(press[m],lb->press_CCLabel,vfindex,patch);
	new_dw->put(vol_frac[m],lb->vol_frac_CCLabel,vfindex,patch);
	new_dw->put(speedSound[m],lb->speedSound_CCLabel,vfindex,patch);
      }
    }
    
  }
    
    
#endif

}

void ICE::actuallyStep1c(const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{

  cout << "Doing actually step1c" << endl;

  int numMatls = d_sharedState->getNumMatls();

  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label());
  Vector dx = patch->dCell();
  Vector gravity = d_sharedState->getGravity();

  // Compute the face centered velocities
  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    if(ice_matl){
      int vfindex = matl->getVFIndex();
      // Get required variables for this patch
      CCVariable<double> rho_CC;
      CCVariable<double> rho_micro_CC;
      CCVariable<double> press_CC;
      CCVariable<double> uvel_CC, vvel_CC, wvel_CC;

      old_dw->get(rho_CC,  lb->rho_CCLabel,  vfindex, patch, Ghost::None, 0);
      old_dw->get(rho_micro_CC, lb->rho_micro_CCLabel,
				vfindex, patch, Ghost::None, 0);
      old_dw->get(press_CC,lb->press_CCLabel, vfindex, patch, Ghost::None, 0);
      old_dw->get(uvel_CC, lb->uvel_CCLabel,  vfindex, patch, Ghost::None, 0);
      old_dw->get(vvel_CC, lb->vvel_CCLabel,  vfindex, patch, Ghost::None, 0);
      old_dw->get(wvel_CC, lb->wvel_CCLabel,  vfindex, patch, Ghost::None, 0);

      // Create variables for the results
      FCVariable<double> uvel_FC;
      FCVariable<double> vvel_FC;
      FCVariable<double> wvel_FC;
      new_dw->allocate(uvel_FC, lb->uvel_FCLabel, vfindex, patch);
      new_dw->allocate(vvel_FC, lb->vvel_FCLabel, vfindex, patch);
      new_dw->allocate(wvel_FC, lb->wvel_FCLabel, vfindex, patch);

      double term1, term2, term3, press_coeff, rho_micro_FC, rho_FC;

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
	IntVector curcell = *iter;

       // Top face
	IntVector adjcell(curcell.x(),curcell.y()+1,curcell.z()); 

	rho_micro_FC = rho_micro_CC[adjcell] + rho_micro_CC[curcell];
	rho_FC       = rho_CC[adjcell]       + rho_CC[curcell];

	term1 = (rho_CC[adjcell] * vvel_CC[adjcell] +
		 rho_CC[curcell] * vvel_CC[curcell])/rho_FC;

	press_coeff = 2.0/(rho_micro_FC);

	term2 =   delT * press_coeff *
			(press_CC[adjcell] - press_CC[curcell])/dx.y();
	term3 =  delT * gravity.y();

	// I don't know what this is going to look like yet
	// but the equations are right I think.
//	uvel_FC[curcell][top] = 0.0;
//	vvel_FC[curcell][top] = term1- term2 + term3;
//	wvel_FC[curcell][top] = 0.0;

       // Right face
	adjcell = IntVector(curcell.x()+1,curcell.y(),curcell.z()); 

	rho_micro_FC = rho_micro_CC[adjcell] + rho_micro_CC[curcell];
	rho_FC       = rho_CC[adjcell]       + rho_CC[curcell];

	term1 = (rho_CC[adjcell] * vvel_CC[adjcell] +
		 rho_CC[curcell] * vvel_CC[curcell])/rho_FC;

	press_coeff = 2.0/(rho_micro_FC);

	term2 =   delT * press_coeff *
			(press_CC[adjcell] - press_CC[curcell])/dx.x();
	term3 =  delT * gravity.x();

	// I don't know what this is going to look like yet
	// but the equations are right I think.
//	uvel_FC[curcell][top] = term1- term2 + term3;
//	vvel_FC[curcell][top] = 0.0;
//	wvel_FC[curcell][top] = 0.0;

       // Front face
	adjcell = IntVector(curcell.x(),curcell.y(),curcell.z()+1); 

	rho_micro_FC = rho_micro_CC[adjcell] + rho_micro_CC[curcell];
	rho_FC       = rho_CC[adjcell]       + rho_CC[curcell];

	term1 = (rho_CC[adjcell] * vvel_CC[adjcell] +
		 rho_CC[curcell] * vvel_CC[curcell])/rho_FC;

	press_coeff = 2.0/(rho_micro_FC);

	term2 =   delT * press_coeff *
			(press_CC[adjcell] - press_CC[curcell])/dx.z();
	term3 =  delT * gravity.z();

	// I don't know what this is going to look like yet
	// but the equations are right I think.
//	uvel_FC[curcell][top] = 0.0;
//	vvel_FC[curcell][top] = 0.0;
//	wvel_FC[curcell][top] = term1- term2 + term3;
      }

      // Put Boundary condition stuff in here
      //
      //

      // Put the result in the datawarehouse
      new_dw->put(uvel_FC, lb->uvel_FCLabel, vfindex, patch);
      new_dw->put(vvel_FC, lb->vvel_FCLabel, vfindex, patch);
      new_dw->put(wvel_FC, lb->wvel_FCLabel, vfindex, patch);
    }
  }
}

void ICE::actuallyStep1d(const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{
  cout << "Doing actually step1d" << endl;

  int numMatls = d_sharedState->getNumMatls();
  int NVFs = d_sharedState->getNumVelFields();

  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label());
  Vector dx = patch->dCell();
  Vector gravity = d_sharedState->getGravity();

  double temp,betamn;

  // Create variables for the required values
  vector<CCVariable<double> > rho_micro_CC(NVFs);
  vector<CCVariable<double> > vol_frac_CC(NVFs);
  vector<FCVariable<double> > uvel_FC(NVFs);
  vector<FCVariable<double> > vvel_FC(NVFs);
  vector<FCVariable<double> > wvel_FC(NVFs);

  // Create variables for the results
  vector<FCVariable<double> > uvel_FCME(NVFs);
  vector<FCVariable<double> > vvel_FCME(NVFs);
  vector<FCVariable<double> > wvel_FCME(NVFs);

  vector<double> b(NVFs);
  DenseMatrix beta(NVFs,NVFs),a(NVFs,NVFs),K(NVFs,NVFs);

  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    if(ice_matl){
      int vfindex = matl->getVFIndex();
      new_dw->get(rho_micro_CC[vfindex], lb->rho_micro_CCLabel,
				vfindex, patch, Ghost::None, 0);
      new_dw->get(vol_frac_CC[vfindex],  lb->vol_frac_CCLabel,
				vfindex, patch, Ghost::None, 0);
      new_dw->get(uvel_FC[vfindex], lb->uvel_FCLabel,
				vfindex, patch, Ghost::None, 0);
      new_dw->get(vvel_FC[vfindex], lb->vvel_FCLabel,
				vfindex, patch, Ghost::None, 0);
      new_dw->get(wvel_FC[vfindex], lb->wvel_FCLabel,
				vfindex, patch, Ghost::None, 0);

      new_dw->allocate(uvel_FC[vfindex], lb->uvel_FCMELabel, vfindex, patch);
      new_dw->allocate(vvel_FC[vfindex], lb->vvel_FCMELabel, vfindex, patch);
      new_dw->allocate(wvel_FC[vfindex], lb->wvel_FCMELabel, vfindex, patch);
    }
  }

  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
    IntVector curcell = *iter;

   // Top face
   IntVector adjcell(curcell.x(),curcell.y()+1,curcell.z()); 

   for(int m = 0; m < NVFs; m++){
     for(int n = 0; n < NVFs; n++){
	temp = (vol_frac_CC[n][adjcell] + vol_frac_CC[n][curcell]) * K.get(n,m);
	betamn = delT * temp/
		       (rho_micro_CC[m][curcell] + rho_micro_CC[m][adjcell]);
	beta.put(m,n,betamn);
	a.put(m,n,-beta.get(m,n));
     }
   }

   for(int m = 0; m < NVFs; m++){
     a.put(m,m,1.);
     for(int n = 0; n < NVFs; n++){
	a.put(m,m, a.get(m,m) +  beta.get(m,n));
     }
   }

   for(int m = 0; m < NVFs; m++){
     b[m] = 0.0;
     for(int n = 0; n < NVFs; n++){
	b[m] += beta.get(m,n) * (vvel_FC[n][*iter] - vvel_FC[m][*iter]);
     }
   }

//  gauss_jordan_elimination(a,  b,  nMaterials);
//   int itworked = a.solve(b);

   for(int m = 0; m < NVFs; m++){
     vvel_FCME[m][*iter] = vvel_FCME[m][*iter] + b[m];
   }

    // I don't know what this is going to look like yet
    // but the equations are right I think.
//	uvel_FC[curcell][top] = 0.0;
//	vvel_FC[curcell][top] = term1- term2 + term3;
//	wvel_FC[curcell][top] = 0.0;

       // Right face
	adjcell = IntVector(curcell.x()+1,curcell.y(),curcell.z()); 

       // Front face
	adjcell = IntVector(curcell.x(),curcell.y(),curcell.z()+1); 
  }

  // Put Boundary condition stuff in here
  //
  //

  // Put the result in the datawarehouse
  for(int m = 0; m < NVFs; m++){
      new_dw->put(uvel_FCME[m], lb->uvel_FCMELabel, m, patch);
      new_dw->put(vvel_FCME[m], lb->vvel_FCMELabel, m, patch);
      new_dw->put(wvel_FCME[m], lb->wvel_FCMELabel, m, patch);
  }
}

void ICE::actuallyStep1e(const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{
  cout << "Doing actually step1e" << endl;

  int numMatls = d_sharedState->getNumMatls();
  Vector dx = patch->dCell();
  double  top, bottom, right, left, front, back;

  // Compute the divergence of the face centered velocities
  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    if(ice_matl){
      int vfindex = matl->getVFIndex();
      // Get required variables for this patch
      FCVariable<double> uvel_FC;
      FCVariable<double> vvel_FC;
      FCVariable<double> wvel_FC;
      CCVariable<double> vol_frac;
      new_dw->get(uvel_FC, lb->uvel_FCMELabel, vfindex, patch, Ghost::None, 0);
      new_dw->get(vvel_FC, lb->vvel_FCMELabel, vfindex, patch, Ghost::None, 0);
      new_dw->get(wvel_FC, lb->wvel_FCMELabel, vfindex, patch, Ghost::None, 0);
      new_dw->get(vol_frac,lb->vol_frac_CCLabel, vfindex,patch,Ghost::None, 0);

      // Create variables for the results
      CCVariable<double> div_velfc_CC;
      new_dw->allocate(div_velfc_CC, lb->div_velfc_CCLabel, vfindex, patch);

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
	IntVector curcell = *iter;
//	top      =  dx.x()*dx.z()* vvel_FC[*iter][TOP];
//	bottom   = -dx.x()*dx.z()* vvel_FC[*iter][BOTTOM];
//	left     = -dx.y()*dx.z()* uvel_FC[*iter][LEFT];
//	right    =  dx.y()*dx.z()* uvel_FC[*iter][RIGHT];
//	front    =  dx.x()*dx.y()* wvel_FC[*iter][FRONT];
//	back     = -dx.x()*dx.y()* wvel_FC[*iter][BACK];
	div_velfc_CC[*iter] = vol_frac[*iter]*
			     (top + bottom + left + right + front  + back );
      }

      new_dw->put(div_velfc_CC, lb->div_velfc_CCLabel, vfindex, patch);
    }
  }

}
void ICE::actuallyStep2(const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{
  cout << "Doing actually step2" << endl;
}



void ICE::actuallyStep3(const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{
  cout << "Doing actually step3" << endl;

}



void ICE::actuallyStep4(const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{
  cout << "Doing actually step4" << endl;
}


void ICE::actuallyStep5(const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{
  cout << "Doing actually step5" << endl;

}


void ICE::actuallyStep6and7(const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw)
{

  cout << "Doing actually step6 and 7" << endl;
  double dT = 0.0001;
  new_dw->put(delt_vartype(dT), lb->delTLabel);

  CCVariable<double> rho_micro, temp, cv, rho_CC;
  CCVariable<double> uvel_CC,vvel_CC,wvel_CC;
  for (int m = 0; m < d_sharedState->getNumMatls(); m++ ) {
    Material* matl = d_sharedState->getMaterial(m);
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    if(ice_matl){
      int vfindex = ice_matl->getDWIndex();
      new_dw->allocate(rho_micro,lb->rho_micro_CCLabel,vfindex,patch);
      new_dw->allocate(rho_CC,lb->rho_CCLabel,vfindex,patch);
      new_dw->allocate(temp,lb->temp_CCLabel,vfindex,patch);
      new_dw->allocate(cv,lb->cv_CCLabel,vfindex,patch);

      new_dw->allocate(uvel_CC,lb->uvel_CCLabel,vfindex,patch);
      new_dw->allocate(vvel_CC,lb->vvel_CCLabel,vfindex,patch);
      new_dw->allocate(wvel_CC,lb->wvel_CCLabel,vfindex,patch);

      new_dw->put(rho_micro,lb->rho_micro_CCLabel,vfindex,patch);
      new_dw->put(rho_CC,lb->rho_CCLabel,vfindex,patch);
      new_dw->put(temp,lb->temp_CCLabel,vfindex,patch);
      new_dw->put(cv,lb->cv_CCLabel,vfindex,patch);

      new_dw->put(uvel_CC,lb->uvel_CCLabel,vfindex,patch);
      new_dw->put(vvel_CC,lb->vvel_CCLabel,vfindex,patch);
      new_dw->put(wvel_CC,lb->wvel_CCLabel,vfindex,patch);
    }
  }
}

//
// $Log$
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
