// MPMICE.cc

#include <Packages/Uintah/CCA/Components/MPMICE/MPMICE.h>
#include <Packages/Uintah/CCA/Components/MPMICE/MPMICELabel.h>
#include <Packages/Uintah/CCA/Components/MPM/SerialMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/Burn/HEBurn.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMPhysicalModules.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>

#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>

#include <Core/Datatypes/DenseMatrix.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

MPMICE::MPMICE(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  Mlb  = scinew MPMLabel();
  Ilb  = scinew ICELabel();
  MIlb = scinew MPMICELabel();
  d_fracture = false;
  d_mpm      = scinew SerialMPM(myworld);
  d_ice      = scinew ICE(myworld);
}

MPMICE::~MPMICE()
{
  delete Mlb;
  delete Ilb;
  delete MIlb;
  delete d_mpm;
  delete d_ice;
}

void MPMICE::problemSetup(const ProblemSpecP& prob_spec, GridP& grid,
			  SimulationStateP& sharedState)
{
   d_sharedState = sharedState;

   d_mpm->setMPMLabel(Mlb);
   d_mpm->problemSetup(prob_spec, grid, d_sharedState);

   d_ice->setICELabel(Ilb);
   d_ice->problemSetup(prob_spec, grid, d_sharedState);

   cerr << "MPMICE::problemSetup passed.\n";
}

void MPMICE::scheduleInitialize(const LevelP& level,
				SchedulerP& sched,
				DataWarehouseP& dw)
{
  d_mpm->scheduleInitialize(level, sched, dw);
  d_ice->scheduleInitialize(level, sched, dw);
}

void MPMICE::scheduleComputeStableTimestep(const LevelP& level,
					   SchedulerP& sched,
					   DataWarehouseP& dw)
{
  // Schedule computing the ICE stable timestep
  d_ice->scheduleComputeStableTimestep(level, sched, dw);
  // MPM stable timestep is a by product of the CM
}

void MPMICE::scheduleFixVolFrac(const Patch* patch,
				SchedulerP& sched,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw)
{
  int numMPMMatls = d_sharedState->getNumMPMMatls();
  int numICEMatls = d_sharedState->getNumICEMatls();

  Task* t = scinew Task("MPMICE::actuallyFixVolFrac",patch, old_dw, new_dw,this,
                          &MPMICE::actuallyFixVolFrac);

  for (int m = 0; m < d_sharedState->getNumMPMMatls(); m++ ) {
      MPMMaterial*  matl = d_sharedState->getMPMMaterial(m);
      int dwindex = matl->getDWIndex();

      t->requires(new_dw, Mlb->gVolumeLabel,    dwindex, patch,
						Ghost::AroundCells, 1);
  }

  for (int m = 0; m < d_sharedState->getNumICEMatls(); m++ ) {
      ICEMaterial*  matl = d_sharedState->getICEMaterial(m);
      int dwindex = matl->getDWIndex();

      t->requires(old_dw, Ilb->rho_micro_CCLabel, dwindex, patch,
						Ghost::AroundCells, 1);
      t->computes(new_dw, Ilb->rho_CCLabel,       dwindex, patch);
  }

  sched->addTask(t);

}

void MPMICE::scheduleTimeAdvance(double t, double dt,
				 const LevelP&         level,
				 SchedulerP&     sched,
				 DataWarehouseP& old_dw, 
				 DataWarehouseP& new_dw)
{
   int numMPMMatls = d_sharedState->getNumMPMMatls();
   for(Level::const_patchIterator iter=level->patchesBegin();
       iter != level->patchesEnd(); iter++){

    const Patch* patch=*iter;
    if(d_fracture) {
       d_mpm->scheduleComputeNodeVisibility(patch,sched,old_dw,new_dw);
    }
    d_mpm->scheduleInterpolateParticlesToGrid(patch,sched,old_dw,new_dw);

//    if(t == 0.){
//      scheduleFixVolFrac(patch,sched,old_dw,new_dw);
//    }

    if (MPMPhysicalModules::thermalContactModel) {
       d_mpm->scheduleComputeHeatExchange(patch,sched,old_dw,new_dw);
    }

    d_mpm->scheduleExMomInterpolated(patch,sched,old_dw,new_dw);
    d_mpm->scheduleComputeStressTensor(patch,sched,old_dw,new_dw);

//    d_ice->scheduleComputeEquilibrationPressure(    patch,sched,old_dw,new_dw);
    // schedule the interpolation of mass and volume to the cell centers
    scheduleInterpolateNCToCC_0(patch,sched,old_dw,new_dw);
    scheduleComputeEquilibrationPressure(    patch,sched,old_dw,new_dw);

    d_ice->scheduleComputeFaceCenteredVelocities(   patch,sched,old_dw,new_dw);
    d_ice->scheduleAddExchangeContributionToFCVel(  patch,sched,old_dw,new_dw);
    d_ice->scheduleComputeDelPressAndUpdatePressCC( patch,sched,old_dw,new_dw);

    scheduleInterpolatePressureToParticles(         patch,sched,old_dw,new_dw);

    d_mpm->scheduleComputeInternalForce(patch,sched,old_dw,new_dw);
    d_mpm->scheduleComputeInternalHeatRate(patch,sched,old_dw,new_dw);
    d_mpm->scheduleSolveEquationsMotion(patch,sched,old_dw,new_dw);
    d_mpm->scheduleSolveHeatEquations(patch,sched,old_dw,new_dw);
    d_mpm->scheduleIntegrateAcceleration(patch,sched,old_dw,new_dw);
    d_mpm->scheduleIntegrateTemperatureRate(patch,sched,old_dw,new_dw);

    d_ice->scheduleComputePressFC(                  patch,sched,old_dw,new_dw);
    d_ice->scheduleAccumulateMomentumSourceSinks(   patch,sched,old_dw,new_dw);
    d_ice->scheduleAccumulateEnergySourceSinks(     patch,sched,old_dw,new_dw);
    d_ice->scheduleComputeLagrangianValues(         patch,sched,old_dw,new_dw);

    scheduleInterpolateNCToCC(patch,sched,old_dw,new_dw);

    // Either do this one
    scheduleCCMomExchange(patch,sched,old_dw,new_dw);
    // OR these
//    d_mpm->scheduleExMomIntegrated(patch,sched,old_dw,new_dw);
//    d_ice->scheduleAddExchangeToMomentumAndEnergy(patch,sched,old_dw,new_dw);

    d_mpm->scheduleInterpolateToParticlesAndUpdate(patch,sched,old_dw,new_dw);
    d_mpm->scheduleComputeMassRate(patch,sched,old_dw,new_dw);
    if(d_fracture) {
      d_mpm->scheduleCrackGrow(patch,sched,old_dw,new_dw);
      d_mpm->scheduleStressRelease(patch,sched,old_dw,new_dw);
      d_mpm->scheduleComputeCrackSurfaceContactForce(patch,sched,old_dw,new_dw);
    }

    d_mpm->scheduleCarryForwardVariables(patch,sched,old_dw,new_dw);

    // Step 6and7 advect and advance in time
    d_ice->scheduleAdvectAndAdvanceInTime( patch,sched,old_dw,new_dw);

  }

    
   sched->scheduleParticleRelocation(level, old_dw, new_dw,
				     Mlb->pXLabel_preReloc, 
				     Mlb->d_particleState_preReloc,
				     Mlb->pXLabel, Mlb->d_particleState,
				     numMPMMatls);
}


void MPMICE::scheduleInterpolatePressureToParticles(const Patch* patch,
                                                SchedulerP& sched,
                                                DataWarehouseP& old_dw,
                                                DataWarehouseP& new_dw)
{

   cout << "scheduleInterpolatePressureToParticles" << endl;
   int numMPMMatls = d_sharedState->getNumMPMMatls();
   Task* t=scinew Task("MPMICE::interpolatePressureToParticles",
		        patch, old_dw, new_dw,
		        this, &MPMICE::interpolatePressureToParticles);

   t->requires(new_dw,Ilb->press_CCLabel,0, patch, Ghost::AroundCells, 1);

   for(int m = 0; m < numMPMMatls; m++){
     MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
     int idx = mpm_matl->getDWIndex();
     t->requires(old_dw, Mlb->pXLabel,        idx, patch, Ghost::None);
     t->computes(new_dw, Mlb->pPressureLabel, idx, patch);
	cout << "matlindex computes = " << idx << endl;
   }

   sched->addTask(t);

}

void MPMICE::scheduleInterpolateNCToCC_0(const Patch* patch,
                                       SchedulerP& sched,
                                       DataWarehouseP& old_dw,
                                       DataWarehouseP& new_dw)
{
   /* interpolateNCToCC */

   cout << "scheduleInterpolateNCToCC_0" << endl;
   int numMPMMatls = d_sharedState->getNumMPMMatls();
   Task* t=scinew Task("MPMICE::interpolateNCToCC_0",
		        patch, old_dw, new_dw,
		        this, &MPMICE::interpolateNCToCC_0);

   for(int m = 0; m < numMPMMatls; m++){
     MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
     int idx = mpm_matl->getDWIndex();
     t->requires(new_dw, Mlb->gMassLabel,         idx, patch,
		Ghost::AroundCells, 1);
     t->requires(new_dw, Mlb->gVolumeLabel,       idx, patch,
		Ghost::AroundCells, 1);
     t->requires(new_dw, Mlb->gVelocityLabel,     idx, patch,
		Ghost::AroundCells, 1);

     t->computes(new_dw, MIlb->cMassLabel,         idx, patch);
     t->computes(new_dw, MIlb->cVolumeLabel,       idx, patch);
     t->computes(new_dw, MIlb->rho_CCLabel,        idx, patch);
     t->computes(new_dw, MIlb->vel_CCLabel,        idx, patch);
//     cout << "schedule cmopute for vel_CC on matl = " << idx << endl;
   }

   sched->addTask(t);

}

void MPMICE::scheduleInterpolateNCToCC(const Patch* patch,
                                       SchedulerP& sched,
                                       DataWarehouseP& old_dw,
                                       DataWarehouseP& new_dw)
{
   cout << "scheduleInterpolateNCToCC" << endl;
   /* interpolateNCToCC */

   int numMPMMatls = d_sharedState->getNumMPMMatls();
   Task* t=scinew Task("MPMICE::interpolateNCToCC",
		        patch, old_dw, new_dw,
		        this, &MPMICE::interpolateNCToCC);

   for(int m = 0; m < numMPMMatls; m++){
     MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
     int idx = mpm_matl->getDWIndex();
     t->requires(new_dw, Mlb->gVelocityStarLabel, idx, patch,
		Ghost::AroundCells, 1);
     t->requires(new_dw, Mlb->gAccelerationLabel, idx, patch,
		Ghost::AroundCells, 1);
     t->requires(new_dw, Mlb->gMassLabel,         idx, patch,
		Ghost::AroundCells, 1);
     t->requires(new_dw, MIlb->mom_source_CCLabel,idx, patch,
		Ghost::AroundCells, 1);

     t->computes(new_dw, MIlb->mom_L_CCLabel, idx, patch);
   }

   sched->addTask(t);

}

void MPMICE::scheduleCCMomExchange(const Patch* patch,
                                   SchedulerP& sched,
                                   DataWarehouseP& old_dw,
                                   DataWarehouseP& new_dw)
{
   cout << "scheduleCCMomExchange" << endl;
   Task* t=scinew Task("MPMICE::doCCMomExchange",
		        patch, old_dw, new_dw,
		        this, &MPMICE::doCCMomExchange);

   for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
     MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
     int mpmidx = mpm_matl->getDWIndex();
     t->requires(new_dw, MIlb->mom_L_CCLabel, mpmidx, patch, Ghost::None, 0);
     t->requires(new_dw, MIlb->rho_CCLabel,   mpmidx, patch, Ghost::None, 0);
     t->requires(new_dw, Mlb->gVelocityStarLabel,
					      mpmidx, patch, Ghost::None, 0);
     t->requires(new_dw, Mlb->gAccelerationLabel,
					      mpmidx, patch, Ghost::None, 0);
     t->requires(new_dw, MIlb->rho_micro_CCLabel,
					      mpmidx, patch, Ghost::None, 0);
     t->requires(new_dw, MIlb->mom_source_CCLabel,
				       mpmidx, patch, Ghost::AroundCells, 1);

     t->computes(new_dw, Mlb->gMomExedVelocityStarLabel, mpmidx, patch);
     t->computes(new_dw, Mlb->gMomExedAccelerationLabel, mpmidx, patch);
   }

  for (int m = 0; m < d_sharedState->getNumICEMatls(); m++) {
    ICEMaterial* matl = d_sharedState->getICEMaterial(m);
    int iceidx = matl->getDWIndex();
    t->requires(old_dw,Ilb->rho_CCLabel,             iceidx,patch,Ghost::None);
    t->requires(new_dw,Ilb->mom_L_CCLabel,           iceidx,patch,Ghost::None);
    t->requires(new_dw,Ilb->int_eng_L_CCLabel,       iceidx,patch,Ghost::None);
    t->requires(new_dw,Ilb->vol_frac_CCLabel,        iceidx,patch,Ghost::None);
    t->requires(old_dw,Ilb->cv_CCLabel,              iceidx,patch,Ghost::None);
    t->requires(new_dw,Ilb->rho_micro_CCLabel,       iceidx,patch,Ghost::None);

    t->computes(new_dw,Ilb->mom_L_ME_CCLabel,        iceidx, patch);
    t->computes(new_dw,Ilb->int_eng_L_ME_CCLabel,    iceidx, patch);
  }

   sched->addTask(t);

}

void MPMICE::scheduleComputeEquilibrationPressure(const Patch* patch,
						  SchedulerP& sched,
						  DataWarehouseP& old_dw,
						  DataWarehouseP& new_dw)
{
   cout << "scheduleComputeEquilibrationPressure" << endl;
  Task* task = scinew Task("MPMICE::computeEquilibrationPressure",
                        patch, old_dw, new_dw,this,
			   &MPMICE::computeEquilibrationPressure);
  
  task->requires(old_dw,Ilb->press_CCLabel, 0,patch,Ghost::None);
  
  int numICEMatls=d_sharedState->getNumICEMatls();
  for (int m = 0; m < numICEMatls; m++)  {
    ICEMaterial*  matl = d_sharedState->getICEMaterial(m);
    int dwindex = matl->getDWIndex();

    task->requires(old_dw,Ilb->rho_CCLabel,       dwindex,patch,Ghost::None);
    task->requires(old_dw,Ilb->temp_CCLabel,      dwindex,patch,Ghost::None);
    task->requires(old_dw,Ilb->cv_CCLabel,        dwindex,patch,Ghost::None);

    task->computes(new_dw,Ilb->speedSound_CCLabel,dwindex, patch);
    task->computes(new_dw,Ilb->vol_frac_CCLabel,  dwindex, patch);
    task->computes(new_dw,Ilb->rho_micro_CCLabel, dwindex, patch);
  }

  for (int m = 0; m < d_sharedState->getNumMPMMatls(); m++)  {
    MPMMaterial* matl = d_sharedState->getMPMMaterial(m);
    int dwindex = matl->getDWIndex();
    task->requires(new_dw, MIlb->rho_CCLabel,        dwindex,patch,Ghost::None);

    task->computes(new_dw, MIlb->rho_micro_CCLabel,  dwindex,patch);
    task->computes(new_dw, MIlb->vol_frac_CCLabel,   dwindex,patch);
    task->computes(new_dw, MIlb->speedSound_CCLabel, dwindex,patch);
  }

  task->computes(new_dw,Ilb->press_equil_CCLabel,0, patch);
  sched->addTask(task);
}

void MPMICE::actuallyFixVolFrac(const ProcessorGroup*,
                                const Patch* patch,
                                DataWarehouseP&,
                                DataWarehouseP& new_dw)
{
  int numMPMMatls = d_sharedState->getNumMPMMatls();
  int numICEMatls = d_sharedState->getNumICEMatls();
  int numALLMatls = numICEMatls + numMPMMatls;
  Vector zero(0.,0.,0.);
  Vector dx = patch->dCell();
  double vol = dx.x()*dx.y()*dx.z();

  vector<CCVariable<double> > rho_CC(numALLMatls),rho_micro_CC(numALLMatls);

  for (int m = 0; m < numALLMatls; m++) {
    Material* matl = d_sharedState->getMaterial( m );
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    if(ice_matl){
      int dwindex = ice_matl->getDWIndex();

      new_dw->allocate(rho_CC[m],  Ilb->rho_CCLabel, dwindex, patch);

      new_dw->get(rho_micro_CC[m], Ilb->rho_micro_CCLabel, dwindex, patch,
							Ghost::None, 0);
    }
  }

  for(int m = 0; m < numALLMatls; m++){
     Material* matl = d_sharedState->getMaterial( m );
     MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
     int matlindex = mpm_matl->getDWIndex();

     // Create arrays for the grid data
     NCVariable<double> gvolume;

     new_dw->get(gvolume,   Mlb->gVolumeLabel,         matlindex, patch,
							Ghost::AroundCells, 1);

     IntVector nodeIdx[8];

     for(CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
       patch->findNodesFromCell(*iter,nodeIdx);
       double cvolume   = 0.;
       for (int in=0;in<8;in++){
	 cvolume   += .125*gvolume[nodeIdx[in]];
       }
       double vf0 = cvolume/vol;
       rho_CC[1][*iter] = (1.0 - vf0)*rho_micro_CC[1][*iter];
     }

  }

  for (int m = 0; m < numALLMatls; m++) {
    Material* matl = d_sharedState->getMaterial( m );
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    if(ice_matl){
      int dwindex = ice_matl->getDWIndex();

      new_dw->put(rho_CC[m], Ilb->rho_CCLabel, dwindex, patch);
    }
  }
}

void MPMICE::interpolatePressureToParticles(const ProcessorGroup*,
                                            const Patch* patch,
                                            DataWarehouseP& old_dw,
                                            DataWarehouseP& new_dw)
{

  cout << "Doing interpolatePressureToParticles" << endl;
  
  CCVariable<double> pressCC;
  NCVariable<double> pressNC;
  IntVector ni[8];
  double S[8];

  new_dw->get(pressCC,       Ilb->press_CCLabel, 0, patch,Ghost::None, 0);
  new_dw->allocate(pressNC, MIlb->press_NCLabel, 0, patch);

  IntVector cIdx[8];
  for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
     patch->findCellsFromNode(*iter,cIdx);
     pressNC[*iter] = 0.0;
     for (int in=0;in<8;in++){
	pressNC[*iter]  += .125*pressCC[cIdx[in]];
     }
  }

  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    int dwindex = mpm_matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwindex, patch);
    ParticleVariable<double> pPressure;
    ParticleVariable<Point> px;

    new_dw->allocate(pPressure, Mlb->pPressureLabel, pset);
    old_dw->get(px,             Mlb->pXLabel,         pset);

    for(ParticleSubset::iterator iter = pset->begin();
				          iter != pset->end(); iter++){
	particleIndex idx = *iter;
	double press = 0.;

	// Get the node indices that surround the cell
	patch->findCellAndWeights(px[idx], ni, S);
	for (int k = 0; k < 8; k++) {
	    press += pressNC[ni[k]] * S[k];
	}
	pPressure[idx] = press;
    }

    new_dw->put(pPressure,  Mlb->pPressureLabel);

  }
}

void MPMICE::interpolateNCToCC_0(const ProcessorGroup*,
                                 const Patch* patch,
                                 DataWarehouseP&,
                                 DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  Vector zero(0.,0.,0.);
  Vector dx = patch->dCell();
  double vol = dx.x()*dx.y()*dx.z();
  double d_SMALL_NUM = 1.e-100;       // TEMPORARY THIS SHOULD BE PRIVATE DATA

  cout << "Doing interpolateNCToCC_0 " << endl;

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    int matlindex = mpm_matl->getDWIndex();

     // Create arrays for the grid data
     NCVariable<double> gmass, gvolume;
     NCVariable<Vector> gvelocity;
     CCVariable<double> cmass, cvolume, rho_CC;
     CCVariable<Vector> vel_CC;

     new_dw->allocate(cmass,     MIlb->cMassLabel,         matlindex, patch);
     new_dw->allocate(cvolume,   MIlb->cVolumeLabel,       matlindex, patch);
     new_dw->allocate(rho_CC,    MIlb->rho_CCLabel,        matlindex, patch);
     new_dw->allocate(vel_CC,    MIlb->vel_CCLabel,        matlindex, patch);
 
     cmass.initialize(0.);
     cvolume.initialize(0.);
     vel_CC.initialize(zero);

     new_dw->get(gmass,     Mlb->gMassLabel,           matlindex, patch,
							Ghost::AroundCells, 1);
     new_dw->get(gvolume,   Mlb->gVolumeLabel,         matlindex, patch,
							Ghost::AroundCells, 1);
     new_dw->get(gvelocity, Mlb->gVelocityLabel,       matlindex, patch,
							Ghost::AroundCells, 1);

     IntVector nodeIdx[8];

     for(CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
       patch->findNodesFromCell(*iter,nodeIdx);
       for (int in=0;in<8;in++){
	 cmass[*iter]    += .125*gmass[nodeIdx[in]];
	 cvolume[*iter]  += .125*gvolume[nodeIdx[in]];
	 vel_CC[*iter]   +=      gvelocity[nodeIdx[in]]*.125*gmass[nodeIdx[in]];
       }
       rho_CC[*iter] = cmass[*iter]/vol;
       vel_CC[*iter]   /= (cmass[*iter] + d_SMALL_NUM);
     }

     new_dw->put(cmass,     MIlb->cMassLabel,         matlindex, patch);
     new_dw->put(cvolume,   MIlb->cVolumeLabel,       matlindex, patch);
     new_dw->put(rho_CC,    MIlb->rho_CCLabel,        matlindex, patch);
     new_dw->put(vel_CC,    MIlb->vel_CCLabel,        matlindex, patch);
  }
}

void MPMICE::interpolateNCToCC(const ProcessorGroup*,
                               const Patch* patch,
                               DataWarehouseP&,
                               DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  Vector zero(0.,0.,0.);

  cout << "Doing interpolateNCToCC " << endl;

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    int matlindex = mpm_matl->getDWIndex();

     // Create arrays for the grid data
     NCVariable<double> gmass, gvolume;
     NCVariable<Vector> gvelocity, gacc;
     CCVariable<Vector> cmomentum, mom_source;

     new_dw->get(gmass,     Mlb->gMassLabel,           matlindex, patch,
							Ghost::AroundCells, 1);
     new_dw->get(gvelocity, Mlb->gVelocityStarLabel,   matlindex, patch,
							Ghost::AroundCells, 1);
     new_dw->get(gacc,      Mlb->gAccelerationLabel,   matlindex, patch,
							Ghost::AroundCells, 1);
     new_dw->get(mom_source,MIlb->mom_source_CCLabel,  matlindex, patch,
							Ghost::AroundCells, 1);

     new_dw->allocate(cmomentum, MIlb->mom_L_CCLabel, matlindex, patch);
 
     cmomentum.initialize(zero);

     IntVector nodeIdx[8];

     for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
       patch->findNodesFromCell(*iter,nodeIdx);
       for (int in=0;in<8;in++){
 	 cmomentum[*iter] += gvelocity[nodeIdx[in]]*gmass[nodeIdx[in]]*.125;
       }
       cmomentum[*iter] += mom_source[*iter];
     }

     new_dw->put(cmomentum, MIlb->mom_L_CCLabel, matlindex, patch);
  }
}

void MPMICE::doCCMomExchange(const ProcessorGroup*,
                             const Patch* patch,
                             DataWarehouseP& old_dw,
                             DataWarehouseP& new_dw)
{

  cout << "Doing CCMomEx" << endl;
  int numMPMMatls = d_sharedState->getNumMPMMatls();
  int numICEMatls = d_sharedState->getNumICEMatls();
  int numALLMatls = numMPMMatls + numICEMatls;

  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label());
  Vector dx = patch->dCell();
  Vector gravity = d_sharedState->getGravity();
  Vector zero(0.,0.,0.);

  // Create arrays for the grid data
  vector<NCVariable<Vector> > gacceleration(numALLMatls);
  vector<NCVariable<Vector> > gvelocity(numALLMatls);
  vector<NCVariable<Vector> > gMEacceleration(numALLMatls);
  vector<NCVariable<Vector> > gMEvelocity(numALLMatls);

  vector<CCVariable<double> > rho_CC(numALLMatls);
  vector<CCVariable<double> > vol_frac_CC(numALLMatls);

  vector<CCVariable<double> > rho_micro_CC(numALLMatls);
  vector<CCVariable<double> > cv_CC(numALLMatls);

  vector<CCVariable<Vector> > mom_L(numALLMatls);
  vector<CCVariable<double> > int_eng_L(numALLMatls);
  // mom_source will contain the pressure and gravity force from ICE
  vector<CCVariable<Vector> > mom_source(numALLMatls);
  vector<CCVariable<double> > cmass(numALLMatls);

  // Create variables for the results
  vector<CCVariable<Vector> > mom_L_ME(numALLMatls);
  vector<CCVariable<Vector> > vel_CC(numALLMatls);
  vector<CCVariable<Vector> > dvdt_CC(numALLMatls);
  vector<CCVariable<double> > int_eng_L_ME(numALLMatls);

  vector<double> b(numALLMatls);
  vector<double> mass(numALLMatls);
  vector<double> density(numALLMatls);
  DenseMatrix beta(numALLMatls,numALLMatls),acopy(numALLMatls,numALLMatls);
  DenseMatrix K(numALLMatls,numALLMatls),H(numALLMatls,numALLMatls);
  DenseMatrix a(numALLMatls,numALLMatls);

//  for (int i = 0; i < numALLMatls; i++ ) {
//      K[numICEMatls-1-i][i] = d_K_mom[i];
//      H[numICEMatls-1-i][i] = d_K_heat[i];
//  }

  // Hardwiring the values for the momentum exchange for now
  K[0][0] = 0.;
  K[0][1] = 1.e6;
  K[1][0] = 1.e6;
//  K[0][1] = 0.;
//  K[1][0] = 0.;
  K[1][1] = 0.;

  for(int m = 0; m < numALLMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int matlindex = mpm_matl->getDWIndex();

      new_dw->get(gvelocity[m],     Mlb->gVelocityStarLabel, matlindex, patch,
							Ghost::None, 0);
      new_dw->get(gacceleration[m], Mlb->gAccelerationLabel, matlindex, patch,
							Ghost::None, 0);
      new_dw->allocate(gMEvelocity[m],     Mlb->gMomExedVelocityStarLabel,
							 matlindex, patch);
      new_dw->allocate(gMEacceleration[m], Mlb->gMomExedAccelerationLabel,
							 matlindex, patch);

      new_dw->get(mom_L[m],      MIlb->mom_L_CCLabel,       matlindex, patch,
							Ghost::None, 0);
      new_dw->get(rho_CC[m],     MIlb->rho_CCLabel,         matlindex, patch,
							Ghost::None, 0);
      new_dw->get(cmass[m],      MIlb->cMassLabel,          matlindex, patch,
							Ghost::None, 0);
      new_dw->get(rho_micro_CC[m], MIlb->rho_micro_CCLabel, matlindex, patch,
							Ghost::None, 0);
      new_dw->get(vol_frac_CC[m],  MIlb->vol_frac_CCLabel,  matlindex, patch,
							Ghost::None, 0);
      new_dw->get(mom_source[m],   MIlb->mom_source_CCLabel,matlindex, patch,
							Ghost::AroundCells, 1);
      new_dw->allocate(vel_CC[m],  MIlb->velstar_CCLabel,   matlindex, patch);
      new_dw->allocate(dvdt_CC[m], MIlb->dvdt_CCLabel,      matlindex, patch);
      dvdt_CC[m].initialize(zero);
    }
  }

  for(int m = 0; m < numALLMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    if(ice_matl){
      int dwindex = ice_matl->getDWIndex();
      old_dw->get(rho_CC[m],      Ilb->rho_CCLabel,
                                dwindex, patch, Ghost::None, 0);
      new_dw->get(mom_L[m],       Ilb->mom_L_CCLabel,
                                dwindex, patch, Ghost::None, 0);
      new_dw->get(int_eng_L[m],    Ilb->int_eng_L_CCLabel,
                                dwindex, patch, Ghost::None, 0);
      new_dw->get(vol_frac_CC[m],  Ilb->vol_frac_CCLabel,
                                dwindex, patch, Ghost::None, 0);
      new_dw->get(rho_micro_CC[m], Ilb->rho_micro_CCLabel,
                                dwindex, patch, Ghost::None, 0);
      old_dw->get(cv_CC[m],        Ilb->cv_CCLabel,
                                dwindex, patch, Ghost::None, 0);

      new_dw->allocate(mom_L_ME[m],    Ilb->mom_L_ME_CCLabel,    dwindex,patch);
      new_dw->allocate(vel_CC[m],      Ilb->vel_CCLabel,         dwindex,patch);
      new_dw->allocate(int_eng_L_ME[m],Ilb->int_eng_L_ME_CCLabel,dwindex,patch);
      new_dw->allocate(dvdt_CC[m],     MIlb->dvdt_CCLabel,       dwindex,patch);
      dvdt_CC[m].initialize(zero);
    }
  }

  double vol = dx.x()*dx.y()*dx.z();
  double SMALL_NUM = 1.e-80;
  int itworked;

  // Convert momenta to velocities.  Slightly different for MPM and ICE.
  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
    for (int m = 0; m < numALLMatls; m++) {
      mass[m]     = rho_CC[m][*iter] * vol + SMALL_NUM;
      vel_CC[m][*iter]  =  mom_L[m][*iter]/mass[m];
    }
  }

  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
    //   Form BETA matrix (a), off diagonal terms
    //  The beta and (a) matrix is common to all momentum exchanges
    for(int m = 0; m < numALLMatls; m++)  {
      density[m]  = rho_micro_CC[m][*iter];
      for(int n = 0; n < numALLMatls; n++) {
	beta[m][n] = delT * vol_frac_CC[n][*iter] * K[n][m]/density[m];
	a[m][n] = -beta[m][n];
      }
    }
    //   Form matrix (a) diagonal terms
    for(int m = 0; m < numALLMatls; m++) {
      a[m][m] = 1.;
      for(int n = 0; n < numALLMatls; n++) {
	a[m][m] +=  beta[m][n];
      }
    }

    //     X - M O M E N T U M  --  F O R M   R H S   (b)
    for(int m = 0; m < numALLMatls; m++) {
      b[m] = 0.0;
      for(int n = 0; n < numALLMatls; n++) {
	b[m] += beta[m][n] * (vel_CC[n][*iter].x() - vel_CC[m][*iter].x());
      }
    }
    //     S O L V E
    //  - Add exchange contribution to orig value
    acopy = a;
    itworked = acopy.solve(b);
    for(int m = 0; m < numALLMatls; m++) {
	vel_CC[m][*iter].x( vel_CC[m][*iter].x() + b[m] );
	dvdt_CC[m][*iter].x( b[m] );
    }

    //     Y - M O M E N T U M  --   F O R M   R H S   (b)
    for(int m = 0; m < numALLMatls; m++) {
      b[m] = 0.0;
      for(int n = 0; n < numALLMatls; n++) {
	b[m] += beta[m][n] * (vel_CC[n][*iter].y() - vel_CC[m][*iter].y());
      }
    }

    //     S O L V E
    //  - Add exchange contribution to orig value
    acopy    = a;
    itworked = acopy.solve(b);
    for(int m = 0; m < numALLMatls; m++)   {
	vel_CC[m][*iter].y( vel_CC[m][*iter].y() + b[m] );
	dvdt_CC[m][*iter].y( b[m] );
    }

    //     Z - M O M E N T U M  --  F O R M   R H S   (b)
    for(int m = 0; m < numALLMatls; m++)  {
      b[m] = 0.0;
      for(int n = 0; n < numALLMatls; n++) {
	b[m] += beta[m][n] * (vel_CC[n][*iter].z() - vel_CC[m][*iter].z());
      }
    }    

    //     S O L V E
    //  - Add exchange contribution to orig value
    acopy    = a;
    itworked = acopy.solve(b);
    for(int m = 0; m < numALLMatls; m++)  {
	vel_CC[m][*iter].z( vel_CC[m][*iter].z() + b[m] );
	dvdt_CC[m][*iter].z( b[m] );
    }

    //---------- E N E R G Y   E X C H A N G E
    //  THIS IS NOT IMPLEMENTED YET, CURRENTLY JUST CARRYING FORWARD FOR ICE
    for(int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      if(ice_matl){
        int_eng_L_ME[m][*iter] = int_eng_L[m][*iter];
      }
    }
  }

  //__________________________________
  //  Set the Boundary condiitions
  for (int m = 0; m < numALLMatls; m++)  {
    Material* matl = d_sharedState->getMaterial( m );
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    if(ice_matl){
      d_ice->setBC(vel_CC[m],"Velocity",patch);
    }
  }
  //__________________________________
  // Convert vars. primitive-> flux  for ICE matls only
  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      if(ice_matl){
        mass[m] = rho_CC[m][*iter] * vol;
        mom_L_ME[m][*iter] = vel_CC[m][*iter] * mass[m];
      }
    }
  }

  // put ONLY the ICE Materials' data in the new_dw
  for(int m = 0; m < numALLMatls; m++){
     Material* matl = d_sharedState->getMaterial( m );
     ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
     if(ice_matl){
       int dwindex = ice_matl->getDWIndex();
       new_dw->put(mom_L_ME[m],    Ilb->mom_L_ME_CCLabel,    dwindex, patch);
       new_dw->put(int_eng_L_ME[m],Ilb->int_eng_L_ME_CCLabel,dwindex, patch);
     }
  }

  // This is where I interpolate the CC changes to NCs for the MPMMatls
  IntVector cIdx[8];

  for(int m = 0; m < numALLMatls; m++){
     Material* matl = d_sharedState->getMaterial( m );
     MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
     if(mpm_matl){
       for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
         patch->findCellsFromNode(*iter,cIdx);
	 gMEvelocity[m][*iter]     = gvelocity[m][*iter];
	 gMEacceleration[m][*iter] = gacceleration[m][*iter];
	 for (int in=0;in<8;in++){
	   gMEvelocity[m][*iter]     +=  dvdt_CC[m][cIdx[in]]*delT*.125;
	   gMEacceleration[m][*iter] += (dvdt_CC[m][cIdx[in]] + 
		   mom_source[m][cIdx[in]]/(cmass[m][cIdx[in]]*delT))*.125;
         }
       }
     }
  }

  for(int m = 0; m < numALLMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int matlindex = mpm_matl->getDWIndex();
      new_dw->put(gMEvelocity[m],
			Mlb->gMomExedVelocityStarLabel,matlindex,patch);
      new_dw->put(gMEacceleration[m],
			Mlb->gMomExedAccelerationLabel,matlindex,patch);
    }
  }

}

void MPMICE::computeEquilibrationPressure(const ProcessorGroup*,
				       const Patch* patch,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw)
{
  double    converg_coeff = 10;              
//  double    convergence_crit = converg_coeff * DBL_EPSILON;
  double convergence_crit = 1e-7;
  double    sum, tmp;

  static int timestep=0;

  int numICEMatls = d_sharedState->getNumICEMatls();
  int numMPMMatls = d_sharedState->getNumMPMMatls();
  int numALLMatls = numICEMatls + numMPMMatls;
  double d_SMALL_NUM = 1.e-100;       // TEMPORARY THIS SHOULD BE PRIVATE DATA
  vector<CCVariable<double> > vol_frac(numALLMatls);
  char warning[100];
  char description[50];
  static int n_passes;                  
  n_passes ++; 
  cout << "Doing calc_equilibration_pressure in MPMICE" << endl;
  cout << timestep << endl;
 
  vector<double> delVol_frac(numALLMatls),press_eos(numALLMatls);
  vector<double> dp_drho(numALLMatls),dp_de(numALLMatls);
  
  vector<CCVariable<double> > rho_micro(numALLMatls);
  vector<CCVariable<double> > rho_CC(numALLMatls);
  vector<CCVariable<double> > cv(numALLMatls);
  vector<CCVariable<double> > Temp(numALLMatls),sS_new(numALLMatls);
  vector<CCVariable<double> > speedSound(numALLMatls);
  CCVariable<double> press,press_new;
  
  old_dw->get(press,         Ilb->press_CCLabel, 0,patch,Ghost::None, 0); 
  new_dw->allocate(press_new,Ilb->press_CCLabel, 0,patch);


  for (int m = 0; m < numALLMatls; m++) {
    Material* matl = d_sharedState->getMaterial( m );
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    if(ice_matl){
      int dwindex = ice_matl->getDWIndex();
      old_dw->get(cv[m],      Ilb->cv_CCLabel,  dwindex, patch, Ghost::None, 0);
      old_dw->get(rho_CC[m],  Ilb->rho_CCLabel, dwindex, patch, Ghost::None, 0);
      old_dw->get(Temp[m],    Ilb->temp_CCLabel,dwindex, patch, Ghost::None, 0);              
      new_dw->allocate(sS_new[m],    Ilb->speedSound_CCLabel,dwindex, patch);
      new_dw->allocate(rho_micro[m], Ilb->rho_micro_CCLabel, dwindex, patch);
      new_dw->allocate(vol_frac[m],  Ilb->vol_frac_CCLabel,  dwindex, patch); 
   }
  }

  for (int m = 0; m < numALLMatls; m++) {
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int dwindex = mpm_matl->getDWIndex();
      new_dw->get(rho_CC[m],  MIlb->cMassLabel, dwindex, patch, Ghost::None, 0);

      new_dw->allocate(sS_new[m],    MIlb->speedSound_CCLabel,dwindex, patch);
      new_dw->allocate(rho_micro[m], MIlb->rho_micro_CCLabel, dwindex, patch);
      new_dw->allocate(vol_frac[m],  MIlb->vol_frac_CCLabel,  dwindex, patch); 
   }
  }

  // Fix up the MPM quantities so that they are what we really need
  Vector dx = patch->dCell();
  double vol = dx.x()*dx.y()*dx.z();
  for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++) {
     for (int m = 0; m < numALLMatls; m++) {
       Material* matl = d_sharedState->getMaterial( m );
       MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
       if(mpm_matl){
	 rho_CC[m][*iter] = rho_CC[m][*iter]/vol + d_SMALL_NUM;
       }
     }
  }

  press_new = press;

  //__________________________________
  // Compute rho_micro, speedSound, and volfrac
  // First for ICE Matls
  for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++) {
     for (int m = 0; m < numALLMatls; m++) {
       Material* matl = d_sharedState->getMaterial( m );
       ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
       if(ice_matl){
         double gamma = ice_matl->getGamma();
                     
          rho_micro[m][*iter] =  ice_matl->getEOS()->computeRhoMicro(
                                            press_new[*iter],gamma,
					         cv[m][*iter],Temp[m][*iter]); 
                                                                     
          ice_matl->getEOS()->computePressEOS(rho_micro[m][*iter],gamma,
                                            cv[m][*iter],Temp[m][*iter],
                                            press_eos[m],dp_drho[m], dp_de[m]);
                                            
          tmp = dp_drho[m] + dp_de[m] * 
                    (press_eos[m]/(rho_micro[m][*iter]*rho_micro[m][*iter]));
          sS_new[m][*iter] = sqrt(tmp);
        
          vol_frac[m][*iter] = rho_CC[m][*iter]/rho_micro[m][*iter];
//	cout << "ice_vf = " << *iter << " " << vol_frac[m][*iter] << endl;
       }
     }
  }

  // Next for MPM Matls
  for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++) {
     for (int m = 0; m < numALLMatls; m++) {
       Material* matl = d_sharedState->getMaterial( m );
       MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
       if(mpm_matl){
          rho_micro[m][*iter] =
	    mpm_matl->getConstitutiveModel()->computeRhoMicro(press_new[*iter]);

          mpm_matl->getConstitutiveModel()->
	     computePressEOS(rho_micro[m][*iter],press_eos[m],
					dp_drho[m],sS_new[m][*iter]);

          vol_frac[m][*iter] = rho_CC[m][*iter]/rho_micro[m][*iter];
//	cout << "mpm_vf " << *iter << " = " << vol_frac[m][*iter] << endl;
       }
     }
 }


 if(timestep==0){
  cout << "First timestep" << endl;
  for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++) {
	vol_frac[1][*iter] = 1.0 - vol_frac[0][*iter];
	rho_CC[0][*iter] = vol_frac[0][*iter]*rho_micro[0][*iter];
	rho_CC[1][*iter] = vol_frac[1][*iter]*rho_micro[1][*iter];
//       cout << "vf0 " << *iter << " = " << vol_frac[0][*iter]  << " vf1 " << *iter << " = " << vol_frac[1][*iter] << endl;
  }
 }
 timestep++;
/*`==========DEBUG============*/
    d_ice->printData( patch, 1, "TOP_equilibration", "Press_CC_top", press);

   for (int m = 0; m < numALLMatls; m++)  {
     Material* matl = d_sharedState->getMaterial( m );
     int dwindex = matl->getDWIndex();
     char description[50];
     sprintf(description, "TOP_equilibration_Mat_%d ",dwindex);
     d_ice->printData( patch, 1, description, "rho_CC",          rho_CC[m]);
     d_ice->printData( patch, 0, description, "speedSound",      sS_new[m]);
     d_ice->printData( patch, 1, description, "rho_micro",    rho_micro[m]);
     d_ice->printData( patch, 1, description, "vol_frac",        vol_frac[m]);
    }
 /*==========DEBUG============`*/

//______________________________________________________________________
// Done with preliminary calcs, now loop over every cell
  int count, test_max_iter = 0;
  int d_max_iter_equilibration = 5;  // TEMPORARY THIS SHOULD COME FROM UPS
  for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++) {
    
    IntVector curcell = *iter;    //So I have a chance at finding the bug
    int i,j,k;
    i   = curcell.x();
    j   = curcell.y();
    k   = curcell.z();
      
    double delPress = 0.;
    bool converged  = false;
    count           = 0;
    while ( count < d_max_iter_equilibration && converged == false) {
      count++;
      double A = 0.;
      double B = 0.;
      double C = 0.;
      
      for (int m = 0; m < numALLMatls; m++) 
        delVol_frac[m] = 0.;
      //__________________________________
     // evaluate press_eos at cell i,j,k
     for (int m = 0; m < numALLMatls; m++)  {
       Material* matl = d_sharedState->getMaterial( m );
       ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
       MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
       if(ice_matl){
          double gamma = ice_matl->getGamma();
       
          ice_matl->getEOS()->computePressEOS(rho_micro[m][*iter],gamma,
                                           cv[m][*iter],Temp[m][*iter],
                                           press_eos[m], dp_drho[m], dp_de[m]);
       }
       if(mpm_matl){
          mpm_matl->getConstitutiveModel()->
             computePressEOS(rho_micro[m][*iter],press_eos[m],
                                        dp_drho[m],sS_new[m][*iter]);
       }
     }
     //__________________________________
     // - compute delPress
     // - update press_CC     
     vector<double> Q(numALLMatls),y(numALLMatls);     
     for (int m = 0; m < numALLMatls; m++)   {
       Q[m] =  press_new[*iter] - press_eos[m];
       y[m] =  dp_drho[m] * ( rho_CC[m][*iter]/
               (vol_frac[m][*iter] * vol_frac[m][*iter] + d_SMALL_NUM) ); 
       A   +=  vol_frac[m][*iter];
       B   +=  Q[m]/(y[m] + d_SMALL_NUM);
       C   +=  1.0/(y[m]  + d_SMALL_NUM);
#if 0
       cout << "dp_drho = " << dp_drho[m] << " rho_CC " << rho_CC[m][*iter] << " vol_frac = " << vol_frac[m][*iter] << endl;
       cout << "Q[m] = " << Q[m] <<  "    y[m] = " << y[m] << endl;
//       cout << "Q[m] = " << Q[m] <<  "    y[m] = " << y[m] << " " ;
       cout << "press_eos[m] = " << press_eos[m] << endl;
#endif
     }
//     cout << "A = " << A << " B = " << B << " C = " << C << endl;
     double vol_frac_not_close_packed = 1.;
     delPress = (A - vol_frac_not_close_packed - B)/C;
//     cout << "delPress = " << delPress << endl;
//     cout << endl;
     
     press_new[*iter] += delPress;
     
     //__________________________________
     // backout rho_micro_CC at this new pressure
     for (int m = 0; m < numALLMatls; m++) {
       Material* matl = d_sharedState->getMaterial( m );
       ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
       MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
       if(ice_matl){
         double gamma = ice_matl->getGamma();
       
         rho_micro[m][*iter] = 
           ice_matl->getEOS()->computeRhoMicro(press_new[*iter],gamma,
                                             cv[m][*iter],Temp[m][*iter]);
       }
       if(mpm_matl){
          rho_micro[m][*iter] =
	    mpm_matl->getConstitutiveModel()->computeRhoMicro(press_new[*iter]);
       }
     }
     //__________________________________
     // - compute the updated volume fractions
     //  There are two different way of doing it
     for (int m = 0; m < numALLMatls; m++)  {
       delVol_frac[m]       = -(Q[m] + delPress)/( y[m] + d_SMALL_NUM );
     //vol_frac[m][*iter]   += delVol_frac[m];
       vol_frac[m][*iter]   = rho_CC[m][*iter]/rho_micro[m][*iter];
//       cout << "vf = " << vol_frac[m][*iter] << " ";
     }
//     cout << endl;
     //__________________________________
     // Find the speed of sound at ijk
     // needed by eos and the the explicit
     // del pressure function
     for (int m = 0; m < numALLMatls; m++)  {
       Material* matl = d_sharedState->getMaterial( m );
       ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
       MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
       if(ice_matl){
         double gamma = ice_matl->getGamma();
         ice_matl->getEOS()->computePressEOS(rho_micro[m][*iter],gamma,
                                            cv[m][*iter],Temp[m][*iter],
                                            press_eos[m],dp_drho[m], dp_de[m]);

         tmp = dp_drho[m] + dp_de[m] * 
                    (press_eos[m]/(rho_micro[m][*iter]*rho_micro[m][*iter]));
         sS_new[m][*iter] = sqrt(tmp);
       }
       if(mpm_matl){
          mpm_matl->getConstitutiveModel()->
             computePressEOS(rho_micro[m][*iter],press_eos[m],
                                        dp_drho[m],sS_new[m][*iter]);
       }
     }
     //__________________________________
     // - Test for convergence 
     //  If sum of vol_frac_CC ~= 1.0 then converged 
     sum = 0.0;
     for (int m = 0; m < numALLMatls; m++)  {
       sum += vol_frac[m][*iter];
     }
     if (fabs(sum-1.0) < convergence_crit){
       converged = true;
     }
    else{
//      cout << setprecision(20) << endl;
      cout << "SUM - 1.0 " << *iter << " = " << sum - 1.0 << endl;
      cout << "press_new = " << press_new[*iter] << endl;
    }
     
    }   // end of converged
    
    test_max_iter = std::max(test_max_iter, count);
    
    //__________________________________
    //      BULLET PROOFING
    if(test_max_iter == d_max_iter_equilibration)
    {
        sprintf(warning, 
        " cell[%d][%d][%d], iter %d, n_passes %d,Now exiting ",
        i,j,k,count,n_passes);
	cerr << " Maximum number of iterations was reached " << endl;
    }
    
     for (int m = 0; m < numALLMatls; m++) {
         ASSERT(( vol_frac[m][*iter] > 0.0 ) ||
                ( vol_frac[m][*iter] < 1.0));
     }
    if ( fabs(sum - 1.0) > convergence_crit)   {
        sprintf(warning, 
        " cell[%d][%d][%d], iter %d, n_passes %d,Now exiting ",
        i,j,k,count,n_passes);
	cerr << " sum(vol_frac_CC) != 1.0" << endl;
    }

    if ( press_new[*iter] < 0.0 )   {
        sprintf(warning, 
        " cell[%d][%d][%d], iter %d, n_passes %d, Now exiting",
         i,j,k, count, n_passes);
	cerr << " press_new[iter*] < 0" << endl;
    }

    for (int m = 0; m < numALLMatls; m++)
    if ( rho_micro[m][*iter] < 0.0 || vol_frac[m][*iter] < 0.0) {
        sprintf(warning, 
        " cell[%d][%d][%d], mat %d, iter %d, n_passes %d,Now exiting ",
        i,j,k,m,count,n_passes);
	cerr << " rho_micro < 0 || vol_frac < 0" << endl;
    }
  }     // end of cell interator

    fprintf(stderr, "\n max number of iterations in any cell %i\n",test_max_iter);
  /*__________________________________
   *   THIS NEEDS TO BE FIXED 
   *   WE NEED TO UPDATE BC_VALUES NOT PRESSURE
   *   SINCE WE USE BC_VALUES LATER ON IN THE CODE
   *___________________________________*/
  d_ice->setBC(press_new,"Pressure",patch);
  
  for (int m = 0; m < numALLMatls; m++)   {
    Material* matl = d_sharedState->getMaterial( m );
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(ice_matl){
      int dwindex = ice_matl->getDWIndex();
      new_dw->put( vol_frac[m],      Ilb->vol_frac_CCLabel,   dwindex, patch);
      new_dw->put( sS_new[m],        Ilb->speedSound_CCLabel, dwindex, patch);
      new_dw->put( rho_micro[m],     Ilb->rho_micro_CCLabel,  dwindex, patch);
    }
    if(mpm_matl){
      int dwindex = mpm_matl->getDWIndex();
      new_dw->put(vol_frac[m],  MIlb->vol_frac_CCLabel,  dwindex, patch);
      new_dw->put(sS_new[m],    MIlb->speedSound_CCLabel,dwindex, patch);
      new_dw->put(rho_micro[m], MIlb->rho_micro_CCLabel, dwindex, patch);
    }
  }
  new_dw->put(press_new,Ilb->press_equil_CCLabel,0,patch);
  
}
