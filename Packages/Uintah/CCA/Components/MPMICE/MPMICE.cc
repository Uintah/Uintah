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
  d_SMALL_NUM = 1.e-100;
}

MPMICE::~MPMICE()
{
  delete MIlb;
  delete d_mpm;
  delete d_ice;
}
//______________________________________________________________________
//
void MPMICE::problemSetup(const ProblemSpecP& prob_spec, GridP& grid,
			  SimulationStateP& sharedState)
{
   d_sharedState = sharedState;

   d_mpm->setMPMLabel(Mlb);
   d_mpm->problemSetup(prob_spec, grid, d_sharedState);

   d_ice->setICELabel(Ilb);
   d_ice->problemSetup(prob_spec, grid, d_sharedState);
   
   finishMPMICEproblemSetup(prob_spec, grid, d_sharedState);

   cerr << "Done with problemSetup \t\t\t MPMICE" <<endl;
   cerr << "--------------------------------\n"<<endl;
}
//______________________________________________________________________
//
void MPMICE::scheduleInitialize(const LevelP& level,
				SchedulerP& sched,
				DataWarehouseP& dw)
{
  d_mpm->scheduleInitialize(      level, sched, dw);
  d_ice->scheduleInitialize(      level, sched, dw);
//  scheduleFinishMPMICEinitialize( level, sched, dw);
   cerr << "Doing Initialization \t\t\t MPMICE" <<endl;
   cerr << "--------------------------------\n"<<endl; 
}
//______________________________________________________________________
//
void MPMICE::scheduleFinishMPMICEinitialize(const LevelP& level, 
                          SchedulerP& sched,
			     DataWarehouseP& dw)      
{
  Level::const_patchIterator iter;
  for(iter=level->patchesBegin(); iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    Task* t = scinew Task("MPMICE::finishMPMICEinitialize", patch, dw, dw,this,
			  &MPMICE::finishMPMICEinitialize);

    sched->addTask(t);
  }
}
//______________________________________________________________________
//
void MPMICE::scheduleComputeStableTimestep(const LevelP& level,
					   SchedulerP& sched,
					   DataWarehouseP& dw)
{
  // Schedule computing the ICE stable timestep
  d_ice->scheduleComputeStableTimestep(level, sched, dw);
  // MPM stable timestep is a by product of the CM
}

//______________________________________________________________________
//
void MPMICE::scheduleTimeAdvance(double, double,
				 const LevelP&   level,
				 SchedulerP&     sched,
				 DataWarehouseP& old_dw, 
				 DataWarehouseP& new_dw)
{
   int numMPMMatls = d_sharedState->getNumMPMMatls();
   for(Level::const_patchIterator iter=level->patchesBegin();
       iter != level->patchesEnd(); iter++){

    const Patch* patch=*iter;
    if(d_fracture) {
      d_mpm->scheduleSetPositions(patch,sched,old_dw,new_dw);
      d_mpm->scheduleComputeBoundaryContact(patch,sched,old_dw,new_dw);
      d_mpm->scheduleComputeVisibility(patch,sched,old_dw,new_dw);
    }
    d_mpm->scheduleInterpolateParticlesToGrid(      patch,sched,old_dw,new_dw);

    if (MPMPhysicalModules::thermalContactModel) {
       d_mpm->scheduleComputeHeatExchange(          patch,sched,old_dw,new_dw);
    }

    // schedule the interpolation of mass and volume to the cell centers
    scheduleInterpolateNCToCC_0(                    patch,sched,old_dw,new_dw);
    scheduleComputeEquilibrationPressure(           patch,sched,old_dw,new_dw);
    d_ice->scheduleComputeFaceCenteredVelocities(   patch,sched,old_dw,new_dw);
    d_ice->scheduleAddExchangeContributionToFCVel(  patch,sched,old_dw,new_dw);
    d_ice->scheduleComputeDelPressAndUpdatePressCC( patch,sched,old_dw,new_dw);

//    scheduleInterpolateVelIncFCToNC(                patch,sched,old_dw,new_dw);

    d_mpm->scheduleExMomInterpolated(               patch,sched,old_dw,new_dw);
    d_mpm->scheduleComputeStressTensor(             patch,sched,old_dw,new_dw);
     
    d_ice->scheduleComputePressFC(                  patch,sched,old_dw,new_dw);
    d_ice->scheduleAccumulateMomentumSourceSinks(   patch,sched,old_dw,new_dw);
    d_ice->scheduleAccumulateEnergySourceSinks(     patch,sched,old_dw,new_dw);
    d_ice->scheduleComputeLagrangianValues(         patch,sched,old_dw,new_dw);

    scheduleInterpolatePAndGradP(                   patch,sched,old_dw,new_dw);
   
    d_mpm->scheduleComputeInternalForce(            patch,sched,old_dw,new_dw);
    d_mpm->scheduleComputeInternalHeatRate(         patch,sched,old_dw,new_dw);
    d_mpm->scheduleSolveEquationsMotion(            patch,sched,old_dw,new_dw);
    d_mpm->scheduleSolveHeatEquations(              patch,sched,old_dw,new_dw);
    d_mpm->scheduleIntegrateAcceleration(           patch,sched,old_dw,new_dw);
    d_mpm->scheduleIntegrateTemperatureRate(        patch,sched,old_dw,new_dw);

    scheduleInterpolateNCToCC(                      patch,sched,old_dw,new_dw);
    scheduleCCMomExchange(                          patch,sched,old_dw,new_dw);
//    d_mpm->scheduleExMomIntegrated(patch,sched,old_dw,new_dw);
//    d_ice->scheduleAddExchangeToMomentumAndEnergy(patch,sched,old_dw,new_dw);
    d_mpm->scheduleInterpolateToParticlesAndUpdate( patch,sched,old_dw,new_dw);
    d_mpm->scheduleComputeMassRate(                 patch,sched,old_dw,new_dw);
    if(d_fracture) {
      d_mpm->scheduleComputeFracture(patch,sched,old_dw,new_dw);
      d_mpm->scheduleStressRelease(patch,sched,old_dw,new_dw);
    }

    d_mpm->scheduleCarryForwardVariables(           patch,sched,old_dw,new_dw);
    d_ice->scheduleAdvectAndAdvanceInTime(          patch,sched,old_dw,new_dw);
  }
  
   sched->scheduleParticleRelocation(level, old_dw, new_dw,
				     Mlb->pXLabel_preReloc, 
				     Mlb->d_particleState_preReloc,
				     Mlb->pXLabel, Mlb->d_particleState,
				     numMPMMatls);
}
//______________________________________________________________________
//
void MPMICE::scheduleInterpolatePAndGradP(const Patch* patch,
                                          SchedulerP& sched,
                                          DataWarehouseP& old_dw,
                                          DataWarehouseP& new_dw)
{
//   cout << "scheduleInterpolatePressureToParticles" << endl;
   int numMPMMatls = d_sharedState->getNumMPMMatls();
   Task* t=scinew Task("MPMICE::interpolatePAndGradP",
		        patch, old_dw, new_dw,
		        this, &MPMICE::interpolatePAndGradP);

   t->requires(new_dw,Ilb->press_CCLabel,0, patch, Ghost::AroundCells, 1);

   for(int m = 0; m < numMPMMatls; m++){
     MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
     int idx = mpm_matl->getDWIndex();
     t->requires(old_dw, Mlb->pXLabel,            idx, patch, Ghost::None);
     t->requires(new_dw, MIlb->mom_source_CCLabel,idx, patch,
							Ghost::AroundCells, 1);

     t->computes(new_dw, Mlb->pPressureLabel,   idx, patch);
     t->computes(new_dw, Mlb->gradPressNCLabel, idx, patch);
   }

   sched->addTask(t);

}
//______________________________________________________________________
//
void MPMICE::scheduleInterpolateVelIncFCToNC(const Patch* patch,
                                      SchedulerP& sched,
                                      DataWarehouseP& old_dw,
                                      DataWarehouseP& new_dw)
{
   int numMPMMatls = d_sharedState->getNumMPMMatls();
   Task* t=scinew Task("MPMICE::interpolateVelIncFCToNC",
                        patch, old_dw, new_dw,
                        this, &MPMICE::interpolateVelIncFCToNC);

   for(int m = 0; m < numMPMMatls; m++){
     MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
     int idx = mpm_matl->getDWIndex();
     t->requires(new_dw,MIlb->uvel_FCLabel,     idx, patch,Ghost::None);
     t->requires(new_dw,MIlb->vvel_FCLabel,     idx, patch,Ghost::None);
     t->requires(new_dw,MIlb->wvel_FCLabel,     idx, patch,Ghost::None);
     t->requires(new_dw,MIlb->uvel_FCMELabel,   idx, patch,Ghost::None);
     t->requires(new_dw,MIlb->vvel_FCMELabel,   idx, patch,Ghost::None);
     t->requires(new_dw,MIlb->wvel_FCMELabel,   idx, patch,Ghost::None);

     t->requires(new_dw,Mlb->gVelocityLabel,    idx, patch, Ghost::None);

     t->computes(new_dw,Mlb->gMomExedVelocityLabel, idx, patch );

   }
   sched->addTask(t);

}
//______________________________________________________________________
//
void MPMICE::scheduleInterpolateNCToCC_0(const Patch* patch,
                                       SchedulerP& sched,
                                       DataWarehouseP& old_dw,
                                       DataWarehouseP& new_dw)
{
   /* interpolateNCToCC */
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
     //t->computes(new_dw, MIlb->rho_CCLabel,        idx, patch);  EXTRA
     t->computes(new_dw, MIlb->vel_CCLabel,        idx, patch);
     t->computes(new_dw, MIlb->temp_CCLabel,       idx, patch);
     t->computes(new_dw, MIlb->cv_CCLabel,         idx, patch);
//     t->computes(new_dw, MIlb->int_eng_L_CCLabel,  idx, patch);
     
   }

   sched->addTask(t);

}
//______________________________________________________________________
//
void MPMICE::scheduleInterpolateNCToCC(const Patch* patch,
                                       SchedulerP& sched,
                                       DataWarehouseP& old_dw,
                                       DataWarehouseP& new_dw)
{
//   cout << "scheduleInterpolateNCToCC" << endl;
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

     t->computes(new_dw, MIlb->mom_L_CCLabel, idx, patch);
   }

   sched->addTask(t);

}
//______________________________________________________________________
//
void MPMICE::scheduleCCMomExchange(const Patch* patch,
                                   SchedulerP& sched,
                                   DataWarehouseP& old_dw,
                                   DataWarehouseP& new_dw)
{
   Task* t=scinew Task("MPMICE::doCCMomExchange",
		        patch, old_dw, new_dw,
		        this, &MPMICE::doCCMomExchange);
  int numALLMatls  = d_sharedState->getNumMatls();                      
  for (int m = 0; m < numALLMatls; m++) {
    Material* matl = d_sharedState->getMaterial( m );
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(ice_matl){                   // I C E
      int iceidx = ice_matl->getDWIndex();
      t->requires(old_dw,Ilb->rho_CCLabel,          iceidx,patch,Ghost::None);
      t->requires(new_dw,Ilb->mom_L_CCLabel,        iceidx,patch,Ghost::None);
      t->requires(new_dw,Ilb->int_eng_L_CCLabel,    iceidx,patch,Ghost::None);
      t->requires(new_dw,Ilb->vol_frac_CCLabel,     iceidx,patch,Ghost::None);
      t->requires(old_dw,Ilb->cv_CCLabel,           iceidx,patch,Ghost::None);
      t->requires(new_dw,Ilb->rho_micro_CCLabel,    iceidx,patch,Ghost::None);

      t->computes(new_dw,Ilb->mom_L_ME_CCLabel,     iceidx,patch);
      t->computes(new_dw,Ilb->int_eng_L_ME_CCLabel, iceidx,patch);
   }
   if(mpm_matl){                    // M P M
     int mpmidx = mpm_matl->getDWIndex();
     t->requires(new_dw, MIlb->mom_L_CCLabel,       mpmidx,patch,Ghost::None,0);
     t->requires(new_dw, MIlb->rho_CCLabel,         mpmidx,patch,Ghost::None,0);
     t->requires(new_dw, MIlb->int_eng_L_CCLabel,   mpmidx,patch,Ghost::None,0);
     t->requires(new_dw, MIlb->cv_CCLabel,          mpmidx,patch,Ghost::None,0);
     t->requires(new_dw, Mlb->gVelocityStarLabel,   mpmidx,patch,Ghost::None,0);
     t->requires(new_dw, Mlb->gAccelerationLabel,   mpmidx,patch,Ghost::None,0);
     t->requires(new_dw, MIlb->rho_micro_CCLabel,   mpmidx,patch,Ghost::None,0);

     t->computes(new_dw, Mlb->gMomExedVelocityStarLabel, mpmidx, patch);
     t->computes(new_dw, Mlb->gMomExedAccelerationLabel, mpmidx, patch);
   }
 }
   sched->addTask(t);

}
/* ---------------------------------------------------------------------
 Function~  MPMICE::scheduleComputeEquilibrationPressure--
 Purpose~   Compute the equilibration pressure
 Note:  This similar to ICE::scheduleComputeEquilibrationPressure
         with the addition of MPM matls
_____________________________________________________________________*/
void MPMICE::scheduleComputeEquilibrationPressure(const Patch* patch,
						  SchedulerP& sched,
						  DataWarehouseP& old_dw,
						  DataWarehouseP& new_dw)
{
  Task* task = scinew Task("MPMICE::computeEquilibrationPressure",
                        patch, old_dw, new_dw,this,
			   &MPMICE::computeEquilibrationPressure);
  
  task->requires(old_dw,Ilb->press_CCLabel, 0,patch,Ghost::None);
  int numALLMatls  = d_sharedState->getNumMatls();    
  for (int m = 0; m < numALLMatls; m++) {
    Material* matl = d_sharedState->getMaterial( m );
    int dwindex = matl->getDWIndex();

    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(ice_matl){                   // I C E
      task->requires(old_dw,Ilb->temp_CCLabel,      dwindex,patch,Ghost::None);
      task->requires(old_dw,Ilb->cv_CCLabel,        dwindex,patch,Ghost::None);
      task->requires(old_dw,Ilb->mass_CCLabel,      dwindex,patch,Ghost::None);
      task->requires(old_dw,Ilb->sp_vol_CCLabel,    dwindex,patch,Ghost::None);
      // Above Might be extra
      
      task->computes(new_dw,Ilb->sp_vol_equilLabel, dwindex, patch);
      // Above might be extra need to test
    }
    if(mpm_matl){                    // M P M
      task->requires(new_dw,MIlb->temp_CCLabel,      dwindex,patch,Ghost::None);
      task->requires(new_dw,MIlb->cv_CCLabel,        dwindex,patch,Ghost::None);
      task->requires(new_dw,MIlb->cVolumeLabel,      dwindex,patch,Ghost::None);
    }
    // For all materials
    task->computes(new_dw,MIlb->speedSound_CCLabel,dwindex,patch);
    task->computes(new_dw,MIlb->rho_micro_CCLabel, dwindex,patch);
    task->computes(new_dw,MIlb->vol_frac_CCLabel,  dwindex,patch);
    task->computes(new_dw,MIlb->rho_CCLabel,       dwindex,patch);
 } 

  task->computes(new_dw,Ilb->press_equil_CCLabel,0, patch);
  sched->addTask(task);
}
//______________________________________________________________________
//       A C T U A L   S T E P S :
//______________________________________________________________________
//
/* --------------------------------------------------------------------- 
 Function~  MPMICE::finishMPMICEproblemSetup--
 Purpose~   Grab the exchange coefficients from the uda file  
_____________________________________________________________________*/ 
void MPMICE::finishMPMICEproblemSetup(const ProblemSpecP& prob_spec, 
                            GridP&, SimulationStateP&)    
{
  ProblemSpecP mat_ps       =  prob_spec->findBlock("MaterialProperties");
  ProblemSpecP mpm_ice_ps   =  mat_ps->findBlock("MPMICE");
  ProblemSpecP exch_ps = mpm_ice_ps->findBlock("exchange_coefficients");
  exch_ps->require("momentum",d_K_mom);
  exch_ps->require("heat",d_K_heat);
  cerr << "Pulled out exchange coefficients of the input file \t\t MPMICE" << endl;
}

/* --------------------------------------------------------------------- 
 Function~  MPMICE::finishMPMICEinitialize--
 Purpose~   Make necessary adjustments
_____________________________________________________________________*/ 
void MPMICE::finishMPMICEinitialize(const ProcessorGroup*, const Patch*,
			     DataWarehouseP& , DataWarehouseP& )    
{

}

//______________________________________________________________________
//
void MPMICE::interpolatePAndGradP(const ProcessorGroup*,
                                  const Patch* patch,
                                  DataWarehouseP& old_dw,
                                  DataWarehouseP& new_dw)
{
//  cout << "Doing interpolatePressureToParticles \t\t MPMICE" << endl;
  
  CCVariable<double> pressCC;
  NCVariable<double> pressNC;
  IntVector ni[8];
  double S[8];
  Vector zero(0.,0.,0.);

  new_dw->get(pressCC,       Ilb->press_CCLabel, 0, patch,Ghost::None, 0);
  new_dw->allocate(pressNC, MIlb->press_NCLabel, 0, patch);

  IntVector cIdx[8];
  // Interpolate CC pressure to nodes
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

    new_dw->allocate(pPressure, Mlb->pPressureLabel,      pset);
    old_dw->get(px,             Mlb->pXLabel,             pset);

    // Interpolate NC pressure to particles
    for(ParticleSubset::iterator iter = pset->begin();
				          iter != pset->end(); iter++){
	particleIndex idx = *iter;
	double press = 0.;

	// Get the node indices that surround the cell
	patch->findCellAndWeights(px[idx], ni, S);
	for (int k = 0; k < 8; k++) {
	    press += pressNC[ni[k]] * S[k];
	}
// HARDWIRING
	pPressure[idx] = press-101325.0;
    }

    CCVariable<Vector> mom_source;
    NCVariable<Vector> gradPressNC;
    new_dw->get(mom_source,      MIlb->mom_source_CCLabel, dwindex, patch,
							Ghost::AroundCells, 1);
    new_dw->allocate(gradPressNC, Mlb->gradPressNCLabel,   dwindex, patch);

    // Interpolate CC pressure gradient (mom_source) to nodes (gradP*dA*dt)
    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
       patch->findCellsFromNode(*iter,cIdx);
       gradPressNC[*iter] = zero;
       for (int in=0;in<8;in++){
	  gradPressNC[*iter]  += mom_source[cIdx[in]]*.125;
       }
    }

    new_dw->put(pPressure,   Mlb->pPressureLabel);
    new_dw->put(gradPressNC, Mlb->gradPressNCLabel, dwindex, patch);
  }
}
//______________________________________________________________________
//
void MPMICE::interpolateVelIncFCToNC(const ProcessorGroup*,
                                     const Patch* patch,
                                     DataWarehouseP&,
                                     DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  Vector zero(0.0,0.0,0.);
  Vector dx = patch->dCell();
  SFCXVariable<double> uvel_FC, uvel_FCME;
  SFCYVariable<double> vvel_FC, vvel_FCME;
  SFCZVariable<double> wvel_FC, wvel_FCME;
  CCVariable<Vector> velInc_CC;
  NCVariable<Vector> velInc_NC;
  NCVariable<Vector> gvelocity;

  for(int m = 0; m < numMatls; m++) {
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    int dwindex = mpm_matl->getDWIndex();
    new_dw->get(uvel_FC,   MIlb->uvel_FCLabel,   dwindex, patch, Ghost::None,0);
    new_dw->get(vvel_FC,   MIlb->vvel_FCLabel,   dwindex, patch, Ghost::None,0);
    new_dw->get(wvel_FC,   MIlb->wvel_FCLabel,   dwindex, patch, Ghost::None,0);
    new_dw->get(uvel_FCME, MIlb->uvel_FCMELabel, dwindex, patch, Ghost::None,0);
    new_dw->get(vvel_FCME, MIlb->vvel_FCMELabel, dwindex, patch, Ghost::None,0);
    new_dw->get(wvel_FCME, MIlb->wvel_FCMELabel, dwindex, patch, Ghost::None,0);

    new_dw->get(gvelocity,Mlb->gVelocityLabel,   dwindex, patch, Ghost::None,0);

    new_dw->allocate(velInc_CC, MIlb->velInc_CCLabel, dwindex, patch);
    new_dw->allocate(velInc_NC, MIlb->velInc_NCLabel, dwindex, patch);
    double xcomp,ycomp,zcomp;

    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
        IntVector cur = *iter;
        IntVector adjx(cur.x()+1,cur.y(),  cur.z());
        IntVector adjy(cur.x(),  cur.y()+1,cur.z());
        IntVector adjz(cur.x(),  cur.y(),  cur.z()+1);
	xcomp = ((uvel_FCME[cur]  - uvel_FC[cur]) +
		 (uvel_FCME[adjx] - uvel_FC[adjx]))*0.5;
	ycomp = ((vvel_FCME[cur]  - vvel_FC[cur]) +
		 (vvel_FCME[adjy] - vvel_FC[adjy]))*0.5;
	zcomp = ((wvel_FCME[cur]  - wvel_FC[cur]) +
		 (wvel_FCME[adjz] - wvel_FC[adjz]))*0.5;

        velInc_CC[*iter] = Vector(xcomp,ycomp,zcomp);
    }

    IntVector cIdx[8];
    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	patch->findCellsFromNode(*iter,cIdx);
	velInc_NC[*iter]  = zero;
	for (int in=0;in<8;in++){
	   velInc_NC[*iter]     +=  velInc_CC[cIdx[in]]*.125;
        }
	gvelocity[*iter] += velInc_NC[*iter];
    }

    new_dw->put(gvelocity, Mlb->gMomExedVelocityLabel, dwindex, patch);

  }

}
//______________________________________________________________________
//
void MPMICE::interpolateNCToCC_0(const ProcessorGroup*,
                                 const Patch* patch,
                                 DataWarehouseP&,
                                 DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  Vector zero(0.0,0.0,0.);
  Vector dx = patch->dCell();
  static int timestep = 0;
//  cout << "Doing interpolateNCToCC_0 \t\t\t MPMICE" << endl;

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    int matlindex = mpm_matl->getDWIndex();

     // Create arrays for the grid data
     NCVariable<double > gmass, gvolume;
     NCVariable<Vector > gvelocity;
     CCVariable<double > cmass, cvolume;
     CCVariable<double > Temp_CC;
     CCVariable<double > cv_CC;
     CCVariable<double > int_eng_L_CC;
     CCVariable<Vector > vel_CC;

     new_dw->allocate(cmass,     MIlb->cMassLabel,         matlindex, patch);
     new_dw->allocate(cvolume,   MIlb->cVolumeLabel,       matlindex, patch);
     new_dw->allocate(vel_CC,    MIlb->vel_CCLabel,        matlindex, patch);
     new_dw->allocate(Temp_CC,   MIlb->temp_CCLabel,       matlindex, patch);
     new_dw->allocate(cv_CC,     MIlb->cv_CCLabel,         matlindex, patch);
     new_dw->allocate(int_eng_L_CC,    
                                  MIlb->int_eng_L_CCLabel, matlindex, patch);
      
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

     cv_CC.initialize(mpm_matl->getSpecificHeat());

#if 0
     Vector nodal_mom(0.,0.,0.);
     for(NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
	nodal_mom+=gvelocity[*iter]*gmass[*iter];
     }
     cout << "Solid matl nodal momentum = " << nodal_mom << endl;
     Vector cell_mom(0.,0.,0.);
     cout << "In NCToCC_0" << endl;
#endif

     for(CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
       patch->findNodesFromCell(*iter,nodeIdx);
       for (int in=0;in<8;in++){
	 cmass[*iter]    += .125*gmass[nodeIdx[in]];
	 cvolume[*iter]  += .125*gvolume[nodeIdx[in]];
	 vel_CC[*iter]   +=      gvelocity[nodeIdx[in]]*.125*gmass[nodeIdx[in]];
       }
//       cell_mom += vel_CC[*iter];
       vel_CC[*iter]      /= (cmass[*iter]     + d_SMALL_NUM);
       Temp_CC[*iter]      =  300.0;           // H A R D W I R E D 
       // int_eng_L_CC[*iter] = Temp_CC[*iter] * cv_CC[*iter] * cmass[*iter]; EXTRA
     }
//     cout << "Solid matl CC momentum = " << cell_mom << endl;

//__________________________________
//   H A R D W I R E
// Jim: note we need the velocity and temperature
//      defined everyhere, even where there are no particles.  If we 
//      don't then the Malloc lib initializes it as a NAN.
    if(timestep==0){
      cout<<"I've hardwired the initial temperature for mpm matl"<<endl;
      Temp_CC.initialize(300.0);
    }
    timestep++;

  //  Set BC's and put into new_dw
     d_ice->setBC(vel_CC,  "Velocity",   patch);
     d_ice->setBC(Temp_CC, "Temperature",patch);
     
     new_dw->put(cmass,     MIlb->cMassLabel,         matlindex, patch);
     new_dw->put(cvolume,   MIlb->cVolumeLabel,       matlindex, patch);
     new_dw->put(vel_CC,    MIlb->vel_CCLabel,        matlindex, patch);
     new_dw->put(Temp_CC,   MIlb->temp_CCLabel,       matlindex, patch);
     new_dw->put(cv_CC,     MIlb->cv_CCLabel,         matlindex, patch);
    // new_dw->put(int_eng_L_CC,    
    //                        MIlb->int_eng_L_CCLabel,  matlindex, patch);    EXTRA    
  }
}
//______________________________________________________________________
//
void MPMICE::interpolateNCToCC(const ProcessorGroup*,
                               const Patch* patch,
                               DataWarehouseP&,
                               DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  Vector zero(0.,0.,0.);
//  cout << "Doing interpolateNCToCC \t\t\t MPMICE" << endl;

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    int matlindex = mpm_matl->getDWIndex();

     // Create arrays for the grid data
     NCVariable<double> gmass, gvolume;
     NCVariable<Vector> gvelocity, gacc;
     CCVariable<Vector> cmomentum;

     new_dw->get(gmass,     Mlb->gMassLabel,           matlindex, patch,
							Ghost::AroundCells, 1);
     new_dw->get(gvelocity, Mlb->gVelocityStarLabel,   matlindex, patch,
							Ghost::AroundCells, 1);
     new_dw->get(gacc,      Mlb->gAccelerationLabel,   matlindex, patch,
							Ghost::AroundCells, 1);

     new_dw->allocate(cmomentum, MIlb->mom_L_CCLabel, matlindex, patch);
 
     cmomentum.initialize(zero);

     IntVector nodeIdx[8];

#if 0
     Vector nodal_mom(0.,0.,0.);
     Vector cell_momnpg(0.,0.,0.);
     Vector cell_momwpg(0.,0.,0.);

     for(NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
        nodal_mom+=gvelocity[*iter]*gmass[*iter];
     }
     cout << "In NCToCC" << endl;
     cout << "Solid matl nodal momentum = " << nodal_mom << endl;
#endif

     for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
       patch->findNodesFromCell(*iter,nodeIdx);
       for (int in=0;in<8;in++){
 	 cmomentum[*iter] += gvelocity[nodeIdx[in]]*gmass[nodeIdx[in]]*.125;
       }
//       cell_momwpg += cmomentum[*iter];
     }
//     cout << "Solid matl CC momentum (wpg) = " << cell_momwpg << endl;

#if 0
/*`==========TESTING==========*/ 
    char description[50];
    sprintf(description, "interpolateNCToCC_%d ",matlindex); 
    d_ice->printVector( patch,1, description, "xmom_L", 0, cmomentum);
    d_ice->printVector( patch,1, description, "ymom_L", 1, cmomentum);
    d_ice->printVector( patch,1, description, "zmom_L", 2, cmomentum);   
 /*==========TESTING==========`*/
 #endif
     new_dw->put(cmomentum, MIlb->mom_L_CCLabel, matlindex, patch);
  }
}

//______________________________________________________________________
//
void MPMICE::doCCMomExchange(const ProcessorGroup*,
                             const Patch* patch,
                             DataWarehouseP& old_dw,
                             DataWarehouseP& new_dw)
{
//  cout << "Doing Heat and momentum exchange \t\t MPMICE" << endl;
  int numMPMMatls = d_sharedState->getNumMPMMatls();
  int numICEMatls = d_sharedState->getNumICEMatls();
  int numALLMatls = numMPMMatls + numICEMatls;

  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label());
  Vector dx = patch->dCell();
  Vector zero(0.,0.,0.);

  // Create arrays for the grid data
  vector<NCVariable<Vector> > gacceleration(numALLMatls);
  vector<NCVariable<Vector> > gvelocity(numALLMatls);
  vector<NCVariable<Vector> > gMEacceleration(numALLMatls);
  vector<NCVariable<Vector> > gMEvelocity(numALLMatls);
  vector<NCVariable<double> > gmass(numALLMatls);

  vector<CCVariable<double> > rho_CC(numALLMatls);
  vector<CCVariable<double> > Temp_CC(numALLMatls);  
  vector<CCVariable<double> > vol_frac_CC(numALLMatls);

  vector<CCVariable<double> > rho_micro_CC(numALLMatls);
  vector<CCVariable<double> > cv_CC(numALLMatls);

  vector<CCVariable<Vector> > mom_L(numALLMatls);
  vector<CCVariable<double> > int_eng_L(numALLMatls);
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
  beta.zero();
  acopy.zero();
  K.zero();
  H.zero();
  a.zero();
  
  for (int i = 0; i < numALLMatls; i++ ) {
      K[numALLMatls-1-i][i] =d_K_mom[i];
      H[numALLMatls-1-i][i] =d_K_heat[i];
  }
 
  for (int m = 0; m < numALLMatls; m++) {
    Material* matl = d_sharedState->getMaterial( m );
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int dwindex = mpm_matl->getDWIndex();

      new_dw->get(gvelocity[m],    Mlb->gVelocityStarLabel,   dwindex, patch,
							Ghost::None, 0);
      new_dw->get(gacceleration[m],Mlb->gAccelerationLabel,   dwindex, patch,
							Ghost::None, 0);
      new_dw->allocate(gMEvelocity[m],     Mlb->gMomExedVelocityStarLabel,
						             dwindex, patch);
      new_dw->allocate(gMEacceleration[m], Mlb->gMomExedAccelerationLabel,
						             dwindex, patch);
      new_dw->get(mom_L[m],        MIlb->mom_L_CCLabel,       dwindex, patch,
							Ghost::None, 0);
      new_dw->get(rho_CC[m],       MIlb->rho_CCLabel,         dwindex, patch,
							Ghost::None, 0);
      new_dw->get(int_eng_L[m],    MIlb->int_eng_L_CCLabel,   dwindex, patch,
							Ghost::None, 0);
      new_dw->get(cv_CC[m],        MIlb->cv_CCLabel,          dwindex, patch,
							Ghost::None, 0);
      new_dw->get(cmass[m],        MIlb->cMassLabel,          dwindex, patch,
							Ghost::None, 0);
      new_dw->get(rho_micro_CC[m], MIlb->rho_micro_CCLabel,   dwindex, patch,
							Ghost::None, 0);
      new_dw->get(vol_frac_CC[m],  MIlb->vol_frac_CCLabel,    dwindex, patch,
							Ghost::None, 0);
     new_dw->get(gmass[m]    ,     Mlb->gMassLabel,           dwindex, patch,
							Ghost::AroundCells, 1);
      new_dw->allocate(vel_CC[m],  MIlb->velstar_CCLabel,     dwindex, patch);
      new_dw->allocate(mom_L_ME[m],MIlb->mom_L_ME_CCLabel,    dwindex, patch);
      new_dw->allocate(dvdt_CC[m], MIlb->dvdt_CCLabel,        dwindex, patch);
      new_dw->allocate(Temp_CC[m], MIlb->temp_CC_scratchLabel,dwindex, patch);
      new_dw->allocate(int_eng_L_ME[m],MIlb->int_eng_L_ME_CCLabel,
                                                              dwindex, patch);

      dvdt_CC[m].initialize(zero);
    }
    if(ice_matl){
      int dwindex = ice_matl->getDWIndex();
      new_dw->get(rho_CC[m],        Ilb->rho_CCLabel,
                                              dwindex, patch, Ghost::None, 0);
      new_dw->get(mom_L[m],         Ilb->mom_L_CCLabel,
                                              dwindex, patch, Ghost::None, 0);
      new_dw->get(int_eng_L[m],     Ilb->int_eng_L_CCLabel,
                                              dwindex, patch, Ghost::None, 0);
      new_dw->get(vol_frac_CC[m],   Ilb->vol_frac_CCLabel,
                                              dwindex, patch, Ghost::None, 0);
      new_dw->get(rho_micro_CC[m],  Ilb->rho_micro_CCLabel,
                                              dwindex, patch, Ghost::None, 0);
      old_dw->get(cv_CC[m],         Ilb->cv_CCLabel,
                                              dwindex, patch, Ghost::None, 0);

      new_dw->allocate(vel_CC[m],      Ilb->vel_CCLabel,         dwindex,patch);
      new_dw->allocate(mom_L_ME[m],    Ilb->mom_L_ME_CCLabel,    dwindex,patch);
      new_dw->allocate(Temp_CC[m],     Ilb->temp_CCLabel,        dwindex,patch);
      new_dw->allocate(int_eng_L_ME[m],Ilb->int_eng_L_ME_CCLabel,dwindex,patch);
      new_dw->allocate(dvdt_CC[m],     MIlb->dvdt_CCLabel,       dwindex,patch);
      dvdt_CC[m].initialize(zero);
    }
  }

  double vol = dx.x()*dx.y()*dx.z();
  double SMALL_NUM = 1.0e-100;
  double tmp;
  int itworked=-9;

  // Convert momenta to velocities.  Slightly different for MPM and ICE.
  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
    for (int m = 0; m < numALLMatls; m++) {
      mass[m]           = rho_CC[m][*iter] * vol + SMALL_NUM;
      Temp_CC[m][*iter] = int_eng_L[m][*iter]/(mass[m]*cv_CC[m][*iter]);
      vel_CC[m][*iter]  = mom_L[m][*iter]/mass[m];
    }
  }


#if 0
  cout << "GRID MOMENTUM BEFORE CCMOMENTUM EXCHANGE" << endl;
  Vector total_mom(0.,0.,0.);
  for (int m = 0; m < numALLMatls; m++) {
    Vector matl_mom(0.,0.,0.);
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
	matl_mom += mom_L[m][*iter];
    }
    cout << "Momentum for material " << m << " = " << matl_mom << endl;
    total_mom+=matl_mom;
  }
  cout << "TOTAL Momentum BEFORE = " << total_mom << endl;
#endif



  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
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
    //         
    for(int m = 0; m < numALLMatls; m++) {
      tmp = cv_CC[m][*iter]*rho_micro_CC[m][*iter];
      for(int n = 0; n < numALLMatls; n++)  {
	beta[m][n] = delT * vol_frac_CC[n][*iter] * H[n][m]/tmp;
	a[m][n] = -beta[m][n];
      }
    }
    //   Form matrix (a) diagonal terms
    for(int m = 0; m < numALLMatls; m++) {
      a[m][m] = 1.;
      for(int n = 0; n < numALLMatls; n++)   {
	a[m][m] +=  beta[m][n];
      }
    }
    // -  F O R M   R H S   (b)
    for(int m = 0; m < numALLMatls; m++)  {
      b[m] = 0.0;
     
     for(int n = 0; n < numALLMatls; n++) {
	b[m] += beta[m][n] *
	  (Temp_CC[n][*iter] - Temp_CC[m][*iter]);
      }
    }
    //     S O L V E, Add exchange contribution to orig value
    itworked = a.solve(b);
    
    for(int m = 0; m < numALLMatls; m++) {
      Temp_CC[m][*iter] = Temp_CC[m][*iter] + b[m];
    }
  }  //end CellIterator loop
  
  //__________________________________
  //  Set the Boundary conditions 
  //   Do this for all matls even though MPM doesn't
  //   care about this.  For two identical ideal gases
  //   mom_L_ME and int_eng_L_ME should be identical and this
  //   is useful when debugging.
  for (int m = 0; m < numALLMatls; m++)  {
      d_ice->setBC(vel_CC[m], "Velocity",   patch);
      d_ice->setBC(Temp_CC[m],"Temperature",patch);
  }
  //__________________________________
  // Convert vars. primitive-> flux 
  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
    for (int m = 0; m < numALLMatls; m++) {
        mass[m] = rho_CC[m][*iter] * vol + SMALL_NUM;
        int_eng_L_ME[m][*iter] = Temp_CC[m][*iter] * cv_CC[m][*iter] * mass[m];
        mom_L_ME[m][*iter]     = vel_CC[m][*iter] * mass[m];
    }
  }
  /*`==========DEBUG============*/ 
  if (d_ice->switchDebugMomentumExchange_CC ) {
    for(int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int dwindex = matl->getDWIndex();
      char description[50];;
      sprintf(description, "MPMICE_momExchange_CC_%d ",dwindex);
      d_ice->printVector( patch,1, description, "xmom_L_ME", 0, mom_L_ME[m]);
      d_ice->printVector( patch,1, description, "ymom_L_ME", 1, mom_L_ME[m]);
      d_ice->printVector( patch,1, description, "zmom_L_ME", 2, mom_L_ME[m]);
      d_ice->printData(   patch,1, description, "int_eng_L_ME",int_eng_L_ME[m]);
    }
  }
    /*==========DEBUG============`*/

#if 0
  cout << "CELL MOMENTUM AFTER CCMOMENTUM EXCHANGE" << endl;
  Vector total_moma(0.,0.,0.);
  for (int m = 0; m < numALLMatls; m++) {
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    Vector matl_mom(0.,0.,0.);
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
	matl_mom += mom_L_ME[m][*iter];
        if(mpm_matl){
	   dvdt_CC[m][*iter] = vel_CC[m][*iter]-vel_CC_old[m][*iter];
        }
    }
    cout << "Momentum for material " << m << " = " << matl_mom << endl;
    total_moma+=matl_mom;
  }
  cout << "TOTAL Momentum AFTER = " << total_moma << endl;
#endif

  //__________________________________
  // This is where I interpolate the CC 
  // changes to NCs for the MPMMatls
  IntVector cIdx[8];

  Vector total_momwithdvdt(0.,0.,0.);
  for(int m = 0; m < numALLMatls; m++){
     Material* matl = d_sharedState->getMaterial( m );
     MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
     if(mpm_matl){
       for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
         patch->findCellsFromNode(*iter,cIdx);
	 gMEvelocity[m][*iter]     = gvelocity[m][*iter];
	 gMEacceleration[m][*iter] = gacceleration[m][*iter];
	 for (int in=0;in<8;in++){
	   gMEvelocity[m][*iter]     +=  dvdt_CC[m][cIdx[in]]*.125;
	   gMEacceleration[m][*iter] += (dvdt_CC[m][cIdx[in]]/delT)*.125;
         }
	 total_momwithdvdt += gMEvelocity[m][*iter]*gmass[m][*iter];
       }
     }
  }
//  cout << "NODE MOMENTUM AFTER CCMOMENTUM EXCHANGE" << endl;
//  cout << "SOLID Momentum WITHDVDT = " << total_momwithdvdt << endl;
  //__________________________________
  //    Put into new_dw
  for (int m = 0; m < numALLMatls; m++) {
    Material* matl = d_sharedState->getMaterial( m );
    int dwindex = matl->getDWIndex();
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
     if(ice_matl){
       new_dw->put(mom_L_ME[m],    Ilb->mom_L_ME_CCLabel,    dwindex, patch);
       new_dw->put(int_eng_L_ME[m],Ilb->int_eng_L_ME_CCLabel,dwindex, patch);
     }
     if(mpm_matl){
      new_dw->put(gMEvelocity[m],
			Mlb->gMomExedVelocityStarLabel,dwindex,patch);
      new_dw->put(gMEacceleration[m],
			Mlb->gMomExedAccelerationLabel,dwindex,patch);
    }
  }  
}

/* --------------------------------------------------------------------- 
 Function~  MPMICE::computeEquilibrationPressure--
 Purpose~   Find the equilibration pressure  
 Reference: Flow of Interpenetrating Material Phases, J. Comp, Phys
               18, 440-464, 1975, see the equilibration section
                   
 Steps
 ----------------
    - Compute rho_micro_CC, SpeedSound, vol_frac for ALL matls

    For each cell
    _ WHILE LOOP(convergence, max_iterations)
        - compute the pressure and dp_drho from the EOS of each material.
        - Compute delta Pressure
        - Compute delta volume fraction and update the 
          volume fraction and the celldensity.
        - Test for convergence of delta pressure and delta volume fraction
    - END WHILE LOOP
    - bulletproofing
    end
 
Note:  The nomenclature follows the reference.
       This is identical to  ICE::computeEquilibrationPressure except
       we now include EOS for MPM matls.                               
_____________________________________________________________________*/
void MPMICE::computeEquilibrationPressure(const ProcessorGroup*,
				       const Patch* patch,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw)
{
  double    converg_coeff = 10;
  double    convergence_crit = converg_coeff * DBL_EPSILON;
  double    sum, tmp;
  int numICEMatls = d_sharedState->getNumICEMatls();
  int numMPMMatls = d_sharedState->getNumMPMMatls();
  int numALLMatls = numICEMatls + numMPMMatls;
  Vector dx       = patch->dCell(); 
  double cell_vol = dx.x()*dx.y()*dx.z();
  char warning[100];
  static int n_passes;                  
  n_passes ++; 
  cout << "Doing calc_equilibration_pressure \t\t MPMICE" << endl;
 
  vector<double> delVol_frac(numALLMatls),press_eos(numALLMatls);
  vector<double> dp_drho(numALLMatls),dp_de(numALLMatls);
  vector<double> mat_volume(numALLMatls);
  vector<double> mat_mass(numALLMatls);
  vector<CCVariable<double> > vol_frac(numALLMatls);
  vector<CCVariable<double> > rho_micro(numALLMatls);
  vector<CCVariable<double> > rho_CC(numALLMatls);
  vector<CCVariable<double> > cv(numALLMatls);
  vector<CCVariable<double> > Temp(numALLMatls);
  vector<CCVariable<double> > speedSound_new(numALLMatls);
  vector<CCVariable<double> > speedSound(numALLMatls);
  vector<CCVariable<double> > sp_vol_CC(numALLMatls);
  vector<CCVariable<double> > sp_vol_equil(numALLMatls);
  vector<CCVariable<double> > mat_vol(numALLMatls);
  vector<CCVariable<double> > mass_CC(numALLMatls);
  CCVariable<double> press, press_new;
  
/*`==========TESTING==========*/ 
  CCVariable<double> scratch;
  new_dw->allocate(scratch,Ilb->press_CCLabel, 0,patch); 
 /*==========TESTING==========`*/ 
  
  old_dw->get(press,         Ilb->press_CCLabel, 0,patch,Ghost::None, 0); 
  new_dw->allocate(press_new,Ilb->press_CCLabel, 0,patch);


  for (int m = 0; m < numALLMatls; m++) {
    Material* matl = d_sharedState->getMaterial( m );
    int dwindex = matl->getDWIndex();
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(ice_matl){                    // I C E
      old_dw->get(cv[m],      Ilb->cv_CCLabel,  dwindex, patch, Ghost::None,0);
   // old_dw->get(rho_CC[m],  Ilb->rho_CCLabel, dwindex, patch, Ghost::None,0);  EXTRA
      old_dw->get(Temp[m],    Ilb->temp_CCLabel,dwindex, patch, Ghost::None,0);
      old_dw->get(mass_CC[m], Ilb->mass_CCLabel,dwindex, patch, Ghost::None,0);
      old_dw->get(sp_vol_CC[m],Ilb->sp_vol_CCLabel,
                                                dwindex, patch, Ghost::None,0);
      new_dw->allocate(speedSound_new[m],
                                     Ilb->speedSound_CCLabel,dwindex, patch);
      new_dw->allocate(rho_micro[m], Ilb->rho_micro_CCLabel, dwindex, patch);
      new_dw->allocate(vol_frac[m],  Ilb->vol_frac_CCLabel,  dwindex, patch);
      new_dw->allocate(rho_CC[m],    Ilb->rho_CCLabel,       dwindex, patch);
      new_dw->allocate(sp_vol_equil[m],
                                     Ilb->sp_vol_equilLabel, dwindex, patch);
   }
    if(mpm_matl){                    // M P M    
      new_dw->get(Temp[m],   MIlb->temp_CCLabel,dwindex, patch, Ghost::None,0);
      new_dw->get(cv[m],     MIlb->cv_CCLabel,  dwindex, patch, Ghost::None,0);
      new_dw->get(mat_vol[m],MIlb->cVolumeLabel,dwindex, patch, Ghost::None,0);
      new_dw->allocate(speedSound_new[m],
                                     MIlb->speedSound_CCLabel,dwindex, patch);
      new_dw->allocate(rho_micro[m], MIlb->rho_micro_CCLabel, dwindex, patch);
      new_dw->allocate(vol_frac[m],  MIlb->vol_frac_CCLabel,  dwindex, patch);  
      new_dw->allocate(rho_CC[m],    MIlb->rho_CCLabel,       dwindex, patch); 
    }
  }

  press_new = press;
/*`==========DEBUG============*/
#if 0
  if(d_ice -> switchDebug_equilibration_press)  { 
       char description[50];
       sprintf(description, "TOPTOP_equilibration_Mat");
       d_ice->printData( patch,1,description, "mass_CC", mass_CC[1]);
       d_ice->printData( patch,1,description, "cVolume", mat_vol[0]);
    }
#endif
 /*==========DEBUG============`*/
  //__________________________________
  // Compute rho_micro, speedSound, volfrac, rho_CC
  for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++) {
    double total_mat_vol = 0.0;
    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);

      if(ice_matl){                // I C E
        double gamma = ice_matl->getGamma();

         rho_micro[m][*iter] =  ice_matl->getEOS()->computeRhoMicro(
                                           press_new[*iter],gamma,
					        cv[m][*iter],Temp[m][*iter]); 

         ice_matl->getEOS()->computePressEOS(rho_micro[m][*iter],gamma,
                                           cv[m][*iter],Temp[m][*iter],
                                           press_eos[m],dp_drho[m], dp_de[m]);

         mat_volume[m] = mass_CC[m][*iter] * sp_vol_CC[m][*iter];
      } 
          
       if(mpm_matl){                //  M P M
/*`==========Ideal gas for testing but need a cleaner way ==========*/         
          //__________________________________
          //  Hardwire for ideal gas 
          double gamma   = mpm_matl->getGamma(); 
          rho_micro[m][*iter] =  
            mpm_matl->getConstitutiveModel()->computeRhoMicro(
                                            press_new[*iter],gamma,
					         cv[m][*iter],Temp[m][*iter]); 

          mpm_matl->getConstitutiveModel()->
            computePressEOS(rho_micro[m][*iter],gamma,
                                            cv[m][*iter],Temp[m][*iter],
                                            press_eos[m],dp_drho[m], dp_de[m]); 

#if 0         
    JOHN:  when I tried this it core dumped.
    //__________________________________
    //  Hardwire for ideal gas 
          double gamma   = mpm_matl->getGamma(); 

          rho_micro[m][*iter] =  
            mpm_matl->getEOSModel()->computeRhoMicro(
                                            press_new[*iter],gamma,
					         cv[m][*iter],Temp[m][*iter]); 
                                                                     
          mpm_matl->getEOSModel()->
            computePressEOS(rho_micro[m][*iter],gamma,
                                            cv[m][*iter],Temp[m][*iter],
                                            press_eos[m],dp_drho[m], dp_de[m]);
#endif 
        mat_volume[m] = mat_vol[m][*iter];
      }              
 /*==========TESTING==========`*/         
      tmp = dp_drho[m] + dp_de[m] * 
                (press_eos[m]/(rho_micro[m][*iter]*rho_micro[m][*iter]));
      speedSound_new[m][*iter] = sqrt(tmp);
     
      total_mat_vol += mat_volume[m];
      
     }
     for (int m = 0; m < numALLMatls; m++) {
       vol_frac[m][*iter] = mat_volume[m]/(total_mat_vol + d_SMALL_NUM);
       rho_CC[m][*iter]   = vol_frac[m][*iter] * rho_micro[m][*iter] + d_SMALL_NUM;
     }
  }
 

/*`==========DEBUG============*/
  if(d_ice -> switchDebug_equilibration_press)  { 
      d_ice->printData( patch, 1, "TOP_equilibration", "Press_CC_top", press);

     for (int m = 0; m < numALLMatls; m++)  {
       Material* matl = d_sharedState->getMaterial( m );
       int dwindex = matl->getDWIndex();
       char description[50];
       sprintf(description, "TOP_equilibration_Mat_%d ",dwindex);
  #if 0
       d_ice->printData( patch,1,description, "rho_CC",     rho_CC[m]);
       d_ice->printData( patch,1,description, "rho_micro",  rho_micro[m]);
  #endif
       d_ice->printData( patch,0,description, "speedSound", speedSound_new[m]);
       d_ice->printData( patch,1,description, "Temp_CC",    Temp[m]);
 #if 0
       d_ice->printData( patch,1,description, "vol_frac_CC",vol_frac[m]);
 #endif
      }
    }
 /*==========DEBUG============`*/

//______________________________________________________________________
// Done with preliminary calcs, now loop over every cell
  int count, test_max_iter = 0;
  for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++) {
    
    IntVector curcell = *iter;    //So I have a chance at finding the bug
    int i,j,k;
    i   = curcell.x();
    j   = curcell.y();
    k   = curcell.z();
      
    double delPress = 0.;
    bool converged  = false;
    count           = 0;
    while ( count < d_ice->d_max_iter_equilibration && converged == false) {
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
/*`==========Need a cleaner way of doing it==========*/ 
       if(mpm_matl){
        //__________________________________
        //  Hardwire for an ideal gas
          double gamma = mpm_matl->getGamma();
          mpm_matl->getConstitutiveModel()->
            computePressEOS(rho_micro[m][*iter],gamma,
                                           cv[m][*iter],Temp[m][*iter],
                                           press_eos[m], dp_drho[m], dp_de[m]);
       }
 /*==========TESTING==========`*/
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
     } 
     double vol_frac_not_close_packed = 1.;
     delPress = (A - vol_frac_not_close_packed - B)/C;
     
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
         sp_vol_equil[m][*iter] = 1.0/rho_micro[m][*iter];
       }
/*`==========Need a cleaner way  to get an EOS in here==========*/ 
       if(mpm_matl){
         //__________________________________
        //  Hardwire ideal gas
            double gamma = mpm_matl->getGamma();
           rho_micro[m][*iter] = 
           mpm_matl->getConstitutiveModel()->computeRhoMicro(press_new[*iter],gamma,
                                             cv[m][*iter],Temp[m][*iter]);
 /*==========TESTING==========`*/
       }
     }
     //__________________________________
     // - compute the updated volume fractions
     //  There are two different way of doing it
     for (int m = 0; m < numALLMatls; m++)  {
       delVol_frac[m]       = -(Q[m] + delPress)/( y[m] + d_SMALL_NUM );
       vol_frac[m][*iter]   = rho_CC[m][*iter]/rho_micro[m][*iter];
     }
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
         speedSound_new[m][*iter] = sqrt(tmp);
       }
/*`========== Need a cleaner way of doing this ==========*/ 
       if(mpm_matl){
         //__________________________________
         //  Hardwire ideal gas
         double gamma = mpm_matl->getGamma();
         mpm_matl->getConstitutiveModel()->
             computePressEOS(rho_micro[m][*iter],gamma,
                                             cv[m][*iter],Temp[m][*iter],
                                             press_eos[m],dp_drho[m], dp_de[m]);

         tmp = dp_drho[m] + dp_de[m] * 
                    (press_eos[m]/(rho_micro[m][*iter]*rho_micro[m][*iter]));
         speedSound_new[m][*iter] = sqrt(tmp);
       }
 /*==========TESTING==========`*/
     }
     //__________________________________
     // - Test for convergence 
     //  If sum of vol_frac_CC ~= 1.0 then converged 
     sum = 0.0;
     for (int m = 0; m < numALLMatls; m++)  {
       sum += vol_frac[m][*iter];
     }
     if (fabs(sum-1.0) < convergence_crit)
       converged = true;
     
    }   // end of converged
    
    
/*`==========TESTING==========*/ 
    scratch[*iter] = delPress;
 /*==========TESTING==========`*/
    
    
    test_max_iter = std::max(test_max_iter, count);
    
    //__________________________________
    //      BULLET PROOFING
    if(test_max_iter == d_ice->d_max_iter_equilibration)
    {
        sprintf(warning, 
        " cell[%d][%d][%d], iter %d, n_passes %d,Now exiting ",
        i,j,k,count,n_passes);
        d_ice->Message(1,"calc_equilibration_press:",
            " Maximum number of iterations was reached ", warning);
    }
    
     for (int m = 0; m < numALLMatls; m++) {
         ASSERT(( vol_frac[m][*iter] > 0.0 ) ||
                ( vol_frac[m][*iter] < 1.0));
     }
    if ( fabs(sum - 1.0) > convergence_crit)   {
        sprintf(warning, 
        " cell[%d][%d][%d], iter %d, n_passes %d,Now exiting ",
        i,j,k,count,n_passes);
        d_ice ->Message(1,"calc_equilibration_press:",
            " sum(vol_frac_CC) != 1.0", warning);
    }

    if ( press_new[*iter] < 0.0 )   {
        sprintf(warning, 
        " cell[%d][%d][%d], iter %d, n_passes %d, Now exiting",
         i,j,k, count, n_passes);
       d_ice->Message(1,"calc_equilibration_press:", 
            " press_new[iter*] < 0", warning);
    }

    for (int m = 0; m < numALLMatls; m++)
    if ( rho_micro[m][*iter] < 0.0 || vol_frac[m][*iter] < 0.0) {
        sprintf(warning, 
        " cell[%d][%d][%d], mat %d, iter %d, n_passes %d,Now exiting ",
        i,j,k,m,count,n_passes);
        d_ice->Message(1," calc_equilibration_press:", 
            " rho_micro < 0 || vol_frac < 0", warning);
    }
  }     // end of cell interator

    fprintf(stderr, "\tmax number of iterations in any cell %i\n",test_max_iter);


/*`==========TESTING==========*/ 
  //__________________________________
  // Now change how rho_CC is defined to 
  // rho_CC = mass_CC/cell_volume  NOT mass/mat_volume 
  for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++) {
     for (int m = 0; m < numALLMatls; m++) {
       Material* matl = d_sharedState->getMaterial( m );
       ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
       MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);

       if(ice_matl){                // I C E                                      
        mat_mass[m] = mass_CC[m][*iter];
       } 
          
       if(mpm_matl){                //  M P M
        mat_mass[m] = rho_micro[m][*iter] * mat_vol[m][*iter];
       }
       rho_CC[m][*iter]   = mat_mass[m]/cell_vol + d_SMALL_NUM;           
     }
  }
 /*==========TESTING==========`*/


  /*__________________________________
   *   THIS NEEDS TO BE FIXED 
   *   WE NEED TO UPDATE BC_VALUES NOT PRESSURE
   *   SINCE WE USE BC_VALUES LATER ON IN THE CODE
   *___________________________________*/
   for (int m = 0; m < numALLMatls; m++)   {
     d_ice->setBC(rho_CC[m],   "Density" ,patch);
  }  
  d_ice->setBC(press_new,"Pressure",patch);

  //__________________________________
  //    Put all matls into new dw
  for (int m = 0; m < numALLMatls; m++)   {
    Material* matl = d_sharedState->getMaterial( m );
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    int dwindex = matl->getDWIndex();
    if(ice_matl){
      new_dw->put( vol_frac[m],      Ilb->vol_frac_CCLabel,   dwindex, patch);
      new_dw->put( speedSound_new[m],Ilb->speedSound_CCLabel, dwindex, patch);
      new_dw->put( rho_micro[m],     Ilb->rho_micro_CCLabel,  dwindex, patch);
      new_dw->put( rho_CC[m],        Ilb->rho_CCLabel,        dwindex, patch);
      new_dw->put( sp_vol_equil[m],  Ilb->sp_vol_equilLabel,  dwindex, patch);
    }
    if(mpm_matl){
      new_dw->put( vol_frac[m],      MIlb->vol_frac_CCLabel,  dwindex, patch);
      new_dw->put( speedSound_new[m],MIlb->speedSound_CCLabel,dwindex, patch);
      new_dw->put( rho_micro[m],     MIlb->rho_micro_CCLabel, dwindex, patch);
      new_dw->put( rho_CC[m],        MIlb->rho_CCLabel,       dwindex, patch);
    }
  }
  new_dw->put(press_new,Ilb->press_equil_CCLabel,0,patch);
  
 /*`==========DEBUG============*/
  if(d_ice -> switchDebug_equilibration_press)  { 
    d_ice->printData( patch, 1, "BOTTOM", "Press_CC_equil", press_new);
    d_ice->printData( patch, 1, "BOTTOM", "delPress",       scratch);
 #if 0                 
    for (int m = 0; m < numALLMatls; m++)  {
       Material* matl = d_sharedState->getMaterial( m );
       int dwindex = matl->getDWIndex(); 
       char description[50];
       sprintf(description, "BOT_equilibration_Mat_%d ",dwindex);
       d_ice->printData( patch,1,description, "rho_CC",      rho_CC[m]);
     //d_ice->printData( patch,1,description, "speedSound", speedSound_new[m]);
       d_ice->printData( patch,1,description, "rho_micro_CC",rho_micro[m]);
       d_ice->printData( patch,1,description, "vol_frac_CC", vol_frac[m]);
    }
   #endif
  }
 /*==========DEBUG============`*/
  
}
