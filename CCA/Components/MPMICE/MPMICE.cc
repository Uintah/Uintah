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
#include <Packages/Uintah/Core/Grid/CellIterator.h>

#include <Core/Datatypes/DenseMatrix.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

#define DOING
#undef DOING

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

/*`==========TESTING==========  HACK: so we can get mass exchange off the ground*/
#define HMX 1
#define GAS 0
 /*==========TESTING==========`*/
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
  
  ProblemSpecP mat_ps       =  prob_spec->findBlock("MaterialProperties");
  ProblemSpecP mpm_ice_ps   =  mat_ps->findBlock("MPMICE");
  ProblemSpecP exch_ps = mpm_ice_ps->findBlock("exchange_coefficients");
  exch_ps->require("momentum",d_K_mom);
  exch_ps->require("heat",d_K_heat);
  cerr << "Pulled out exchange coefficients of the input file \t\t MPMICE" 
       << endl;
  

   cerr << "Done with problemSetup \t\t\t MPMICE" <<endl;
   cerr << "--------------------------------\n"<<endl;
}
//______________________________________________________________________
//
void MPMICE::scheduleInitialize(const LevelP& level,
				SchedulerP& sched,
				DataWarehouseP& dw)
{
  d_mpm->scheduleInitialize(level, sched, dw);
  d_ice->scheduleInitialize(level, sched, dw);
  cerr << "Doing Initialization \t\t\t MPMICE" <<endl;
  cerr << "--------------------------------\n"<<endl; 
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
      d_mpm->scheduleComputeConnectivity(patch,sched,old_dw,new_dw);
    }
    d_mpm->scheduleInterpolateParticlesToGrid(patch,sched,old_dw,new_dw);

    if (MPMPhysicalModules::thermalContactModel) {
       d_mpm->scheduleComputeHeatExchange(patch,sched,old_dw,new_dw);
    }

    // schedule the interpolation of mass and volume to the cell centers
    scheduleInterpolateNCToCC_0(patch,sched,old_dw,new_dw);
    scheduleComputeEquilibrationPressure(patch,sched,old_dw,new_dw);

    d_ice->scheduleComputeFaceCenteredVelocities(patch,sched,old_dw,new_dw);
    d_ice->scheduleAddExchangeContributionToFCVel(patch,sched,old_dw,new_dw);
    d_ice->scheduleComputeDelPressAndUpdatePressCC(patch,sched,old_dw,new_dw);

    // scheduleInterpolateVelIncFCToNC(patch,sched,old_dw,new_dw);

    d_mpm->scheduleExMomInterpolated(               patch,sched,old_dw,new_dw);
    d_mpm->scheduleComputeStressTensor(             patch,sched,old_dw,new_dw);
     
    d_ice->scheduleComputePressFC(                  patch,sched,old_dw,new_dw);
  //  scheduleMassExchange(                           patch,sched,old_dw,new_dw);
    d_ice->scheduleAccumulateMomentumSourceSinks(   patch,sched,old_dw,new_dw);
    d_ice->scheduleAccumulateEnergySourceSinks(     patch,sched,old_dw,new_dw);
    d_ice->scheduleComputeLagrangianValues(         patch,sched,old_dw,new_dw);

    scheduleInterpolatePressCCToPressNC(            patch,sched,old_dw,new_dw);
    scheduleInterpolatePAndGradP(                   patch,sched,old_dw,new_dw);
   
    d_mpm->scheduleComputeInternalForce(            patch,sched,old_dw,new_dw);
    d_mpm->scheduleComputeInternalHeatRate(         patch,sched,old_dw,new_dw);
    d_mpm->scheduleSolveEquationsMotion(            patch,sched,old_dw,new_dw);
    d_mpm->scheduleSolveHeatEquations(              patch,sched,old_dw,new_dw);
    d_mpm->scheduleIntegrateAcceleration(           patch,sched,old_dw,new_dw);
    d_mpm->scheduleIntegrateTemperatureRate(        patch,sched,old_dw,new_dw);

    scheduleInterpolateNCToCC(                      patch,sched,old_dw,new_dw);
    scheduleCCMomExchange(                          patch,sched,old_dw,new_dw);
    scheduleInterpolateCCToNC(                      patch,sched,old_dw,new_dw);
//    d_mpm->scheduleExMomIntegrated(patch,sched,old_dw,new_dw);
//    d_ice->scheduleAddExchangeToMomentumAndEnergy(patch,sched,old_dw,new_dw);
    d_mpm->scheduleInterpolateToParticlesAndUpdate( patch,sched,old_dw,new_dw);
    d_mpm->scheduleComputeMassRate(                 patch,sched,old_dw,new_dw);
    if(d_fracture) {
      d_mpm->scheduleComputeFracture(patch,sched,old_dw,new_dw);
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


void MPMICE::scheduleInterpolatePressCCToPressNC(const Patch* patch,
						 SchedulerP& sched,
						 DataWarehouseP& old_dw,
						 DataWarehouseP& new_dw)
{
  Task* t=scinew Task("MPMICE::interpolatePressCCToPressNC",
		      patch, old_dw, new_dw,
		      this, &MPMICE::interpolatePressCCToPressNC);
  
  t->requires(new_dw,Ilb->press_CCLabel,0, patch, Ghost::AroundCells, 1);
  t->computes(new_dw, MIlb->press_NCLabel,0,patch);
  
  sched->addTask(t);

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

   t->requires(new_dw,MIlb->press_NCLabel,0, patch, Ghost::AroundCells, 1);

   for(int m = 0; m < numMPMMatls; m++){
     MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
     int idx = mpm_matl->getDWIndex();
     t->requires(old_dw, Mlb->pXLabel,            idx, patch, Ghost::None);
     t->requires(new_dw, Ilb->mom_source_CCLabel,idx, patch,
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
     t->requires(new_dw, Ilb->uvel_FCLabel,     idx, patch,Ghost::None);
     t->requires(new_dw, Ilb->vvel_FCLabel,     idx, patch,Ghost::None);
     t->requires(new_dw, Ilb->wvel_FCLabel,     idx, patch,Ghost::None);
     t->requires(new_dw, Ilb->uvel_FCMELabel,   idx, patch,Ghost::None);
     t->requires(new_dw, Ilb->vvel_FCMELabel,   idx, patch,Ghost::None);
     t->requires(new_dw, Ilb->wvel_FCMELabel,   idx, patch,Ghost::None);

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
     t->computes(new_dw, MIlb->vel_CCLabel,        idx, patch);
     t->computes(new_dw, MIlb->temp_CCLabel,       idx, patch);
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
     t->requires(new_dw, Mlb->gTemperatureStarLabel,idx, patch,
		Ghost::AroundCells, 1);

     t->computes(new_dw, Ilb->mom_L_CCLabel,     idx, patch);
     t->computes(new_dw, Ilb->int_eng_L_CCLabel, idx, patch);
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
    int idx = matl->getDWIndex();
    if(ice_matl){                   // I C E
     t->computes(new_dw,Ilb->mom_L_ME_CCLabel,     idx,patch);
     t->computes(new_dw,Ilb->int_eng_L_ME_CCLabel, idx,patch);
    }
    if(mpm_matl){                    // M P M
     t->requires(new_dw, Mlb->gVelocityStarLabel,   idx,patch,Ghost::None,0);
     t->requires(new_dw, Mlb->gAccelerationLabel,   idx,patch,Ghost::None,0);
     t->computes(new_dw, MIlb->dTdt_CCLabel,              idx, patch);
     t->computes(new_dw, MIlb->dvdt_CCLabel,              idx, patch);
    }
    t->requires(new_dw,  Ilb->rho_CCLabel,         idx,patch,Ghost::None);
    t->requires(new_dw,  Ilb->mom_L_CCLabel,       idx,patch,Ghost::None,0);
    t->requires(new_dw,  Ilb->int_eng_L_CCLabel,   idx,patch,Ghost::None,0);
    t->requires(new_dw,  Ilb->rho_micro_CCLabel,   idx,patch,Ghost::None,0);
    t->requires(new_dw,  Ilb->vol_frac_CCLabel,    idx,patch,Ghost::None);
  }
   sched->addTask(t);

}

void MPMICE::scheduleInterpolateCCToNC(const Patch* patch,
				       SchedulerP& sched,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw)
{
  Task* t=scinew Task("MPMICE::interpolateCCToNC",
		      patch, old_dw, new_dw,
		      this, &MPMICE::interpolateCCToNC);
  int numALLMatls  = d_sharedState->getNumMatls();                      
  for (int m = 0; m < numALLMatls; m++) {
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    int idx = matl->getDWIndex();
    if(mpm_matl){                    // M P M
      t->requires(new_dw, Mlb->gVelocityStarLabel,   idx,patch,Ghost::None,0);
      t->requires(new_dw, Mlb->gAccelerationLabel,   idx,patch,Ghost::None,0);
      t->requires(new_dw, MIlb->dTdt_CCLabel,idx,patch,Ghost::AroundCells,1);
      t->requires(new_dw, MIlb->dvdt_CCLabel,idx,patch,Ghost::AroundCells,1);

      
      t->computes(new_dw, Mlb->gMomExedVelocityStarLabel, idx, patch);
      t->computes(new_dw, Mlb->gMomExedAccelerationLabel, idx, patch);
      t->computes(new_dw, Mlb->dTdt_NCLabel,              idx, patch);
    }
  }
  sched->addTask(t);

}
/* ---------------------------------------------------------------------
 Function~  MPMICE::scheduleComputeEquilibrationPressure--
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
      task->requires(old_dw,Ilb->temp_CCLabel,dwindex,patch,Ghost::None);
      task->requires(old_dw,Ilb->mass_CCLabel,dwindex,patch,Ghost::None);
      task->requires(old_dw,Ilb->sp_vol_CCLabel,dwindex,patch,Ghost::None);
    }
    if(mpm_matl){                    // M P M
      task->requires(new_dw,MIlb->temp_CCLabel,dwindex,patch,Ghost::None);
      task->requires(new_dw,MIlb->cVolumeLabel,dwindex,patch,Ghost::None);
    }
    // For all materials
    task->computes(new_dw, Ilb->speedSound_CCLabel,dwindex,patch);
    task->computes(new_dw, Ilb->rho_micro_CCLabel, dwindex,patch);
    task->computes(new_dw, Ilb->vol_frac_CCLabel,  dwindex,patch);
    task->computes(new_dw, Ilb->rho_CCLabel,       dwindex,patch);
 } 

  task->computes(new_dw,Ilb->press_equil_CCLabel,0, patch);
  sched->addTask(task);
}
/* ---------------------------------------------------------------------
 Function~  MPMICE::scheduleMassExchange--
_____________________________________________________________________*/
void  MPMICE::scheduleMassExchange(const Patch* patch,
					SchedulerP& sched,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw)

{
  Task* task = scinew Task("MPMICE::massExchange",
                        patch, old_dw, new_dw, this, &MPMICE::massExchange);
  int numMatls=d_sharedState->getNumMatls(); 

  for (int m = 0; m < numMatls; m++)  {
    Material* matl = d_sharedState->getMaterial(m);
    int dwindex = matl->getDWIndex();  
    task->computes(new_dw,  Ilb->mass_sourceLabel, dwindex, patch);    
  }
  sched->addTask(task);
}
//______________________________________________________________________
//       A C T U A L   S T E P S :
//______________________________________________________________________
//


void MPMICE::interpolatePressCCToPressNC(const ProcessorGroup*,
				       const Patch* patch,
				       DataWarehouseP&,
				       DataWarehouseP& new_dw)
{			       


//  cout << "Doing interpolatePressCCToPressNC \t\t MPMICE" << endl;
  
  CCVariable<double> pressCC;
  NCVariable<double> pressNC;

  new_dw->get(pressCC, Ilb->press_CCLabel, 0, patch,Ghost::AroundCells, 1);
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
  new_dw->put(pressNC,MIlb->press_NCLabel,0,patch);

}




//______________________________________________________________________
//
void MPMICE::interpolatePAndGradP(const ProcessorGroup*,
                                  const Patch* patch,
                                  DataWarehouseP& old_dw,
                                  DataWarehouseP& new_dw)
{
//  cout << "Doing interpolatePressureToParticles \t\t MPMICE" << endl;
  
  NCVariable<double> pressNC;
  IntVector ni[8];
  double S[8];
  Vector zero(0.,0.,0.);
  IntVector cIdx[8];
  
  new_dw->get(pressNC,MIlb->press_NCLabel,0,patch,Ghost::AroundCells,1);

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
    new_dw->get(mom_source,       Ilb->mom_source_CCLabel, dwindex, patch,
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
  SFCXVariable<double> uvel_FC, uvel_FCME;
  SFCYVariable<double> vvel_FC, vvel_FCME;
  SFCZVariable<double> wvel_FC, wvel_FCME;
  CCVariable<Vector> velInc_CC;
  NCVariable<Vector> velInc_NC;
  NCVariable<Vector> gvelocity;

  for(int m = 0; m < numMatls; m++) {
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    int dwindex = mpm_matl->getDWIndex();
    new_dw->get(uvel_FC,   Ilb->uvel_FCLabel,   dwindex, patch, Ghost::None,0);
    new_dw->get(vvel_FC,   Ilb->vvel_FCLabel,   dwindex, patch, Ghost::None,0);
    new_dw->get(wvel_FC,   Ilb->wvel_FCLabel,   dwindex, patch, Ghost::None,0);
    new_dw->get(uvel_FCME, Ilb->uvel_FCMELabel, dwindex, patch, Ghost::None,0);
    new_dw->get(vvel_FCME, Ilb->vvel_FCMELabel, dwindex, patch, Ghost::None,0);
    new_dw->get(wvel_FCME, Ilb->wvel_FCMELabel, dwindex, patch, Ghost::None,0);

    new_dw->get(gvelocity,Mlb->gVelocityLabel, dwindex, patch, Ghost::None,0);

    new_dw->allocate(velInc_CC, MIlb->velInc_CCLabel, dwindex, patch);
    new_dw->allocate(velInc_NC, MIlb->velInc_NCLabel, dwindex, patch);
    double xcomp,ycomp,zcomp;

    for(CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
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
  double cell_vol = dx.x()*dx.y()*dx.z();
//  cout << "Doing interpolateNCToCC_0 \t\t\t MPMICE" << endl;

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    int matlindex = mpm_matl->getDWIndex();
    
    // Create arrays for the grid data
    NCVariable<double> gmass, gvolume;
    NCVariable<Vector> gvelocity;
    CCVariable<double> cmass, cvolume,Temp_CC;
    CCVariable<Vector> vel_CC;
    
    new_dw->allocate(cmass,     MIlb->cMassLabel,       matlindex, patch);
    new_dw->allocate(cvolume,   MIlb->cVolumeLabel,     matlindex, patch);
    new_dw->allocate(vel_CC,    MIlb->vel_CCLabel,      matlindex, patch);
    new_dw->allocate(Temp_CC,   MIlb->temp_CCLabel,     matlindex, patch);
    
    cmass.initialize(d_SMALL_NUM*cell_vol);
    cvolume.initialize(d_SMALL_NUM);
    vel_CC.initialize(zero); 
    
    new_dw->get(gmass,Mlb->gMassLabel,matlindex, patch,Ghost::AroundCells, 1);
    new_dw->get(gvolume,Mlb->gVolumeLabel,matlindex, patch,
		Ghost::AroundCells, 1);
    new_dw->get(gvelocity,Mlb->gVelocityLabel,matlindex, patch,
		Ghost::AroundCells, 1);
    
    IntVector nodeIdx[8];
    
    // This temp is only used in computeEquilibrationPressure
    // so for now it's hardwired
    Temp_CC.initialize(300.0);
    
    
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
      vel_CC[*iter]      /= cmass[*iter];
    }
//    cout << "Solid matl CC momentum = " << cell_mom << endl;
    
    //  Set BC's and put into new_dw
    d_ice->setBC(vel_CC,  "Velocity",   patch);
    d_ice->setBC(Temp_CC, "Temperature",patch);
    
    new_dw->put(cmass,    MIlb->cMassLabel,       matlindex, patch);
    new_dw->put(cvolume,  MIlb->cVolumeLabel,     matlindex, patch);
    new_dw->put(vel_CC,   MIlb->vel_CCLabel,      matlindex, patch);
    new_dw->put(Temp_CC,  MIlb->temp_CCLabel,     matlindex, patch);
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
     NCVariable<double> gmass, gvolume,gtempstar;
     NCVariable<Vector> gvelocity, gacc;
     CCVariable<Vector> cmomentum;
     CCVariable<double> int_eng;

     new_dw->get(gmass,     Mlb->gMassLabel,           matlindex, patch,
							Ghost::AroundCells, 1);
     new_dw->get(gvelocity, Mlb->gVelocityStarLabel,   matlindex, patch,
							Ghost::AroundCells, 1);
     new_dw->get(gacc,      Mlb->gAccelerationLabel,   matlindex, patch,
							Ghost::AroundCells, 1);
     new_dw->get(gtempstar, Mlb->gTemperatureStarLabel,matlindex, patch,
							Ghost::AroundCells, 1);

     new_dw->allocate(cmomentum,  Ilb->mom_L_CCLabel,      matlindex, patch);
     new_dw->allocate(int_eng,    Ilb->int_eng_L_CCLabel,  matlindex, patch);
 
     cmomentum.initialize(zero);
     int_eng.initialize(0.); 
     double cv = mpm_matl->getSpecificHeat();

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
 	 int_eng[*iter]   += gtempstar[nodeIdx[in]]*gmass[nodeIdx[in]]*cv*.125;
       }
     }

     new_dw->put(cmomentum,     Ilb->mom_L_CCLabel, matlindex, patch);
     new_dw->put(int_eng,       Ilb->int_eng_L_CCLabel,  matlindex, patch);
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

  vector<CCVariable<double> > rho_CC(numALLMatls);
  vector<CCVariable<double> > Temp_CC(numALLMatls);  
  vector<CCVariable<double> > vol_frac_CC(numALLMatls);
  vector<CCVariable<double> > rho_micro_CC(numALLMatls);

  vector<CCVariable<Vector> > mom_L(numALLMatls);
  vector<CCVariable<double> > int_eng_L(numALLMatls);
  vector<CCVariable<double> > cmass(numALLMatls);

  // Create variables for the results
  vector<CCVariable<Vector> > mom_L_ME(numALLMatls);
  vector<CCVariable<Vector> > vel_CC(numALLMatls);
  vector<CCVariable<Vector> > dvdt_CC(numALLMatls);
  vector<CCVariable<double> > dTdt_CC(numALLMatls);
  vector<NCVariable<double> > dTdt_NC(numALLMatls);
  vector<CCVariable<double> > int_eng_L_ME(numALLMatls);

  vector<double> b(numALLMatls);
  vector<double> mass(numALLMatls);
  vector<double> density(numALLMatls);
  vector<double> cv(numALLMatls);
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
    int dwindex = matl->getDWIndex();
    if(mpm_matl){
      new_dw->get(gvelocity[m],Mlb->gVelocityStarLabel, dwindex, patch,
		  Ghost::None, 0);
      new_dw->get(gacceleration[m],Mlb->gAccelerationLabel,dwindex, patch,
		  Ghost::None, 0);
      new_dw->get(cmass[m], MIlb->cMassLabel, dwindex, patch,
		  Ghost::None, 0);
      new_dw->allocate(vel_CC[m],  MIlb->velstar_CCLabel, dwindex, patch);
      new_dw->allocate(Temp_CC[m], MIlb->temp_CC_scratchLabel,dwindex, patch);
      cv[m] = mpm_matl->getSpecificHeat();
    }
    if(ice_matl){
      new_dw->allocate(vel_CC[m], Ilb->vel_CCLabel,  dwindex,patch);
      new_dw->allocate(Temp_CC[m],Ilb->temp_CCLabel, dwindex,patch);
      cv[m] = ice_matl->getSpecificHeat();
    }

    new_dw->get(rho_CC[m],        Ilb->rho_CCLabel,       dwindex, patch,
							Ghost::None, 0);
    new_dw->get(rho_micro_CC[m],  Ilb->rho_micro_CCLabel, dwindex, patch,
							Ghost::None, 0);
    new_dw->get(mom_L[m],         Ilb->mom_L_CCLabel,     dwindex, patch,
							Ghost::None, 0);
    new_dw->get(int_eng_L[m],     Ilb->int_eng_L_CCLabel, dwindex, patch,
							Ghost::None, 0);
    new_dw->get(vol_frac_CC[m],   Ilb->vol_frac_CCLabel,  dwindex, patch,
							Ghost::None, 0);
    new_dw->allocate(dvdt_CC[m], MIlb->dvdt_CCLabel,      dwindex, patch);
    new_dw->allocate(dTdt_CC[m], MIlb->dTdt_CCLabel,      dwindex, patch);
    new_dw->allocate(mom_L_ME[m], Ilb->mom_L_ME_CCLabel,  dwindex,patch);
    new_dw->allocate(int_eng_L_ME[m],Ilb->int_eng_L_ME_CCLabel,dwindex,patch);

    dvdt_CC[m].initialize(zero);
    dTdt_CC[m].initialize(0.);
  }

  double vol = dx.x()*dx.y()*dx.z();
  double tmp;

  // Convert momenta to velocities.  Slightly different for MPM and ICE.
  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
    for (int m = 0; m < numALLMatls; m++) {
      mass[m]           = rho_CC[m][*iter] * vol;
      Temp_CC[m][*iter] = int_eng_L[m][*iter]/(mass[m]*cv[m]);
      vel_CC[m][*iter]  = mom_L[m][*iter]/mass[m];
    }
  }


#if 0
  cout << "GRID MOMENTUM BEFORE CCMOMENTUM EXCHANGE" << endl;
  Vector total_mom(0.,0.,0.);
  for (int m = 0; m < numALLMatls; m++) {
    Vector matl_mom(0.,0.,0.);
    for(CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
	matl_mom += mom_L[m][*iter];
    }
    cout << "Momentum for material " << m << " = " << matl_mom << endl;
    total_mom+=matl_mom;
  }
  cout << "TOTAL Momentum BEFORE = " << total_mom << endl;
#endif

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
    acopy.solve(b);
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
    acopy.solve(b);
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
    acopy.solve(b);
    for(int m = 0; m < numALLMatls; m++)  {
	vel_CC[m][*iter].z( vel_CC[m][*iter].z() + b[m] );
	dvdt_CC[m][*iter].z( b[m] );
    }

    //---------- E N E R G Y   E X C H A N G E
    //         
    for(int m = 0; m < numALLMatls; m++) {
      tmp = cv[m]*rho_micro_CC[m][*iter];
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
    a.solve(b);
    
    for(int m = 0; m < numALLMatls; m++) {
      Temp_CC[m][*iter] = Temp_CC[m][*iter] + b[m];
      dTdt_CC[m][*iter] = b[m];
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
  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
    for (int m = 0; m < numALLMatls; m++) {
        mass[m] = rho_CC[m][*iter] * vol;
        int_eng_L_ME[m][*iter] = Temp_CC[m][*iter] * cv[m] * mass[m];
        mom_L_ME[m][*iter]     = vel_CC[m][*iter] * mass[m];
    }
  }
  //---- P R I N T   D A T A ------ 
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

  // Cut and now have to do the interpolation in a separate function.

#if 0
  cout << "CELL MOMENTUM AFTER CCMOMENTUM EXCHANGE" << endl;
  Vector total_moma(0.,0.,0.);
  for (int m = 0; m < numALLMatls; m++) {
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    Vector matl_mom(0.,0.,0.);
    for(CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
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


  //HARDWIRING FOR NEUMANN TEMPERATURE BCS
  for(int m = 0; m < numALLMatls; m++){
    for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
	dTdt_CC[m].fillFace(face,0);
    }
  }

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
      new_dw->put(dvdt_CC[m],MIlb->dvdt_CCLabel,dwindex,patch);
      new_dw->put(dTdt_CC[m],MIlb->dTdt_CCLabel,dwindex,patch);
    }
  }  
}

void MPMICE::interpolateCCToNC(const ProcessorGroup*,
			       const Patch* patch,
			       DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw)
{
//  cout << "Doing interpolation of CC to NC  \t\t MPMICE" << endl;
  //__________________________________
  // This is where I interpolate the CC 
  // changes to NCs for the MPMMatls

  int numALLMatls = d_sharedState->getNumMPMMatls() + 
    d_sharedState->getNumICEMatls();

  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label());

  vector<CCVariable<Vector> > dvdt_CC(numALLMatls);
  vector<CCVariable<double> > dTdt_CC(numALLMatls);
  vector<NCVariable<Vector> > gacceleration(numALLMatls);
  vector<NCVariable<Vector> > gvelocity(numALLMatls);

  vector<NCVariable<Vector> > gMEacceleration(numALLMatls);
  vector<NCVariable<Vector> > gMEvelocity(numALLMatls);
  vector<NCVariable<double> > dTdt_NC(numALLMatls);

  for (int m = 0; m < numALLMatls; m++) {
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    int dwindex = matl->getDWIndex();
    if(mpm_matl){
      new_dw->get(gvelocity[m],Mlb->gVelocityStarLabel, dwindex, patch,
		  Ghost::None, 0);
      new_dw->get(gacceleration[m],Mlb->gAccelerationLabel,dwindex, patch,
		  Ghost::None, 0);
      new_dw->get(dvdt_CC[m], MIlb->dvdt_CCLabel,dwindex, patch,
		  Ghost::AroundCells,1);
      new_dw->get(dTdt_CC[m], MIlb->dTdt_CCLabel,dwindex, patch,
		  Ghost::AroundCells,1);      
      new_dw->allocate(gMEvelocity[m], Mlb->gMomExedVelocityStarLabel,
		       dwindex, patch);
      new_dw->allocate(gMEacceleration[m], Mlb->gMomExedAccelerationLabel,
		       dwindex, patch);
      new_dw->allocate(dTdt_NC[m], Mlb->dTdt_NCLabel,        dwindex, patch);

    }
  }

  
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
	 dTdt_NC[m][*iter]         = 0.0;
	 for (int in=0;in<8;in++){
	   gMEvelocity[m][*iter]     +=  dvdt_CC[m][cIdx[in]]*.125;
	   gMEacceleration[m][*iter] += (dvdt_CC[m][cIdx[in]]/delT)*.125;
	   dTdt_NC[m][*iter]         += (dTdt_CC[m][cIdx[in]]/delT)*.125;
         }
       }
     }
  }
  //__________________________________
  //    Put into new_dw
  for (int m = 0; m < numALLMatls; m++) {
    Material* matl = d_sharedState->getMaterial( m );
    int dwindex = matl->getDWIndex();
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      new_dw->put(gMEvelocity[m],Mlb->gMomExedVelocityStarLabel,dwindex,patch);
      new_dw->put(gMEacceleration[m],Mlb->gMomExedAccelerationLabel,dwindex,
		  patch);
      new_dw->put(dTdt_NC[m],Mlb->dTdt_NCLabel, dwindex,patch);
    }
  }  

}



/* --------------------------------------------------------------------- 
 Function~  MPMICE::computeEquilibrationPressure-- 
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
  double    converg_coeff = 100.;
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
//  cout << "Doing calc_equilibration_pressure \t\t MPMICE" << endl;
 
  vector<double> delVol_frac(numALLMatls),press_eos(numALLMatls);
  vector<double> dp_drho(numALLMatls),dp_de(numALLMatls);
  vector<double> mat_volume(numALLMatls);
  vector<double> mat_mass(numALLMatls);
  vector<double> cv(numALLMatls);
  vector<CCVariable<double> > vol_frac(numALLMatls);
  vector<CCVariable<double> > rho_micro(numALLMatls);
  vector<CCVariable<double> > rho_CC(numALLMatls);
  vector<CCVariable<double> > Temp(numALLMatls);
  vector<CCVariable<double> > speedSound_new(numALLMatls);
  vector<CCVariable<double> > speedSound(numALLMatls);
  vector<CCVariable<double> > sp_vol_CC(numALLMatls);
  vector<CCVariable<double> > mat_vol(numALLMatls);
  vector<CCVariable<double> > mass_CC(numALLMatls);
  CCVariable<double> press, press_new;
  
/**/  CCVariable<double> scratch;
/**/  new_dw->allocate(scratch,Ilb->press_CCLabel, 0,patch); 
  
  old_dw->get(press,         Ilb->press_CCLabel, 0,patch,Ghost::None, 0); 
  new_dw->allocate(press_new,Ilb->press_CCLabel, 0,patch);


  for (int m = 0; m < numALLMatls; m++) {
    Material* matl = d_sharedState->getMaterial( m );
    int dwindex = matl->getDWIndex();
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(ice_matl){                    // I C E
      old_dw->get(Temp[m],Ilb->temp_CCLabel,dwindex, patch, Ghost::None,0);
      old_dw->get(mass_CC[m],Ilb->mass_CCLabel,dwindex, patch, Ghost::None,0);
      old_dw->get(sp_vol_CC[m],Ilb->sp_vol_CCLabel,dwindex,patch,
		  Ghost::None,0);
      cv[m] = ice_matl->getSpecificHeat();
    }
    if(mpm_matl){                    // M P M    
      new_dw->get(Temp[m],   MIlb->temp_CCLabel,dwindex, patch, Ghost::None,0);
      new_dw->get(mat_vol[m],MIlb->cVolumeLabel,dwindex, patch, Ghost::None,0);
      new_dw->get(mass_CC[m],MIlb->cMassLabel,  dwindex, patch, Ghost::None,0);
      cv[m] = mpm_matl->getSpecificHeat();
    }
    new_dw->allocate(rho_CC[m],    Ilb->rho_CCLabel,       dwindex, patch);
    new_dw->allocate(vol_frac[m],  Ilb->vol_frac_CCLabel,  dwindex, patch);
    new_dw->allocate(rho_micro[m], Ilb->rho_micro_CCLabel, dwindex, patch);
    new_dw->allocate(speedSound_new[m],Ilb->speedSound_CCLabel,dwindex, patch);
  }

  press_new = press;

//---- P R I N T   D A T A ------
#if 0

  if(d_ice -> switchDebug_equilibration_press)  { 
    for (int m = 0; m < numALLMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      int dwindex = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      char description[50];
      sprintf(description, "TOPTOP_equilibration_Mat_%d ",dwindex);
      d_ice->printData( patch,1,description, "mass_CC", mass_CC[m]);
      if (ice_matl) {
	d_ice->printData( patch,1,description, "sp_vol_CC", sp_vol_CC[m]);
      }
      if (mpm_matl) {
	d_ice->printData( patch,1,description, "cVolume", mat_vol[m]);
      }
    }
  }
#endif
  //__________________________________
  // Compute rho_micro, speedSound, volfrac, rho_CC
  for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++) {
    double total_mat_vol = 0.0;
    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      
      if(ice_matl){                // I C E
	rho_micro[m][*iter] = 1.0/sp_vol_CC[m][*iter];
	double gamma   = ice_matl->getGamma(); 
	ice_matl->getEOS()->computePressEOS(rho_micro[m][*iter],gamma,
					    cv[m],Temp[m][*iter],
					    press_eos[m],dp_drho[m], dp_de[m]);
	
	mat_volume[m] = mass_CC[m][*iter] * sp_vol_CC[m][*iter];
	
	tmp = dp_drho[m] + dp_de[m] * 
	  (press_eos[m]/(rho_micro[m][*iter]*rho_micro[m][*iter]));
      } 
      
      if(mpm_matl){                //  M P M
#if 0
	rho_micro[m][*iter] =  mpm_matl->getConstitutiveModel()->
	  computeRhoMicroCM(press_new[*iter],mpm_matl);

	mpm_matl->getConstitutiveModel()->
	  computePressEOSCM(rho_micro[m][*iter],press_eos[m],dp_drho[m],
			    tmp,mpm_matl);
#endif
	mat_volume[m] = mat_vol[m][*iter];

//    This is the IDEAL GAS stuff
#if 1
	double gamma   = mpm_matl->getGamma(); 
	rho_micro[m][*iter] = mpm_matl->
	  getConstitutiveModel()->computeRhoMicro(press_new[*iter],gamma,
						  cv[m],Temp[m][*iter]); 
	mpm_matl->getConstitutiveModel()->
	  computePressEOS(rho_micro[m][*iter],gamma,cv[m],Temp[m][*iter],
			  press_eos[m],dp_drho[m], dp_de[m]);

	tmp = dp_drho[m] + dp_de[m] * 
	  (press_eos[m]/(rho_micro[m][*iter]*rho_micro[m][*iter]));
#endif
      }              
      speedSound_new[m][*iter] = sqrt(tmp);
     
      total_mat_vol += mat_volume[m];

     }

     for (int m = 0; m < numALLMatls; m++) {
       vol_frac[m][*iter] = mat_volume[m]/total_mat_vol;
       rho_CC[m][*iter] = vol_frac[m][*iter]*rho_micro[m][*iter];
     }
  }
 

//---- P R I N T   D A T A ------
  if(d_ice -> switchDebug_equilibration_press)  { 
      d_ice->printData( patch, 1, "TOP_equilibration", "Press_CC_top", press);

     for (int m = 0; m < numALLMatls; m++)  {
       Material* matl = d_sharedState->getMaterial( m );
       int dwindex = matl->getDWIndex();
       char description[50];
       sprintf(description, "TOP_equilibration_Mat_%d ",dwindex);
       d_ice->printData( patch,1,description, "rho_CC",     rho_CC[m]);
       d_ice->printData( patch,1,description, "rho_micro",  rho_micro[m]);
       d_ice->printData( patch,0,description, "speedSound", speedSound_new[m]);
       d_ice->printData( patch,1,description, "Temp_CC",    Temp[m]);
       d_ice->printData( patch,1,description, "vol_frac_CC",vol_frac[m]);
      }
    }


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
                                           cv[m],Temp[m][*iter],
                                           press_eos[m], dp_drho[m], dp_de[m]);
       }
       if(mpm_matl){
        //__________________________________
        //  Hardwire for an ideal gas
#if 0
          mpm_matl->getConstitutiveModel()->
               computePressEOSCM(rho_micro[m][*iter],press_eos[m],dp_drho[m],
								tmp,mpm_matl);
#endif
//    This is the IDEAL GAS stuff
#if 1
          double gamma = mpm_matl->getGamma();
          mpm_matl->getConstitutiveModel()->
            computePressEOS(rho_micro[m][*iter],gamma, cv[m],Temp[m][*iter],
			    press_eos[m], dp_drho[m], dp_de[m]);
#endif
       }
     }

     //__________________________________
     // - compute delPress
     // - update press_CC     
     vector<double> Q(numALLMatls),y(numALLMatls);     
     for (int m = 0; m < numALLMatls; m++)   {
       Q[m] =  press_new[*iter] - press_eos[m];
       y[m] =  dp_drho[m] * ( rho_CC[m][*iter]/
               (vol_frac[m][*iter] * vol_frac[m][*iter]) ); 
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
                                             cv[m],Temp[m][*iter]);
       }
       if(mpm_matl){
#if 0
         rho_micro[m][*iter] =  
           mpm_matl->getConstitutiveModel()->computeRhoMicroCM(press_new[*iter],
								mpm_matl);
#endif
//    This is the IDEAL GAS stuff
#if 1
         double gamma = mpm_matl->getGamma();
         rho_micro[m][*iter] = 
         mpm_matl->getConstitutiveModel()->computeRhoMicro(press_new[*iter],
					gamma, cv[m],Temp[m][*iter]);
#endif
       }
     }
     //__________________________________
     // - compute the updated volume fractions
     //  There are two different way of doing it
     for (int m = 0; m < numALLMatls; m++)  {
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
                                            cv[m],Temp[m][*iter],
                                            press_eos[m],dp_drho[m], dp_de[m]);

         tmp = dp_drho[m] + dp_de[m] * 
                    (press_eos[m]/(rho_micro[m][*iter]*rho_micro[m][*iter]));
       }
       if(mpm_matl){
#if 0
          mpm_matl->getConstitutiveModel()->
               computePressEOSCM(rho_micro[m][*iter],press_eos[m],dp_drho[m],
								tmp,mpm_matl);
#endif
//    This is the IDEAL GAS stuff
#if 1
         double gamma = mpm_matl->getGamma();
         mpm_matl->getConstitutiveModel()->
             computePressEOS(rho_micro[m][*iter],gamma,
                                          cv[m],Temp[m][*iter],
                                          press_eos[m],dp_drho[m], dp_de[m]);
         tmp = dp_drho[m] + dp_de[m] * 
                    (press_eos[m]/(rho_micro[m][*iter]*rho_micro[m][*iter]));
#endif
       }
       speedSound_new[m][*iter] = sqrt(tmp);
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
        
/**/    scratch[*iter] = delPress;
     
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

    fprintf(stderr,"\tmax number of iterations in any cell %i\n",test_max_iter);


/*`==========TESTING==========*/ 
  //__________________________________
  // Now change how rho_CC is defined to 
  // rho_CC = mass_CC/cell_volume  NOT mass/mat_volume 
  for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++) {
     for (int m = 0; m < numALLMatls; m++) {
       mat_mass[m] = mass_CC[m][*iter];
          
       rho_CC[m][*iter]   = mat_mass[m]/cell_vol;
     }
  }
 /*==========TESTING==========`*/

   for (int m = 0; m < numALLMatls; m++)   {
     d_ice->setBC(rho_CC[m],   "Density" ,patch);
  }  
  
  d_ice->setBC(press_new, rho_micro[SURROUND_MAT], "Pressure",patch);

  //__________________________________
  //    Put all matls into new dw
  for (int m = 0; m < numALLMatls; m++)   {
    Material* matl = d_sharedState->getMaterial( m );
    int dwindex = matl->getDWIndex();
    new_dw->put( vol_frac[m],      Ilb->vol_frac_CCLabel,   dwindex, patch);
    new_dw->put( speedSound_new[m],Ilb->speedSound_CCLabel, dwindex, patch);
    new_dw->put( rho_micro[m],     Ilb->rho_micro_CCLabel,  dwindex, patch);
    new_dw->put( rho_CC[m],        Ilb->rho_CCLabel,        dwindex, patch);
  }
  new_dw->put(press_new,Ilb->press_equil_CCLabel,0,patch);
  
  
//---- P R I N T   D A T A ------
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
}

/* ---------------------------------------------------------------------
 Function~  MPMICE::massExchange--
 ---------------------------------------------------------------------  */
void MPMICE::massExchange(const ProcessorGroup*,  const Patch* patch,
			  DataWarehouseP& ,  DataWarehouseP& new_dw)
{
#ifdef DOING
  cout << "Doing massExchange on patch " <<
    patch->getID() << "\t MPMICE" << endl;
#endif
 double misha_change_in_mass_from_particles = 0.001;  //hardwired
 int numMatls=d_sharedState->getNumMatls();
 vector<CCVariable<double> > mass_source(numMatls);

  for(int m = 0; m < numMatls; m++) {
    Material* matl = d_sharedState->getMaterial( m );
    int dwindex = matl->getDWIndex();
    new_dw->allocate(mass_source[m],Ilb->mass_sourceLabel,dwindex,patch);
  }
    
  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
    mass_source[HMX][*iter] =  -misha_change_in_mass_from_particles;
    mass_source[GAS][*iter] =  mass_source[HMX][*iter];
  } 
}
