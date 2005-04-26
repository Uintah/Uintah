// MPMICE.cc
#include <Packages/Uintah/CCA/Components/MPMICE/MPMICE.h>
#include <Packages/Uintah/CCA/Components/MPM/SerialMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/RigidMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/ShellMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/FractureMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Components/MPM/ThermalContact/ThermalContact.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/EquationOfState.h>
#include <Packages/Uintah/Core/Labels/ICELabel.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Labels/MPMICELabel.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/ModelMaker.h>

#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/TemperatureBoundCond.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Math/MiscMath.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Exceptions/MaxIteration.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>

#include <Core/Containers/StaticArray.h>
#include <float.h>
#include <stdio.h>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "MPMICE_NORMAL_COUT:+,MPMICE_DOING_COUT".....
//  MPMICE_NORMAL_COUT:  dumps out during problemSetup 
//  MPMICE_DOING_COUT:   dumps when tasks are scheduled and performed
//  default is OFF


static DebugStream cout_norm("MPMICE_NORMAL_COUT", false);  
static DebugStream cout_doing("MPMICE_DOING_COUT", false);


MPMICE::MPMICE(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  Mlb  = scinew MPMLabel();
  Ilb  = scinew ICELabel();
  MIlb = scinew MPMICELabel();
  d_mpm      = scinew SerialMPM(myworld);
  d_ice      = scinew ICE(myworld);
  d_SMALL_NUM = d_ice->d_SMALL_NUM; 
  d_TINY_RHO  = d_ice->d_TINY_RHO;
  d_rigidMPM = false;
  dataArchiver = 0;
  
  // Turn off all the debuging switches
  switchDebug_InterpolateNCToCC_0 = false;
  switchDebug_InterpolateCCToNC   = false;
  switchDebug_InterpolatePAndGradP= false;
}

MPMICE::MPMICE(const ProcessorGroup* myworld, MPMType mpmtype)
  : UintahParallelComponent(myworld)
{
  Mlb  = scinew MPMLabel();
  Ilb  = scinew ICELabel();
  MIlb = scinew MPMICELabel();
  d_rigidMPM = false;
  
  switch(mpmtype) {
  case RIGID_MPMICE:
    d_mpm = scinew RigidMPM(myworld);
    d_rigidMPM = true;
    break;
  case SHELL_MPMICE:
    d_mpm = scinew ShellMPM(myworld);
    break;
  case FRACTURE_MPMICE:
    d_mpm = scinew FractureMPM(myworld);
    break;
  default:
    d_mpm = scinew SerialMPM(myworld);
  }

  d_ice      = scinew ICE(myworld);
  d_SMALL_NUM = d_ice->d_SMALL_NUM; 
  d_TINY_RHO  = d_ice->d_TINY_RHO;
  
  // Turn off all the debuging switches
  switchDebug_InterpolateNCToCC_0 = false;
  switchDebug_InterpolateCCToNC   = false;
  switchDebug_InterpolatePAndGradP= false;
}

MPMICE::~MPMICE()
{
  delete MIlb;
  delete d_mpm;
  delete d_ice;
}

//__________________________________
//    For restarting timesteps
bool MPMICE::restartableTimesteps()
{
  return true;
}

double MPMICE::recomputeTimestep(double current_dt)
{
  return current_dt/2;
} 
//______________________________________________________________________
//
void MPMICE::problemSetup(const ProblemSpecP& prob_spec, GridP& grid,
                          SimulationStateP& sharedState)
{
  d_sharedState = sharedState;
  dataArchiver = dynamic_cast<Output*>(getPort("output"));

  //__________________________________
  //  M P M
  d_mpm->setMPMLabel(Mlb);
  d_mpm->setWithICE();
  d_mpm->attachPort("output",dataArchiver);
  d_mpm->problemSetup(prob_spec, grid, d_sharedState);
  d_8or27 = d_mpm->flags->d_8or27; 
  if(d_8or27==8){
    NGN=1;
  } else if(d_8or27==27){
    NGN=2;
  }

  //__________________________________
  //  I C E

  if(!dataArchiver){
    throw InternalError("MPMICE needs a dataArchiever component to work");
  }
  d_ice->attachPort("output", dataArchiver);
  
  SolverInterface* solver = dynamic_cast<SolverInterface*>(getPort("solver"));
  if(!solver){
    throw InternalError("ICE needs a solver component to work");
  }
  d_ice->attachPort("solver", solver);
  
//  port* models = getPort("modelmaker");
  ModelMaker* models = dynamic_cast<ModelMaker*>(getPort("modelmaker"));

//  port* models = dynamic_cast<port*>(getPort("modelmaker"));
  if(models){  // of there are models then push the port down to ICE
    d_ice->attachPort("modelmaker",models);
  }
  
  d_ice->setICELabel(Ilb);
  d_ice->problemSetup(prob_spec, grid, d_sharedState);


  if(models){  // some models may need to have access to MPMLabels
    for(vector<ModelInterface*>::iterator iter = d_ice->d_models.begin();
       iter != d_ice->d_models.end(); iter++){
      (*iter)->setMPMLabel(Mlb);
    }
  }
  
//  Need to add some means of checking whether or not this is a
//  restart, and if it is, execute the following
//  d_mpm->addMaterial(prob_spec, grid, d_sharedState);
//  d_ice->updateExchangeCoefficients(prob_spec, grid, d_sharedState);
//  cout << "Adding an MPM material" << endl;
  
  //__________________________________
  //  M P M I C E
  ProblemSpecP debug_ps = prob_spec->findBlock("Debug");
  if (debug_ps) {   
    for (ProblemSpecP child = debug_ps->findBlock("debug"); child != 0;
        child = child->findNextBlock("debug")) {
      map<string,string> debug_attr;
      child->getAttributes(debug_attr);
      if (debug_attr["label"]      == "switchDebug_InterpolateNCToCC_0")
        switchDebug_InterpolateNCToCC_0 = true;
      else if (debug_attr["label"] == "switchDebug_InterpolateCCToNC")
        switchDebug_InterpolateCCToNC   = true;
      else if (debug_attr["label"] == "switchDebug_InterpolatePAndGradP")
        switchDebug_InterpolatePAndGradP   = true;       
    }
  }  
  if (cout_norm.active()) {
    cout_norm << "Done with problemSetup \t\t\t MPMICE" <<endl;
    cout_norm << "--------------------------------\n"<<endl;
  }
}
//______________________________________________________________________
//
void MPMICE::scheduleInitialize(const LevelP& level,
                            SchedulerP& sched)
{
  if (cout_doing.active())
    cout_doing << "\nDoing scheduleInitialize \t\t\t MPMICE" << endl;

  d_mpm->scheduleInitialize(level, sched);
  d_ice->scheduleInitialize(level, sched);

  //__________________________________
  //  What isn't initialized in either ice or mpm
  Task* t = scinew Task("MPMICE::actuallyInitialize",
                  this, &MPMICE::actuallyInitialize);
                  
  MaterialSubset* one_matl = scinew MaterialSubset();
  one_matl->add(0);
  one_matl->addReference();
  t->computes(MIlb->vel_CCLabel);
  t->computes(Ilb->rho_CCLabel); 
  t->computes(Ilb->temp_CCLabel);
  t->computes(Ilb->sp_vol_CCLabel);
  t->computes(Ilb->speedSound_CCLabel); 
  t->computes(MIlb->NC_CCweightLabel, one_matl);
    
  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());
  if (cout_doing.active()) {
    cout_doing << "Done with Initialization \t\t\t MPMICE" <<endl;
    cout_norm << "--------------------------------\n"<<endl;   
  }
  if (one_matl->removeReference())
    delete one_matl; // shouln't happen, but...  
}

void MPMICE::restartInitialize()
{
  d_mpm->restartInitialize();
  d_ice->restartInitialize();
}

//______________________________________________________________________
//
void MPMICE::scheduleComputeStableTimestep(const LevelP& level,
                                      SchedulerP& sched)
{
  // Schedule computing the ICE stable timestep
  d_ice->scheduleComputeStableTimestep(level, sched);
  // MPM stable timestep is a by product of the CM
}

//______________________________________________________________________
//
void
MPMICE::scheduleTimeAdvance(const LevelP& level, SchedulerP& sched, 
                            int step , int nsteps )
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* ice_matls = d_sharedState->allICEMaterials();
  const MaterialSet* mpm_matls = d_sharedState->allMPMMaterials();
  const MaterialSet* all_matls = d_sharedState->allMaterials();
  MaterialSubset* press_matl   = d_ice->d_press_matl;
  MaterialSubset* one_matl     = d_ice->d_press_matl;

  const MaterialSubset* ice_matls_sub = ice_matls->getUnion();
  const MaterialSubset* mpm_matls_sub = mpm_matls->getUnion();
  double AMR_subCycleVar = double(step)/double(nsteps);
 //__________________________________
 // Scheduling
  d_ice->scheduleComputeThermoTransportProperties(sched, level,  ice_matls);
   
  d_mpm->scheduleApplyExternalLoads(              sched, patches, mpm_matls);
  // Fracture
  d_mpm->scheduleParticleVelocityField(           sched, patches, mpm_matls);
  d_mpm->scheduleInterpolateParticlesToGrid(      sched, patches, mpm_matls);

  d_mpm->scheduleComputeHeatExchange(             sched, patches, mpm_matls);

  // Fracture
  d_mpm->scheduleAdjustCrackContactInterpolated(  sched,patches,  mpm_matls);

  // schedule the interpolation of mass and volume to the cell centers
  scheduleInterpolateNCToCC_0(                    sched, patches, one_matl, 
                                                                  mpm_matls);

  scheduleComputePressure(                        sched, patches, ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  press_matl,
                                                                  all_matls);

  if (d_ice->d_RateForm) {
    d_ice->schedulecomputeDivThetaVel_CC(         sched, patches, ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  all_matls);
  }
  
  d_ice->scheduleComputeTempFC(                   sched, patches, ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  all_matls);
  d_ice->scheduleComputeModelSources(             sched, level,   all_matls);

  d_ice->scheduleUpdateVolumeFraction(            sched, level,   all_matls);
  
  d_ice->scheduleComputeVel_FC(                   sched, patches, ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  press_matl, 
                                                                  all_matls, 
                                                                  false);
                                                               
  d_ice->scheduleAddExchangeContributionToFCVel(  sched, patches, ice_matls_sub,
                                                                  all_matls,
                                                                  false);  
  
  if(d_ice->d_impICE) {        //  I M P L I C I T 
    d_ice->scheduleSetupRHS(                      sched, patches, one_matl, 
                                                                  all_matls,
                                                                  false);
                                                                  
    d_ice->scheduleImplicitPressureSolve(         sched, level,   patches,
                                                                  one_matl, 
                                                                  press_matl,
                                                                  ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  all_matls);
                                                           
    d_ice->scheduleComputeDel_P(                  sched, level,   patches, 
                                                                  one_matl, 
                                                                  press_matl,
                                                                  all_matls);
  }                           //  IMPLICIT AND EXPLICIT

                                                                  
  if(!(d_ice->d_impICE)){       //  E X P L I C I T 
    d_ice->scheduleComputeDelPressAndUpdatePressCC(sched,patches, press_matl,
                                                                  ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  all_matls);
  } 
  
  d_mpm->scheduleExMomInterpolated(               sched, patches, mpm_matls);
  d_mpm->scheduleComputeStressTensor(             sched, patches, mpm_matls);

  scheduleInterpolateMassBurnFractionToNC(        sched, patches, mpm_matls);

  d_ice->scheduleComputePressFC(                  sched, patches, press_matl,
                                                                  all_matls);
  d_ice->scheduleAccumulateMomentumSourceSinks(   sched, patches, press_matl,
                                                                  ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  all_matls);
                                                                  
  d_ice->scheduleAccumulateEnergySourceSinks(     sched, patches, ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  press_matl,
                                                                  all_matls);

  if(!d_rigidMPM){
    scheduleInterpolatePressCCToPressNC(          sched, patches, press_matl,
                                                                  mpm_matls);
    scheduleInterpolatePAndGradP(                 sched, patches, press_matl,
                                                                  one_matl,
                                                                  mpm_matls_sub,
                                                                  mpm_matls);
  }
   
  d_mpm->scheduleComputeInternalForce(            sched, patches, mpm_matls);
  d_mpm->scheduleComputeInternalHeatRate(         sched, patches, mpm_matls);
  scheduleSolveEquationsMotion(                   sched, patches, mpm_matls);
  d_mpm->scheduleSolveHeatEquations(              sched, patches, mpm_matls);
  d_mpm->scheduleIntegrateAcceleration(           sched, patches, mpm_matls);
  d_mpm->scheduleIntegrateTemperatureRate(        sched, patches, mpm_matls);
  d_mpm->scheduleAdjustCrackContactIntegrated(    sched, patches, mpm_matls);
  
  scheduleComputeLagrangianValuesMPM(             sched, patches, one_matl,
                                                                  mpm_matls); 

  d_ice->scheduleComputeLagrangianValues(         sched, patches, ice_matls);

  d_ice->scheduleAddExchangeToMomentumAndEnergy(  sched, patches, ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  press_matl,
                                                                  all_matls); 

  d_ice->scheduleComputeLagrangianSpecificVolume( sched, patches, ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  press_matl,
                                                                  all_matls);
                                                                  
  d_ice->scheduleComputeLagrangian_Transported_Vars(sched,patches,ice_matls);


  scheduleInterpolateCCToNC(                      sched, patches, mpm_matls);
  d_mpm->scheduleExMomIntegrated(                 sched, patches, mpm_matls);
  d_mpm->scheduleSetGridBoundaryConditions(       sched, patches, mpm_matls);
  d_mpm->scheduleCalculateDampingRate(            sched, patches, mpm_matls);
  d_mpm->scheduleAddNewParticles(                 sched, patches, mpm_matls);
  d_mpm->scheduleConvertLocalizedParticles(       sched, patches, mpm_matls);
  d_mpm->scheduleInterpolateToParticlesAndUpdate( sched, patches, mpm_matls);
  //d_mpm->scheduleApplyExternalLoads(              sched, patches, mpm_matls);

  d_mpm->scheduleCalculateFractureParameters(    sched, patches, mpm_matls);
  d_mpm->scheduleDoCrackPropagation(             sched, patches, mpm_matls);
  d_mpm->scheduleMoveCracks(                     sched, patches, mpm_matls);
  d_mpm->scheduleUpdateCrackFront(               sched, patches, mpm_matls);

  vector<PatchSubset*> maxMach_PSS(Patch::numFaces);
  d_ice->scheduleMaxMach_on_Lodi_BC_Faces(sched, level,ice_matls,maxMach_PSS);
                                   
  d_ice->scheduleAdvectAndAdvanceInTime(         sched, patches, AMR_subCycleVar,
                                                                 ice_matls_sub,
                                                                 mpm_matls_sub,
                                                                 press_matl,
                                                                 ice_matls);
                                                                  
  d_ice->scheduleTestConservation(               sched, patches, ice_matls_sub,
                                                                 all_matls); 

  if(d_ice->d_canAddICEMaterial){
    //  This checks to see if the model on THIS patch says that it's
    //  time to add a new material
    d_ice->scheduleCheckNeedAddMaterial(    sched, level,   all_matls);
                                                                                
    //  This one checks to see if the model on ANY patch says that it's
    //  time to add a new material
    d_ice->scheduleSetNeedAddMaterialFlag(  sched, level,   all_matls);
  }
  if(d_mpm->flags->d_canAddMPMMaterial){
    //  This checks to see if the model on THIS patch says that it's
    //  time to add a new material
    d_mpm->scheduleCheckNeedAddMPMMaterial( sched, patches, mpm_matls);
                                                                                
    //  This one checks to see if the model on ANY patch says that it's
    //  time to add a new material
    d_mpm->scheduleSetNeedAddMaterialFlag(  sched, level,   mpm_matls);
  }

  sched->scheduleParticleRelocation(level,
                                Mlb->pXLabel_preReloc, 
                                Mlb->d_particleState_preReloc,
                                Mlb->pXLabel, Mlb->d_particleState,
                                Mlb->pParticleIDLabel, mpm_matls);

  //__________________________________
  // clean up memory
  if(d_ice->d_customBC_var_basket->usingLodi){
    for(int f=0;f<Patch::numFaces;f++){
      if(maxMach_PSS[f]->removeReference()){
        delete maxMach_PSS[f];
      }
    }
  }
} // end scheduleTimeAdvance()


//______________________________________________________________________
//
void MPMICE::scheduleInterpolatePressCCToPressNC(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSubset* press_matl,
                                           const MaterialSet* matls)
{
  if (cout_doing.active())
    cout_doing << "MPMICE::scheduleInterpolatePressCCToPressNC" << endl;

  Task* t=scinew Task("MPMICE::interpolatePressCCToPressNC",
                    this, &MPMICE::interpolatePressCCToPressNC);

  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::NewDW,Ilb->press_CCLabel, press_matl, gac, 1);
  t->computes(MIlb->press_NCLabel, press_matl);
  
  sched->addTask(t, patches, matls);
}
//______________________________________________________________________
//
void MPMICE::scheduleInterpolatePAndGradP(SchedulerP& sched,
                                     const PatchSet* patches,
                                     const MaterialSubset* press_matl,
                                     const MaterialSubset* /*one_matl*/,
                                     const MaterialSubset* mpm_matl,
                                     const MaterialSet* all_matls)
{
  if (cout_doing.active())
    cout_doing << "MPMICE::scheduleInterpolatePAndGradP" << endl;
 
   Task* t=scinew Task("MPMICE::interpolatePAndGradP",
                 this, &MPMICE::interpolatePAndGradP);
   Ghost::GhostType  gac = Ghost::AroundCells;
   t->requires(Task::OldDW, d_sharedState->get_delt_label());
   
   t->requires(Task::NewDW, MIlb->press_NCLabel,       press_matl,gac, NGN);
   t->requires(Task::NewDW, MIlb->cMassLabel,          mpm_matl,  gac, 1);
   t->requires(Task::NewDW, Ilb->press_force_CCLabel,  mpm_matl,  gac, 1);
   t->requires(Task::OldDW, Mlb->pXLabel,              mpm_matl,  Ghost::None);
   t->requires(Task::OldDW, Mlb->pSizeLabel,         mpm_matl,  Ghost::None);
   
   t->computes(Mlb->pPressureLabel,   mpm_matl);
   t->computes(Mlb->gradPAccNCLabel,  mpm_matl);
   sched->addTask(t, patches, all_matls);
}



//______________________________________________________________________
//
void MPMICE::scheduleInterpolateNCToCC_0(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSubset* one_matl,
                                    const MaterialSet* mpm_matls)
{
  if (cout_doing.active())
    cout_doing << "MPMICE::scheduleInterpolateNCToCC_0" << endl;
 
   /* interpolateNCToCC */
   Task* t=scinew Task("MPMICE::interpolateNCToCC_0",
                 this, &MPMICE::interpolateNCToCC_0);
   const MaterialSubset* mss = mpm_matls->getUnion();
   t->requires(Task::NewDW, Mlb->gMassLabel,       Ghost::AroundCells, 1);
   t->requires(Task::NewDW, Mlb->gVolumeLabel,     Ghost::AroundCells, 1);
   t->requires(Task::NewDW, Mlb->gVelocityLabel,   Ghost::AroundCells, 1); 
   t->requires(Task::NewDW, Mlb->gTemperatureLabel,Ghost::AroundCells, 1);
   t->requires(Task::NewDW, Mlb->gSp_volLabel,     Ghost::AroundCells, 1);
   t->requires(Task::OldDW, MIlb->NC_CCweightLabel,one_matl,
                                                   Ghost::AroundCells, 1);
   t->requires(Task::OldDW, Ilb->sp_vol_CCLabel,   Ghost::None, 0); 
   t->requires(Task::OldDW, MIlb->temp_CCLabel,    Ghost::None, 0);
   t->requires(Task::OldDW, MIlb->vel_CCLabel,     Ghost::None, 0);
    
   t->computes(MIlb->cMassLabel);
   t->computes(MIlb->cVolumeLabel);

   t->computes(MIlb->vel_CCLabel);
   t->computes(MIlb->temp_CCLabel);
   t->computes(Ilb->sp_vol_CCLabel, mss);
   t->computes(Ilb->rho_CCLabel, mss); 
   
   sched->addTask(t, patches, mpm_matls);
}

//______________________________________________________________________
//
void MPMICE::scheduleComputeLagrangianValuesMPM(SchedulerP& sched,
                                   const PatchSet* patches,
                                   const MaterialSubset* one_matl,
                                   const MaterialSet* mpm_matls)
{
  if (cout_doing.active())
    cout_doing << "MPMICE::scheduleComputeLagrangianValuesMPM" << endl;

   /* interpolateNCToCC */

   Task* t=scinew Task("MPMICE::computeLagrangianValuesMPM",
                 this, &MPMICE::computeLagrangianValuesMPM);

   const MaterialSubset* mss = mpm_matls->getUnion();
   Ghost::GhostType  gac = Ghost::AroundCells;
   Ghost::GhostType  gn  = Ghost::None;
   t->requires(Task::NewDW, Mlb->gVelocityStarLabel, mss, gac,1);
   t->requires(Task::NewDW, Mlb->gMassLabel,              gac,1);
   t->requires(Task::NewDW, Mlb->gTemperatureStarLabel,   gac,1);
   t->requires(Task::OldDW, MIlb->NC_CCweightLabel,       one_matl, gac,1);
   t->requires(Task::NewDW, MIlb->cMassLabel,             gn);

   t->requires(Task::NewDW, MIlb->temp_CCLabel,           gn);
   t->requires(Task::NewDW, MIlb->vel_CCLabel,            gn);

   if(d_ice->d_models.size() > 0){
     t->requires(Task::NewDW, Ilb->modelMass_srcLabel,   gn);
     t->requires(Task::NewDW, Ilb->modelMom_srcLabel,    gn);
     t->requires(Task::NewDW, Ilb->modelEng_srcLabel,    gn);
   }

   t->modifies( Ilb->rho_CCLabel); 
   t->computes( Ilb->mass_L_CCLabel);
   t->computes( Ilb->mom_L_CCLabel);
   t->computes( Ilb->int_eng_L_CCLabel);
   t->computes(MIlb->NC_CCweightLabel, one_matl);

   sched->addTask(t, patches, mpm_matls);
}

//______________________________________________________________________
//
void MPMICE::scheduleInterpolateCCToNC(SchedulerP& sched,
                                   const PatchSet* patches,
                                   const MaterialSet* mpm_matls)
{

  if (cout_doing.active())
    cout_doing << "MPMICE::scheduleInterpolateCCToNC" << endl;

  Task* t=scinew Task("MPMICE::interpolateCCToNC",
                this, &MPMICE::interpolateCCToNC);               
  const MaterialSubset* mss = mpm_matls->getUnion();
  Ghost::GhostType  gac = Ghost::AroundCells;

  t->requires(Task::OldDW, d_sharedState->get_delt_label());
  t->requires(Task::NewDW, Ilb->mass_L_CCLabel,         gac,1);
  t->requires(Task::NewDW, Ilb->mom_L_CCLabel,          gac,1);  
  t->requires(Task::NewDW, Ilb->int_eng_L_CCLabel,      gac,1);  
  t->requires(Task::NewDW, Ilb->mom_L_ME_CCLabel,       gac,1);
  t->requires(Task::NewDW, Ilb->eng_L_ME_CCLabel,       gac,1);
  t->requires(Task::NewDW, Ilb->sp_vol_src_CCLabel,     gac,1); 
  
  t->modifies(Mlb->gVelocityStarLabel, mss);             
  t->modifies(Mlb->gAccelerationLabel, mss);             
  t->computes(Mlb->dTdt_NCLabel);

  sched->addTask(t, patches, mpm_matls);
}
/* ---------------------------------------------------------------------
 Function~  MPMICE::scheduleComputePressure--
 Note:  Handles both Rate and Equilibration form of solution technique
_____________________________________________________________________*/
void MPMICE::scheduleComputePressure(SchedulerP& sched,
                                     const PatchSet* patches,
                                     const MaterialSubset* ice_matls,
                                     const MaterialSubset* mpm_matls,
                                     const MaterialSubset* press_matl,
                                     const MaterialSet* all_matls)
{
  Task* t = NULL;
  if (d_ice->d_RateForm) {     // R A T E   F O R M
    if (cout_doing.active())
      cout_doing << "MPMICE::scheduleComputeRateFormPressure" << endl;

    t = scinew Task("MPMICE::computeRateFormPressure",
                     this, &MPMICE::computeRateFormPressure);
  }
  if (d_ice->d_EqForm) {       // E Q   F O R M
    if (cout_doing.active())
      cout_doing << "MPMICE::scheduleComputeEquilibrationPressure" << endl;

    t = scinew Task("MPMICE::computeEquilibrationPressure",
                    this, &MPMICE::computeEquilibrationPressure);
  }
                              // I C E
  t->requires(Task::OldDW,Ilb->temp_CCLabel,         ice_matls, Ghost::None);
  t->requires(Task::OldDW,Ilb->rho_CCLabel,          ice_matls, Ghost::None);
  t->requires(Task::OldDW,Ilb->sp_vol_CCLabel,       ice_matls, Ghost::None);
  t->requires(Task::NewDW,Ilb->specific_heatLabel,   ice_matls, Ghost::None);
  t->requires(Task::NewDW,Ilb->gammaLabel,           ice_matls, Ghost::None);

                              // M P M
  t->requires(Task::NewDW,MIlb->temp_CCLabel,        mpm_matls, Ghost::None);
  t->requires(Task::NewDW,Ilb->rho_CCLabel,          mpm_matls, Ghost::None);
  t->requires(Task::NewDW,Ilb->sp_vol_CCLabel,       mpm_matls, Ghost::None);
 
  if (d_ice->d_EqForm) {      // E Q   F O R M
    t->requires(Task::OldDW,Ilb->press_CCLabel,      press_matl, Ghost::None);
    t->requires(Task::OldDW,Ilb->vel_CCLabel,        ice_matls,  Ghost::None);
    t->requires(Task::NewDW,MIlb->vel_CCLabel,       mpm_matls,  Ghost::None);
  }

  computesRequires_CustomBCs(t, "EqPress", Ilb, ice_matls, 
                            d_ice->d_customBC_var_basket);
                              
                              //  A L L _ M A T L S
  if (d_ice->d_RateForm) {   
    t->computes(Ilb->matl_press_CCLabel);
  }
  t->computes(Ilb->f_theta_CCLabel);

  t->computes(Ilb->speedSound_CCLabel); 
  t->computes(Ilb->vol_frac_CCLabel);
  t->computes(Ilb->press_equil_CCLabel, press_matl);  // diagnostic variable
  t->computes(Ilb->press_CCLabel,       press_matl);  // needed by implicit ICE
  t->modifies(Ilb->sp_vol_CCLabel,      mpm_matls);
  t->modifies(Ilb->rho_CCLabel,         mpm_matls); 
  t->computes(Ilb->sp_vol_CCLabel,      ice_matls);
  t->computes(Ilb->rho_CCLabel,         ice_matls);
  sched->addTask(t, patches, all_matls);
}
//______________________________________________________________________
//
void MPMICE::scheduleInterpolateMassBurnFractionToNC(SchedulerP& sched,
                                           const PatchSet* patches,
                                            const MaterialSet* mpm_matls)
{
  if (cout_doing.active())
    cout_doing << "MPMICE::scheduleInterpolateMassBurnFractionToNC" << endl;

  Task* t = scinew Task("MPMICE::interpolateMassBurnFractionToNC",
                  this, &MPMICE::interpolateMassBurnFractionToNC);

  t->requires(Task::OldDW, d_sharedState->get_delt_label());  
  t->requires(Task::NewDW, MIlb->cMassLabel,        Ghost::AroundCells,1);
 
  if(d_ice->d_models.size() > 0){
    t->requires(Task::NewDW,Ilb->modelMass_srcLabel, Ghost::AroundCells,1);
  }
  t->computes(Mlb->massBurnFractionLabel);

  sched->addTask(t, patches, mpm_matls);
}

//______________________________________________________________________
void MPMICE::scheduleSolveEquationsMotion(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSet* matls)
{
  d_mpm->scheduleSolveEquationsMotion(sched,patches,matls);

  if (!d_rigidMPM) {
    
    Task* t = scinew Task("MPMICE::solveEquationsMotion",
                          this, &MPMICE::solveEquationsMotion);
    
    t->requires(Task::NewDW, Mlb->gradPAccNCLabel,  Ghost::None);
    t->modifies(Mlb->gAccelerationLabel);
    sched->addTask(t,patches,matls);
  }
}
//______________________________________________________________________
void MPMICE::solveEquationsMotion(const ProcessorGroup* pg,
                                     const PatchSubset* patches,
                                     const MaterialSubset* ms,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      constNCVariable<Vector> gradPAccNC; 
      NCVariable<Vector> acceleration;

      new_dw->getModifiable(acceleration,Mlb->gAccelerationLabel,dwi, patch);
      new_dw->get(gradPAccNC,Mlb->gradPAccNCLabel,dwi,patch,Ghost::None,0);
      
      for(NodeIterator iter = patch->getNodeIterator(d_mpm->flags->d_8or27);
          !iter.done();iter++)
        acceleration[*iter] += gradPAccNC[*iter];
    }
    
  }
  
}



//______________________________________________________________________
//       A C T U A L   S T E P S :
//______________________________________________________________________
void MPMICE::actuallyInitialize(const ProcessorGroup*, 
                            const PatchSubset* patches,
                            const MaterialSubset* ,
                            DataWarehouse*,
                            DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){ 
    const Patch* patch = patches->get(p);
    if (cout_doing.active()) {
      cout_doing << "Doing Initialize on patch " << patch->getID() 
                 << "\t\t\t MPMICE" << endl;
    }

    NCVariable<double> NC_CCweight;
    new_dw->allocateAndPut(NC_CCweight, MIlb->NC_CCweightLabel,    0, patch);
   //__________________________________
   // - Initialize NC_CCweight = 0.125
   // - Find the walls with symmetry BC and
   //   double NC_CCweight
   NC_CCweight.initialize(0.125);
   for(Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
        face=Patch::nextFace(face)){
      int mat_id = 0; 
#if 0
      const BoundCondBase *sym_bcs =
        patch->getBCValues(mat_id,"Symmetric",face);
      if (sym_bcs != 0) {
#endif
      if (patch->haveBC(face,mat_id,"symmetry","Symmetric")) {
        for(CellIterator iter = patch->getFaceCellIterator(face,"NC_vars"); 
                                                  !iter.done(); iter++) {
          NC_CCweight[*iter] = 2.0*NC_CCweight[*iter];
        }
      }
    }
    //__________________________________
    //  Initialize CCVaribles for MPM Materials
    //  Even if mass = 0 in a cell you still need
    //  CC Variables defined.
    double junk=-9, tmp;
    int numALL_matls = d_sharedState->getNumMatls();
    int numMPM_matls = d_sharedState->getNumMPMMatls();
    double p_ref = d_sharedState->getRefPress();
    for (int m = 0; m < numMPM_matls; m++ ) {
      
      CCVariable<double> rho_micro, sp_vol_CC, rho_CC, Temp_CC, speedSound;
      CCVariable<Vector> vel_CC;
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      int indx= mpm_matl->getDWIndex();
      new_dw->allocateTemporary(rho_micro, patch);
      new_dw->allocateAndPut(sp_vol_CC, Ilb->sp_vol_CCLabel,    indx,patch);
      new_dw->allocateAndPut(rho_CC,    Ilb->rho_CCLabel,       indx,patch);
      new_dw->allocateAndPut(speedSound,Ilb->speedSound_CCLabel,indx,patch);
      new_dw->allocateAndPut(Temp_CC,  MIlb->temp_CCLabel,      indx,patch);
      new_dw->allocateAndPut(vel_CC,   MIlb->vel_CCLabel,       indx,patch);

      // Ignore the dummy materials that are used when particles are
      // localized
      if (d_mpm->flags->d_createNewParticles) {
        if (m%2 == 0) // The actual materials
          mpm_matl->initializeCCVariables(rho_micro,   rho_CC,
                                          Temp_CC,     vel_CC,  
                                          numALL_matls,patch);  
        else // The dummy materials
          mpm_matl->initializeDummyCCVariables(rho_micro,   rho_CC,
                                               Temp_CC,     vel_CC,  
                                               numALL_matls,patch);  
      } else {
        mpm_matl->initializeCCVariables(rho_micro,   rho_CC,
                                        Temp_CC,     vel_CC,  
                                        numALL_matls,patch);  
      }

      setBC(rho_CC,    "Density",      patch, d_sharedState, indx, new_dw);    
      setBC(rho_micro, "Density",      patch, d_sharedState, indx, new_dw);    
      setBC(Temp_CC,   "Temperature",  patch, d_sharedState, indx, new_dw);    
      setBC(vel_CC,    "Velocity",     patch, d_sharedState, indx, new_dw);
      for (CellIterator iter = patch->getExtraCellIterator();
                                                        !iter.done();iter++){
        IntVector c = *iter;
        sp_vol_CC[c] = 1.0/rho_micro[c];

        mpm_matl->getConstitutiveModel()->
            computePressEOSCM(rho_micro[c],junk, p_ref, junk, tmp,mpm_matl); 
        speedSound[c] = sqrt(tmp);
      }
      
      //__________________________________
      //    B U L L E T   P R O O F I N G
      IntVector neg_cell;
      ostringstream warn;
      if( !d_ice->areAllValuesPositive(rho_CC, neg_cell) ) {
        warn<<"ERROR MPMICE::actuallyInitialize, mat "<<indx<< " cell "
            <<neg_cell << " rho_CC is negative\n";
        throw ProblemSetupException(warn.str() );
      }
      if( !d_ice->areAllValuesPositive(Temp_CC, neg_cell) ) {
        warn<<"ERROR MPMICE::actuallyInitialize, mat "<<indx<< " cell "
            <<neg_cell << " Temp_CC is negative\n";
        throw ProblemSetupException(warn.str() );
      }
      if( !d_ice->areAllValuesPositive(sp_vol_CC, neg_cell) ) {
        warn<<"ERROR MPMICE::actuallyInitialize, mat "<<indx<< " cell "
            <<neg_cell << " sp_vol_CC is negative\n";
        throw ProblemSetupException(warn.str() );
      }
      
      //---- P R I N T   D A T A ------        
      if (d_ice->switchDebugInitialize){      
        ostringstream desc;
        desc << "MPMICE_Initialization_Mat_" << indx << "_patch_"
             << patch->getID();
        d_ice->printData(indx, patch,  1, desc.str(), "rho_CC",      rho_CC);
        d_ice->printData(indx, patch,  1, desc.str(), "rho_micro_CC",rho_micro);
        d_ice->printData(indx, patch,  1, desc.str(), "sp_vol_CC",   sp_vol_CC);
        d_ice->printData(indx, patch,  1, desc.str(), "Temp_CC",     Temp_CC);
        d_ice->printVector(indx, patch,1, desc.str(), "vel_CC", 0,   vel_CC);
      }             
    }  // num_MPM_matls loop 
  } // Patch loop
}

//______________________________________________________________________
//
void MPMICE::interpolatePressCCToPressNC(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* ,
                                    DataWarehouse*,
                                    DataWarehouse* new_dw)
{                            
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    if (cout_doing.active()) {
      cout_doing<<"Doing interpolatePressCCToPressNC on patch "<<patch->getID()
                <<"\t\t MPMICE" << endl;
    }

    constCCVariable<double> pressCC;
    NCVariable<double> pressNC;

    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(pressCC, Ilb->press_CCLabel, 0, patch, gac, 1);
    new_dw->allocateAndPut(pressNC, MIlb->press_NCLabel, 0, patch);
    pressNC.initialize(0.0);
    
    IntVector cIdx[8];
    // Interpolate CC pressure to nodes
    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
       patch->findCellsFromNode(*iter,cIdx);
      for (int in=0;in<8;in++){
        pressNC[*iter]  += .125*pressCC[cIdx[in]];
      }
    }
  }
}

//______________________________________________________________________
//
void MPMICE::interpolatePAndGradP(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* ,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    if (cout_doing.active()) {
      cout_doing<<"Doing interpolatePressureToParticles on patch "<<
        patch->getID()<<"\t\t MPMICE" << endl;
    }

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());

    ParticleInterpolator* interpolator = d_mpm->flags->d_interpolator->clone(patch);
    vector<IntVector> ni;
    ni.reserve(interpolator->size());
    vector<double> S;
    S.reserve(interpolator->size());

    IntVector cIdx[8];
    double p_ref = d_sharedState->getRefPress();
    constNCVariable<double>   pressNC;    
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(pressNC, MIlb->press_NCLabel,  0, patch, gac, NGN);

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int indx = mpm_matl->getDWIndex();
      constCCVariable<Vector> press_force;
      constCCVariable<double> mass;
      NCVariable<Vector> gradPAccNC; 
      new_dw->get(press_force,      Ilb->press_force_CCLabel,indx,patch,gac,1);
      new_dw->get(mass,             MIlb->cMassLabel,        indx,patch,gac,1); 
      new_dw->allocateAndPut(gradPAccNC, Mlb->gradPAccNCLabel,    indx,patch);
      gradPAccNC.initialize(Vector(0.,0.,0.));    

      ParticleSubset* pset = old_dw->getParticleSubset(indx, patch);
      ParticleVariable<double> pPressure;
      constParticleVariable<Point> px;
      constParticleVariable<Vector> psize;
      old_dw->get(psize,              Mlb->pSizeLabel,     pset);     
      
      new_dw->allocateAndPut(pPressure, Mlb->pPressureLabel, pset);     
      old_dw->get(px,                   Mlb->pXLabel,        pset);     

     //__________________________________
     // Interpolate NC pressure to particles
      for(ParticleSubset::iterator iter = pset->begin();
         iter != pset->end(); iter++){
        particleIndex idx = *iter;
        double press = 0.;

        // Get the node indices that surround the cell
        interpolator->findCellAndWeights(px[idx], ni, S,psize[idx]);

        for (int k = 0; k < d_8or27; k++) {
          press += pressNC[ni[k]] * S[k];
        }
        pPressure[idx] = press-p_ref;
      }
      //__________________________________
      // gradPAccNC
     for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
        patch->findCellsFromNode(*iter,cIdx);
        for (int in=0;in<8;in++){
          IntVector c = cIdx[in];
          double mass_CC = mass[c];      
                                    // force /mass
          gradPAccNC[*iter][0] += (press_force[c][0]/mass_CC) * .125;
          gradPAccNC[*iter][1] += (press_force[c][1]/mass_CC) * .125;
          gradPAccNC[*iter][2] += (press_force[c][2]/mass_CC) * .125;      
        }
      }
      //---- P R I N T   D A T A ------ 
      if(switchDebug_InterpolatePAndGradP) {
        ostringstream desc;
        desc<< "BOT_MPMICE::interpolatePAndGradP_mat_"<< indx<<"_patch_"
            <<patch->getID();                   
        printNCVector(0, patch, 1,desc.str(),"gradPAccNC",0,gradPAccNC);
      }
    }  // numMPMMatls
    delete interpolator;
  } //patches
}

//______________________________________________________________________
//
void MPMICE::interpolateNCToCC_0(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* ,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    if (cout_doing.active()) {
      cout_doing << "Doing interpolateNCToCC_0 on patch "<< patch->getID()
                 <<"\t\t\t MPMICE" << endl;
    }

    int numMatls = d_sharedState->getNumMPMMatls();
    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z(); 
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn = Ghost::None;     
    
    constNCVariable<double> NC_CCweight;
    old_dw->get(NC_CCweight, MIlb->NC_CCweightLabel,  0, patch, gac, 1);

    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int indx = mpm_matl->getDWIndex();
      // Create arrays for the grid data
      constNCVariable<double> gmass, gvolume, gtemperature, gSp_vol;
      constNCVariable<Vector> gvelocity;
      CCVariable<double> cmass, cvolume,Temp_CC, sp_vol_CC, rho_CC;
      CCVariable<Vector> vel_CC;
      constCCVariable<double> Temp_CC_ice, sp_vol_CC_ice;
      constCCVariable<Vector> vel_CC_ice;

      new_dw->allocateAndPut(cmass,    MIlb->cMassLabel,     indx, patch);  
      new_dw->allocateAndPut(cvolume,  MIlb->cVolumeLabel,   indx, patch);  
      new_dw->allocateAndPut(vel_CC,   MIlb->vel_CCLabel,    indx, patch);  
      new_dw->allocateAndPut(Temp_CC,  MIlb->temp_CCLabel,   indx, patch);  
      new_dw->allocateAndPut(sp_vol_CC, Ilb->sp_vol_CCLabel, indx, patch); 
      new_dw->allocateAndPut(rho_CC,    Ilb->rho_CCLabel,    indx, patch);
      
      double rho_orig = mpm_matl->getInitialDensity();
      double very_small_mass = d_TINY_RHO * cell_vol;
      cmass.initialize(very_small_mass);
      cvolume.initialize( very_small_mass/rho_orig);
         
      new_dw->get(gmass,        Mlb->gMassLabel,        indx, patch,gac, 1);
      new_dw->get(gvolume,      Mlb->gVolumeLabel,      indx, patch,gac, 1);
      new_dw->get(gvelocity,    Mlb->gVelocityLabel,    indx, patch,gac, 1);
      new_dw->get(gtemperature, Mlb->gTemperatureLabel, indx, patch,gac, 1);
      new_dw->get(gSp_vol,      Mlb->gSp_volLabel,      indx, patch,gac, 1);
      old_dw->get(sp_vol_CC_ice,Ilb->sp_vol_CCLabel,    indx, patch,gn, 0); 
      old_dw->get(Temp_CC_ice,  MIlb->temp_CCLabel,     indx, patch,gn, 0);
      old_dw->get(vel_CC_ice,   MIlb->vel_CCLabel,      indx, patch,gn, 0);
      IntVector nodeIdx[8];
      
      //---- P R I N T   D A T A ------ 
#if 0
      if(switchDebug_InterpolateNCToCC_0) {
        ostringstream desc;
        desc<< "TOP_MPMICE::interpolateNCToCC_0_mat_"<<indx<<"_patch_"
            <<  patch->getID();
        printData(     indx, patch, 1,desc.str(), "gmass",       gmass);
        printData(     indx, patch, 1,desc.str(), "gvolume",     gvolume);
        printData(     indx, patch, 1,desc.str(), "gtemperatue", gtemperature);
        printNCVector( indx, patch, 1,desc.str(), "gvelocity", 0, gvelocity);
      }
#endif 
      //__________________________________
      //  compute CC Variables
      for(CellIterator iter =patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        patch->findNodesFromCell(*iter,nodeIdx);
 
        double Temp_CC_mpm = 0.0;  
        double sp_vol_mpm = 0.0;   
        Vector vel_CC_mpm  = Vector(0.0, 0.0, 0.0);
        
        for (int in=0;in<8;in++){
          double NC_CCw_mass = NC_CCweight[nodeIdx[in]] * gmass[nodeIdx[in]];
          cmass[c]    += NC_CCw_mass;
          cvolume[c]  += NC_CCweight[nodeIdx[in]]  * gvolume[nodeIdx[in]];
          sp_vol_mpm  += gSp_vol[nodeIdx[in]]      * NC_CCw_mass;
          vel_CC_mpm  += gvelocity[nodeIdx[in]]    * NC_CCw_mass;
          Temp_CC_mpm += gtemperature[nodeIdx[in]] * NC_CCw_mass;
        } 
        double inv_cmass = 1.0/cmass[c];
        vel_CC_mpm  *= inv_cmass;    
        Temp_CC_mpm *= inv_cmass;
        sp_vol_mpm  *= inv_cmass;
        
        //__________________________________
        // set *_CC = to either vel/Temp_CC_ice or vel/Temp_CC_mpm
        // depending upon if there is cmass.  You need
        // a well defined vel/temp_CC even if there isn't any mass
        // If you change this you must also change 
        // MPMICE::computeLagrangianValuesMPM
        double one_or_zero = (cmass[c] - very_small_mass)/cmass[c];

        Temp_CC[c]  =(1.0-one_or_zero)*Temp_CC_ice[c]  +one_or_zero*Temp_CC_mpm;
//      This allows propagation of messed up solid matl velocities,
//      and has thus been abandoned for now.
//      vel_CC[c]   =(1.0-one_or_zero)*vel_CC_ice[c]   +one_or_zero*vel_CC_mpm;
        vel_CC[c]   =vel_CC_mpm;
        sp_vol_CC[c]=(1.0-one_or_zero)*sp_vol_CC_ice[c]+one_or_zero*sp_vol_mpm;
        rho_CC[c]    = cmass[c]/cell_vol;
      }
      //  Set BC's
      setBC(Temp_CC, "Temperature",patch, d_sharedState, indx, new_dw);
      setBC(rho_CC,  "Density",    patch, d_sharedState, indx, new_dw);
      setBC(vel_CC,  "Velocity",   patch, d_sharedState, indx, new_dw);
      //  Set if symmetric Boundary conditions
      setBC(cmass,    "set_if_sym_BC",patch, d_sharedState, indx, new_dw);
      setBC(cvolume,  "set_if_sym_BC",patch, d_sharedState, indx, new_dw);
      setBC(sp_vol_CC,"set_if_sym_BC",patch, d_sharedState, indx, new_dw); 
      
     //---- P R I N T   D A T A ------
     if(switchDebug_InterpolateNCToCC_0) {
        ostringstream desc;
        desc<< "BOT_MPMICE::interpolateNCToCC_0_Mat_"<< indx <<"_patch_"
            <<  patch->getID();
        d_ice->printData(   indx, patch, 1,desc.str(), "sp_vol",    sp_vol_CC); 
        d_ice->printData(   indx, patch, 1,desc.str(), "cmass",     cmass);
        d_ice->printData(   indx, patch, 1,desc.str(), "cvolume",   cvolume);
        d_ice->printData(   indx, patch, 1,desc.str(), "Temp_CC",   Temp_CC);
        d_ice->printData(   indx, patch, 1,desc.str(), "rho_CC",    rho_CC);
        d_ice->printVector( indx, patch, 1,desc.str(), "vel_CC", 0, vel_CC);
      }
      //---- B U L L E T   P R O O F I N G------
      IntVector neg_cell;
      ostringstream warn;
      if (!d_ice->areAllValuesPositive(Temp_CC, neg_cell)) {
        warn <<"ERROR MPMICE::interpolateNCToCC_0, mat "<< indx <<" cell "
             << neg_cell << " Temp_CC " << Temp_CC[neg_cell] << "\n ";
        throw InvalidValue(warn.str());
      }
      if (!d_ice->areAllValuesPositive(rho_CC, neg_cell)) {
        warn <<"ERROR MPMICE::interpolateNCToCC_0, mat "<< indx <<" cell "
             << neg_cell << " rho_CC " << rho_CC[neg_cell]<< "\n ";
        throw InvalidValue(warn.str());
      }
      if (!d_ice->areAllValuesPositive(sp_vol_CC, neg_cell)) {
        warn <<"ERROR MPMICE::interpolateNCToCC_0, mat "<< indx <<" cell "
             << neg_cell << " sp_vol_CC " << sp_vol_CC[neg_cell]<<"\n ";
        throw InvalidValue(warn.str());
      } 
    }
  }  //patches
}
//______________________________________________________________________
//
void MPMICE::computeLagrangianValuesMPM(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* ,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    if (cout_doing.active()) {
      cout_doing << "Doing computeLagrangianValuesMPM on patch "
                 << patch->getID() <<"\t\t MPMICE" << endl;
    }

    int numMatls = d_sharedState->getNumMPMMatls();

    Vector dx = patch->dCell();
    double cellVol = dx.x()*dx.y()*dx.z();
    double inv_cellVol = 1.0/cellVol;
    double very_small_mass = d_TINY_RHO * cellVol; 
    Ghost::GhostType  gn = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;         
         
    constNCVariable<double> NC_CCweight;
    NCVariable<double>NC_CCweight_copy;
    new_dw->allocateAndPut(NC_CCweight_copy, MIlb->NC_CCweightLabel, 0,patch);
    old_dw->get(NC_CCweight,                 MIlb->NC_CCweightLabel, 0,patch, 
                                                                      gac, 1);
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int indx = mpm_matl->getDWIndex();

      // Create arrays for the grid data
      constNCVariable<double> gmass, gvolume,gtempstar;
      constNCVariable<Vector> gvelocity;
      CCVariable<Vector> cmomentum;
      CCVariable<double> int_eng_L, rho_CC, mass_L;              
      constCCVariable<double> cmass, Temp_CC_sur;
      constCCVariable<Vector> vel_CC_sur; 
      new_dw->get(gmass,       Mlb->gMassLabel,           indx,patch,gac,1);
      new_dw->get(gvelocity,   Mlb->gVelocityStarLabel,   indx,patch,gac,1);
      new_dw->get(gtempstar,   Mlb->gTemperatureStarLabel,indx,patch,gac,1);
      new_dw->get(cmass,       MIlb->cMassLabel,          indx,patch,gn,0);    
      new_dw->get(Temp_CC_sur, MIlb->temp_CCLabel,        indx,patch,gn,0);    
      new_dw->get(vel_CC_sur,  MIlb->vel_CCLabel,         indx,patch,gn,0);
                                                           
      new_dw->getModifiable(rho_CC,     Ilb->rho_CCLabel,      indx,patch);
      new_dw->allocateAndPut(mass_L,    Ilb->mass_L_CCLabel,   indx,patch); 
      new_dw->allocateAndPut(cmomentum, Ilb->mom_L_CCLabel,    indx,patch);
      new_dw->allocateAndPut(int_eng_L, Ilb->int_eng_L_CCLabel,indx,patch);

      cmomentum.initialize(Vector(0.0, 0.0, 0.0));
      int_eng_L.initialize(0.);
      mass_L.initialize(0.); 
      double cv = mpm_matl->getSpecificHeat();

      IntVector nodeIdx[8];

      //---- P R I N T   D A T A ------ 
      if(d_ice->switchDebugLagrangianValues) {
        ostringstream desc;
        desc <<"TOP_MPMICE::computeLagrangianValuesMPM_mat_"<<indx<<"_patch_"
             <<  indx<<patch->getID();
        d_ice->printData(indx, patch,1,desc.str(), "cmass",    cmass);
        printData(     indx, patch,  1,desc.str(), "gmass",    gmass);
        d_ice->printData(indx, patch,1,desc.str(), "rho_CC",   rho_CC);
        printData(     indx, patch,  1,desc.str(), "gtemStar", gtempstar);
        //printNCVector( indx, patch,  1,desc.str(), "gvelocityStar", 0,
        //                                                       gvelocity);
      }
 
      for(CellIterator iter = patch->getExtraCellIterator();!iter.done();
                                                          iter++){ 
        patch->findNodesFromCell(*iter,nodeIdx);
        IntVector c = *iter;
        double int_eng_L_mpm = 0.0;
        double int_eng_L_sur = cmass[c] * Temp_CC_sur[c] * cv;
        Vector cmomentum_mpm = Vector(0.0, 0.0, 0.0);
        Vector cmomentum_sur = vel_CC_sur[c] * cmass[c];
        
        for (int in=0;in<8;in++){
          double NC_CCw_mass = NC_CCweight[nodeIdx[in]] * gmass[nodeIdx[in]];
          cmomentum_mpm +=gvelocity[nodeIdx[in]]      * NC_CCw_mass;
          int_eng_L_mpm +=gtempstar[nodeIdx[in]] * cv * NC_CCw_mass;
        }
        //__________________________________
        // set cmomentum/int_eng_L to either 
        // what's calculated from mpm or 
        // the surrounding value.
        // If you change this you must also change MPMICE::interpolateNCToCC_0  
        double one_or_zero = (cmass[c] - very_small_mass)/cmass[c];
        cmomentum[c] = (1.0 - one_or_zero)* cmomentum_sur + 
                              one_or_zero * cmomentum_mpm;
        int_eng_L[c] = (1.0 - one_or_zero)* int_eng_L_sur + 
                              one_or_zero * int_eng_L_mpm;
      }
      //__________________________________
      //  NO REACTION
      if(d_ice->d_models.size() == 0)  { 
        for(CellIterator iter = patch->getExtraCellIterator();!iter.done();
                                                    iter++){ 
         IntVector c = *iter;
         mass_L[c]    = cmass[c];
         rho_CC[c]    = mass_L[c] * inv_cellVol;
        }
      }
      //__________________________________
      //   M O D E L   B A S E D   E X C H A N G E
      // The reaction can't completely eliminate 
      //  all the mass, momentum and internal E.
      // If it does then we'll get erroneous vel,
      // and temps in CCMomExchange.  If the mass
      // goes to min_mass then cmomentum and int_eng_L
      // need to be scaled by min_mass to avoid inf temp and vel_CC
      // in 
      if(d_ice->d_models.size() > 0)  { 
        constCCVariable<double> modelMass_src;
        constCCVariable<double> modelEng_src;
        constCCVariable<Vector> modelMom_src;
        new_dw->get(modelMass_src,Ilb->modelMass_srcLabel,indx, patch, gn, 0);
        new_dw->get(modelMom_src, Ilb->modelMom_srcLabel, indx, patch, gn, 0);
        new_dw->get(modelEng_src, Ilb->modelEng_srcLabel, indx, patch, gn, 0);
                
        for(CellIterator iter = patch->getExtraCellIterator();!iter.done();
                                                    iter++){ 
          IntVector c = *iter;
          //  must have a minimum mass
          double min_mass = d_TINY_RHO * cellVol;
          double inv_cmass = 1.0/cmass[c];
          mass_L[c] = std::max( (cmass[c] + modelMass_src[c] ), min_mass);
          rho_CC[c] = mass_L[c] * inv_cellVol;
              
          //  must have a minimum momentum 
          for (int dir = 0; dir <3; dir++) {  //loop over all three directons
            double min_mom_L = min_mass * cmomentum[c][dir] * inv_cmass;
            double mom_L_tmp = cmomentum[c][dir] + modelMom_src[c][dir];

            // Preserve the original sign on momemtum     
            // Use d_SMALL_NUMs to avoid nans when mom_L_temp = 0.0
            double plus_minus_one = (mom_L_tmp+d_SMALL_NUM)/
                                    (fabs(mom_L_tmp+d_SMALL_NUM));

            mom_L_tmp = (mom_L_tmp/mass_L[c] ) * (cmass[c] + modelMass_src[c] );
                                
            cmomentum[c][dir] = plus_minus_one *
                                std::max( fabs(mom_L_tmp), fabs(min_mom_L) );
          }
          // must have a minimum int_eng   
          double min_int_eng = min_mass * int_eng_L[c] * inv_cmass;
          int_eng_L[c] = (int_eng_L[c]/mass_L[c]) * (cmass[c]+modelMass_src[c]);
          int_eng_L[c] = std::max((int_eng_L[c] + modelEng_src[c]),min_int_eng);
        }
      }  // if(model.size() >0)      
       
       //__________________________________
       //  Set Boundary conditions
       setBC(cmomentum, "set_if_sym_BC",patch, d_sharedState, indx, new_dw);
       setBC(int_eng_L, "set_if_sym_BC",patch, d_sharedState, indx, new_dw);
       setBC(rho_CC,    "Density",      patch, d_sharedState, indx, new_dw);  

      //---- P R I N T   D A T A ------ 
      if(d_ice->switchDebugLagrangianValues) {
        ostringstream desc;
        desc<<"BOT_MPMICE::computeLagrangianValuesMPM_mat_"<<indx<<"_patch_"
            <<  patch->getID();
        d_ice->printData(  indx,patch, 1,desc.str(), "rho_CC",        rho_CC);
        d_ice->printData(  indx,patch, 1,desc.str(), "int_eng_L",    int_eng_L);
        d_ice->printVector(indx,patch, 1,desc.str(), "mom_L_CC", 0,  cmomentum);
      }
      
      //---- B U L L E T   P R O O F I N G------
      IntVector neg_cell;
      ostringstream warn;
      if (!d_ice->areAllValuesPositive(int_eng_L, neg_cell)) {
        warn <<"ERROR MPMICE::computeLagrangianValuesMPM, mat "<< indx<<" cell "
             << neg_cell << " int_eng_L " << int_eng_L[neg_cell] << "\n ";
        throw InvalidValue(warn.str());
      }
      if (!d_ice->areAllValuesPositive(rho_CC, neg_cell)) {
        warn <<"ERROR MPMICE::computeLagrangianValuesMPM, mat "<<indx<<" cell "
             << neg_cell << " rho_CC " << rho_CC[neg_cell]<< "\n ";
        throw InvalidValue(warn.str());
      }
    }  //numMatls
    //__________________________________
    // carry forward interpolation weight 
    IntVector low = patch->getNodeLowIndex();
    IntVector hi  = patch->getNodeHighIndex();
    NC_CCweight_copy.copyPatch(NC_CCweight, low,hi);
  }  //patches
}

//______________________________________________________________________
//
void MPMICE::interpolateCCToNC(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* ,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw)
{ 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    if (cout_doing.active()) {
      cout_doing << "Doing interpolateCCToNC on patch "<< patch->getID()
                 <<"\t\t\t MPMICE" << endl;
    }

    //__________________________________
    // This is where I interpolate the CC 
    // changes to NCs for the MPMMatls
    int numMPMMatls = d_sharedState->getNumMPMMatls();

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());

    for (int m = 0; m < numMPMMatls; m++) {
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int indx = mpm_matl->getDWIndex();
      NCVariable<Vector> gacceleration, gvelocity;
      NCVariable<double> dTdt_NC;

      constCCVariable<double> mass_L_CC;
      constCCVariable<Vector> mom_L_ME_CC, old_mom_L_CC;
      constCCVariable<double> eng_L_ME_CC, old_int_eng_L_CC; 
      
      new_dw->getModifiable(gvelocity,    Mlb->gVelocityStarLabel,indx,patch);
      new_dw->getModifiable(gacceleration,Mlb->gAccelerationLabel,indx,patch);
                  
      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(old_mom_L_CC,    Ilb->mom_L_CCLabel,       indx,patch,gac,1);
      new_dw->get(old_int_eng_L_CC,Ilb->int_eng_L_CCLabel,   indx,patch,gac,1);
      new_dw->get(mass_L_CC,       Ilb->mass_L_CCLabel,      indx,patch,gac,1);
      new_dw->get(mom_L_ME_CC,     Ilb->mom_L_ME_CCLabel,    indx,patch,gac,1);
      new_dw->get(eng_L_ME_CC,     Ilb->eng_L_ME_CCLabel,    indx,patch,gac,1);

      double cv = mpm_matl->getSpecificHeat();     

      new_dw->allocateAndPut(dTdt_NC,     Mlb->dTdt_NCLabel,    indx, patch);
      dTdt_NC.initialize(0.0);
      IntVector cIdx[8];
      Vector dvdt_tmp;
      double dTdt_tmp;
      //__________________________________
      //  Take care of momentum and specific volume source
     if(!d_rigidMPM){
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
        patch->findCellsFromNode(*iter,cIdx);
        for(int in=0;in<8;in++){
          dvdt_tmp  = (mom_L_ME_CC[cIdx[in]] - old_mom_L_CC[cIdx[in]])
                    / (mass_L_CC[cIdx[in]] * delT); 
          gvelocity[*iter]     +=  dvdt_tmp*.125*delT;
          gacceleration[*iter] +=  dvdt_tmp*.125;
        }
      }
     }    
      
      //__________________________________
      //  E Q  F O R M
      if (d_ice->d_EqForm){
        for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
         patch->findCellsFromNode(*iter,cIdx);
         for(int in=0;in<8;in++){
                         // eng_L_ME = internal energy
            dTdt_tmp  = ( eng_L_ME_CC[cIdx[in]] - old_int_eng_L_CC[cIdx[in]])
                      / (mass_L_CC[cIdx[in]] * cv * delT);
            dTdt_NC[*iter]  +=  dTdt_tmp*.125;
          }
        } 
      }
      //__________________________________
      //   R A T E   F O R M
      if (d_ice->d_RateForm){
        double KE, int_eng_L_ME;
        Vector vel_CC;
      
        for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
          patch->findCellsFromNode(*iter,cIdx);
          for(int in=0;in<8;in++){
            // convert total energy to internal energy
            vel_CC = mom_L_ME_CC[cIdx[in]]/mass_L_CC[cIdx[in]];
            KE = 0.5 * mass_L_CC[cIdx[in]] * vel_CC.length() * vel_CC.length();
            int_eng_L_ME = eng_L_ME_CC[cIdx[in]] - KE;

            dTdt_tmp  = ( int_eng_L_ME - old_int_eng_L_CC[cIdx[in]])
                        / (mass_L_CC[cIdx[in]] * cv * delT); 
            dTdt_NC[*iter]   +=  dTdt_tmp*.125;
          }
        }
      }
  
      //---- P R I N T   D A T A ------ 
      if(switchDebug_InterpolateCCToNC) {
        ostringstream desc;
        desc<< "BOT_MPMICE::interpolateCCToNC_mat_"<< indx<<"_patch_"
            <<patch->getID();                   
        printData(    indx,patch, 1,desc.str(), "dTdt_NC",     dTdt_NC);
        printNCVector(indx,patch, 1,desc.str(),"gvelocity",    0,gvelocity);
        printNCVector(indx,patch, 1,desc.str(),"gacceleration",0,gacceleration);
      }
    }  
  }  //patches
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
                                     const PatchSubset* patches,
                                     const MaterialSubset* ,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    if (cout_doing.active()) {
      cout_doing<<"Doing computeEquilibrationPressure on patch "
                << patch->getID() <<"\t\t MPMICE" << endl;
    }

    double    converg_coeff = 100.;
    double    convergence_crit = converg_coeff * DBL_EPSILON;
    double    c_2;
    double press_ref= d_sharedState->getRefPress();
    int numICEMatls = d_sharedState->getNumICEMatls();
    int numMPMMatls = d_sharedState->getNumMPMMatls();
    int numALLMatls = numICEMatls + numMPMMatls;

    Vector dx       = patch->dCell(); 
    double cell_vol = dx.x()*dx.y()*dx.z();
    char warning[100];
    static int n_passes = 0;
    n_passes ++; 

    StaticArray<double> press_eos(numALLMatls);
    StaticArray<double> dp_drho(numALLMatls),dp_de(numALLMatls);
    StaticArray<double> mat_volume(numALLMatls);
    StaticArray<double> kappa(numALLMatls);

    StaticArray<CCVariable<double> > vol_frac(numALLMatls);
    StaticArray<CCVariable<double> > rho_micro(numALLMatls);
    StaticArray<CCVariable<double> > rho_CC_new(numALLMatls);
    StaticArray<CCVariable<double> > speedSound(numALLMatls);
    StaticArray<CCVariable<double> > sp_vol_new(numALLMatls);
    StaticArray<CCVariable<double> > f_theta(numALLMatls);
    StaticArray<constCCVariable<double> > placeHolder(0);
    StaticArray<constCCVariable<double> > cv(numALLMatls);
    StaticArray<constCCVariable<double> > gamma(numALLMatls);
    StaticArray<constCCVariable<double> > sp_vol_CC(numALLMatls); 
    StaticArray<constCCVariable<double> > Temp(numALLMatls);
    StaticArray<constCCVariable<double> > rho_CC_old(numALLMatls);
    StaticArray<constCCVariable<double> > mass_CC(numALLMatls);
    StaticArray<constCCVariable<Vector> > vel_CC(numALLMatls);
    constCCVariable<double> press;    
    CCVariable<double> press_new, delPress_tmp, press_copy;    
    Ghost::GhostType  gn = Ghost::None;
    //__________________________________
    //  Implicit press calc. needs two copies of the pressure
    old_dw->get(press,                Ilb->press_CCLabel, 0,patch,gn, 0); 
    new_dw->allocateAndPut(press_new, Ilb->press_equil_CCLabel, 0,patch);
    new_dw->allocateAndPut(press_copy,Ilb->press_CCLabel,       0,patch);
    new_dw->allocateTemporary(delPress_tmp, patch); 

    StaticArray<MPMMaterial*> mpm_matl(numALLMatls);
    StaticArray<ICEMaterial*> ice_matl(numALLMatls);
    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ice_matl[m] = dynamic_cast<ICEMaterial*>(matl);
      mpm_matl[m] = dynamic_cast<MPMMaterial*>(matl);
    }

    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      if(ice_matl[m]){                    // I C E
        old_dw->get(Temp[m],      Ilb->temp_CCLabel,      indx,patch,gn,0);
        old_dw->get(rho_CC_old[m],Ilb->rho_CCLabel,       indx,patch,gn,0);
        old_dw->get(sp_vol_CC[m], Ilb->sp_vol_CCLabel,    indx,patch,gn,0);
        old_dw->get(vel_CC[m],    Ilb->vel_CCLabel,       indx,patch,gn,0);
        new_dw->get(cv[m],        Ilb->specific_heatLabel,indx,patch,gn,0);
        new_dw->get(gamma[m],     Ilb->gammaLabel,        indx,patch,gn,0);
      }
      if(mpm_matl[m]){                    // M P M
        new_dw->get(Temp[m],     MIlb->temp_CCLabel, indx,patch,gn,0);
        new_dw->get(mass_CC[m],  MIlb->cMassLabel,   indx,patch,gn,0);
        new_dw->get(vel_CC[m],   MIlb->vel_CCLabel,  indx,patch,gn,0);
        new_dw->get(sp_vol_CC[m],Ilb->sp_vol_CCLabel,indx,patch,gn,0); 
        new_dw->get(rho_CC_old[m],Ilb->rho_CCLabel,  indx,patch,gn,0);

      }
      new_dw->allocateTemporary(rho_micro[m],  patch);
      new_dw->allocateAndPut(rho_CC_new[m], Ilb->rho_CCLabel,indx, patch);
      new_dw->allocateAndPut(vol_frac[m],   Ilb->vol_frac_CCLabel,  indx,patch);
      new_dw->allocateAndPut(f_theta[m],    Ilb->f_theta_CCLabel,   indx,patch);
      new_dw->allocateAndPut(speedSound[m], Ilb->speedSound_CCLabel,indx,patch);
      new_dw->allocateAndPut(sp_vol_new[m], Ilb->sp_vol_CCLabel,    indx,patch);
    }

    press_new.copyData(press);

    //__________________________________
    // Compute rho_micro, speedSound, volfrac, rho_CC
    // see Docs/MPMICE.txt for explaination of why we ONlY
    // use eos evaulations for rho_micro_mpm

    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      const IntVector& c = *iter;
      double total_mat_vol = 0.0;
      for (int m = 0; m < numALLMatls; m++) {
        if(ice_matl[m]){                // I C E
         rho_micro[m][c] = 1.0/sp_vol_CC[m][c];
        } else if(mpm_matl[m]){                //  M P M
          rho_micro[m][c] =  mpm_matl[m]->getConstitutiveModel()->
            computeRhoMicroCM(press_new[c],press_ref, mpm_matl[m]); 
        }
        mat_volume[m] = (rho_CC_old[m][c]*cell_vol)/rho_micro[m][c];
        total_mat_vol += mat_volume[m];
      }  // numAllMatls loop

      for (int m = 0; m < numALLMatls; m++) {
        vol_frac[m][c] = mat_volume[m]/total_mat_vol;
        rho_CC_new[m][c] = vol_frac[m][c]*rho_micro[m][c];
      }
    }  // cell iterator
    //---- P R I N T   D A T A ------
    if(d_ice -> switchDebug_EQ_RF_press)  {
      ostringstream desc1;
      desc1<< "TOP_equilibration_patch_"<< patch->getID();
      d_ice->printData( 0, patch, 1, desc1.str(), "Press_CC_top", press); 

      for (int m = 0; m < numALLMatls; m++)  {
        Material* matl = d_sharedState->getMaterial( m );
        int indx = matl->getDWIndex();
        ostringstream desc;
        desc<<"TOP_equilibration_Mat_"<< indx<<"_patch_"<<patch->getID();
        d_ice->printData( indx,patch,1,desc.str(),"rho_CC_new",rho_CC_new[m]);
        d_ice->printData( indx,patch,1,desc.str(),"rho_micro", rho_micro[m]);
        d_ice->printData( indx,patch,1,desc.str(),"Temp_CC",   Temp[m]);     
        d_ice->printData( indx,patch,1,desc.str(),"vol_frac_CC",vol_frac[m]);
      }
    }
    //______________________________________________________________________
    // Done with preliminary calcs, now loop over every cell
    int count, test_max_iter = 0;

    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      const IntVector& c = *iter;  
      double delPress = 0.;
      bool converged  = false;
      double sum;
      count           = 0;

      while ( count < d_ice->d_max_iter_equilibration && converged == false) {
        count++;
        //__________________________________
        // evaluate press_eos at cell i,j,k
        for (int m = 0; m < numALLMatls; m++)  {
          if(ice_matl[m]){    // ICE
            ice_matl[m]->getEOS()->computePressEOS(rho_micro[m][c],gamma[m][c],
                                                   cv[m][c],Temp[m][c],
                                                   press_eos[m],
                                                   dp_drho[m],dp_de[m]);
          } else if(mpm_matl[m]){    // MPM
            mpm_matl[m]->getConstitutiveModel()->
              computePressEOSCM(rho_micro[m][c],press_eos[m],press_ref,
                                dp_drho[m], c_2,mpm_matl[m]);
          }
        }

        //__________________________________
        // - compute delPress
        // - update press_CC     
        double A = 0., B = 0., C = 0.;

        for (int m = 0; m < numALLMatls; m++)   {
          double Q =  press_new[c] - press_eos[m];
          double inv_y =  (vol_frac[m][c] * vol_frac[m][c])
            / (dp_drho[m] * rho_CC_new[m][c] + d_SMALL_NUM);
                                 
          A   +=  vol_frac[m][c];
          B   +=  Q * inv_y;
          C   +=  inv_y;
        } 
        double vol_frac_not_close_packed = 1.;
        delPress = (A - vol_frac_not_close_packed - B)/C;

        press_new[c] += delPress;

        if(press_new[c] < convergence_crit ){
          press_new[c] = fabs(delPress);
        }

       //__________________________________
       // backout rho_micro_CC at this new pressure
       // - compute the updated volume fractions
       sum = 0;
       for (int m = 0; m < numALLMatls; m++) {
         if(ice_matl[m]){
           rho_micro[m][c] = 
             ice_matl[m]->getEOS()->computeRhoMicro(press_new[c],gamma[m][c],
                                           cv[m][c],Temp[m][c],rho_micro[m][c]);
         } if(mpm_matl[m]){
           rho_micro[m][c] =  
             mpm_matl[m]->getConstitutiveModel()->computeRhoMicroCM(
                                          press_new[c],press_ref,mpm_matl[m]);
         }
         vol_frac[m][c]   = rho_CC_new[m][c]/rho_micro[m][c];
         sum += vol_frac[m][c];
       }

       //__________________________________
       // - Test for convergence 
       //  If sum of vol_frac_CC ~= 1.0 then converged 
       if (fabs(sum-vol_frac_not_close_packed) < convergence_crit){
         converged = true;
         //__________________________________
         // Find the speed of sound based on the converged solution
         for (int m = 0; m < numALLMatls; m++)  {
           if(ice_matl[m]){
             ice_matl[m]->getEOS()->computePressEOS(rho_micro[m][c],gamma[m][c],
                                              cv[m][c],Temp[m][c],press_eos[m],
                                              dp_drho[m],dp_de[m]);

             c_2 = dp_drho[m] + dp_de[m] * 
                        (press_eos[m]/(rho_micro[m][c]*rho_micro[m][c]));
           } else if(mpm_matl[m]){
              mpm_matl[m]->getConstitutiveModel()->
                   computePressEOSCM(rho_micro[m][c],press_eos[m],press_ref,
                                     dp_drho[m],c_2,mpm_matl[m]);
           }
           speedSound[m][c] = sqrt(c_2);         // Isentropic speed of sound
         }
       }
     }   // end of converged

      delPress_tmp[c] = delPress;

     //__________________________________
     // If the pressure solution has stalled out 
     //  then try a binary search
     if(count >= d_ice->d_max_iter_equilibration) {

        binaryPressureSearch( Temp, rho_micro, vol_frac, rho_CC_new,
                              speedSound,  dp_drho,  dp_de, 
                              press_eos, press, press_new, press_ref,
                              cv, gamma, convergence_crit, 
                              numALLMatls, count, sum, c);

     }
     test_max_iter = std::max(test_max_iter, count);

      //__________________________________
      //      BULLET PROOFING
      if(test_max_iter == d_ice->d_max_iter_equilibration) 
       throw MaxIteration(c,count,n_passes,"MaxIterations reached");

      for (int m = 0; m < numALLMatls; m++) {
           ASSERT(( vol_frac[m][c] > 0.0 ) ||
                  ( vol_frac[m][c] < 1.0));
      }
      if ( fabs(sum - 1.0) > convergence_crit) {  
         throw MaxIteration(c,count,n_passes,
                         "MaxIteration reached vol_frac != 1");
      }
      if ( press_new[c] < 0.0 ) {
         throw MaxIteration(c,count,n_passes,
                         "MaxIteration reached press_new < 0");
      }

      for (int m = 0; m < numALLMatls; m++)
      if ( rho_micro[m][c] < 0.0 || vol_frac[m][c] < 0.0) {
        int i = c.x(), j=c.y(), k=c.z();
          sprintf(warning, 
          " cell[%d][%d][%d], mat %d, iter %d, n_passes %d,Now exiting ",
          i,j,k,m,count,n_passes);
          cout << "r_m " << rho_micro[m][c] << endl;
          cout << "v_f " << vol_frac[m][c]  << endl;
          cout << "tmp " << Temp[m][c]      << endl;
          cout << "p_n " << press_new[c]    << endl;
          cout << "m_v " << mat_volume[m]       << endl;
          d_ice->Message(1," calc_equilibration_press:", 
              " rho_micro < 0 || vol_frac < 0", warning);
      }
    }     // end of cell interator
    if (cout_norm.active())
      cout_norm<<"max number of iterations in any cell \t"<<test_max_iter<<endl;

    //__________________________________
    // Now change how rho_CC is defined to 
    // rho_CC = mass/cell_volume  NOT mass/mat_volume 
    for (int m = 0; m < numALLMatls; m++) {
      // Is this necessary? - Steve
      rho_CC_new[m].copyData(rho_CC_old[m]);
    }

    //__________________________________
    // - update Boundary conditions
    //   Don't set Lodi bcs, we already compute Press
    //   in all the extra cells.
    // - make copy of press for implicit calc.
    preprocess_CustomBCs("EqPressMPMICE",old_dw, new_dw, Ilb, patch, 999,
                          d_ice->d_customBC_var_basket);
    
    setBC(press_new,   rho_micro, placeHolder,d_ice->d_surroundingMatl_indx,
          "rho_micro", "Pressure", patch , d_sharedState, 0, new_dw, 
          d_ice->d_customBC_var_basket);
    
    delete_CustomBCs(d_ice->d_customBC_var_basket);
    
    // Is this necessary?  - Steve
    press_copy.copyData(press_new);
     
    //__________________________________
    // compute sp_vol_CC
    for (int m = 0; m < numALLMatls; m++)   {
    
/*`==========TESTING==========*/
#if 0
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      setBC_rho_micro( patch,mpm_matl[m], ice_matl[m], indx,
                       cv[m], gamma[m],  press_new, Temp[m],  
                       press_ref, rho_micro[m]);
#endif   
/*===========TESTING==========`*/
      
      for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
        const IntVector& c = *iter;
        sp_vol_new[m][c] = 1.0/rho_micro[m][c];
      }
    }
    //__________________________________
    //  compute f_theta  
    for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
      const IntVector& c = *iter;
      double sumVolFrac_kappa = 0.0;
      for (int m = 0; m < numALLMatls; m++) {
        kappa[m] = sp_vol_new[m][c]/(speedSound[m][c]*speedSound[m][c]);
        sumVolFrac_kappa += vol_frac[m][c]*kappa[m];
      }
      for (int m = 0; m < numALLMatls; m++) {
        f_theta[m][c] = vol_frac[m][c]*kappa[m]/sumVolFrac_kappa;
      }
    }
    
  //---- P R I N T   D A T A ------
    if(d_ice -> switchDebug_EQ_RF_press)  { 
      ostringstream desc1;
      desc1<< "BOT_equilibration_patch_"<<patch->getID();

      d_ice->printData( 0, patch, 1, desc1.str(),"Press_CC_equil",press_new);
      d_ice->printData( 0, patch, 1, desc1.str(),"delPress",      delPress_tmp);
      for (int m = 0; m < numALLMatls; m++)  {
         Material* matl = d_sharedState->getMaterial( m );
         int indx = matl->getDWIndex();
         ostringstream desc; 
         desc<< "BOT_equilibration_Mat_"<<indx<<"_patch_"<< patch->getID();
         d_ice->printData(indx,patch,1,desc.str(),"rho_CC",      rho_CC_new[m]);
         d_ice->printData(indx,patch,1,desc.str(),"rho_micro_CC",rho_micro[m]);
         d_ice->printData(indx,patch,1,desc.str(),"vol_frac_CC", vol_frac[m]);
      }
    }
  }  //patches
}

/* --------------------------------------------------------------------- 
 Function~  MPMICE::binaryPressureSearch-- 
 Purpose:   When the technique for find the equilibration pressure
            craps out then try this method.
 Reference:  See Jim.
_____________________________________________________________________*/ 
void MPMICE::binaryPressureSearch(  StaticArray<constCCVariable<double> >& Temp,
                            StaticArray<CCVariable<double> >& rho_micro, 
                            StaticArray<CCVariable<double> >& vol_frac, 
                            StaticArray<CCVariable<double> >& rho_CC_new,
                            StaticArray<CCVariable<double> >& speedSound,
                            StaticArray<double> & dp_drho, 
                            StaticArray<double> & dp_de, 
                            StaticArray<double> & press_eos,
                            constCCVariable<double> & press,
                            CCVariable<double> & press_new, 
                            double press_ref,
                            StaticArray<constCCVariable<double> > & cv,
                            StaticArray<constCCVariable<double> > & gamma,
                            double convergence_crit,
                            int numALLMatls,
                            int & count,
                            double & sum,
                            IntVector c )
{
   // Start over for this cell using a binary search
//   cout << " cell " << c << " Starting binary pressure search "<< endl;
   count = 0;
   bool converged = false;
   double c_2;
   double Pleft=0., Pright=0., Ptemp=0., Pm=0.;
   double rhoMicroR=0., rhoMicroL=0.;
   StaticArray<double> vfR(numALLMatls);
   StaticArray<double> vfL(numALLMatls);
   Pm = press[c];

   while ( count < d_ice->d_max_iter_equilibration && converged == false) {
   count++;
   sum = 0.;
   for (int m = 0; m < numALLMatls; m++) {
     Material* matl = d_sharedState->getMaterial( m );
     ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
     MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
     if(ice_matl){        // ICE
       rho_micro[m][c] =
         ice_matl->getEOS()->computeRhoMicro(Pm,gamma[m][c],
                                           cv[m][c],Temp[m][c],rho_micro[m][c]);
     }
     if(mpm_matl){        // MPM
       rho_micro[m][c] =
         mpm_matl->getConstitutiveModel()->computeRhoMicroCM(
                                      Pm,press_ref,mpm_matl);
     }
     vol_frac[m][c] = rho_CC_new[m][c]/rho_micro[m][c];
     sum += vol_frac[m][c];
   }  // loop over matls
   double residual = 1. - sum;

   if(fabs(residual) <= convergence_crit){
      converged = true;
      press_new[c] = Pm;
      //__________________________________
      // Find the speed of sound at ijk
      // needed by eos and the the explicit
      // del pressure function
      for (int m = 0; m < numALLMatls; m++)  {
        Material* matl = d_sharedState->getMaterial( m );
        ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
        MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
        if(ice_matl){       // ICE
          ice_matl->getEOS()->computePressEOS(rho_micro[m][c],gamma[m][c],
                                              cv[m][c],Temp[m][c],press_eos[m],
                                              dp_drho[m],dp_de[m]);
          c_2 = dp_drho[m] + dp_de[m] *
                     (press_eos[m]/(rho_micro[m][c]*rho_micro[m][c]));
        }
        if(mpm_matl){       // MPM
           mpm_matl->getConstitutiveModel()->
                computePressEOSCM(rho_micro[m][c],press_eos[m],press_ref,
                                  dp_drho[m],c_2,mpm_matl);
        }
        speedSound[m][c] = sqrt(c_2);     // Isentropic speed of sound
     }
   }
   if(count == 1){
    if(residual < 0){
     Pleft  = press[c];
     Pright = 3.*press[c];
     Ptemp  = max(10.*press[c],press_ref);
    }
    else{
     Pleft  = DBL_EPSILON;
     Pright = press[c];
    }
   }

   double sumR=0., sumL=0.;
   for (int m = 0; m < numALLMatls; m++) {
     Material* matl = d_sharedState->getMaterial( m );
     ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
     MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
     
     if(ice_matl){        //  ICE
       rhoMicroR = ice_matl->getEOS()->
        computeRhoMicro(Pright,gamma[m][c],cv[m][c],Temp[m][c],rho_micro[m][c]);
       rhoMicroL = ice_matl->getEOS()->
        computeRhoMicro(Pleft, gamma[m][c],cv[m][c],Temp[m][c],rho_micro[m][c]);
     }
     if(mpm_matl){        //  MPM
       rhoMicroR =
         mpm_matl->getConstitutiveModel()->computeRhoMicroCM(
                                      Pright,press_ref,mpm_matl);
       rhoMicroL =
         mpm_matl->getConstitutiveModel()->computeRhoMicroCM(
                                      Pleft, press_ref,mpm_matl);
     }
     vfR[m] = rho_CC_new[m][c]/rhoMicroR;
     vfL[m] = rho_CC_new[m][c]/rhoMicroL;
     sumR+=vfR[m];
     sumL+=vfL[m];
   }  // all matls
   
   double prod = (1.- sumR)*(1. - sumL);
   if(prod < 0.){
     Ptemp = Pleft;
     Pleft = .5*(Pleft + Pright);
   }
   else{
     Pleft  = Pright;
     Pright = Ptemp;
     if(Pleft == Pright){
        Pright = 4.*Pleft;
     }
     Ptemp = 2.*Ptemp;
   }
   Pm = .5*(Pleft + Pright);
//   cout << setprecision(15);
//   cout << "Pm = " << Pm << " 1.-sum " << residual << endl;
  }   // end of converged
}

//______________________________________________________________________
//
void MPMICE::interpolateMassBurnFractionToNC(const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset* ,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    if (cout_doing.active()) {
      cout_doing << "Doing interpolateMassBurnFractionToNC on patch "
                 << patch->getID() <<"\t MPMICE" << endl;
    }

    // Interpolate the CC burn fraction to the nodes

    int numALLMatls = d_sharedState->getNumMPMMatls() + 
      d_sharedState->getNumICEMatls();

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());
    Ghost::GhostType  gac = Ghost::AroundCells;
    
    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      int indx = matl->getDWIndex();
      if(mpm_matl){
        constCCVariable<double> massCC;
        NCVariable<double> massBurnFraction;
        new_dw->get(massCC,       MIlb->cMassLabel,          indx,patch, gac,1);
        new_dw->allocateAndPut(massBurnFraction, 
                                  Mlb->massBurnFractionLabel,indx,patch);
        massBurnFraction.initialize(0.);
        if(d_ice->d_models.size() > 0)  { 
          constCCVariable<double> modelMass_src;
          new_dw->get(modelMass_src,Ilb->modelMass_srcLabel,indx,patch, gac,1);
          
          IntVector cIdx[8];  
          for(NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
             patch->findCellsFromNode(*iter,cIdx);
            for (int in=0;in<8;in++){
              massBurnFraction[*iter] +=
                       (fabs(modelMass_src[cIdx[in]])/massCC[cIdx[in]])*.125;

            }
          }
        }  // if(models >0 )
      }  //if(mpm_matl)
    }  //ALLmatls  
  }  //patches
}
//______________________________________________________________________
bool MPMICE::needRecompile(double time, double dt, const GridP& grid) {
  if(d_recompile){
    d_recompile = false;
    return true;
  }
  else{
    return false;
  }
}
                                                                                
void MPMICE::addMaterial(const ProblemSpecP& prob_spec, GridP& grid,
                         SimulationStateP&   sharedState)
{
  d_recompile = true;
  if(d_sharedState->needAddMaterial() > 0){
    d_ice->addMaterial(prob_spec, grid, d_sharedState);
    cout << "Adding an ICE material" << endl;
  }
  if(d_sharedState->needAddMaterial() < 0){
    d_mpm->addMaterial(prob_spec, grid, d_sharedState);
    d_ice->updateExchangeCoefficients(prob_spec, grid, d_sharedState);
    cout << "Adding an MPM material" << endl;
  }
}
//______________________________________________________________________
void MPMICE::scheduleInitializeAddedMaterial(const LevelP& level,
                                             SchedulerP& sched)
{
  cout_doing << "Entering MPMICE::scheduleInitializeAddedMaterial" << endl;

  if(d_sharedState->needAddMaterial() > 0){
    d_ice->scheduleInitializeAddedMaterial(level,sched);
  }
  if(d_sharedState->needAddMaterial() < 0){
    d_mpm->scheduleInitializeAddedMaterial(level,sched);

    Task* t = scinew Task("MPMICE::actuallyInitializeAddedMPMMaterial",
                           this, &MPMICE::actuallyInitializeAddedMPMMaterial);

    int addedMaterialIndex = d_sharedState->getNumMatls() - 1;

    MaterialSubset* add_matl = scinew MaterialSubset();
    add_matl->add(addedMaterialIndex);
    add_matl->addReference();
    t->computes(MIlb->vel_CCLabel,       add_matl);
    t->computes(Ilb->rho_CCLabel,        add_matl);
    t->computes(Ilb->temp_CCLabel,       add_matl);
    t->computes(Ilb->sp_vol_CCLabel,     add_matl);
    t->computes(Ilb->speedSound_CCLabel, add_matl);

    sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());

    if (add_matl->removeReference())
      delete add_matl; // shouln't happen, but...
  }

  cout_doing  << "Leaving MPMICE::scheduleInitializeAddedMaterial" << endl;

}
/*______________________________________________________________________
 Function~  setBC_rho_micro
 Purpose: set the boundary conditions for the microscopic density in 
 MPMICE computeEquilibration Press
______________________________________________________________________*/
void MPMICE::setBC_rho_micro(const Patch* patch,
                             MPMMaterial* mpm_matl,
                             ICEMaterial* ice_matl,
                             const int indx,
                             const CCVariable<double>& cv,
                             const CCVariable<double>& gamma,
                             const CCVariable<double>& press_new,
                             const CCVariable<double>& Temp,
                             const double press_ref,
                             CCVariable<double>& rho_micro)
{
  //__________________________________
  //  loop over faces on the boundary
  vector<Patch::FaceType>::const_iterator f;
  for (f  = patch->getBoundaryFaces()->begin(); 
       f != patch->getBoundaryFaces()->end(); ++f){
    Patch::FaceType face = *f;
    
    int numChildren = patch->getBCDataArray(face)->getNumberChildren(indx);
    for (int child = 0;  child < numChildren; child++) {
      double bc_value = -9;
      string bc_kind = "NotSet";
      vector<IntVector> bound;
      
      bool foundIterator =
        getIteratorBCValueBCKind<double>( patch, face, child, "Density", indx,
                                          bc_value, bound, bc_kind); 
      
      //cout << " face " << face << " bc_kind " << bc_kind << " \t indx " << indx
      //     <<"\t bound limits = "<< *bound.begin()<< " "<< *(bound.end()-1) << endl;
      
      if (foundIterator && bc_kind == "Dirichlet") { 
        // what do you put here?
      }
      if (foundIterator && bc_kind != "Dirichlet") {      // Everything else
        vector<IntVector>::const_iterator iter;
        
        if(ice_matl){             //  I C E
          for (iter=bound.begin(); iter != bound.end(); iter++) {
            IntVector c = *iter;
            rho_micro[c] = 
              ice_matl->getEOS()->computeRhoMicro(press_new[c],gamma[c],
                                           cv[c],Temp[c],rho_micro[c]);
          }
        } else if(mpm_matl){     //  M P M
          for (iter=bound.begin(); iter != bound.end(); iter++) {
            IntVector c = *iter;
            rho_micro[c] =  
              mpm_matl->getConstitutiveModel()->computeRhoMicroCM(
                                         press_new[c],press_ref,mpm_matl);
          }
        }  // mpm
      } // if not
    } // child loop
  } // face loop
}
//______________________________________________________________________
void MPMICE::actuallyInitializeAddedMPMMaterial(const ProcessorGroup*, 
                                                const PatchSubset* patches,
                                                const MaterialSubset* ,
                                                DataWarehouse*,
                                                DataWarehouse* new_dw)
{
  new_dw->unfinalize();
  for(int p=0;p<patches->size();p++){ 
    const Patch* patch = patches->get(p);
    cout_doing << "Doing actuallyInitializeAddedMPMMaterial on patch "
               << patch->getID() << "\t\t\t MPMICE" << endl;
    double junk=-9, tmp;
    int m    = d_sharedState->getNumMPMMatls() - 1;
    int indx = d_sharedState->getNumMatls() - 1;
    double p_ref = d_sharedState->getRefPress();
    CCVariable<double> rho_micro, sp_vol_CC, rho_CC, Temp_CC, speedSound;
    CCVariable<Vector> vel_CC;
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    new_dw->allocateTemporary(rho_micro, patch);
    new_dw->allocateAndPut(sp_vol_CC, Ilb->sp_vol_CCLabel,    indx,patch);
    new_dw->allocateAndPut(rho_CC,    Ilb->rho_CCLabel,       indx,patch);
    new_dw->allocateAndPut(speedSound,Ilb->speedSound_CCLabel,indx,patch);
    new_dw->allocateAndPut(Temp_CC,  MIlb->temp_CCLabel,      indx,patch);
    new_dw->allocateAndPut(vel_CC,   MIlb->vel_CCLabel,       indx,patch);

    mpm_matl->initializeDummyCCVariables(rho_micro,   rho_CC,
                                         Temp_CC,     vel_CC, 0,patch);

    setBC(rho_CC,    "Density",      patch, d_sharedState, indx, new_dw);
    setBC(rho_micro, "Density",      patch, d_sharedState, indx, new_dw);
    setBC(Temp_CC,   "Temperature",  patch, d_sharedState, indx, new_dw);
    setBC(vel_CC,    "Velocity",     patch, d_sharedState, indx, new_dw);
    for (CellIterator iter = patch->getExtraCellIterator();
                                                      !iter.done();iter++){
      IntVector c = *iter;
      sp_vol_CC[c] = 1.0/rho_micro[c];
                                                                              
      mpm_matl->getConstitutiveModel()->
          computePressEOSCM(rho_micro[c],junk, p_ref, junk, tmp,mpm_matl);
      speedSound[c] = sqrt(tmp);
    }
  }
  new_dw->refinalize();
}
