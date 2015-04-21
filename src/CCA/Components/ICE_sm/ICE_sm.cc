/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/ICE_sm/ICE_sm.h>
#include <CCA/Components/ICE_sm/ConservationTest.h>
#include <CCA/Components/ICE_sm/Diffusion.h>
#include <CCA/Components/ICE_sm/ICEMaterial.h>
#include <CCA/Components/ICE_sm/BoundaryCond.h>
#include <CCA/Components/ICE_sm/Advection/AdvectionFactory.h>
#include <CCA/Components/ICE_sm/EOS/EquationOfState.h>
#include <CCA/Components/ICE_sm/SpecificHeatModel/SpecificHeat.h>

#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/DbgOutput.h>

#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/InvalidValue.h>

#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Util/DebugStream.h>

#include <sstream>
#include <iostream>
#include <cfloat>

using namespace std;
using namespace Uintah;

//__________________________________
//  To turn on internal debugging code
//  setenv SCI_DEBUG "ICE_SM_DOING_COUT:+"
//  ICE_DOING_COUT:   output when tasks are scheduled and performed
static DebugStream iceCout("ICE_SM", false);


ICE_sm::ICE_sm(const ProcessorGroup* myworld, const bool doAMR) :
  UintahParallelComponent(myworld)
{
  lb   = scinew ICELabel();

  d_useCompatibleFluxes   = true;
  d_viscousFlow           = false;
  d_gravity               = Vector(0,0,0);     // hardwired for mini app
  d_EVIL_NUM              = -9.99e30;
  d_SMALL_NUM             = 1.0e-100;
  d_matl                  = 0;                 // ICE material index


  d_conservationTest      = scinew conservationTest_flags();
  d_conservationTest->onOff = false;
}
//______________________________________________________________________
//
//______________________________________________________________________
ICE_sm::~ICE_sm()
{
  iceCout << d_myworld->myrank() << " Doing: ICE destructor " << endl;

  delete d_conservationTest;
  delete lb;
  delete d_advector;
}
//______________________________________________________________________
//
bool ICE_sm::restartableTimesteps()
{
  return true;
}
//______________________________________________________________________
//
double ICE_sm::recomputeTimestep(double current_dt)
{
  return current_dt * 0.75;
}

/* _____________________________________________________________________
  Purpose~  Read in the xml file and set global variables
_____________________________________________________________________*/
void ICE_sm::problemSetup(const ProblemSpecP& prob_spec,
                          const ProblemSpecP& restart_prob_spec,
                          GridP& grid,
                          SimulationStateP&   sharedState)
{
  iceCout << d_myworld->myrank() << " Doing ICE_sm::problemSetup " << "\t\t\t ICE" << endl;
  d_sharedState = sharedState;

  dataArchiver = dynamic_cast<Output*>(getPort("output"));
  if(!dataArchiver){
    throw InternalError("ICE:couldn't get output port", __FILE__, __LINE__);
  }

  //__________________________________
  // Pull out from CFD-ICE section
  ProblemSpecP cfd_ps = prob_spec->findBlock("CFD");

  if(!cfd_ps){
    throw ProblemSetupException(
     "\n Could not find the <CFD> section in the input file\n",__FILE__, __LINE__);
  }

  cfd_ps->require("cfl",d_CFL);
  ProblemSpecP cfd_ice_ps = cfd_ps->findBlock("ICE");
  if(!cfd_ice_ps){
    throw ProblemSetupException(
     "\n Could not find the <CFD> <ICE> section in the input file\n",__FILE__, __LINE__);
  }

  d_advector = AdvectionFactory::create(cfd_ice_ps, d_useCompatibleFluxes, d_OrderOfAdvection);


  //__________________________________
  // register ICE material
  // on a restart you use a different problem spec
  ProblemSpecP mat_ps = 0;

  if (prob_spec->findBlockWithOutAttribute("MaterialProperties")){
    mat_ps = prob_spec->findBlockWithOutAttribute("MaterialProperties");
  }else if (restart_prob_spec){
    mat_ps = restart_prob_spec->findBlockWithOutAttribute("MaterialProperties");
  }

  ProblemSpecP ice_mat_ps   = mat_ps->findBlock("ICE");
  ProblemSpecP ps = ice_mat_ps->findBlock("material");

  oneICEMaterial *mat = scinew oneICEMaterial(ps, sharedState);
  sharedState->registerOneICEMaterial(mat);

  //__________________________________
  //  Are we going to perform a conservationTest
  if (dataArchiver->isLabelSaved("TotalMass") ){
    d_conservationTest->mass     = true;
    d_conservationTest->onOff    = true;
  }
  if (dataArchiver->isLabelSaved("TotalMomentum") ){
    d_conservationTest->momentum = true;
    d_conservationTest->onOff    = true;
  }
  if (dataArchiver->isLabelSaved("TotalIntEng")   ||
      dataArchiver->isLabelSaved("KineticEnergy") ){
    d_conservationTest->energy   = true;
    d_conservationTest->onOff    = true;
  }


  //__________________________________
  //  boundary condition warnings
  BC_bulletproofing(prob_spec,sharedState);
}

/*______________________________________________________________________
 Purpose~   outputs material state during a checkpoint.  The material
            state can change throughout the simulation
 _____________________________________________________________________*/
void ICE_sm::outputProblemSpec(ProblemSpecP& root_ps)
{
  iceCout << d_myworld->myrank() << " Doing ICE_sm::outputProblemSpec " << "\t\t\t ICE" << endl;

  ProblemSpecP root = root_ps->getRootNode();

  ProblemSpecP mat_ps = 0;
  mat_ps = root->findBlockWithOutAttribute("MaterialProperties");

  if (mat_ps == 0){
    mat_ps = root->appendChild("MaterialProperties");
  }

  ProblemSpecP ice_ps = mat_ps->appendChild("ICE");
  oneICEMaterial* ice_mat = d_sharedState->getOneICEMaterial(d_matl);
  ice_mat->outputProblemSpec(ice_ps);
}

/* _____________________________________________________________________
    Purpose~     initialize variables
_____________________________________________________________________*/
void ICE_sm::scheduleInitialize(const LevelP& level,
                                 SchedulerP& sched)
{
  printSchedule(level,iceCout,"actuallyInitialize");

  Task* t = scinew Task("ICE_sm::actuallyInitialize",
                  this, &ICE_sm::actuallyInitialize);
                  
  t->computes( lb->vel_CCLabel );
  t->computes( lb->rho_CCLabel );
  t->computes( lb->sp_vol_CCLabel );
  t->computes( lb->temp_CCLabel );
  t->computes( lb->speedSound_CCLabel);
  
  t->computes( lb->thermalCondLabel );
  t->computes( lb->viscosityLabel );
  t->computes( lb->gammaLabel );
  t->computes( lb->specific_heatLabel );
  
  const MaterialSet* ice_matls = d_sharedState->allICE_smMaterials();

  sched->addTask(t, level->eachPatch(), ice_matls);
}

//______________________________________________________________________
//
void ICE_sm::scheduleRestartInitialize(const LevelP& level,
                                        SchedulerP& sched)
{
  // do nothing for now
}

/* _____________________________________________________________________
  Purpose~   Set variables that are normally set during the initialization
             phase, but get wiped clean when you restart
_____________________________________________________________________*/
void ICE_sm::restartInitialize()
{
  iceCout << d_myworld->myrank() << " Doing restartInitialize "<< "\t\t\t ICE" << endl;

  //__________________________________
  // ICE: Material specific flags
  oneICEMaterial* ice_matl = d_sharedState->getOneICEMaterial( d_matl );
  if(ice_matl->getViscosity() > 0.0){
    d_viscousFlow = true;
  }
}

/* _____________________________________________________________________
    Purpose~  Compute the timestep
_____________________________________________________________________*/
void
ICE_sm::scheduleComputeStableTimestep(const LevelP& level,
                                      SchedulerP& sched)
{
  printSchedule(level,iceCout,"ICE_sm::actuallyComputeStableTimestep");

  Task* t = scinew Task("ICE_sm::actuallyComputeStableTimestep",
                   this, &ICE_sm::actuallyComputeStableTimestep);

  Ghost::GhostType  gn = Ghost::None;
  const MaterialSet* ice_matls = d_sharedState->allICE_smMaterials();

  t->requires(Task::NewDW, lb->vel_CCLabel,        gn );
  t->requires(Task::NewDW, lb->speedSound_CCLabel, gn );
  t->requires(Task::NewDW, lb->thermalCondLabel,   gn );
  t->requires(Task::NewDW, lb->gammaLabel,         gn );
  t->requires(Task::NewDW, lb->specific_heatLabel, gn );
  t->requires(Task::NewDW, lb->sp_vol_CCLabel,     gn );
  t->requires(Task::NewDW, lb->viscosityLabel,     gn );

  t->computes(d_sharedState->get_delt_label(),level.get_rep());
  sched->addTask(t,level->eachPatch(), ice_matls);
}


/* _____________________________________________________________________
    Purpose~  Specify the scheduling of tasks in a timestep
_____________________________________________________________________*/
void
ICE_sm::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched)
{
  MALLOC_TRACE_TAG_SCOPE("ICE_sm::scheduleTimeAdvance()");

  iceCout << d_myworld->myrank() << " --------------------------------------------------------L-"<<level->getIndex()<< endl;
  printSchedule(level,iceCout,"scheduleTimeAdvance");

  const PatchSet* patches = level->eachPatch();
  const MaterialSet* ice_matls = d_sharedState->allICE_smMaterials();

  sched_ComputeThermoTransportProperties(sched, level,   ice_matls);

  sched_ComputePressure(                sched, patches,  ice_matls);

  sched_ComputeVel_FC(                   sched, patches, ice_matls);

  sched_ComputeDelPressAndUpdatePressCC(sched, patches,  ice_matls);


  sched_ComputePressFC(                 sched, patches,  ice_matls);

  sched_VelTau_CC(                      sched, patches,  ice_matls);

  sched_ViscousShearStress(             sched, patches,  ice_matls);

  sched_AccumulateMomentumSourceSinks(  sched, patches,  ice_matls);

  sched_AccumulateEnergySourceSinks(    sched, patches,  ice_matls);

  sched_ComputeLagrangianValues(        sched, patches,  ice_matls);


  sched_AdvectAndAdvanceInTime(         sched, patches,  ice_matls);

  sched_ConservedtoPrimitive_Vars(      sched, patches,  ice_matls);
                                        
  sched_TestConservation(               sched, patches, ice_matls);

  iceCout << "---------------------------------------------------------"<<endl;
}

/* _____________________________________________________________________
  Purpose~  Compute the thermodynamice and transport properties
_____________________________________________________________________*/
void
ICE_sm::sched_ComputeThermoTransportProperties(SchedulerP& sched,
                                               const LevelP& level,
                                               const MaterialSet* ice_matls)
{

  printSchedule(level,iceCout,"sched_ComputeThermoTransportProperties");

  Task* t = scinew Task("ICE_sm::computeThermoTransportProperties",
                  this, &ICE_sm::computeThermoTransportProperties);

  t->requires(Task::OldDW,lb->temp_CCLabel, ice_matls->getUnion(), Ghost::None, 0);

  t->computes(lb->viscosityLabel);
  t->computes(lb->thermalCondLabel);
  t->computes(lb->gammaLabel);
  t->computes(lb->specific_heatLabel);

  sched->addTask(t, level->eachPatch(), ice_matls);
}

/* _____________________________________________________________________
  Purpose~  Compute the pressure based on a equation of state
_____________________________________________________________________*/
void
ICE_sm::sched_ComputePressure(SchedulerP& sched,
                              const PatchSet* patches,
                              const MaterialSet* ice_matls)
{
  printSchedule(patches,iceCout,"sched_ComputePressure");

  Task* t = scinew Task("ICE_sm::computeEquilPressure_1_matl",
                 this, &ICE_sm::computeEquilPressure_1_matl);


  Ghost::GhostType  gn = Ghost::None;

  t->requires( Task::OldDW, lb->delTLabel, getLevel(patches) );
  t->requires( Task::OldDW, lb->rho_CCLabel,        gn );
  t->requires( Task::OldDW, lb->temp_CCLabel,       gn );
  t->requires( Task::NewDW, lb->gammaLabel,         gn );
  t->requires( Task::NewDW, lb->specific_heatLabel, gn );

  t->computes( lb->speedSound_CCLabel );
  t->computes( lb->vol_frac_CCLabel );
  t->computes( lb->sp_vol_CCLabel );
  t->computes( lb->compressibilityLabel );
  t->computes( lb->press_equil_CCLabel );

  sched->addTask(t, patches, ice_matls);
}

/* _____________________________________________________________________
  Purpose~  Compute the face-centered velocities
            also compute gradient of P for diagnostics
_____________________________________________________________________*/
void
ICE_sm::sched_ComputeVel_FC(SchedulerP& sched,
                            const PatchSet* patches,
                            const MaterialSet* all_matls)
{
  printSchedule(patches,iceCout,"sched_ComputeVel_FC");

  Task* t = scinew Task("ICE_sm::computeVel_FC", this,
                        &ICE_sm::computeVel_FC);

  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires( Task::OldDW, lb->delTLabel, getLevel(patches));
  t->requires( Task::NewDW, lb->press_equil_CCLabel,  gac,1);
  t->requires( Task::NewDW, lb->sp_vol_CCLabel,       gac,1);
  t->requires( Task::OldDW, lb->rho_CCLabel,          gac,1);
  t->requires( Task::OldDW, lb->vel_CCLabel,          gac,1);

  t->computes( lb->uvel_FCLabel );
  t->computes( lb->vvel_FCLabel );
  t->computes( lb->wvel_FCLabel );

  // used for diagnostics
  t->computes( lb->grad_P_XFCLabel );
  t->computes( lb->grad_P_YFCLabel );
  t->computes( lb->grad_P_ZFCLabel );
  
  sched->addTask(t, patches, all_matls);
}


/* _____________________________________________________________________
  Purpose~  Compute change in pressure and update the pressure
_____________________________________________________________________*/
void
ICE_sm::sched_ComputeDelPressAndUpdatePressCC(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  printSchedule(patches,iceCout, "sched_ComputeDelPressAndUpdatePressCC");

  Task *task = scinew Task("ICE_sm::computeDelPressAndUpdatePressCC", this,
                           &ICE_sm::computeDelPressAndUpdatePressCC);

  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn = Ghost::None;

  task->requires( Task::OldDW, lb->delTLabel,getLevel(patches));
  
  task->requires( Task::NewDW, lb->vol_frac_CCLabel,   gac,2);
  task->requires( Task::NewDW, lb->uvel_FCLabel,       gac,2);
  task->requires( Task::NewDW, lb->vvel_FCLabel,       gac,2);
  task->requires( Task::NewDW, lb->wvel_FCLabel,       gac,2);

  task->requires( Task::NewDW, lb->compressibilityLabel,gn);
  task->requires( Task::NewDW, lb->press_equil_CCLabel, gn);

  task->computes(lb->press_CCLabel);
  task->computes(lb->delP_DilatateLabel);
  task->computes(lb->term2Label);

  sched->setRestartable(true);
  sched->addTask(task, patches, matls);
}

/* _____________________________________________________________________
  Purpose~  Compute face-centered pressure
_____________________________________________________________________*/
void
ICE_sm::sched_ComputePressFC(SchedulerP& sched,
                             const PatchSet* patches,
                             const MaterialSet* matls)
{
  printSchedule(patches,iceCout, "sched_ComputePressFC");

  Task* task = scinew Task("ICE_sm::computePressFC",
                     this, &ICE_sm::computePressFC);

  Ghost::GhostType  gac = Ghost::AroundCells;
  task->requires(Task::NewDW,lb->press_CCLabel,    gac,1);
  task->requires(Task::OldDW,lb->rho_CCLabel,      gac,1);

  task->computes( lb->pressX_FCLabel );
  task->computes( lb->pressY_FCLabel );
  task->computes( lb->pressZ_FCLabel );

  sched->addTask(task, patches, matls);
}

/* _____________________________________________________________________
  Purpose~  Compute a CC Velocity which is used to compute the viscous
            fluxes
_____________________________________________________________________*/
void
ICE_sm::sched_VelTau_CC( SchedulerP& sched,
                         const PatchSet* patches,
                         const MaterialSet* ice_matls )
{
  if( !d_viscousFlow ){
    return;
  }
  printSchedule(patches,iceCout,"sched_VelTau_CC");

  Task* t = scinew Task("ICE_sm::VelTau_CC",
                  this, &ICE_sm::VelTau_CC);

  Ghost::GhostType  gn= Ghost::None;
  t->requires( Task::OldDW, lb->vel_CCLabel, gn );
  t->computes( lb->velTau_CCLabel );

  sched->addTask(t, patches, ice_matls);
}

/*_____________________________________________________________________
  Purpose~  Compute the viscous stress terms on each cell face
_____________________________________________________________________*/
void
ICE_sm::sched_ViscousShearStress(SchedulerP& sched,
                                 const PatchSet* patches,
                                 const MaterialSet* ice_matls)
{
  printSchedule(patches,iceCout,"sched_ViscousShearStress");

  Task* t = scinew Task("ICE_sm::viscousShearStress",
                  this, &ICE_sm::viscousShearStress);

  Ghost::GhostType  gac = Ghost::AroundCells;

  if(d_viscousFlow){
    t->requires( Task::NewDW, lb->viscosityLabel, gac, 2);  
    t->requires( Task::NewDW, lb->velTau_CCLabel, gac, 2);  

    t->computes( lb->tau_X_FCLabel );
    t->computes( lb->tau_Y_FCLabel );
    t->computes( lb->tau_Z_FCLabel );
  }

  t->computes( lb->viscous_src_CCLabel );
  sched->addTask(t, patches, ice_matls);
}


/*_____________________________________________________________________
 Purpose~  Accumulate the sources/sinks of momentum or the RHS
_____________________________________________________________________*/
void
ICE_sm::sched_AccumulateMomentumSourceSinks(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  printSchedule(patches,iceCout,"sched_AccumulateMomentumSourceSinks");

  Task* t = scinew Task("ICE_sm::accumulateMomentumSourceSinks",this,
                        &ICE_sm::accumulateMomentumSourceSinks);

  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn  = Ghost::None;

  t->requires(Task::OldDW, lb->delTLabel,getLevel(patches));
  t->requires( Task::NewDW, lb->pressX_FCLabel,      gac, 1);
  t->requires( Task::NewDW, lb->pressY_FCLabel,      gac, 1);
  t->requires( Task::NewDW, lb->pressZ_FCLabel,      gac, 1);
  t->requires( Task::NewDW, lb->viscous_src_CCLabel, gn );
  t->requires( Task::OldDW, lb->rho_CCLabel,         gn );

  t->computes(lb->mom_source_CCLabel);

  sched->addTask(t, patches, matls);
}

/* _____________________________________________________________________
 Purpose~  Accumulate the sources/sinks of internal energy or the RHS
_____________________________________________________________________*/
void
ICE_sm::sched_AccumulateEnergySourceSinks(SchedulerP& sched,
                                         const PatchSet* patches,
                                         const MaterialSet* matls)

{
  printSchedule(patches,iceCout,"sched_AccumulateEnergySourceSinks");

  Task* t = scinew Task("ICE_sm::accumulateEnergySourceSinks",
                  this, &ICE_sm::accumulateEnergySourceSinks);

  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn  = Ghost::None;

  t->requires(Task::OldDW, lb->delTLabel,getLevel(patches));
  t->requires(Task::NewDW, lb->press_CCLabel,        gn);
  t->requires(Task::NewDW, lb->delP_DilatateLabel,   gn);
  t->requires(Task::NewDW, lb->compressibilityLabel, gn);
  t->requires(Task::OldDW, lb->temp_CCLabel,         gac,1);
  t->requires(Task::NewDW, lb->thermalCondLabel,     gac,1);

  t->computes(lb->int_eng_source_CCLabel);
  t->computes(lb->heatCond_src_CCLabel);

  sched->addTask(t, patches, matls);
}

/* _____________________________________________________________________
 Purpose~  Add the sources/sinks of momenta and energy to the old values
_____________________________________________________________________*/
void
ICE_sm::sched_ComputeLagrangianValues(SchedulerP& sched,
                                     const PatchSet* patches,
                                     const MaterialSet* ice_matls)
{
  printSchedule(patches,iceCout,"sched_ComputeLagrangianValues");

  Task* t = scinew Task("ICE_sm::computeLagrangianValues", this,
                        &ICE_sm::computeLagrangianValues);

  Ghost::GhostType  gn  = Ghost::None;
  t->requires(Task::OldDW,  lb->delTLabel,getLevel(patches));
  t->requires(Task::NewDW,  lb->specific_heatLabel,    gn);
  t->requires(Task::OldDW,  lb->rho_CCLabel,           gn);
  t->requires(Task::OldDW,  lb->vel_CCLabel,           gn);
  t->requires(Task::OldDW,  lb->temp_CCLabel,          gn);
  t->requires(Task::NewDW,  lb->mom_source_CCLabel,    gn);
  t->requires(Task::NewDW,  lb->int_eng_source_CCLabel,gn);

  t->computes(lb->mom_L_CCLabel);
  t->computes(lb->int_eng_L_CCLabel);
  t->computes(lb->mass_L_CCLabel);

  sched->addTask(t, patches, ice_matls);
}

/* _____________________________________________________________________
  Purpose~  Advect the conserved quantities and advance in time
_____________________________________________________________________*/
void
ICE_sm::sched_AdvectAndAdvanceInTime(SchedulerP& sched,
                                    const PatchSet* patch_set,
                                    const MaterialSet* ice_matls)
{
  printSchedule(patch_set,iceCout,"sched_AdvectAndAdvanceInTime");

  Task* task = scinew Task("ICE_sm::advectAndAdvanceInTime",this,
                           &ICE_sm::advectAndAdvanceInTime);

  task->requires(Task::OldDW, lb->delTLabel,getLevel(patch_set));
  Ghost::GhostType  gac  = Ghost::AroundCells;
  
  task->requires(Task::NewDW, lb->uvel_FCLabel,      gac,2);
  task->requires(Task::NewDW, lb->vvel_FCLabel,      gac,2);
  task->requires(Task::NewDW, lb->wvel_FCLabel,      gac,2);

  task->requires(Task::NewDW, lb->mom_L_CCLabel,     gac,2);
  task->requires(Task::NewDW, lb->mass_L_CCLabel,    gac,2);
  task->requires(Task::NewDW, lb->int_eng_L_CCLabel, gac,2);

  task->computes(lb->mass_advLabel);
  task->computes(lb->mom_advLabel);
  task->computes(lb->eng_advLabel);

  sched->setRestartable(true);
  sched->addTask(task, patch_set, ice_matls);
}
/* _____________________________________________________________________
 Purpose~  compute primitive variables from conserved quantites
_____________________________________________________________________*/
void
ICE_sm::sched_ConservedtoPrimitive_Vars(SchedulerP& sched,
                                        const PatchSet* patch_set,
                                        const MaterialSet* ice_matls)
{
  printSchedule(patch_set, iceCout, "ICE_sm::conservedtoPrimitive_Vars");

  Task* task = scinew Task("ICE_sm::conservedtoPrimitive_Vars", this, 
                           &ICE_sm::conservedtoPrimitive_Vars);

  task->requires(Task::OldDW, lb->delTLabel,getLevel(patch_set));
  Ghost::GhostType  gn   = Ghost::None;

  task->requires(Task::NewDW, lb->mass_advLabel,      gn );
  task->requires(Task::NewDW, lb->mom_advLabel,       gn );
  task->requires(Task::NewDW, lb->eng_advLabel,       gn );

  task->requires(Task::NewDW, lb->specific_heatLabel, gn );
  task->requires(Task::NewDW, lb->speedSound_CCLabel, gn );

  task->computes(lb->temp_CCLabel);
  task->computes(lb->vel_CCLabel);
  task->computes(lb->rho_CCLabel);
  task->computes(lb->machLabel);

  sched->addTask(task, patch_set, ice_matls);
}
/* _____________________________________________________________________
 Function~  ICE_sm::scheduleTestConservation--
_____________________________________________________________________*/
void ICE_sm::sched_TestConservation(SchedulerP& sched,
                                   const PatchSet* patches,
                                   const MaterialSet* all_matls)
{
  int levelIndex = getLevel(patches)->getIndex();
  if(d_conservationTest->onOff && levelIndex == 0) {

     printSchedule(patches,iceCout,"sched_ConservedtoPrimitive_Vars");

    Task* t= scinew Task("ICE_sm::TestConservation",
                   this, &ICE_sm::TestConservation);

    Ghost::GhostType  gn  = Ghost::None;
    t->requires(Task::OldDW, lb->delTLabel,getLevel(patches));
    t->requires(Task::NewDW,lb->rho_CCLabel,         gn);
    t->requires(Task::NewDW,lb->vel_CCLabel,         gn);
    t->requires(Task::NewDW,lb->temp_CCLabel,        gn);
    t->requires(Task::NewDW,lb->specific_heatLabel,  gn);
    t->requires(Task::NewDW,lb->uvel_FCLabel,        gn);
    t->requires(Task::NewDW,lb->vvel_FCLabel,        gn);
    t->requires(Task::NewDW,lb->wvel_FCLabel,        gn);
    t->requires(Task::NewDW,lb->mom_L_CCLabel,       gn);
    t->requires(Task::NewDW,lb->int_eng_L_CCLabel,   gn);

    if(d_conservationTest->mass){
      t->computes(lb->TotalMassLabel);
    }
    if(d_conservationTest->energy){
      t->computes(lb->KineticEnergyLabel);
      t->computes(lb->TotalIntEngLabel);
    }
    if(d_conservationTest->momentum){
      t->computes(lb->TotalMomentumLabel);
    }
    sched->addTask(t, patches, all_matls);
  }
}

/* _____________________________________________________________________
 Function~  ICE_sm::actuallyComputeStableTimestep--
_____________________________________________________________________*/
void ICE_sm::actuallyComputeStableTimestep(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* /*matls*/,
                                    DataWarehouse* /*old_dw*/,
                                    DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch, iceCout, "Doing ICE_sm::actuallyComputeStableTimestep" );

    Vector dx = patch->dCell();
    double delX = dx.x();
    double delY = dx.y();
    double delZ = dx.z();
    double delt_CFL;
    double delt_diff;
    double delt;
    double inv_sum_invDelx_sqr = 1.0/( 1.0/(delX * delX)
                                     + 1.0/(delY * delY)
                                     + 1.0/(delZ * delZ) );

    constCCVariable<double> speedSound, sp_vol_CC, thermalCond, viscosity;
    constCCVariable<double> cv, gamma;
    constCCVariable<Vector> vel_CC;
    Ghost::GhostType  gn  = Ghost::None;

    IntVector badCell(0,0,0);
    delt_CFL  = 1000.0;
    delt_diff = 1000;
    delt      = 1000;

    oneICEMaterial* ice_matl = d_sharedState->getOneICEMaterial( d_matl );
    int indx = ice_matl->getDWIndex();

    new_dw->get(speedSound, lb->speedSound_CCLabel, indx, patch, gn, 0 );
    new_dw->get(vel_CC,     lb->vel_CCLabel,        indx, patch, gn, 0 );
    new_dw->get(sp_vol_CC,  lb->sp_vol_CCLabel,     indx, patch, gn, 0 );
    new_dw->get(viscosity,  lb->viscosityLabel,     indx, patch, gn, 0 );
    new_dw->get(thermalCond,lb->thermalCondLabel,   indx, patch, gn, 0 );
    new_dw->get(gamma,      lb->gammaLabel,         indx, patch, gn, 0 );
    new_dw->get(cv,         lb->specific_heatLabel, indx, patch, gn, 0 );

    for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      double speed_Sound = speedSound[c];

      double A = d_CFL*delX/(speed_Sound + fabs(vel_CC[c].x()) + d_SMALL_NUM);
      double B = d_CFL*delY/(speed_Sound + fabs(vel_CC[c].y()) + d_SMALL_NUM);
      double C = d_CFL*delZ/(speed_Sound + fabs(vel_CC[c].z()) + d_SMALL_NUM);

      delt_CFL = std::min(A, delt_CFL);
      delt_CFL = std::min(B, delt_CFL);
      delt_CFL = std::min(C, delt_CFL);
      
      if (A < 1e-20 || B < 1e-20 || C < 1e-20) {
        if (badCell == IntVector(0,0,0)) {
          badCell = c;
        }
        cout << d_myworld->myrank() << " Bad cell " << c << " (" << patch->getID() << "-" << level->getIndex() << "): " << vel_CC[c]<< endl;
      }

      // cout << " Aggressive delT Based on currant number "<< delt_CFL << endl;
      //__________________________________
      // stability constraint due to diffusion
      double thermalCond_test = ice_matl->getThermalConductivity();
      double viscosity_test   = ice_matl->getViscosity();
      if (thermalCond_test !=0 || viscosity_test !=0) {

        for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
          IntVector c = *iter;
          double cp = cv[c] * gamma[c];
          double inv_thermalDiffusivity = cp/(sp_vol_CC[c] * thermalCond[c]);
          double kinematicViscosity     = viscosity[c] * sp_vol_CC[c];
          double inv_diffusionCoeff     = min(inv_thermalDiffusivity, 1.0/kinematicViscosity);
          
          double A  = d_CFL * 0.5 * inv_sum_invDelx_sqr * inv_diffusionCoeff;
          delt_diff = std::min(A, delt_diff);
          if (delt_diff < 1e-20 && badCell == IntVector(0,0,0)) {
            badCell = c;
          }
        }
      }
      // cout << "delT based on diffusion  "<< delt_diff<<endl;
      delt = std::min(delt_CFL, delt_diff);
    } // aggressive Timestep
    
    //__________________________________
    //  Bullet proofing
    if(delt < 1e-20) {
      ostringstream warn;
      const Level* level = getLevel(patches);
      warn << "ERROR ICE:(L-"<< level->getIndex()
           << "):ComputeStableTimestep: delT < 1e-20 on cell " << badCell;
      throw InvalidValue(warn.str(), __FILE__, __LINE__);
    }
    new_dw->put(delt_vartype(delt), lb->delTLabel, level);
  }  // patch loop
}

/* _____________________________________________________________________
 Function~  ICE_sm::actuallyInitialize--
 Purpose~  Initialize CC variables and the pressure
 Note that rho_micro, sp_vol, temp and velocity must be defined
 everywhere in the domain
_____________________________________________________________________*/
void ICE_sm::actuallyInitialize(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* /*matls*/,
                          DataWarehouse*,
                          DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  int L_indx = level->getIndex();

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, iceCout, "Doing ICE_sm::actuallyInitialize" );

    CCVariable<double>  rho_CC;
    CCVariable<double>  sp_vol_CC;
    CCVariable<double>  Temp_CC;
    CCVariable<Vector>  vel_CC;
    CCVariable<double>  speedSound;
    CCVariable<double>  cv;
    CCVariable<double>  gamma;

    //__________________________________
    //  Thermo and transport properties
    CCVariable<double> viscosity, thermalCond;
    oneICEMaterial* ice_matl = d_sharedState->getOneICEMaterial(d_matl);
    int indx = ice_matl->getDWIndex();

    new_dw->allocateAndPut(viscosity,    lb->viscosityLabel,    indx, patch);
    new_dw->allocateAndPut(thermalCond,  lb->thermalCondLabel,  indx, patch);
    new_dw->allocateAndPut(cv,           lb->specific_heatLabel,indx, patch);
    new_dw->allocateAndPut(gamma,        lb->gammaLabel,        indx, patch);

    viscosity.initialize  ( ice_matl->getViscosity());
    thermalCond.initialize( ice_matl->getThermalConductivity());

    if(ice_matl->getViscosity() > 0.0){
      d_viscousFlow = true;
    }
    
    SpecificHeat *cvModel = ice_matl->getSpecificHeatModel();
    if(cvModel != 0) {
      for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        gamma[c] = cvModel->getGamma(Temp_CC[c]);
        cv[c]    = cvModel->getSpecificHeat(Temp_CC[c]);
      }
    } else {
      gamma.initialize(ice_matl->getGamma());              
      cv.initialize(   ice_matl->getSpecificHeat());       
    }

    //__________________________________
    new_dw->allocateAndPut(rho_CC,     lb->rho_CCLabel,         indx, patch);
    new_dw->allocateAndPut(sp_vol_CC,  lb->sp_vol_CCLabel,      indx, patch);
    new_dw->allocateAndPut(Temp_CC,    lb->temp_CCLabel,        indx, patch);
    new_dw->allocateAndPut(vel_CC,     lb->vel_CCLabel,         indx, patch);
    new_dw->allocateAndPut(speedSound,  lb->speedSound_CCLabel, indx, patch);

    ice_matl->initializeCells(rho_CC, Temp_CC, vel_CC,  patch, new_dw);

    setBC( rho_CC,   "Density",     patch, indx );  
    setBC( Temp_CC,  "Temperature", patch, indx );  
    setBC( vel_CC,   "Velocity",    patch, indx );  

    //__________________________________
    //  compute the speed of sound
    // set sp_vol_CC
    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;

      sp_vol_CC[c] = 1./rho_CC[c];
      
      double dp_drho, dp_de, press;
      ice_matl->getEOS()->computePressEOS(rho_CC[c],gamma[c],
                                        cv[c], Temp_CC[c], press,
                                        dp_drho, dp_de);

      double c_2 = dp_drho + dp_de * press/(rho_CC[c] * rho_CC[c]);
      speedSound[c] = sqrt(c_2);
    }
    //____ B U L L E T   P R O O F I N G----
    IntVector neg_cell;
    ostringstream warn, base;
    base <<"ERROR ICE:(L-"<<L_indx<<"):actuallyInitialize, mat "<< d_matl <<" cell ";

    if( !areAllValuesPositive(rho_CC, neg_cell) ) {
      warn << base.str()<< neg_cell << " rho_CC is negative\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
    }
    if( !areAllValuesPositive(Temp_CC, neg_cell) ) {
      warn << base.str()<< neg_cell << " Temp_CC is negative\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
    }
  }  // patch loop
}

/* _____________________________________________________________________
      Purpose~  Compute the thermodynamic and transport properties
 _____________________________________________________________________  */
void ICE_sm::computeThermoTransportProperties(const ProcessorGroup*,
                                              const PatchSubset* patches,
                                              const MaterialSubset*,
                                              DataWarehouse* old_dw,
                                              DataWarehouse* new_dw)
{

  const Level* level = getLevel(patches);
  int levelIndex = level->getIndex();


  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    iceCout << " ---------------------------------------------- L-"<< levelIndex<< endl;
    printTask(patches, patch, iceCout, "Doing ICE_sm::computeThermoTransportProperties" );

    oneICEMaterial* ice_matl = d_sharedState->getOneICEMaterial(d_matl);
    int indx = ice_matl->getDWIndex();

    constCCVariable<double> temp_CC;
    CCVariable<double> viscosity;
    CCVariable<double> thermalCond;
    CCVariable<double> gamma;
    CCVariable<double> cv;

    old_dw->get(temp_CC, lb->temp_CCLabel, indx, patch, Ghost::None,0);
    new_dw->allocateAndPut(thermalCond, lb->thermalCondLabel,  indx, patch);
    new_dw->allocateAndPut(viscosity,   lb->viscosityLabel,    indx, patch);
    new_dw->allocateAndPut(cv,          lb->specific_heatLabel,indx, patch);
    new_dw->allocateAndPut(gamma,       lb->gammaLabel,        indx, patch);

    // set to a constant value
    viscosity.initialize  ( ice_matl->getViscosity());
    thermalCond.initialize( ice_matl->getThermalConductivity());
    gamma.initialize  (     ice_matl->getGamma());
    cv.initialize(          ice_matl->getSpecificHeat());
    SpecificHeat *cvModel = ice_matl->getSpecificHeatModel();

    if(cvModel != 0) {
      // loop through cells and compute pointwise
      for(CellIterator iter = patch->getCellIterator();!iter.done();iter++) {
        IntVector c = *iter;
        cv[c] = cvModel->getSpecificHeat(temp_CC[c]);
        gamma[c] = cvModel->getGamma(temp_CC[c]);
      }
    }
  }
}


/* _____________________________________________________________________
 Purpose~   Compute the pressure using a equation of  state evaluation
_____________________________________________________________________*/
void ICE_sm::computeEquilPressure_1_matl(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset* matls,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, iceCout, "Doing ICE_sm::computeEquilPressure_1_matl" );

    CCVariable<double> vol_frac;
    CCVariable<double> sp_vol_CC;
    CCVariable<double> speedSound;
    CCVariable<double> kappa;
    CCVariable<double> press_eq;
    constCCVariable<double> Temp,rho_CC, cv, gamma;
    StaticArray<CCVariable<double> > rho_micro(1);

    Ghost::GhostType  gn = Ghost::None;
    oneICEMaterial* ice_matl = d_sharedState->getOneICEMaterial( d_matl );
    int indx = ice_matl->getDWIndex();

    //__________________________________
    old_dw->get(Temp,      lb->temp_CCLabel,      indx,patch, gn,0);
    old_dw->get(rho_CC,    lb->rho_CCLabel,       indx,patch, gn,0);
    new_dw->get(cv,        lb->specific_heatLabel,indx,patch, gn,0);
    new_dw->get(gamma,     lb->gammaLabel,        indx,patch, gn,0);

    new_dw->allocateTemporary(rho_micro[0],  patch);

    new_dw->allocateAndPut(press_eq,     lb->press_equil_CCLabel, indx,  patch);
    new_dw->allocateAndPut(kappa,        lb->compressibilityLabel,indx, patch);
    new_dw->allocateAndPut(vol_frac,     lb->vol_frac_CCLabel,    indx, patch);
    new_dw->allocateAndPut(sp_vol_CC,    lb->sp_vol_CCLabel,      indx, patch);
    new_dw->allocateAndPut(speedSound,   lb->speedSound_CCLabel,  indx, patch);

    //______________________________________________________________________
    //  Main loop
    for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++) {
      IntVector c = *iter;
      vol_frac[c]      = 1.0;
      rho_micro[0][c]  = rho_CC[c];
      sp_vol_CC[c]     = 1.0/rho_CC[c];
      double dp_drho, dp_de, c_2;
      //__________________________________
      // evaluate EOS
      ice_matl->getEOS()->computePressEOS(rho_micro[0][c],gamma[c],
                                          cv[c], Temp[c], press_eq[c],
                                          dp_drho, dp_de);

      c_2 = dp_drho + dp_de * press_eq[c]/(rho_micro[0][c] * rho_micro[0][c]);
      speedSound[c] = sqrt(c_2);

      kappa[c]    = sp_vol_CC[c]/(speedSound[c]*speedSound[c]);
    }
    //__________________________________
    // - apply Boundary conditions
    setBC(press_eq, "Pressure", patch, indx );

  }  // patch loop
}

/* _____________________________________________________________________
 Purpose~   compute the face centered velocities.
_____________________________________________________________________*/
template<class T>
void ICE_sm::computeVelFace(int dir,
                            CellIterator it,
                            IntVector adj_offset,
                            double dx,
                            double delT, double gravity,
                            constCCVariable<double>& rho_CC,
                            constCCVariable<double>& sp_vol_CC,
                            constCCVariable<Vector>& vel_CC,
                            constCCVariable<double>& press_CC,
                            T& vel_FC,
                            T& grad_P_FC)
{
  double inv_dx = 1.0/dx;

  for(;!it.done(); it++){
    IntVector R = *it;
    IntVector L = R + adj_offset;

    double rho_FC = rho_CC[L] + rho_CC[R];
#if SCI_ASSERTION_LEVEL >=2
    if (rho_FC <= 0.0) {
      cout << d_myworld->myrank() << " rho_fc <= 0: " << rho_FC << " with L= " << L << " ("
           << rho_CC[L] << ") R= " << R << " (" << rho_CC[R]<< ")\n";
    }
#endif
    ASSERT(rho_FC > 0.0);

    //__________________________________
    // interpolation to the face
    double term1 = (rho_CC[L] * vel_CC[L][dir] +
                    rho_CC[R] * vel_CC[R][dir])/(rho_FC);
    //__________________________________
    // pressure term
    double sp_vol_brack = 2.*(sp_vol_CC[L] * sp_vol_CC[R])/
                             (sp_vol_CC[L] + sp_vol_CC[R]);

    grad_P_FC[R] = (press_CC[R] - press_CC[L]) * inv_dx;
    double term2 = delT * sp_vol_brack * grad_P_FC[R];

    //__________________________________
    // gravity term
    double term3 =  delT * gravity;

    vel_FC[R] = term1 - term2 + term3;
  }
}

//______________________________________________________________________
//
void ICE_sm::computeVel_FC(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* /*matls*/,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);

  for(int p = 0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch, iceCout, "Doing ICE_sm::computeVel_FCVel" );

    Vector dx      = patch->dCell();
    Vector gravity = getGravity();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(),level);

    // Compute the face centered velocities
    constCCVariable<double> press_CC;
    constCCVariable<double> rho_CC;
    constCCVariable<double> sp_vol_CC;
    constCCVariable<Vector> vel_CC;

    Ghost::GhostType  gac = Ghost::AroundCells;
    oneICEMaterial* ice_matl = d_sharedState->getOneICEMaterial(d_matl);
    int indx = ice_matl->getDWIndex();

    new_dw->get( press_CC,   lb->press_equil_CCLabel, indx, patch, gac, 1 );
    new_dw->get( sp_vol_CC,  lb->sp_vol_CCLabel,      indx, patch, gac, 1 );
    old_dw->get( rho_CC,     lb->rho_CCLabel,         indx, patch, gac, 1 );
    old_dw->get( vel_CC,     lb->vel_CCLabel,         indx, patch, gac, 1 );

    SFCXVariable<double> uvel_FC, grad_P_XFC;
    SFCYVariable<double> vvel_FC, grad_P_YFC;
    SFCZVariable<double> wvel_FC, grad_P_ZFC;

    new_dw->allocateAndPut( uvel_FC, lb->uvel_FCLabel, indx, patch );
    new_dw->allocateAndPut( vvel_FC, lb->vvel_FCLabel, indx, patch );
    new_dw->allocateAndPut( wvel_FC, lb->wvel_FCLabel, indx, patch );

    // debugging variables
    new_dw->allocateAndPut( grad_P_XFC, lb->grad_P_XFCLabel, indx, patch );
    new_dw->allocateAndPut( grad_P_YFC, lb->grad_P_YFCLabel, indx, patch );
    new_dw->allocateAndPut( grad_P_ZFC, lb->grad_P_ZFCLabel, indx, patch );

    uvel_FC.initialize( 0.0);
    vvel_FC.initialize( 0.0);
    wvel_FC.initialize( 0.0);

    grad_P_XFC.initialize(0.0);
    grad_P_YFC.initialize(0.0);
    grad_P_ZFC.initialize(0.0);

    vector<IntVector> adj_offset(3);
    adj_offset[0] = IntVector(-1, 0, 0);    // X faces
    adj_offset[1] = IntVector(0, -1, 0);    // Y faces
    adj_offset[2] = IntVector(0,  0, -1);   // Z faces

    CellIterator XFC_iterator = patch->getSFCXIterator();
    CellIterator YFC_iterator = patch->getSFCYIterator();
    CellIterator ZFC_iterator = patch->getSFCZIterator();

    //__________________________________
    //  Compute vel_FC for each face
    computeVelFace<SFCXVariable<double> >(0, XFC_iterator,
                                     adj_offset[0],dx[0],delT,gravity[0],
                                     rho_CC,sp_vol_CC,vel_CC,press_CC,
                                     uvel_FC, grad_P_XFC );

    computeVelFace<SFCYVariable<double> >(1, YFC_iterator,
                                     adj_offset[1],dx[1],delT,gravity[1],
                                     rho_CC,sp_vol_CC,vel_CC,press_CC,
                                     vvel_FC, grad_P_YFC );

    computeVelFace<SFCZVariable<double> >(2, ZFC_iterator,
                                     adj_offset[2],dx[2],delT,gravity[2],
                                     rho_CC,sp_vol_CC,vel_CC,press_CC,
                                     wvel_FC, grad_P_ZFC );

    //________________________________
    //  Boundary Conditons
    setBC_FC<SFCXVariable<double> >(uvel_FC, "Velocity", patch, indx);
    setBC_FC<SFCYVariable<double> >(vvel_FC, "Velocity", patch, indx);
    setBC_FC<SFCZVariable<double> >(wvel_FC, "Velocity", patch, indx);
  }  // patch loop
}

/*_____________________________________________________________________
 Function~  ICE_sm::computeDelPressAndUpdatePressCC--
 Purpose~
   This function calculates the change in pressure explicitly.
 _____________________________________________________________________  */
void ICE_sm::computeDelPressAndUpdatePressCC(const ProcessorGroup*,
                                             const PatchSubset* patches,
                                             const MaterialSubset* /*matls*/,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, iceCout, "Doing ICE_sm::computeDelPressAndUpdatePressCC" );

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(),level);

    bool newGrid = d_sharedState->isRegridTimestep();
    Advector* advector = d_advector->clone(new_dw,patch,newGrid );

    //intermediate values
    CCVariable<double> q_advected;
    CCVariable<double> delP_Dilatate;
    CCVariable<double> press_CC;
    CCVariable<double> term1, term2;

    oneICEMaterial* ice_matl = d_sharedState->getOneICEMaterial(d_matl);
    int indx = ice_matl->getDWIndex();

    new_dw->allocateAndPut( press_CC,    lb->press_CCLabel,      indx, patch);
    new_dw->allocateAndPut(delP_Dilatate,lb->delP_DilatateLabel, indx, patch);
    new_dw->allocateAndPut(term2,        lb->term2Label,         indx, patch);

    new_dw->allocateTemporary(q_advected, patch);
    new_dw->allocateTemporary(term1,      patch);

    term1.initialize(0.);
    term2.initialize(0.);
    delP_Dilatate.initialize(0.0);

    //__________________________________
    //  Pull data from the DW
    constCCVariable<double> vol_frac;
    constCCVariable<double> rho_CC;
    constCCVariable<double> Kappa;
    constCCVariable<double> press_equil;

    constSFCXVariable<double> uvel_FC;
    constSFCYVariable<double> vvel_FC;
    constSFCZVariable<double> wvel_FC;

    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;

    new_dw->get(uvel_FC,     lb->uvel_FCLabel,        indx, patch, gac, 2);
    new_dw->get(vvel_FC,     lb->vvel_FCLabel,        indx, patch, gac, 2);
    new_dw->get(wvel_FC,     lb->wvel_FCLabel,        indx, patch, gac, 2);
    new_dw->get(vol_frac,    lb->vol_frac_CCLabel,    indx, patch, gac, 2);

    new_dw->get(Kappa,       lb->compressibilityLabel,indx, patch, gn, 0);
    new_dw->get(press_equil, lb->press_equil_CCLabel, indx, patch, gn, 0);

    //__________________________________
    // Advection preprocessing
    // - divide vol_frac_cc/vol
    bool bulletProof_test=true;
    advectVarBasket* varBasket = scinew advectVarBasket();

    advector->inFluxOutFluxVolume(uvel_FC, vvel_FC, wvel_FC, delT, patch, d_matl,
                                  bulletProof_test, new_dw, varBasket);
    //__________________________________
    //   advect vol_frac
    varBasket->doRefluxing = false;  // don't need to reflux here
    advector->advectQ(vol_frac, patch, q_advected, varBasket, new_dw);

    delete varBasket;

    for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
      IntVector c = *iter;
      term2[c] -= q_advected[c];
    }

    delete advector;

    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      delP_Dilatate[c] = -term2[c]/Kappa[c];
      press_CC[c]      =  press_equil[c]  + delP_Dilatate[c];
      press_CC[c]      = max(1.0e-12, press_CC[c]);  // CLAMP
    }

    //__________________________________
    //  set boundary conditions
    setBC(press_CC, "Pressure", patch, indx );
  }  // patch loop
}

/* _____________________________________________________________________
 Purpose~
    This function calculates the face centered pressure on each of the
    cell faces for every cell in the computational domain and a single
    layer of ghost cells.
  _____________________________________________________________________  */
template <class T>
void ICE_sm::computePressFace(CellIterator iter,
                              IntVector adj_offset,
                              constCCVariable<double>& rho_CC,
                              constCCVariable<double>& press_CC,
                              T& press_FC)
{
  for(;!iter.done(); iter++){
    IntVector R = *iter;
    IntVector L = R + adj_offset;

    press_FC[R] = (press_CC[R] * rho_CC[L] + press_CC[L] * rho_CC[R])/
                  (rho_CC[R] + rho_CC[L]);
  }
}

/* _____________________________________________________________________
 Purpose~
    This task calculates the face centered pressure on each of the
    cell faces for every cell in the computational domain and a single
    layer of ghost cells.
  _____________________________________________________________________  */
void ICE_sm::computePressFC(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* /*matls*/,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, iceCout, "Doing ICE_sm::computePressFC" );

    Ghost::GhostType  gac = Ghost::AroundCells;


    constCCVariable<double> press_CC;
    constCCVariable<double> rho_CC;
    oneICEMaterial* ice_matl = d_sharedState->getOneICEMaterial(d_matl);
    int indx = ice_matl->getDWIndex();

    new_dw->get(press_CC,  lb->press_CCLabel,   indx, patch, gac, 1);
    old_dw->get(rho_CC,    lb->rho_CCLabel,     indx, patch, gac, 1);

    SFCXVariable<double> pressX_FC;
    SFCYVariable<double> pressY_FC;
    SFCZVariable<double> pressZ_FC;
    new_dw->allocateAndPut(pressX_FC, lb->pressX_FCLabel, indx, patch);
    new_dw->allocateAndPut(pressY_FC, lb->pressY_FCLabel, indx, patch);
    new_dw->allocateAndPut(pressZ_FC, lb->pressZ_FCLabel, indx, patch);

    vector<IntVector> adj_offset(3);
    adj_offset[0] = IntVector(-1, 0, 0);    // X faces
    adj_offset[1] = IntVector(0, -1, 0);    // Y faces
    adj_offset[2] = IntVector(0,  0, -1);   // Z faces

    //__________________________________
    //  For each face compute the pressure
    computePressFace<SFCXVariable<double> >(patch->getSFCXIterator(),
                                       adj_offset[0], rho_CC, press_CC,
                                       pressX_FC);

    computePressFace<SFCYVariable<double> >(patch->getSFCYIterator(),
                                       adj_offset[1], rho_CC, press_CC,
                                       pressY_FC);

    computePressFace<SFCZVariable<double> >(patch->getSFCZIterator(),
                                       adj_offset[2], rho_CC, press_CC,
                                       pressZ_FC);
  }  // patch loop
}



//______________________________________________________________________
//    See comments in ICE_sm::VelTau_CC()
//______________________________________________________________________
void ICE_sm::computeVelTau_CCFace( const Patch* patch,
                                const Patch::FaceType face,
                                constCCVariable<Vector>& vel_CC,
                                CCVariable<Vector>& velTau_CC)
{
  CellIterator iterLimits=patch->getFaceIterator(face, Patch::ExtraMinusEdgeCells);
  IntVector oneCell = patch->faceDirection(face);

  for(CellIterator iter = iterLimits;!iter.done();iter++){
    IntVector c = *iter;        // extra cell index
    IntVector in = c - oneCell; // interior cell index

    velTau_CC[c] = 2. * vel_CC[c] - vel_CC[in];
  }
}

//______________________________________________________________________
//  Modify the vel_CC in the extra cells so that it behaves
//     vel_FC[FC] = (vel_CC(c) + vel_CC(ec) )/2        (1)
//
//  Note that at the edge of the domain we assume that vel_FC = vel_CC[ec]
//  so (1) becomes:
//     vel_CC[ec]    = (vel_CC(c) + velTau_CC(ec) )/2
//            or
//     velTau_CC[ec] = (2 * vel_CC(ec) - velTau_CC(c) );
//
//  You need this so the viscous shear stress terms tau_xy = tau_yx.
//
//               |           |
//    ___________|___________|_______________
//               |           |
//         o     |     o     |      o         Vel_CC
//               |     c     |
//               |           |
//               |           |
//    ===========|====FC=====|==============      Edge of computational Domain
//               |           |
//               |           |
//         *     |     o     |      o          Vel_CC in extraCell
//               |    ec     |
//               |           |
//    ___________|___________|_______________
//
//   A fundamental assumption is that the boundary conditions
//  have been vel_CC[ec] in the old_dw.
//______________________________________________________________________
void ICE_sm::VelTau_CC(const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset* /*matls*/,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch, iceCout, "Doing VelTau_CC" );


    Ghost::GhostType  gn  = Ghost::None;

    oneICEMaterial* ice_matl = d_sharedState->getOneICEMaterial(d_matl);
    int indx = ice_matl->getDWIndex();

    constCCVariable<Vector> vel_CC;
    CCVariable<Vector>   velTau_CC;
    old_dw->get(             vel_CC,    lb->vel_CCLabel,    indx, patch, gn, 0);
    new_dw->allocateAndPut( velTau_CC,  lb->velTau_CCLabel, indx, patch);

    velTau_CC.copyData( vel_CC );           // copy interior values over

    if( patch->hasBoundaryFaces() ){

      // Iterate over the faces encompassing the domain
      vector<Patch::FaceType> bf;
      patch->getBoundaryFaces(bf);

      for( vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ){
        Patch::FaceType face = *iter;

        //__________________________________
        //           X faces
        if (face == Patch::xminus || face == Patch::xplus) {
          computeVelTau_CCFace( patch, face, vel_CC, velTau_CC );
          continue;
        }

        //__________________________________
        //           Y faces
        if (face == Patch::yminus || face == Patch::yplus) {
          computeVelTau_CCFace( patch, face, vel_CC, velTau_CC );
          continue;
        }

        //__________________________________
        //           Z faces
        if (face == Patch::zminus || face == Patch::zplus) {
          computeVelTau_CCFace( patch, face, vel_CC, velTau_CC );
          continue;
        }

      }  // face loop
    }  // has boundary face
  }  // patch loop
}



/* _____________________________________________________________________
 Purpose~   This task computes the viscous shear stress terms
 _____________________________________________________________________  */
void ICE_sm::viscousShearStress(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch, iceCout, "Doing ICE_sm::viscousShearStress" );

    IntVector right, left, top, bottom, front, back;

    Vector dx    = patch->dCell();
    double areaX = dx.y() * dx.z();
    double areaY = dx.x() * dx.z();
    double areaZ = dx.x() * dx.y();

    //__________________________________
    oneICEMaterial* ice_matl = d_sharedState->getOneICEMaterial(d_matl);
    int indx = ice_matl->getDWIndex();

    CCVariable<Vector>   viscous_src;
    new_dw->allocateAndPut( viscous_src,  lb->viscous_src_CCLabel,  indx, patch);
    viscous_src.initialize( Vector(0.,0.,0.) );

    //__________________________________
    // Compute Viscous diffusion
    if( d_viscousFlow ){
      Ghost::GhostType  gac = Ghost::AroundCells;

      SFCXVariable<Vector> tau_X_FC, Ttau_X_FC;
      SFCYVariable<Vector> tau_Y_FC, Ttau_Y_FC;
      SFCZVariable<Vector> tau_Z_FC, Ttau_Z_FC;

      new_dw->allocateAndPut(tau_X_FC, lb->tau_X_FCLabel, indx,patch);
      new_dw->allocateAndPut(tau_Y_FC, lb->tau_Y_FCLabel, indx,patch);
      new_dw->allocateAndPut(tau_Z_FC, lb->tau_Z_FCLabel, indx,patch);

      tau_X_FC.initialize( Vector(0.0) );  // DEFAULT VALUE
      tau_Y_FC.initialize( Vector(0.0) );
      tau_Z_FC.initialize( Vector(0.0) );

      //__________________________________
      //  compute the shear stress terms
      double viscosity_test = ice_matl->getViscosity();

      if(viscosity_test != 0.0) {
        constCCVariable<double>   viscosity;
        constCCVariable<Vector>   velTau_CC;

        new_dw->get(viscosity, lb->viscosityLabel, indx, patch, gac,2);
        new_dw->get(velTau_CC, lb->velTau_CCLabel, indx, patch, gac,2);

        // Use temporary arrays to eliminate the communication of shear stress components
        // across the network.  Normally you would compute them in a separate task and then
        // require them with ghostCells to compute the divergence.
        SFCXVariable<Vector> Ttau_X_FC;
        SFCYVariable<Vector> Ttau_Y_FC;
        SFCZVariable<Vector> Ttau_Z_FC;

        Ghost::GhostType  gac = Ghost::AroundCells;
        new_dw->allocateTemporary(Ttau_X_FC, patch, gac, 1);
        new_dw->allocateTemporary(Ttau_Y_FC, patch, gac, 1);
        new_dw->allocateTemporary(Ttau_Z_FC, patch, gac, 1);

        Vector evilNum(-9e30);
        Ttau_X_FC.initialize( evilNum );
        Ttau_Y_FC.initialize( evilNum );
        Ttau_Z_FC.initialize( evilNum );

        computeTauComponents( patch, velTau_CC,viscosity, Ttau_X_FC, Ttau_Y_FC, Ttau_Z_FC);

        for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
          IntVector c = *iter;
          right    = c + IntVector(1,0,0);    left     = c + IntVector(0,0,0);
          top      = c + IntVector(0,1,0);    bottom   = c + IntVector(0,0,0);
          front    = c + IntVector(0,0,1);    back     = c + IntVector(0,0,0);

          viscous_src[c].x(  (Ttau_X_FC[right].x() - Ttau_X_FC[left].x())  * areaX +
                             (Ttau_Y_FC[top].x()   - Ttau_Y_FC[bottom].x())* areaY +
                             (Ttau_Z_FC[front].x() - Ttau_Z_FC[back].x())  * areaZ  );

          viscous_src[c].y(  (Ttau_X_FC[right].y() - Ttau_X_FC[left].y())  * areaX +
                             (Ttau_Y_FC[top].y()   - Ttau_Y_FC[bottom].y())* areaY +
                             (Ttau_Z_FC[front].y() - Ttau_Z_FC[back].y())  * areaZ  );

          viscous_src[c].z(  (Ttau_X_FC[right].z() - Ttau_X_FC[left].z())  * areaX +
                             (Ttau_Y_FC[top].z()   - Ttau_Y_FC[bottom].z())* areaY +
                             (Ttau_Z_FC[front].z() - Ttau_Z_FC[back].z())  * areaZ  );
        }
        // copy the temporary data
        tau_X_FC.copyPatch( Ttau_X_FC, tau_X_FC.getLowIndex(), tau_X_FC.getHighIndex() );
        tau_Y_FC.copyPatch( Ttau_Y_FC, tau_Y_FC.getLowIndex(), tau_Y_FC.getHighIndex() );
        tau_Z_FC.copyPatch( Ttau_Z_FC, tau_Z_FC.getLowIndex(), tau_Z_FC.getHighIndex() );

      }
    }  // hasViscosity
  }  // patch loop
}


/* _____________________________________________________________________
 Purpose~   This task accumulates all of the sources/sinks of momentum
 _____________________________________________________________________  */
void ICE_sm::accumulateMomentumSourceSinks(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset* /*matls*/,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, iceCout, "Doing ICE_sm::accumulateMomentumSourceSinks" );

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(),level);

    Vector dx      = patch->dCell();
    double vol     = dx.x() * dx.y() * dx.z();
    Vector gravity = getGravity();

    double areaX = dx.y() * dx.z();
    double areaY = dx.x() * dx.z();
    double areaZ = dx.x() * dx.y();

    //__________________________________
    constSFCXVariable<double> pressX_FC;
    constSFCYVariable<double> pressY_FC;
    constSFCZVariable<double> pressZ_FC;
    constCCVariable<double>  rho_CC;
    CCVariable<Vector>   mom_source;
    
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;
    
    oneICEMaterial* ice_matl = d_sharedState->getOneICEMaterial(d_matl);
    int indx = ice_matl->getDWIndex();

    new_dw->get(pressX_FC, lb->pressX_FCLabel, indx, patch, gac, 1);
    new_dw->get(pressY_FC, lb->pressY_FCLabel, indx, patch, gac, 1);
    new_dw->get(pressZ_FC, lb->pressZ_FCLabel, indx, patch, gac, 1);
    old_dw->get(rho_CC,    lb->rho_CCLabel,    indx, patch, gn,  0);
    
    new_dw->allocateAndPut(mom_source,  lb->mom_source_CCLabel,  indx, patch);
    mom_source.initialize( Vector(0.,0.,0.) );

    //__________________________________
    //  accumulate sources
    for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      
      IntVector right, left, top, bottom, front, back;
      right    = c + IntVector(1,0,0);    left     = c + IntVector(0,0,0);
      top      = c + IntVector(0,1,0);    bottom   = c + IntVector(0,0,0);
      front    = c + IntVector(0,0,1);    back     = c + IntVector(0,0,0);

      double press_src_X = ( pressX_FC[right] - pressX_FC[left] );
      double press_src_Y = ( pressY_FC[top]   - pressY_FC[bottom] );
      double press_src_Z = ( pressZ_FC[front] - pressZ_FC[back] );

      mom_source[c].x( -press_src_X * areaX );
      mom_source[c].y( -press_src_Y * areaY );
      mom_source[c].z( -press_src_Z * areaZ );
    }

    //__________________________________
    constCCVariable<Vector> viscous_src;
    new_dw->get(viscous_src, lb->viscous_src_CCLabel, indx, patch,gn,0);

    for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      double mass = rho_CC[c] * vol;

      mom_source[c] = (mom_source[c] + viscous_src[c] + mass * gravity );
    }

    //__________________________________
    //  All Matls
    for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      mom_source[c] *= delT;
    }
  }  //patches
}

/* _____________________________________________________________________
 Purpose~   This task accumulates all of the sources/sinks of energy
            Currently the kinetic energy isn't included.
 _____________________________________________________________________  */
void ICE_sm::accumulateEnergySourceSinks(const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* /*matls*/,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch, iceCout, "Doing ICE_sm::accumulateEnergySourceSinks" );

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(),level);
    Vector dx = patch->dCell();
    double vol=dx.x()*dx.y()*dx.z();

    constCCVariable<double> kappa;
    constCCVariable<double> press_CC;
    constCCVariable<double> delP_Dilatate;
    constCCVariable<double> matl_press;
    constCCVariable<double> rho_CC;

    oneICEMaterial* ice_matl = d_sharedState->getOneICEMaterial( d_matl );
    int indx = ice_matl->getDWIndex();

    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;

    new_dw->get(press_CC,     lb->press_CCLabel,        indx, patch, gn, 0);
    new_dw->get(delP_Dilatate,lb->delP_DilatateLabel,   indx, patch, gn, 0);
    new_dw->get(kappa,        lb->compressibilityLabel, indx, patch, gn, 0);

    CCVariable<double> int_eng_source;
    CCVariable<double> heatCond_src;
    new_dw->allocateAndPut(int_eng_source,  lb->int_eng_source_CCLabel,indx,patch);
    new_dw->allocateAndPut(heatCond_src,    lb->heatCond_src_CCLabel,  indx,patch);

    int_eng_source.initialize(0.0);
    heatCond_src.initialize(0.0);

    //__________________________________
    //  Source due to conduction
    double thermalCond_test = ice_matl->getThermalConductivity();

    if(thermalCond_test != 0.0 ){
      constCCVariable<double> Temp_CC;
      constCCVariable<double> thermalCond;
      new_dw->get(thermalCond, lb->thermalCondLabel, indx,patch,gac,1);
      old_dw->get(Temp_CC,     lb->temp_CCLabel,     indx,patch,gac,1);

      scalarDiffusionOperator(new_dw, patch, Temp_CC, heatCond_src, thermalCond, delT);
    }

    //__________________________________
    //   Compute source from volume dilatation
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      double A = vol * kappa[c] * press_CC[c];
      int_eng_source[c] += A * delP_Dilatate[c] + heatCond_src[c];
    }
  }  // patch loop
}

/* _____________________________________________________________________
 Purpose~ Computes lagrangian mass momentum and energy
 _____________________________________________________________________  */
void ICE_sm::computeLagrangianValues(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* /*matls*/,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label(),level);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, iceCout, "Doing ICE_sm::computeLagrangianValues" );
    Vector  dx = patch->dCell();
    double vol = dx.x()*dx.y()*dx.z();

    //__________________________________
    //  Compute the Lagrangian quantities
    oneICEMaterial* ice_matl = d_sharedState->getOneICEMaterial( d_matl );
    int indx = ice_matl->getDWIndex();

    CCVariable<Vector> mom_L;
    CCVariable<double> int_eng_L;
    CCVariable<double> mass_L;

    constCCVariable<double> rho_CC, oldTemp_CC, cv, int_eng_source;
    constCCVariable<Vector> vel_CC, mom_source;

    Ghost::GhostType  gn = Ghost::None;
    new_dw->get(cv,             lb->specific_heatLabel,      indx, patch, gn, 0);
    old_dw->get(rho_CC,         lb->rho_CCLabel,             indx, patch, gn, 0);
    old_dw->get(vel_CC,         lb->vel_CCLabel,             indx, patch, gn, 0);
    old_dw->get(oldTemp_CC,     lb->temp_CCLabel,            indx, patch, gn, 0);
    new_dw->get(mom_source,     lb->mom_source_CCLabel,      indx, patch, gn, 0);
    new_dw->get(int_eng_source, lb->int_eng_source_CCLabel,  indx, patch, gn, 0);

    new_dw->allocateAndPut(mom_L,     lb->mom_L_CCLabel,     indx, patch);
    new_dw->allocateAndPut(int_eng_L, lb->int_eng_L_CCLabel, indx, patch);
    new_dw->allocateAndPut(mass_L,    lb->mass_L_CCLabel,    indx, patch);

    //__________________________________
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++) {
     IntVector c = *iter;
      double mass = rho_CC[c] * vol;

      mass_L[c]    = mass;
      mom_L[c]     = vel_CC[c] * mass + mom_source[c];
      int_eng_L[c] = mass*  cv[c] * oldTemp_CC[c] + int_eng_source[c];
    }

    //____ B U L L E T   P R O O F I N G----
    // catch negative internal energies
    // ignore BP if timestep restart has already been requested
    IntVector neg_cell;
    bool tsr = new_dw->timestepRestarted();

    if (!areAllValuesPositive(int_eng_L, neg_cell) && !tsr ) {
     ostringstream warn;
     int idx = level->getIndex();
     warn<<"ICE:(L-"<<idx<<"):computeLagrangianValues, mat "<<indx<<" cell "
         <<neg_cell<<" Negative int_eng_L: " << int_eng_L[neg_cell] <<  "\n";
     throw InvalidValue(warn.str(), __FILE__, __LINE__);
    }
  }  // patch loop
}

/* _____________________________________________________________________
 Purpose~
   This task calculates the The cell-centered, time n+1, mass, momentum
   internal energy, sp_vol
 _____________________________________________________________________  */
void ICE_sm::advectAndAdvanceInTime(const ProcessorGroup* /*pg*/,
                                    const PatchSubset* patches,
                                    const MaterialSubset* /*matls*/,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);

  // the advection calculations care about the position of the old dw subcycle
  double AMR_subCycleProgressVar = getSubCycleProgress(old_dw);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, iceCout, "Doing ICE_sm::advectAndAdvanceInTime" );
    iceCout << " progressVar " << AMR_subCycleProgressVar << endl;

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(),level);

    bool newGrid = d_sharedState->isRegridTimestep();
    Advector* advector = d_advector->clone(new_dw,patch,newGrid );


    CCVariable<double>  q_advected;
    CCVariable<Vector>  qV_advected;
    new_dw->allocateTemporary(q_advected,   patch);
    new_dw->allocateTemporary(qV_advected,  patch);

    oneICEMaterial* ice_matl = d_sharedState->getOneICEMaterial( d_matl );
    int indx = ice_matl->getDWIndex();

    CCVariable<double> mass_adv;
    CCVariable<double> int_eng_adv;
    CCVariable<Vector> mom_adv;

    constCCVariable<double>  int_eng_L;
    constCCVariable<double>  mass_L;
    constCCVariable<Vector>  mom_L;

    constSFCXVariable<double > uvel_FC;
    constSFCYVariable<double > vvel_FC;
    constSFCZVariable<double > wvel_FC;

    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(uvel_FC,     lb->uvel_FCLabel,          indx, patch, gac, 2);
    new_dw->get(vvel_FC,     lb->vvel_FCLabel,          indx, patch, gac, 2);
    new_dw->get(wvel_FC,     lb->wvel_FCLabel,          indx, patch, gac, 2);

    new_dw->get(mass_L,      lb->mass_L_CCLabel,        indx, patch, gac, 2);
    new_dw->get(mom_L,       lb->mom_L_CCLabel,         indx, patch, gac, 2);
    new_dw->get(int_eng_L,   lb->int_eng_L_CCLabel,     indx, patch, gac, 2);

    new_dw->allocateAndPut(mass_adv,    lb->mass_advLabel,   indx, patch);
    new_dw->allocateAndPut(mom_adv,     lb->mom_advLabel,    indx, patch);
    new_dw->allocateAndPut(int_eng_adv, lb->eng_advLabel,    indx, patch);

    mass_adv.initialize(0.0);
    mom_adv.initialize(Vector(0.0,0.0,0.0));
    int_eng_adv.initialize(0.0);
    q_advected.initialize(0.0);
    qV_advected.initialize(Vector(0.0,0.0,0.0));

    //__________________________________
    // common variables that get passed into the advection operators
    advectVarBasket* varBasket = scinew advectVarBasket();
    varBasket->new_dw = new_dw;
    varBasket->old_dw = old_dw;
    varBasket->indx   = indx;
    varBasket->patch  = patch;
    varBasket->level  = level;
    varBasket->lb     = lb;
    varBasket->useCompatibleFluxes = d_useCompatibleFluxes;

    //__________________________________
    //   Advection preprocessing
    bool bulletProof_test=true;
    advector->inFluxOutFluxVolume(uvel_FC, vvel_FC, wvel_FC, delT, patch,indx,
                                  bulletProof_test, new_dw, varBasket);
    //__________________________________
    // mass
    advector->advectMass(mass_L, q_advected,  varBasket);

    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      mass_adv[c]  = (mass_L[c] + q_advected[c]);
    }
    
    //__________________________________
    // momentum
    varBasket->is_Q_massSpecific   = true;
    varBasket->desc = "mom";
    advector->advectQ(mom_L, mass_L, qV_advected, varBasket);

    for(CellIterator iter = patch->getCellIterator(); !iter.done();  iter++){
      IntVector c = *iter;
      mom_adv[c] = (mom_L[c] + qV_advected[c]) ;
    }
    
    //__________________________________
    // internal energy
    varBasket->is_Q_massSpecific = true;
    varBasket->desc = "int_eng";
    advector->advectQ(int_eng_L, mass_L, q_advected, varBasket);

    for(CellIterator iter = patch->getCellIterator(); !iter.done();  iter++){
      IntVector c = *iter;
      int_eng_adv[c] = (int_eng_L[c] + q_advected[c]) ;
    }
    
    delete varBasket;
    delete advector;
  }  // patch loop
}
/*_____________________________________________________________________

 Purpose~ This task computes the primitive variables (rho,T,vel,sp_vol,...)
          at time n+1, from the conserved variables mass, momentum, energy...
 _____________________________________________________________________  */
void ICE_sm::conservedtoPrimitive_Vars(const ProcessorGroup* /*pg*/,
                                       const PatchSubset* patches,
                                       const MaterialSubset* /*matls*/,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  int L_indx = level->getIndex();

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, iceCout, "Doing ICE_sm::conservedtoPrimitive_Vars" );

    Vector dx = patch->dCell();
    double invvol = 1.0/(dx.x()*dx.y()*dx.z());
    Ghost::GhostType  gn  = Ghost::None;

    oneICEMaterial* ice_matl = d_sharedState->getOneICEMaterial( d_matl );
    int indx = ice_matl->getDWIndex();

    CCVariable<double> rho_CC, temp_CC, mach;
    CCVariable<Vector> vel_CC;
    constCCVariable<double> int_eng_adv;
    constCCVariable<double> mass_adv;
    constCCVariable<double> speedSound;
    constCCVariable<double> cv;
    constCCVariable<Vector> mom_adv;

    new_dw->get( speedSound,  lb->speedSound_CCLabel, indx, patch, gn, 0 );
    new_dw->get( cv,          lb->specific_heatLabel, indx, patch, gn, 0 );

    new_dw->get( mass_adv,    lb->mass_advLabel,      indx, patch, gn, 0 );
    new_dw->get( mom_adv,     lb->mom_advLabel,       indx, patch, gn, 0 );
    new_dw->get( int_eng_adv, lb->eng_advLabel,       indx, patch, gn, 0 );

    new_dw->allocateAndPut( rho_CC, lb->rho_CCLabel,   indx,patch );
    new_dw->allocateAndPut( temp_CC,lb->temp_CCLabel,  indx,patch );
    new_dw->allocateAndPut( vel_CC, lb->vel_CCLabel,   indx,patch );
    new_dw->allocateAndPut( mach,   lb->machLabel,     indx,patch );

    rho_CC.initialize(-d_EVIL_NUM);
    temp_CC.initialize(-d_EVIL_NUM);
    vel_CC.initialize(Vector(0.0,0.0,0.0));

    //__________________________________
    // Backout primitive quantities from
    // the conserved ones.
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      double inv_mass_adv = 1.0/mass_adv[c];
      rho_CC[c]    = mass_adv[c] * invvol;
      vel_CC[c]    = mom_adv[c]  * inv_mass_adv;
    }

    //__________________________________
    // Backout primitive quantities from
    // the conserved ones.
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      temp_CC[c] = int_eng_adv[c]/ (mass_adv[c]*cv[c]);
    }

    //__________________________________
    // set the boundary conditions
    setBC(rho_CC, "Density",     patch, indx );
    setBC(vel_CC, "Velocity",    patch, indx );
    setBC(temp_CC,"Temperature", patch, indx );
    
    //__________________________________
    // Compute Auxilary quantities
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      mach[c]  = vel_CC[c].length()/speedSound[c];
    }

    //____ B U L L E T   P R O O F I N G----
    // ignore BP if timestep restart has already been requested
    IntVector neg_cell;
    bool tsr = new_dw->timestepRestarted();

    ostringstream base, warn;
    base <<"ERROR ICE:(L-"<<L_indx<<"):conservedtoPrimitive_Vars, mat "<< indx <<" cell ";
    if (!areAllValuesPositive(rho_CC, neg_cell) && !tsr) {
      warn << base.str() << neg_cell << " negative rho_CC\n ";
      throw InvalidValue(warn.str(), __FILE__, __LINE__);
    }
    if (!areAllValuesPositive(temp_CC, neg_cell) && !tsr) {
      warn << base.str() << neg_cell << " negative temp_CC\n ";
      throw InvalidValue(warn.str(), __FILE__, __LINE__);
    }
  }  // patch loop
}


/*_______________________________________________________________________
 Purpose:   Test for conservation of mass, momentum, energy.
_______________________________________________________________________ */
void ICE_sm::TestConservation(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* /*matls*/,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw)
{


  const Level* level = getLevel(patches);
  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label(),level);

  double total_mass     = 0.0;
  double total_KE       = 0.0;
  double total_int_eng  = 0.0;
  Vector total_mom(0.0, 0.0, 0.0);

  for(int p=0; p<patches->size(); p++)  {
    const Patch* patch = patches->get(p);

    printTask(patches, patch, iceCout, "Doing ICE_sm::TestConservation" );

    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();

    Ghost::GhostType  gn  = Ghost::None;
    //__________________________________
    // get face centered velocities to
    // to compute what's being fluxed through the domain
    constSFCXVariable<double> uvel_FC;
    constSFCYVariable<double> vvel_FC;
    constSFCZVariable<double> wvel_FC;

    oneICEMaterial* ice_matl = d_sharedState->getOneICEMaterial(d_matl);
    int indx = ice_matl->getDWIndex();

    new_dw->get(uvel_FC, lb->uvel_FCLabel, indx, patch, gn, 0);
    new_dw->get(vvel_FC, lb->vvel_FCLabel, indx, patch, gn, 0);
    new_dw->get(wvel_FC, lb->wvel_FCLabel, indx, patch, gn, 0);

    //__________________________________
    // conservation of mass
    constCCVariable<double> rho_CC;
    new_dw->get(rho_CC, lb->rho_CCLabel,   indx, patch, gn,0);

    CCVariable<double>  mass;
    new_dw->allocateTemporary(mass,patch);

    for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      mass[c] = rho_CC[c] * cell_vol;
    }

    if(d_conservationTest->mass){
      conservationTest<double>(patch, delT, mass, uvel_FC, vvel_FC, wvel_FC, total_mass);
      total_mass = total_mass * cell_vol;
    }
    //__________________________________
    // conservation of momentum
    if(d_conservationTest->momentum){
      CCVariable<Vector> mom;
      constCCVariable<Vector> vel_CC;
      new_dw->allocateTemporary(mom,patch);

      new_dw->get(vel_CC, lb->vel_CCLabel,indx, patch, gn,0);

      for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        mom[c] = mass[c] * vel_CC[c];
      }
      conservationTest<Vector>(patch, delT, mom,  uvel_FC,vvel_FC,wvel_FC, total_mom);
    }
    //__________________________________
    // conservation of internal_energy
    if(d_conservationTest->energy){
      CCVariable<double> int_eng;
      constCCVariable<double> temp_CC;
      constCCVariable<double> cv;
      new_dw->allocateTemporary(int_eng,patch);

      new_dw->get(temp_CC, lb->temp_CCLabel,      indx, patch, gn,0);
      new_dw->get(cv,      lb->specific_heatLabel,indx, patch, gn,0);

      for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        int_eng[c] = mass[c] * cv[c] * temp_CC[c];
      }
      conservationTest<double>(patch, delT, int_eng, uvel_FC, vvel_FC, wvel_FC, total_int_eng);
    }
    //__________________________________
    // conservation of kinetic_energy
    if(d_conservationTest->energy){
      CCVariable<double> KE;
      constCCVariable<Vector> vel_CC;
      new_dw->allocateTemporary(KE,patch);

      new_dw->get(vel_CC, lb->vel_CCLabel,indx, patch, gn,0);

      for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        double vel_mag = vel_CC[c].length();
        KE[c] = 0.5 * mass[c] * vel_mag * vel_mag;
      }

      conservationTest<double>(patch, delT, KE, uvel_FC, vvel_FC, wvel_FC, total_KE);
    }
  }  // patch loop

  if(d_conservationTest->mass){
    new_dw->put(sum_vartype(total_mass),        lb->TotalMassLabel);
  }
  if(d_conservationTest->energy){
    new_dw->put(sum_vartype(total_KE),          lb->KineticEnergyLabel);
    new_dw->put(sum_vartype(total_int_eng),     lb->TotalIntEngLabel);
  }
  if(d_conservationTest->momentum){
    new_dw->put(sumvec_vartype(total_mom),      lb->TotalMomentumLabel);
  }
}

/*_____________________________________________________________________
 purpose:   find the upwind cell in each direction  This is a knock off
            of Bucky's logic
 _____________________________________________________________________  */
IntVector ICE_sm::upwindCell_X(const IntVector& c,
                            const double& var,
                            double is_logical_R_face )
{
  double  plus_minus_half = 0.5 * (var + d_SMALL_NUM)/fabs(var + d_SMALL_NUM);
  int one_or_zero = int(-0.5 - plus_minus_half + is_logical_R_face);
  IntVector tmp = c + IntVector(one_or_zero,0,0);
  return tmp;
}

IntVector ICE_sm::upwindCell_Y(const IntVector& c,
                            const double& var,
                            double is_logical_R_face )
{
  double  plus_minus_half = 0.5 * (var + d_SMALL_NUM)/fabs(var + d_SMALL_NUM);
  int one_or_zero = int(-0.5 - plus_minus_half + is_logical_R_face);
  IntVector tmp = c + IntVector(0,one_or_zero,0);
  return tmp;
}

IntVector ICE_sm::upwindCell_Z(const IntVector& c,
                            const double& var,
                            double is_logical_R_face )
{
  double  plus_minus_half = 0.5 * (var + d_SMALL_NUM)/fabs(var + d_SMALL_NUM);
  int one_or_zero = int(-0.5 - plus_minus_half + is_logical_R_face);
  IntVector tmp = c + IntVector(0,0,one_or_zero);
  return tmp;
}



