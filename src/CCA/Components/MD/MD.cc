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


#include <CCA/Ports/Scheduler.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Thread/Thread.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Mutex.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <CCA/Components/MD/MD.h>
#include <CCA/Components/MD/MDSystem.h>
#include <CCA/Components/MD/Electrostatics/ElectrostaticsFactory.h>
#include <CCA/Components/MD/CoordinateSystems/CoordinateSystemFactory.h>
#include <CCA/Components/MD/Electrostatics/SPME/SPME.h>
#include <CCA/Components/MD/Forcefields/Forcefield.h>
#include <CCA/Components/MD/Forcefields/ForcefieldFactory.h>
#include <CCA/Components/MD/Forcefields/TwoBodyForceField.h>
#include <CCA/Components/MD/Nonbonded/TwoBodyDeterministic.h>
#include <CCA/Components/MD/atomMap.h>
#include <CCA/Components/MD/atomFactory.h>
#include <CCA/Components/MD/Nonbonded/NonbondedFactory.h>

using namespace Uintah;

extern SCIRun::Mutex cerrLock;

static DebugStream md_dbg("MDDebug", false);
static DebugStream md_cout("MDCout", false);

MD::MD(const ProcessorGroup* myworld) :
    UintahParallelComponent(myworld)
{
  d_label = scinew MDLabel();
}

MD::~MD()
{
  delete d_label;
  delete d_system;
  delete d_nonbonded;
  delete d_electrostatics;
}

void MD::problemSetup(const ProblemSpecP& params,
                      const ProblemSpecP& restart_prob_spec,
                      GridP& grid,
                      SimulationStateP& shared_state)
{
  printTask(md_cout, "MD::problemSetup");

// Inherit shared state into the component
  d_sharedState = shared_state;

// Store the problem spec
  d_problemSpec = params;
  d_restartSpec = restart_prob_spec;

// Initialize output stream
  d_dataArchiver = dynamic_cast<Output*>(getPort("output"));
  if (!d_dataArchiver) {
    throw InternalError("MD: couldn't get output port", __FILE__, __LINE__);
  }

// Initialize base scheduler and attach the position variable
  dynamic_cast<Scheduler*>(getPort("scheduler"))->setPositionVar(d_label->global->pX);

  // create the coordinate system interface
  d_coordinate = CoordinateSystemFactory::create(params, shared_state, grid);
  d_coordinate->markCellChanged();

  // create and populate the MD System object
  d_system = scinew MDSystem(params, grid, shared_state);
//  d_system->attachForcefield(d_forcefield);
//  std::cerr << "Created system object" << std::endl;

  // Parse the forcefield
    // Material (atom) types should be registered on parsing the forcefield.
    Forcefield* tempFF = ForcefieldFactory::create(params, shared_state);

    switch(tempFF->getInteractionClass()) { // Set generic interface based on FF type
      case(TwoBody):
        d_forcefield=dynamic_cast<TwoBodyForcefield*> (tempFF);
        break;
      case(ThreeBody):
      case(NBody):
      default:
        throw InternalError("MD:  Attempted to instantiate a forcefield type which is not yet implemented", __FILE__, __LINE__);
    }

    std::cerr << "Forcefield created: "
              << d_forcefield->getForcefieldDescriptor() << std::endl;

  // Instantiate the integrator
//  Integrator* tempIntegrator = IntegratorFactory::create(XXX);
//  switch(tempIntegrator->getInteractionModel()) { // Set generic interface based on integrator type
//    case(Deterministic):
//      d_Integrator=dynamic_cast<DeterministicIntegrator*> (tempIntegrator);
//      break;
//    case(Stochastic):
//    case(Mixed):
//    default:
//      throw InternalError("MD:  Attempted to instantiate an integrator type which is not yet implemented", __FILE__, __LINE__);
//  }

  // For now set the interaction model explicitly
  //   Full implementation would have us inheriting this from the integrator:
  //     i.e. a Langevin-Dynamics simulation would be interactionModel = Mixed
  //          a Monte-Carlo model would be interactionModel = Stochastic
  //          basic MD is interactionModel = Deterministic
  interactionModel integratorModel = Deterministic;

  //bool doesFFSupportInteractionModel = d_forcefield->checkInteractionModelSupport(integratorModel);

  // create the Nonbonded object via factory method
  Nonbonded* tempNB = NonbondedFactory::create(params,
                                               d_coordinate,
                                               d_label,
                                               d_forcefield->getInteractionClass(),
                                               integratorModel);

  if (tempNB->getNonbondedType() == "TwoBodyDeterministic") {
    d_nonbonded = dynamic_cast<TwoBodyDeterministic*> (tempNB);
  }
  std::cerr << "Created nonbonded object" << std::endl;

  // create the Electrostatics object via factory method
  //electrostaticsModel elecCapability = d_forcefield->getElectrostaticsCapability();
  d_electrostatics = ElectrostaticsFactory::create(params, d_coordinate);
  if (d_electrostatics->getType() == Electrostatics::SPME) {
//    dynamic_cast<SPME*>(d_electrostatics)->setMDLabel(d_label);

    // create subscheduler for convergence loop in SPME::calculate
    Scheduler* sched = dynamic_cast<Scheduler*>(getPort("scheduler"));

    d_electrostaticSubscheduler = sched->createSubScheduler();
    d_electrostaticSubscheduler->initialize(3,1);
    d_electrostaticSubscheduler->clearMappings();
    d_electrostaticSubscheduler->mapDataWarehouse(Task::ParentOldDW, 0);
    d_electrostaticSubscheduler->mapDataWarehouse(Task::ParentNewDW, 1);
    d_electrostaticSubscheduler->mapDataWarehouse(Task::OldDW, 2);
    d_electrostaticSubscheduler->mapDataWarehouse(Task::NewDW, 3);
  }
  std::cerr << "created electrostatic object" << std::endl;

  // Add labels from our forcefield (nonbonded)
  MDSubcomponent* d_electrostaticInterface = dynamic_cast<MDSubcomponent*> (d_electrostatics);
  MDSubcomponent* d_nonbondedInterface     = dynamic_cast<MDSubcomponent*> (d_nonbonded);
//  MDSubcomponent* d_integratorInterface    = dynamic_cast<MDSubcomponent*> (d_integrator);
//  MDSubcomponent* d_valenceInterface       = dynamic_cast<MDSubcomponent*> (d_valence);


// Register the general labels that all MD simulations will use
   createBasePermanentParticleState();
   // And then add the labels that each created subcomponent will require
   d_electrostaticInterface->registerRequiredParticleStates(d_particleState, d_particleState_preReloc, d_label);
   d_nonbondedInterface->registerRequiredParticleStates(d_particleState, d_particleState_preReloc, d_label);
   // NYI:  d_integrator->registerRequiredParticleState(d_particleState, d_particleState_preReloc, d_label);

   // We must wait to register our atom (material) types until the
   // subcomponents have provided the per-particle labels
   d_forcefield->registerAtomTypes(d_particleState,
                                   d_particleState_preReloc,
                                   d_label,
                                   d_sharedState);

  std::cerr << "End of MD::Setup" << std::endl;
}

void MD::scheduleInitialize(const LevelP& level,
                            SchedulerP& sched)
{
  CoordinateSystem* coordSys;
  /*
   * Note there are multiple tasks scheduled here. All three need only ever happen once.
   *
   * 1.) MD::initialize
   * 2.) Nonbonded::initialize
   * 3.) SPME::initialize
   */
  std::cerr << "Enter:  Scheduled Initialization" << std::endl;

  // Get Forcefield related label pointers
  printSchedule(level, md_cout, "MD::scheduleInitialize");

  Task* task = scinew Task("MD::initialize", this, &MD::initialize);
  std::cerr << "MD::ScheduleInitialize -> Created new task." << std::endl;

  // Initialize will load position, velocity, and ID tags
  task->computes(d_label->global->pX);
  task->computes(d_label->global->pV);
  task->computes(d_label->global->pID);

  // Add computes from our forcefield (nonbonded) and
//  task->computes(d_label->nonbonded->pF_nonbonded);
//  task->computes(d_label->electrostatic->pF_electroInverse);
//  task->computes(d_label->electrostatic->pF_electroReal);
//  task->computes(d_label->electrostatic->pMu);
//  task->computes(d_label->electrostatic->pE_electroInverse);
//  task->computes(d_label->electrostatic->pE_electroReal);

//  task->computes(d_label->pValenceForceLabel);

//  std::cerr << "MD::Schedule computes particleID" << std::endl;

  task->computes(d_label->electrostatic->dSubschedulerDependency);
//  task->computes(d_label->subSchedulerDependencyLabel);
  std::cerr << "MD::Schedule computes subschedulerDependency" << std::endl;

  const MaterialSet* materials = d_sharedState->allMaterials();
  LoadBalancer* loadBal = sched->getLoadBalancer();
  const PatchSet* perProcPatches = loadBal->getPerProcessorPatchSet(level);

  // FIXME -- Original, no longer correct?
  sched->addTask(task, level->eachPatch(), materials);
  //sched->addTask(task, perProcPatches, materials);

  std::cerr << "MD::ScheduleInitialize -> Added MD::Initialize to task graph." << std::endl;

  // Nonbonded initialization - OncePerProc, during initial (0th) timestep.
  // The required pXlabel is available to this OncePerProc task in the new_dw from the computes above
  scheduleNonbondedInitialize(sched, perProcPatches, materials, level);
//  scheduleNonbondedInitialize()
  std::cerr << "MD::ScheduleInitialize -> Called scheduleNonbondedInitialize." << std::endl;

  // Nonbonded initialization - OncePerProc, during initial (0th) timestep.
  //   This OncePerProc task requires nothing
  scheduleElectrostaticsInitialize(sched, perProcPatches, materials, level);
  std::cerr << "MD::ScheduleInitialize -> Created scheduleElectrostaticsInitialize." << std::endl;

  std::cerr << "Exit:  Scheduled Initialization" << std::endl;
}

void MD::scheduleComputeStableTimestep(const LevelP& level,
                                       SchedulerP& sched)
{
  printSchedule(level, md_cout, "MD::scheduleComputeStableTimestep");

  Task* task = scinew Task("MD::computeStableTimestep", this, &MD::computeStableTimestep);

  task->requires(Task::NewDW, d_label->nonbonded->rNonbondedEnergy);
  task->requires(Task::NewDW, d_label->electrostatic->rElectrostaticInverseEnergy);
  task->requires(Task::NewDW, d_label->electrostatic->rElectrostaticRealEnergy);

  // We only -need- stress tensors if we're doing NPT
  if ( NPT == d_system->getEnsemble()) {
    task->requires(Task::NewDW, d_label->nonbonded->rNonbondedStress);
    task->requires(Task::NewDW, d_label->electrostatic->rElectrostaticInverseStress);
    task->requires(Task::NewDW, d_label->electrostatic->rElectrostaticRealStress);
    if ( d_electrostatics->isPolarizable() ) {
      task->requires(Task::NewDW,
                     d_label->electrostatic->rElectrostaticInverseStressDipole);
    }
  }

  task->computes(d_sharedState->get_delt_label(), level.get_rep());
  task->setType(Task::OncePerProc);

  LoadBalancer* loadBal = sched->getLoadBalancer();
  const PatchSet* perProcPatches = loadBal->getPerProcessorPatchSet(level);

  sched->addTask(task, perProcPatches, d_sharedState->allMaterials());
}

void MD::scheduleTimeAdvance(const LevelP& level,
                             SchedulerP& sched)
{
  printSchedule(level, md_cout, "MD::scheduleTimeAdvance");

  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_sharedState->allMaterials();

  scheduleNonbondedSetup(sched, patches, matls, level);

  scheduleElectrostaticsSetup(sched, patches, matls, level);

  scheduleNonbondedCalculate(sched, patches, matls, level);

  scheduleElectrostaticsCalculate(sched, patches, matls, level);

  // Should probably move the Finalizes into the appropriate clean-up step on MD.  (Destructor?)
  //   and appropriately modify the finalize routines.  !FIXME
  scheduleNonbondedFinalize(sched, patches, matls, level);

  scheduleElectrostaticsFinalize(sched, patches, matls, level);

  scheduleUpdatePosition(sched, patches, matls, level);

  sched->scheduleParticleRelocation(level, d_label->global->pX_preReloc,
                                           d_sharedState->d_particleState_preReloc,
                                           d_label->global->pX,
                                           d_sharedState->d_particleState,
                                           d_label->global->pID,
                                           matls, 1);


//  sched->scheduleParticleRelocation(level, d_label->pXLabel_preReloc, d_sharedState->d_particleState_preReloc, d_label->pXLabel,
//                                    d_sharedState->d_particleState, d_label->pParticleIDLabel, matls, 1);
}

void MD::scheduleNonbondedInitialize(SchedulerP& sched,
                                     const PatchSet* perProcPatches,
                                     const MaterialSet* matls,
                                     const LevelP& level)
{

  printSchedule(perProcPatches, md_cout, "MD::scheduleNonbondedInitialize");

  Task* task = scinew Task("MD::nonbondedInitialize", this, &MD::nonbondedInitialize);

  // This is during the initial timestep... no OldDW exists
  task->requires(Task::NewDW, d_label->global->pX, Ghost::None, 0);
//  task->requires(Task::NewDW, d_label->pXLabel, Ghost::None, 0);

  MDSubcomponent* d_nonbondedInterface = dynamic_cast<MDSubcomponent*> (d_nonbonded);

  task->requires(Task::NewDW, d_label->global->pX, Ghost::None, 0);
  d_nonbondedInterface->addInitializeRequirements(task, d_label);
  d_nonbondedInterface->addInitializeComputes(task, d_label);

   // initialize reduction variable; van der Waals energy
//  task->computes(d_label->nonbonded->rNonbondedEnergy);
//  task->computes(d_label->nonbonded->dNonbondedDependency);
//  task->computes(d_label->nonbonded->rNonbondedStress);

//  task->computes(d_label->nonbondedEnergyLabel);
//  task->computes(d_label->nonbondedDependencyLabel);

  task->setType(Task::OncePerProc);
  sched->addTask(task, perProcPatches, matls);
}

void MD::scheduleNonbondedSetup(SchedulerP& sched,
                                const PatchSet* patches,
                                const MaterialSet* matls,
                                const LevelP& level)
{
  printSchedule(patches, md_cout, "MD::scheduleNonbondedSetup");

  Task* task = scinew Task("MD::nonbondedSetup", this, &MD::nonbondedSetup);

  task->requires(Task::OldDW, d_label->nonbonded->dNonbondedDependency);
  MDSubcomponent* d_nonbondedInterface = dynamic_cast<MDSubcomponent*> (d_nonbonded);

  d_nonbondedInterface->addSetupRequirements(task, d_label);
  d_nonbondedInterface->addSetupComputes(task, d_label);

//  task->computes(d_label->nonbonded->dNonbondedDependency);

//  task->requires(Task::OldDW, d_label->nonbondedDependencyLabel);
//  task->computes(d_label->nonbondedDependencyLabel);

  sched->addTask(task, patches, matls);
}

void MD::scheduleNonbondedCalculate(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls,
                                    const LevelP& level)
{
  printSchedule(patches, md_cout, "MD::scheduleNonbondedCalculate");

  Task* task = scinew Task("MD::nonbondedCalculate",this,&MD::nonbondedCalculate);

//  int CUTOFF_RADIUS = d_system->getNonbondedGhostCells();
  int CUTOFF_CELLS = d_nonbonded->requiredGhostCells();
  task->requires(Task::OldDW, d_label->global->pX, Ghost::AroundNodes, CUTOFF_CELLS);
  task->requires(Task::OldDW, d_label->global->pID, Ghost::AroundNodes, CUTOFF_CELLS);

  MDSubcomponent* d_nonbondedInterface = dynamic_cast<MDSubcomponent*> (d_nonbonded);
  d_nonbondedInterface->addCalculateRequirements(task,d_label);
  d_nonbondedInterface->addCalculateComputes(task,d_label);

  // ??? FIXME
  //  Turns out we don't even need the old forces!
  //  ^^^Do we really need ghost nodes for the force if we're only calculating local forces on a patch?
//  task->requires(Task::OldDW, d_label->nonbonded->pF_nonbonded, Ghost::AroundNodes, CUTOFF_RADIUS);
//  task->requires(Task::OldDW, d_label->nonbonded->dNonbondedDependency, Ghost::None, 0);

  //  task->requires(Task::OldDW, d_label->pXLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
//  task->requires(Task::OldDW, d_label->pNonbondedForceLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
//  task->requires(Task::OldDW, d_lb->pEnergyLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
//  task->requires(Task::OldDW, d_label->pParticleIDLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
//  task->requires(Task::OldDW, d_label->nonbondedDependencyLabel, Ghost::None, 0);

//  task->computes(d_label->nonbonded->pF_nonbonded_preReloc);
//  task->computes(d_label->nonbonded->rNonbondedEnergy);
//  task->computes(d_label->nonbonded->rNonbondedStress);

  sched->addTask(task, patches, matls);
//  task->computes(d_label->pNonbondedForceLabel_preReloc);
//  task->computes(d_lb->pEnergyLabel_preReloc);
//  task->computes(d_label->nonbondedEnergyLabel);

}

void MD::scheduleNonbondedFinalize(SchedulerP& sched,
                                   const PatchSet* patches,
                                   const MaterialSet* matls,
                                   const LevelP& level)
{
  printSchedule(patches, md_cout, "MD::scheduleNonbondedFinalize");

  Task* task = scinew Task("MD::nonbondedFinalize", this, &MD::nonbondedFinalize);

  sched->addTask(task, patches, matls);
}

void MD::scheduleElectrostaticsInitialize(SchedulerP& sched,
                                          const PatchSet* perProcPatches,
                                          const MaterialSet* matls,
                                          const LevelP& level)
{
  printSchedule(perProcPatches, md_cout, "MD::scheduleElectrostaticsInitialize");

  // initialize electrostatics instance; if we're doing electrostatics
  if (d_electrostatics->getType() != Electrostatics::NONE) {

    Task* task = scinew Task("MD::electrostaticsInitialize",
                             this,
                             &MD::electrostaticsInitialize);

    task->requires(Task::NewDW, d_label->global->pX, Ghost::None, 0);

    // cast electrostatics to a subcomponent interface
    MDSubcomponent* d_electroInterface = dynamic_cast<MDSubcomponent*> (d_electrostatics);
    d_electroInterface->addInitializeRequirements(task, d_label);
    d_electroInterface->addInitializeComputes(task, d_label);

    task->setType(Task::OncePerProc);
    sched->addTask(task, perProcPatches, matls);
  }

//    if (d_electrostatics->getType() == Electrostatics::SPME) {
//
//      // FFTW related sole variables
//      task->computes(d_label->electrostatic->sForwardTransformPlan);
//      task->computes(d_label->electrostatic->sBackwardTransformPlan);
//
//      // dependency
//      task->computes(d_label->electrostatic->dElectrostaticDependency);
//
////      task->computes(d_label->electrostaticReciprocalEnergyLabel);
////      task->computes(d_label->electrostaticReciprocalStressLabel);
//
//      // sole variables
////      task->computes(d_label->electrostaticsDependencyLabel);
//      task->setType(Task::OncePerProc);
//      sched->addTask(task, perProcPatches, matls);
//    }
//
//  }
}

void MD::scheduleElectrostaticsSetup(SchedulerP& sched,
                                     const PatchSet* patches,
                                     const MaterialSet* matls,
                                     const LevelP& level)
{
  printSchedule(patches, md_cout, "MD::scheduleElectrostaticsSetup");

  if (d_electrostatics->getType() != Electrostatics::NONE) {
    Task* task = scinew Task("MD::electrostaticsSetup", this, &MD::electrostaticsSetup);

    // cast electrostatics to a subcomponent interface
    MDSubcomponent* d_electroInterface = dynamic_cast<MDSubcomponent*> (d_electrostatics);

    d_electroInterface->addSetupRequirements(task, d_label);
    d_electroInterface->addSetupComputes(task, d_label);
    sched->addTask(task, patches, matls);

//    task->requires(Task::NewDW, d_label->electrostatic->dElectrostaticDependency);
//
//
//    // particle variables
//    task->computes(d_label->electrostatic->pE_electroInverse);
//    task->computes(d_label->electrostatic->pE_electroReal);
//    task->computes(d_label->electrostatic->pF_electroInverse);
//    task->computes(d_label->electrostatic->pF_electroReal);
//
//    // dependency variable (really necessary at this point?)
//    task->modifies(d_label->electrostatic->dElectrostaticDependency);
  }
//  task->computes(d_label->electrostaticsDependencyLabel);

}

void MD::scheduleElectrostaticsCalculate(SchedulerP& sched,
                                         const PatchSet* patches,
                                         const MaterialSet* matls,
                                         const LevelP& level)
{
  printSchedule(patches, md_cout, "MD::scheduleElectrostaticsCalculate");

  if (d_electrostatics->getType() != Electrostatics::NONE) {
    Task* task = scinew Task("electrostaticsCalculate", this, &MD::electrostaticsCalculate, level);

    task->requires(Task::OldDW, d_label->global->pX, Ghost::AroundNodes, d_electrostatics->requiredGhostCells());
    task->requires(Task::OldDW, d_label->global->pID, Ghost::AroundNodes, d_electrostatics->requiredGhostCells());

    MDSubcomponent* d_electroInterface = dynamic_cast<MDSubcomponent*> (d_electrostatics);

    d_electroInterface->addCalculateRequirements(task, d_label);
    d_electroInterface->addCalculateComputes(task, d_label);
//    task->requires(Task::NewDW, d_label->electrostatic->pE_electroInverse, Ghost::None, 0);
//    task->requires(Task::NewDW, d_label->electrostatic->pE_electroReal, Ghost::None, 0);
//
//    task->requires(Task::NewDW, d_label->electrostatic->dElectrostaticDependency);

  //  task->requires(Task::OldDW, d_label->pXLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
  //  task->requires(Task::OldDW, d_lb->pChargeLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
  //  task->requires(Task::OldDW, d_label->pParticleIDLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
  //  task->requires(Task::OldDW, d_label->electrostaticsDependencyLabel);
  //  task->requires(Task::OldDW, d_label->subSchedulerDependencyLabel, Ghost::None, 0);

//    task->computes(d_label->electrostatic->rElectrostaticInverseEnergy);
//    task->computes(d_label->electrostatic->rElectrostaticRealEnergy);
//    task->computes(d_label->electrostatic->rElectrostaticInverseStress);
//    task->computes(d_label->electrostatic->rElectrostaticRealStress);
//
//    task->computes(d_label->electrostatic->pMu);

    task->hasSubScheduler(true);
    task->setType(Task::OncePerProc);

    LoadBalancer* loadBal = sched->getLoadBalancer();
    const PatchSet* perProcPatches = loadBal->getPerProcessorPatchSet(level);

    sched->addTask(task, perProcPatches, matls);
  }

}

void MD::scheduleElectrostaticsFinalize(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls,
                                        const LevelP& level)
{
  printSchedule(patches, md_cout, "MD::scheduleElectrostaticsFinalize");

  if (d_electrostatics->getType() != Electrostatics::NONE) {
    Task* task = scinew Task("MD::electrostaticsFinalize", this, &MD::electrostaticsFinalize);

    MDSubcomponent* d_electroInterface = dynamic_cast<MDSubcomponent*> (d_electrostatics);
    d_electroInterface->addFinalizeRequirements(task, d_label);
    d_electroInterface->addFinalizeComputes(task, d_label);


  // particle variables
//  task->requires(Task::NewDW, d_label->electrostatic->pF_electroInverse, Ghost::None, 0);
//  task->requires(Task::NewDW, d_label->electrostatic->pF_electroReal, Ghost::None, 0);
//  task->requires(Task::NewDW, d_label->electrostatic->dSubschedulerDependency);

//  task->requires(Task::NewDW, d_label->subSchedulerDependencyLabel, Ghost:: Ghost::None, 0);

//  task->computes(d_label->electrostatic->pF_electroInverse_preReloc);
//  task->computes(d_label->electrostatic->pF_electroReal_preReloc);

//  task->computes(d_label->pElectrostaticsReciprocalForce_preReloc);
//  task->computes(d_label->pElectrostaticsRealForce_preReloc);
//  task->computes(d_lb->pElectrostaticsForceLabel_preReloc);
//  task->computes(d_lb->pChargeLabel_preReloc);

    sched->addTask(task, patches, matls);
  }
}

void MD::scheduleUpdatePosition(SchedulerP& sched,
                                const PatchSet* patches,
                                const MaterialSet* matls,
                                const LevelP& level)
{
  printSchedule(patches, md_cout, "MD::scheduleUpdatePosition");

  // This should eventually schedule a call of the integrator.  Something like d_Integrator->advanceTimestep()
  Task* task = scinew Task("updatePosition", this, &MD::updatePosition);


  // Integration requires the position and particle ID from last time step
  task->requires(Task::OldDW, d_label->global->pX, Ghost::None, 0);
  task->requires(Task::OldDW, d_label->global->pID, Ghost::None, 0);

  // d_forcefield->addUpdateRequires();
  task->requires(Task::NewDW, d_label->nonbonded->pF_nonbonded_preReloc, Ghost::None, 0);
  // d_electrostatics->addUpdateRequires();
  task->requires(Task::NewDW, d_label->electrostatic->pF_electroInverse_preReloc, Ghost::None, 0);
  task->requires(Task::NewDW, d_label->electrostatic->pF_electroReal_preReloc, Ghost::None, 0);

//  task->requires(Task::OldDW, d_label->pXLabel, Ghost::None, 0);
//  task->requires(Task::OldDW, d_label->pParticleIDLabel, Ghost::None, 0);

  // And the newly calculated forces
//  task->requires(Task::NewDW, d_label->pNonbondedForceLabel_preReloc, Ghost::None, 0);
//  task->requires(Task::NewDW, d_label->pElectrostaticsReciprocalForce_preReloc, Ghost::None, 0);
//  task->requires(Task::NewDW, d_label->pElectrostaticsRealForce_preReloc, Ghost::None, 0);
//  task->requires(Task::NewDW, d_lb->pElectrostaticsForceLabel_preReloc, Ghost::None, 0);
//  task->requires(Task::OldDW, d_lb->pAccelLabel, Ghost::None, 0);
//  task->requires(Task::OldDW, d_lb->pVelocityLabel, Ghost::None, 0);
//  task->requires(Task::OldDW, d_lb->pMassLabel, Ghost::None, 0);

  // Not sure what this does atm - JBH, 4-7-14
  task->requires(Task::OldDW, d_sharedState->get_delt_label());

  // From integration we get new positions and velocities
  task->computes(d_label->global->pX_preReloc);
  task->computes(d_label->global->pID_preReloc);
//  task->computes(d_label->pXLabel_preReloc);
//  task->computes(d_label->pVelocityLabel_preReloc);
//  task->modifies(d_lb->pNonbondedForceLabel_preReloc);
//  task->modifies(d_lb->pElectrostaticsForceLabel_preReloc);
//  task->computes(d_lb->pAccelLabel_preReloc);
//  task->computes(d_lb->pMassLabel_preReloc);
//  task->computes(d_lb->pParticleIDLabel_preReloc);

  sched->addTask(task, patches, matls);
}

void MD::initialize(const ProcessorGroup* pg,
                    const PatchSubset* perProcPatches,
                    const MaterialSubset* matls,
                    DataWarehouse* /* old_dw */,
                    DataWarehouse* new_dw)
{
  printTask(perProcPatches, md_cout, "MD::initialize");

  Matrix3   systemInverseCell = d_coordinate->getInverseCell();
  IntVector totalSystemExtent = d_coordinate->getCellExtent();

  SCIRun::Vector inverseExtentVector;
  inverseExtentVector[0]=1.0/static_cast<double> (totalSystemExtent[0]);
  inverseExtentVector[1]=1.0/static_cast<double> (totalSystemExtent[1]);
  inverseExtentVector[2]=1.0/static_cast<double> (totalSystemExtent[2]);

  SCIRun::Vector cellDimensions = d_coordinate->getUnitCell()*inverseExtentVector;

  // Loop through each patch
  size_t numPatches = perProcPatches->size();
  size_t numMaterials = matls->size();

  // Input coordinates from problem spec
  atomMap* parsedCoordinates = atomFactory::create(d_problemSpec, d_sharedState);
  size_t numTypesParsed = parsedCoordinates->getNumberAtomTypes();
  size_t numMaterialTypes = d_sharedState->allMDMaterials()->size();

//  if (numTypesParsed != numMaterialTypes) {
//    std::stringstream errorOut;
//    errorOut << " ERROR:  Expected to find " << numMaterialTypes << " types of materials in the coordinate file." << std::endl;
//    errorOut << "     However, the coordinate file only parsed " << numTypesParsed << std::endl;
//    throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
//  }

  for (size_t matlIndex = 0; matlIndex < numMaterialTypes; ++matlIndex) {
    std::string materialLabel = d_sharedState->getMDMaterial(matlIndex)->getMaterialLabel();

    std::vector<atomData*>* currAtomList = parsedCoordinates->getAtomList(materialLabel);
    size_t numAtoms = currAtomList->size();

    d_system->registerAtomCount(numAtoms,matlIndex);

  }

  std::cerr << "Constructed particle map in MD::initialize" << std::endl;
  SCIRun::Vector VectorZero(0.0, 0.0, 0.0);

  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) { // Loop over perProcPatches
    const Patch* currPatch = perProcPatches->get(patchIndex);
    SCIRun::IntVector lowCellBoundary = currPatch->getCellLowIndex();
    SCIRun::IntVector highCellBoundary = currPatch->getCellHighIndex();
    (const_cast<Patch*> (currPatch))->getLevel(true)->setdCell(cellDimensions);

    for (size_t materialIndex = 0; materialIndex < numMaterials; ++materialIndex) { // Loop over materials
      size_t materialID = matls->get(materialIndex);
      std::string materialLabel = d_sharedState->getMDMaterial(materialID)->getMaterialLabel();

      // Match coordinates to material and extract coordinate list
      std::vector<atomData*>* currAtomList = parsedCoordinates->getAtomList(materialLabel);
      size_t numMaterialAtoms = parsedCoordinates->getAtomListSize(materialLabel);
      std::vector<Point> localAtomCoordinates;
      std::vector<size_t> localAtomID;
      std::vector<SCIRun::Vector> localAtomVelocity;

      for (size_t atomIndex = 0; atomIndex < numMaterialAtoms; ++atomIndex) {  // Loop over all atoms of material
        atomData* currAtom = (*currAtomList)[atomIndex];
        Point currPosition = currAtom->getPosition();
        IntVector currCell = currPatch->getLevel()->getCellIndex(currPosition);

        // Build local atom list for atoms of material in current patch
        bool atomInPatch = containsAtom(lowCellBoundary, highCellBoundary, currCell);
        if (atomInPatch) { // Atom is on this patch
          size_t currID = currAtom->getID();
          SCIRun::Vector currVelocity = currAtom->getVelocity();

          localAtomCoordinates.push_back(currPosition);
          localAtomID.push_back(currID);
          localAtomVelocity.push_back(currVelocity);
        }
      }

      // Create this patch's particle set for atoms of current material
      size_t numAtoms = localAtomCoordinates.size();
      ParticleSubset* currPset = new_dw->createParticleSubset(numAtoms, materialID, currPatch);

    // ----> Variables from parsing the input coordinate file
    // --> Position
      ParticleVariable<Point> pX;
      new_dw->allocateAndPut(pX, d_label->global->pX, currPset);

    // --> Velocity
      ParticleVariable<Vector> pV;
      new_dw->allocateAndPut(pV, d_label->global->pV, currPset);

    // --> Index
      ParticleVariable<long64> pID;
      new_dw->allocateAndPut(pID, d_label->global->pID, currPset);

//    // ----> Initialization variables
//    // -->  Force subcomponents
//      ParticleVariable<Vector> pF_nb, pF_eReal, pF_eRecip, pF_v;
//      new_dw->allocateAndPut(pF_nb, d_label->nonbonded->pF_nonbonded, currPset);
//      new_dw->allocateAndPut(pF_eReal, d_label->electrostatic->pF_electroReal, currPset);
//      new_dw->allocateAndPut(pF_eRecip, d_label->electrostatic->pF_electroInverse, currPset);

//      new_dw->allocateAndPut(pF_nb, d_label->pNonbondedForceLabel, currPset);
//      new_dw->allocateAndPut(pF_eReal, d_label->pElectrostaticsRealForce, currPset);
//      new_dw->allocateAndPut(pF_eRecip, d_label->pElectrostaticsReciprocalForce, currPset);
//      new_dw->allocateAndPut(pF_v, d_label->pValenceForceLabel, currPset);
    // --> Dipoles
//      ParticleVariable<Vector> pDipoles;
//      new_dw->allocateAndPut(pDipoles, d_label->electrostatic->pMu, currPset);
//      new_dw->allocateAndPut(pDipoles, d_label->pTotalDipoles, currPset);
    // --> Field subcomponents
//      ParticleVariable<Vector> pFieldReal, pFieldReciprocal;
//      new_dw->allocateAndPut(pFieldReal, d_label->electrostatic->pE_electroReal, currPset);
//      new_dw->allocateAndPut(pFieldReciprocal, d_label->electrostatic->pE_electroInverse, currPset);
//      new_dw->allocateAndPut(pFieldReal, d_label->pElectrostaticsRealField, currPset);
//      new_dw->allocateAndPut(pFieldReciprocal, d_label->pElectrostaticsReciprocalField, currPset);

      for (size_t atomIndex = 0; atomIndex < numAtoms; ++atomIndex) { // Loop over atoms in this matl in this patch
        // Transfer over currently defined atom data
        pX[atomIndex]    = localAtomCoordinates[atomIndex];
        pV[atomIndex]    = localAtomVelocity[atomIndex];
        pID[atomIndex]   = localAtomID[atomIndex];

        // Initialize the rest
//        pF_nb[atomIndex]                    = VectorZero;
//        pF_eReal[atomIndex]                 = VectorZero;
//        pF_eRecip[atomIndex]                = VectorZero;
//        pF_v[atomIndex]                     = VectorZero;
//        pDipoles[atomIndex]                 = VectorZero;
//        pFieldReal[atomIndex]               = VectorZero;
//        pFieldReciprocal[atomIndex]         = VectorZero;

        if (md_dbg.active()) { // Output for debug..
          cerrLock.lock();
          std::cout.setf(std::ios_base::showpoint);  // print decimal and trailing zeros
          std::cout.setf(std::ios_base::left);  // pad after the value
          std::cout.setf(std::ios_base::uppercase);  // use upper-case scientific notation
          std::cout << std::setw(10) << " Patch_ID: " << std::setw(4) << currPatch->getID();
          std::cout << std::setw(14) << " Particle_ID: " << std::setw(4) << pID[atomIndex];
          std::cout << std::setw(12) << " Position: " << pX[atomIndex];
          std::cout << std::endl;
          cerrLock.unlock();
        }


      }

      CCVariable<int> subSchedulerDependency;
      new_dw->allocateAndPut(subSchedulerDependency, d_label->electrostatic->dSubschedulerDependency, materialID, currPatch, Ghost::None, 0);
//      new_dw->allocateAndPut(subSchedulerDependency, d_label->subSchedulerDependencyLabel, materialID, currPatch, Ghost::None, 0);
      subSchedulerDependency.initialize(0);


    } // Loop over materials
  } // Loop over patches

//  // Parse the input coordinates
// size_t atomMapEntries = parsedCoordinates->getNumberAtomTypes();
//
//  // Pull the newly registered materials into a material set
//  const MaterialSet* allMaterials = d_sharedState->allMaterials();
//
//  size_t materialEntries = allMaterials->size();
//  if (materialEntries != atomMapEntries) {
//    std::stringstream errorOut;
//    errorOut << "ERROR:  There are " << materialEntries << " atom types registered, but only " << atomMapEntries
//             << " unique atom types were parsed from the coordinate file(s)." << std::endl;
//    throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
//  }



//
//  // loop through all patches
//  unsigned int numAtoms = d_system->getNumAtoms();
//  unsigned int numPatches = patches->size();
//  for (unsigned int p = 0; p < numPatches; ++p) {
//    const Patch* patch = patches->get(p);
//
//    // get bounds of current patch to correctly initialize particles (atoms)
//    IntVector low = patch->getCellLowIndex();
//    IntVector high = patch->getCellHighIndex();
//
//    // do this for each material
//    unsigned int numMatls = matls->size();
//    for (unsigned int m = 0; m < numMatls; ++m) {
//      int matl = matls->get(m);
//
//      ParticleVariable<Point> px;
//      ParticleVariable<Vector> pforceNonbonded;
//      ParticleVariable<Vector> pforceElectrostatics;
//      ParticleVariable<Vector> paccel;
//      ParticleVariable<Vector> pvelocity;
//      ParticleVariable<double> penergy;
//      ParticleVariable<double> pmass;
//      ParticleVariable<double> pcharge;
//      ParticleVariable<long64> pids;
//      CCVariable<int> subSchedulerDependency;
//
//      // eventually we'll need to use PFS for this
//      std::vector<Atom> localAtoms;
//      for (unsigned int i = 0; i < numAtoms; ++i) {
////        Vector reducedCoordinates = ((d_atomList[i].coords).asVector() * systemInverseCell);
////        IntVector cellCoordinates((reducedCoordinates * totalSystemExtent.asVector()).asPoint());
//        // TODO make sure this is correct before deleting the above lines
//        IntVector ptIndex = patch->getLevel()->getCellIndex(d_atomList[i].coords);
//        if (containsAtom(low, high, ptIndex)) {
//          localAtoms.push_back(d_atomList[i]);
//        }
//      }
//
//      // insert particle type counting loop here
//
//      ParticleSubset* pset = new_dw->createParticleSubset(localAtoms.size(), matl, patch);
//      new_dw->allocateAndPut(px, d_lb->pXLabel, pset);
//      new_dw->allocateAndPut(pforceNonbonded, d_lb->pNonbondedForceLabel, pset);
//      new_dw->allocateAndPut(pforceElectrostatics, d_lb->pElectrostaticsForceLabel, pset);
//      new_dw->allocateAndPut(paccel, d_lb->pAccelLabel, pset);
//      new_dw->allocateAndPut(pvelocity, d_lb->pVelocityLabel, pset);
//      new_dw->allocateAndPut(penergy, d_lb->pEnergyLabel, pset);
//      new_dw->allocateAndPut(pmass, d_lb->pMassLabel, pset);
//      new_dw->allocateAndPut(pcharge, d_lb->pChargeLabel, pset);
//      new_dw->allocateAndPut(pids, d_lb->pParticleIDLabel, pset);
//      new_dw->allocateAndPut(subSchedulerDependency, d_lb->subSchedulerDependencyLabel, matl, patch, Ghost::None, 0);
//      subSchedulerDependency.initialize(0);
//
//      int numParticles = pset->numParticles();
//      for (int i = 0; i < numParticles; ++i) {
//        Point pos = localAtoms[i].coords;
//        px[i] = pos;
//        pforceNonbonded[i] = Vector(0.0, 0.0, 0.0);
//        pforceElectrostatics[i] = Vector(0.0, 0.0, 0.0);
//        paccel[i] = Vector(0.0, 0.0, 0.0);
//        pvelocity[i] = Vector(0.0, 0.0, 0.0);
//        penergy[i] = 1.1;
//        pmass[i] = 2.5;
//        pcharge[i] = localAtoms[i].charge;
//        pids[i] = patch->getID() * numAtoms + i;
//
}

void MD::computeStableTimestep(const ProcessorGroup* pg,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::computeStableTimestep");

  sum_vartype vdwEnergy;
  sum_vartype spmeFourierEnergy;
  matrix_sum spmeFourierStress;
  matrix_sum spmeRealStress;
  matrix_sum spmeFourierStressDipole;

  double energyTest = 0.1;
  new_dw->get(vdwEnergy, d_label->nonbonded->rNonbondedEnergy);
  new_dw->get(spmeFourierEnergy,
              d_label->electrostatic->rElectrostaticInverseEnergy);

  proc0cout << std::endl;
  proc0cout << "-----------------------------------------------------"           << std::endl;
  proc0cout << "Total Energy   = " << std::setprecision(16) << vdwEnergy         << std::endl;
  proc0cout << "-----------------------------------------------------"           << std::endl;
  proc0cout << "Fourier Energy = " << std::setprecision(16) << spmeFourierEnergy << std::endl;
  proc0cout << "-----------------------------------------------------"           << std::endl;


  if (NPT == d_system->getEnsemble()) {
    new_dw->get(spmeFourierStress,
                d_label->electrostatic->rElectrostaticInverseStress);
    Uintah::Matrix3 test(0.1);

    new_dw->get(spmeRealStress,
                d_label->electrostatic->rElectrostaticRealStress);
    proc0cout << "Fourier Stress = " << std::setprecision(16) << spmeFourierStress << std::endl;
    proc0cout << "-----------------------------------------------------"           << std::endl;

    if (d_electrostatics->isPolarizable()) {
      new_dw->get(spmeFourierStressDipole,
                  d_label->electrostatic->rElectrostaticInverseStressDipole);
    }
  }

  proc0cout << std::endl;

  new_dw->put(delt_vartype(1), d_sharedState->get_delt_label(), getLevel(patches));
}

void MD::nonbondedInitialize(const ProcessorGroup* pg,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::nonbondedInitialize");

  d_nonbonded->initialize(pg, patches, matls, old_dw, new_dw,
                          d_sharedState, d_system, d_label, d_coordinate);
}

void MD::nonbondedSetup(const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::nonbondedSetup");

  if (d_coordinate->queryCellChanged()) {
    d_nonbonded->setup(pg, patches, matls, old_dw, new_dw,
                       d_sharedState, d_system, d_label, d_coordinate);
  }
}

void MD::nonbondedCalculate(const ProcessorGroup*   pg,
                            const PatchSubset*      patches,
                            const MaterialSubset*   matls,
                            DataWarehouse*          oldDW,
                            DataWarehouse*          newDW)
{
  printTask(patches, md_cout, "MD::nonbondedCalculate");

  d_nonbonded->calculate(pg, patches, matls, oldDW, newDW,
                         d_sharedState, d_system, d_label, d_coordinate);
}

void MD::nonbondedFinalize(const ProcessorGroup* pg,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::nonbondedFinalize");

  d_nonbonded->finalize(pg, patches, matls, old_dw, new_dw,
                        d_sharedState, d_system, d_label, d_coordinate);
}

void MD::electrostaticsInitialize(const ProcessorGroup* pg,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::electrostaticsInitialize");

  d_electrostatics->initialize(pg, patches, matls, old_dw, new_dw,
                               &d_sharedState,
                               d_system,
                               d_label,
                               d_coordinate);
}

void MD::electrostaticsSetup(const ProcessorGroup* pg,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::electrostaticsSetup");

  d_electrostatics->setup(pg, patches, matls, old_dw, new_dw,
                          &d_sharedState,
                          d_system,
                          d_label,
                          d_coordinate);
}

void MD::electrostaticsCalculate(const ProcessorGroup* pg,
                                 const PatchSubset* perProcPatches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* parentOldDW,
                                 DataWarehouse* parentNewDW,
                                 const LevelP level)
{
  printTask(perProcPatches, md_cout, "MD::electrostaticsCalculate");

//  delt_vartype dt;
//  DataWarehouse* subNewDW = subscheduler->get_dw(3);
//  parentOldDW->get(dt, d_sharedState->get_delt_label(),level.get_rep());
//  subNewDW->put(dt, d_sharedState->get_delt_label(),level.get_rep());

  d_electrostatics->calculate(pg,perProcPatches,matls,parentOldDW,parentNewDW,
                              &d_sharedState,
                              d_system,
                              d_label,
                              d_coordinate,
                              d_electrostaticSubscheduler,
                              level);
}

void MD::electrostaticsFinalize(const ProcessorGroup* pg,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::electrostaticsFinalize");

  d_electrostatics->finalize(pg, patches, matls, old_dw, new_dw,
                             &d_sharedState,
                             d_system,
                             d_label,
                             d_coordinate);
}

void MD::updatePosition(const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::updatePosition");

  // loop through all patches
  unsigned int numPatches = patches->size();
  SimulationStateP simState = d_system->getStatePointer();

  for (unsigned int p = 0; p < numPatches; ++p) {
    const Patch* patch = patches->get(p);
    // do this for each material; for this example, there is only a single material, material "0"
    unsigned int numMatls = matls->size();
    for (unsigned int m = 0; m < numMatls; ++m) {
      int matl = matls->get(m);
      double massInv = 1.0/(simState->getMDMaterial(matl)->getMass());

      ParticleSubset* pset = old_dw->getParticleSubset(matl, patch);
      ParticleSubset* delset = scinew ParticleSubset(0, matl, patch);

      // Variables required in order to integrate
      // --> Position at last time step
      constParticleVariable<Point> pX;
      old_dw->get(pX, d_label->global->pX, pset);
      // --> Velocity at last time step (velocity verlet algorithm)
      constParticleVariable<SCIRun::Vector> pV;
      old_dw->get(pX, d_label->global->pV, pset);
//      // --> Acceleration at last time step (velocity verlet algorithm)
//      constParticleVariable<SCIRun::Vector> pA;
//      old_dw->get(pA, d_lb->pAccelLabel, pset);
      // --> Forces for this time step
      constParticleVariable<SCIRun::Vector> pForceElectroReal;
      constParticleVariable<SCIRun::Vector> pForceElectroRecip;
      constParticleVariable<SCIRun::Vector> pForceNonbonded;
      new_dw->get(pForceElectroReal, d_label->electrostatic->pF_electroReal_preReloc, pset);
      new_dw->get(pForceElectroRecip, d_label->electrostatic->pF_electroInverse_preReloc, pset);
      new_dw->get(pForceNonbonded, d_label->nonbonded->pF_nonbonded_preReloc, pset);
//      new_dw->get(pForceElectroReal,d_label->pElectrostaticsRealForce_preReloc, pset);
//      new_dw->get(pForceElectroRecip, d_label->pElectrostaticsReciprocalForce_preReloc, pset);
//      new_dw->get(pForceNonbonded, d_label->pNonbondedForceLabel_preReloc, pset);

      // Variables which the integrator calculates
      // --> New position
      ParticleVariable<Point> pXNew;
      new_dw->allocateAndPut(pXNew, d_label->global->pX_preReloc, pset);
//      new_dw->allocateAndPut(pXNew, d_label->pXLabel_preReloc, pset);
      // --> New velocity
      ParticleVariable<SCIRun::Vector> pVNew;
      new_dw->allocateAndPut(pVNew, d_label->global->pV_preReloc, pset);
//      new_dw->allocateAndPut(pVNew, d_label->pVelocityLabel_preReloc, pset);
//      // --> New acceleration
//      ParticleVariable<SCIRun::Vector> pANew;
//      new_dw->allocateAndPut(pANew, d_lb->pAccelLabel_preReloc, pset);

      // get delT
      delt_vartype delT;
      old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches));

      size_t numAtoms = pset->numParticles();
      // Loop over the atom set
      for (size_t atomIndex = 0; atomIndex < numAtoms; ++atomIndex) {
        SCIRun::Vector totalForce = pForceElectroReal[atomIndex] + pForceElectroRecip[atomIndex] + pForceNonbonded[atomIndex];
        // pX = X_n; pV = V_n-1/2; we will now calculate A_n
        // --> Force is calculated for position X_n, therefore the acceleration is A_n
        SCIRun::Vector A_n = totalForce*massInv;
        // pV is velocity at time n - 1/2
        pVNew[atomIndex] = pV[atomIndex] + 0.5 * (A_n) * delT; // pVNew is therefore actually V_n
        // Calculate velocity related things here, based on pVNew;  N?T integration temperature determination goes here
        // --> This may eventually be the end of this routine to allow for reduction to gather the total temperature for N?T and Isokinetic integrators
        // -->  For now we simply integrate again to get to V_n+1/2
        pVNew[atomIndex] = pVNew[atomIndex] + 0.5 * (A_n) * delT; // pVNew = V_n+1/2
        // pXNew = X_n+1
        pXNew[atomIndex] = pX[atomIndex] + pVNew[atomIndex] * delT;
//        if (md_dbg.active()) {
//          cerrLock.lock();
//          std::cout << "PatchID: " << std::setw(4) << patch->getID() << std::setw(6);
//          std::cout << "ParticleID: " << std::setw(6) << pidsnew[idx] << std::setw(6);
//          std::cout << "New Position: [";
//          std::cout << std::setw(10) << std::setprecision(6) << pxnew[idx].x();
//          std::cout << std::setw(10) << std::setprecision(6) << pxnew[idx].y();
//          std::cout << std::setprecision(6) << pxnew[idx].z() << std::setw(4) << "]";
//          std::cout << std::endl;
//          cerrLock.unlock();
      } // end Atom Loop

      new_dw->deleteParticles(delset);

    }  // end materials loop

  }  // end patch loop

  //d_coordinate->clearCellChanged();
}

void MD::createBasePermanentParticleState() {
  // The base particle state which must be tracked when particles move across patch boundaries are
  //   the position and velocity.  Everything else bears some dependence on the form of the forcefield,
  //   electrostatic interaction, and valence interactions employed.
  //
  // Note that position is registered elsewhere since it is the position variable.

  d_particleState.push_back(d_label->global->pID);
  d_particleState.push_back(d_label->global->pV);

  d_particleState_preReloc.push_back(d_label->global->pID_preReloc);
  d_particleState_preReloc.push_back(d_label->global->pV_preReloc);
}

void MD::registerPermanentParticleState()
{
  // register the particle states with the shared SimulationState for persistence across timesteps
  d_sharedState->d_particleState_preReloc.push_back(d_particleState_preReloc);
  d_sharedState->d_particleState.push_back(d_particleState);
}

//void MD::extractCoordinates()
//{
//  std::ifstream inputFile;
//  inputFile.open(d_coordinateFile.c_str());
//  if (!inputFile.is_open()) {
//    std::string message = "\tCannot open input file: " + d_coordinateFile;
//    throw ProblemSetupException(message, __FILE__, __LINE__);
//  }
//
//  // do file IO to extract atom coordinates and charge
//  std::string line;
//  unsigned int numRead;
//  unsigned int numAtoms = d_system->getNumAtoms();
//  for (unsigned int i = 0; i < numAtoms; ++i) {
//    // get the atom coordinates
//    getline(inputFile, line);
//    double x, y, z;
//    double charge;
//    numRead = sscanf(line.c_str(), "%lf %lf %lf %lf", &x, &y, &z, &charge);
//    if (numRead != 4) {
//      std::string message = "\tMalformed input file. Should have [x,y,z] coordinates and [charge] per line: ";
//      throw ProblemSetupException(message, __FILE__, __LINE__);
//    }
//
//    //FIXME This is hacky!! Fix for generic case of wrapping arbitrary coordinates into arbitrary unit cells using
//    //  reduced coordinate transformation!  -- JBH 5/9/13
//    Vector box = d_system->getBox();
//    if (x < 0) { x += box.x(); }
//    if (y < 0) { y += box.y(); }
//    if (z < 0) { z += box.z(); }
//
//    if (x >= box.x()) { x -= box.x(); }
//    if (y >= box.y()) { y -= box.y(); }
//    if (z >= box.z()) { z -= box.z(); }
//
//    Atom atom(Point(x, y, z), charge);
//
//    d_atomList.push_back(atom);
//  }
//  inputFile.close();
//}
