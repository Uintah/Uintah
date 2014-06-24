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

#include <CCA/Components/MD/Electrostatics/ElectrostaticsFactory.h>
#include <CCA/Components/MD/Electrostatics/SPME/SPME.h>

#include <CCA/Components/MD/CoordinateSystems/CoordinateSystemFactory.h>

#include <CCA/Components/MD/Forcefields/ForcefieldFactory.h>
#include <CCA/Components/MD/Forcefields/TwoBodyForceField.h>

#include <CCA/Components/MD/Nonbonded/NonbondedFactory.h>
#include <CCA/Components/MD/Nonbonded/TwoBodyDeterministic.h>

#include <CCA/Components/MD/atomMap.h>
#include <CCA/Components/MD/atomFactory.h>



using namespace Uintah;

extern SCIRun::Mutex cerrLock;

static DebugStream md_dbg("MDDebug", false);
static DebugStream md_cout("MDCout", false);
static DebugStream particleDebug("MDParticleVariableDebug", false);
static DebugStream electrostaticDebug("MDElectrostaticDebug", false);
static DebugStream mdFlowDebug("MDLogicFlowDebug", false);


MD::MD(const ProcessorGroup* myworld) :
    UintahParallelComponent(myworld)
{
  d_label = scinew MDLabel();
}

MD::~MD()
{
  if (d_label) {
    delete d_label;
  }
  if (d_system) {
    delete d_system;
  }
  if (d_forcefield) {
    delete d_forcefield;
  }
  if (d_nonbonded) {
    delete d_nonbonded;
  }
  if (d_electrostatics) {
    delete d_electrostatics;
  }
  if (d_coordinate) {
    delete d_coordinate;
  }
}

void MD::problemSetup(const ProblemSpecP&   params,
                      const ProblemSpecP&   restart_prob_spec,
                      GridP&                grid,
                      SimulationStateP&     shared_state)
{
  const std::string flowLocation = "MD::problemSetup | ";
  const std::string particleLocation = "MD::problemSetup P ";
  const std::string electrostaticLocation = "MD::problemSetup E ";

  printTask(md_cout, flowLocation);

//-----> Set up components inherited from Uintah
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
  dynamic_cast<Scheduler*> (getPort("scheduler"))
                            ->setPositionVar(d_label->global->pX);

//------> Set up components inherent to MD
  // create the coordinate system interface
  d_coordinate = CoordinateSystemFactory::create(params, shared_state, grid);
  d_coordinate->markCellChanged();

  // Parse the forcefield
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

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "Created forcefield object: \""
                << d_forcefield->getForcefieldDescriptor()
                << "\"" << std::endl;
  }

  // create and populate the MD System object
  d_system = scinew MDSystem(params, grid, tempFF);

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

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "Created nonbonded object: \""
                << d_nonbonded->getNonbondedType()
                << "\""
                << std::endl;
  }

  // create the Electrostatics object via factory method
  //electrostaticsModel elecCapability = d_forcefield->getElectrostaticsCapability();
  d_electrostatics = ElectrostaticsFactory::create(params, d_coordinate);
  if (d_electrostatics->getType() == Electrostatics::SPME) {
    if (mdFlowDebug.active()) {
      mdFlowDebug << flowLocation
                  << "Created electrostatics object: \"SPME\""
                  << std::endl;
    }
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
  if (electrostaticDebug.active()) {
    electrostaticDebug << electrostaticLocation;
    if (d_electrostatics->isPolarizable()) {
      electrostaticDebug << "Electrostatic model IS polarizable.";
    }
    else
    {
      electrostaticDebug << "Electrostatic model IS NOT polarizable.";
    }
    electrostaticDebug << std::endl;
  }

  // Add labels from our forcefield (nonbonded)
  MDSubcomponent* d_electrostaticInterface;
  d_electrostaticInterface  = dynamic_cast<MDSubcomponent*> (d_electrostatics);

  MDSubcomponent* d_nonbondedInterface;
  d_nonbondedInterface      = dynamic_cast<MDSubcomponent*> (d_nonbonded);

//  MDSubcomponent* d_integratorInterface    = dynamic_cast<MDSubcomponent*> (d_integrator);
//  MDSubcomponent* d_valenceInterface       = dynamic_cast<MDSubcomponent*> (d_valence);

// Register the general labels that all MD simulations will use
   createBasePermanentParticleState();
   // And then add the labels that each created subcomponent will require
   d_electrostaticInterface
     ->registerRequiredParticleStates(d_particleState,
                                      d_particleState_preReloc,
                                      d_label);
   d_nonbondedInterface
     ->registerRequiredParticleStates(d_particleState,
                                      d_particleState_preReloc,
                                      d_label);
   // NYI:  d_integrator->registerRequiredParticleState(d_particleState, d_particleState_preReloc, d_label);

   // We must wait to register our atom (material) types until the
   // subcomponents have provided the per-particle labels
   //
   // For now we're assuming all atom types have the same tracked states.
   size_t stateSize, preRelocSize;
   stateSize = d_particleState.size();
   preRelocSize = d_particleState_preReloc.size();
   if (stateSize != preRelocSize)
   {
     std::cerr << "ERROR:  Mismatch in number of per particle variable labels." << std::endl;
   }

   if (particleDebug.active()) {
     particleDebug << particleLocation
                   << "  Registered particle variables: "
                   << std::endl;
     for (size_t index = 0; index < stateSize; ++index)
     {
       particleDebug << particleLocation
                     << "  " << d_particleState[index]->getName() << "/"
                     << d_particleState_preReloc[index]->getName()
                     << std::endl;
     }
   }
   d_forcefield->registerAtomTypes(d_particleState,
                                   d_particleState_preReloc,
                                   d_label,
                                   d_sharedState);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END" << std::endl;
  }
}

void MD::scheduleInitialize(const LevelP& level,
                            SchedulerP& sched)
{
  const std::string flowLocation = "MD::scheduleInitialize | ";
  printSchedule(level, md_cout, flowLocation);

  /*
   * Note there are multiple tasks scheduled here. All three need only ever happen once.
   *
   * 1.) MD::initialize
   * 2.) Nonbonded::initialize
   * 3.) SPME::initialize
   */

  // Get list of MD materials for scheduling
  const MaterialSet*    materials       =   d_sharedState->allMDMaterials();
  LoadBalancer*         loadBal         =   sched->getLoadBalancer();
  const PatchSet*       perProcPatches  =
                            loadBal->getPerProcessorPatchSet(level);

  Task* task = scinew Task("MD::initialize", this, &MD::initialize);

  // Initialize will load position, velocity, and ID tags
  task->computes(d_label->global->pX);
  task->computes(d_label->global->pV);
  task->computes(d_label->global->pID);

  //FIXME:  Do we still need this here?
  task->computes(d_label->electrostatic->dSubschedulerDependency);

  // FIXME -- Original, no longer correct?
  sched->addTask(task, level->eachPatch(), materials);
  //sched->addTask(task, perProcPatches, materials);

  // Nonbonded initialization - OncePerProc, during initial (0th) timestep.
  // The required pXlabel is available to this OncePerProc task in the new_dw from the computes above
  scheduleNonbondedInitialize(sched, perProcPatches, materials, level);
//  scheduleNonbondedInitialize()

  // Nonbonded initialization - OncePerProc, during initial (0th) timestep.
  //   This OncePerProc task requires nothing
  scheduleElectrostaticsInitialize(sched, perProcPatches, materials, level);
  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }
}

void MD::scheduleComputeStableTimestep(const LevelP& level,
                                       SchedulerP& sched)
{
  const std::string flowLocation = "MD::scheduleComputeStableTimestep | ";
  printSchedule(level, md_cout, flowLocation);

  Task* task = scinew Task("MD::computeStableTimestep", this, &MD::computeStableTimestep);

  task->requires(Task::NewDW,
                 d_label->nonbonded->rNonbondedEnergy);
  task->requires(Task::NewDW,
                 d_label->electrostatic->rElectrostaticInverseEnergy);
  task->requires(Task::NewDW,
                 d_label->electrostatic->rElectrostaticRealEnergy);

  // We only -need- stress tensors if we're doing NPT
  if ( NPT == d_system->getEnsemble()) {
    task->requires(Task::NewDW,
                   d_label->nonbonded->rNonbondedStress);
    task->requires(Task::NewDW,
                   d_label->electrostatic->rElectrostaticInverseStress);
    task->requires(Task::NewDW,
                   d_label->electrostatic->rElectrostaticRealStress);
    if ( d_electrostatics->isPolarizable() ) {
      task->requires(Task::NewDW,
                     d_label->electrostatic->rElectrostaticInverseStressDipole);
    }
  }

  task->computes(d_sharedState->get_delt_label(),
                 level.get_rep());

  task->setType(Task::OncePerProc);

  LoadBalancer* loadBal = sched->getLoadBalancer();
  const PatchSet* perProcPatches = loadBal->getPerProcessorPatchSet(level);

  sched->addTask(task,
                 perProcPatches,
                 d_sharedState->allMaterials());

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }

}

void MD::scheduleTimeAdvance(const LevelP& level,
                             SchedulerP& sched)
{
  const std::string flowLocation = "MD::scheduleTimeAdvance | ";
  printSchedule(level, md_cout, flowLocation);

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

  sched->scheduleParticleRelocation(level,
                                    d_label->global->pX_preReloc,
                                    d_sharedState->d_particleState_preReloc,
                                    d_label->global->pX,
                                    d_sharedState->d_particleState,
                                    d_label->global->pID,
                                    matls);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }

//  sched->scheduleParticleRelocation(level, d_label->pXLabel_preReloc, d_sharedState->d_particleState_preReloc, d_label->pXLabel,
//                                    d_sharedState->d_particleState, d_label->pParticleIDLabel, matls, 1);
}

// Schedule indirect calls to subcomponents
// Note:    Taskgraph can't schedule direct object reference tasks, since
//          we don't know the explicit form of the subcomponents at compile
//          time.
void MD::scheduleNonbondedInitialize(SchedulerP&        sched,
                                     const PatchSet*    perProcPatches,
                                     const MaterialSet* matls,
                                     const LevelP&      level)
{
  const std::string flowLocation = "MD::scheduleNonbondedInitialize | ";
  printSchedule(perProcPatches, md_cout, flowLocation);

  Task* task = scinew Task("MD::nonbondedInitialize",
                           this,
                           &MD::nonbondedInitialize);

  MDSubcomponent* d_nonbondedInterface =
                      dynamic_cast<MDSubcomponent*> (d_nonbonded);

  d_nonbondedInterface->addInitializeRequirements(task, d_label);
  d_nonbondedInterface->addInitializeComputes(task, d_label);

  task->setType(Task::OncePerProc);
  sched->addTask(task, perProcPatches, matls);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }

}

void MD::scheduleNonbondedSetup(SchedulerP&         sched,
                                const PatchSet*     patches,
                                const MaterialSet*  matls,
                                const LevelP&       level)
{
  const std::string flowLocation = "MD::scheduleNonbondedSetup | ";
  printSchedule(patches, md_cout, flowLocation);

  Task* task = scinew Task("MD::nonbondedSetup", this, &MD::nonbondedSetup);

  MDSubcomponent* d_nonbondedInterface =
                      dynamic_cast<MDSubcomponent*> (d_nonbonded);

  d_nonbondedInterface->addSetupRequirements(task, d_label);
  d_nonbondedInterface->addSetupComputes(task, d_label);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }

}

void MD::scheduleNonbondedCalculate(SchedulerP&         sched,
                                    const PatchSet*     patches,
                                    const MaterialSet*  matls,
                                    const LevelP&       level)
{
  const std::string flowLocation = "MD::scheduleNonbondedCalculate | ";
  printSchedule(patches, md_cout, flowLocation);

  Task* task = scinew Task("MD::nonbondedCalculate",
                           this,
                           &MD::nonbondedCalculate);

  MDSubcomponent* d_nonbondedInterface =
                      dynamic_cast<MDSubcomponent*> (d_nonbonded);

  d_nonbondedInterface->addCalculateRequirements(task,d_label);
  d_nonbondedInterface->addCalculateComputes(task,d_label);

  sched->addTask(task, patches, matls);
  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }

}

void MD::scheduleNonbondedFinalize(SchedulerP&          sched,
                                   const PatchSet*      patches,
                                   const MaterialSet*   matls,
                                   const LevelP&        level)
{
  const std::string flowLocation = "MD::scheduleNonbondedFinalize | ";
  printSchedule(patches, md_cout, "MD::scheduleNonbondedFinalize");

  Task* task = scinew Task("MD::nonbondedFinalize",
                           this,
                           &MD::nonbondedFinalize);

  MDSubcomponent* d_nonbondedInterface =
                      dynamic_cast<MDSubcomponent*> (d_nonbonded);

  d_nonbondedInterface->addFinalizeRequirements(task, d_label);
  d_nonbondedInterface->addFinalizeComputes(task, d_label);

  sched->addTask(task, patches, matls);
  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }

}

void MD::scheduleElectrostaticsInitialize(SchedulerP& sched,
                                          const PatchSet* perProcPatches,
                                          const MaterialSet* matls,
                                          const LevelP& level)
{
  const std::string flowLocation = "MD::scheduleElectrostaticsInitialize | ";
  printSchedule(perProcPatches, md_cout, flowLocation);

  // initialize electrostatics instance; if we're doing electrostatics
  if (d_electrostatics->getType() != Electrostatics::NONE) {

    Task* task = scinew Task("MD::electrostaticsInitialize",
                             this,
                             &MD::electrostaticsInitialize);

    // cast electrostatics to a subcomponent interface
    MDSubcomponent* d_electroInterface =
                        dynamic_cast<MDSubcomponent*> (d_electrostatics);

    d_electroInterface->addInitializeRequirements(task, d_label);
    d_electroInterface->addInitializeComputes(task, d_label);

    task->setType(Task::OncePerProc);
    sched->addTask(task, perProcPatches, matls);
  }
  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }

}

void MD::scheduleElectrostaticsSetup(SchedulerP& sched,
                                     const PatchSet* patches,
                                     const MaterialSet* matls,
                                     const LevelP& level)
{
  const std::string flowLocation = "MD::scheduleElectrostaticsSetup | ";
  printSchedule(patches, md_cout, flowLocation);

  if (d_electrostatics->getType() != Electrostatics::NONE) {
    Task* task = scinew Task("MD::electrostaticsSetup", this, &MD::electrostaticsSetup);

    // cast electrostatics to a subcomponent interface
    MDSubcomponent* d_electroInterface = dynamic_cast<MDSubcomponent*> (d_electrostatics);

    d_electroInterface->addSetupRequirements(task, d_label);
    d_electroInterface->addSetupComputes(task, d_label);
    sched->addTask(task, patches, matls);

  }
  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }

}

void MD::scheduleElectrostaticsCalculate(SchedulerP& sched,
                                         const PatchSet* patches,
                                         const MaterialSet* matls,
                                         const LevelP& level)
{
  const std::string flowLocation = "MD::scheduleElectrostaticsCalculate | ";
  printSchedule(patches, md_cout, flowLocation);

  if (d_electrostatics->getType() != Electrostatics::NONE) {
    Task* task = scinew Task("electrostaticsCalculate", this, &MD::electrostaticsCalculate, level);

    task->requires(Task::OldDW, d_label->global->pX, Ghost::AroundNodes, d_electrostatics->requiredGhostCells());
    task->requires(Task::OldDW, d_label->global->pID, Ghost::AroundNodes, d_electrostatics->requiredGhostCells());

    // Need delT for the subscheduler timestep
    task->requires(Task::OldDW, d_sharedState->get_delt_label());

    MDSubcomponent* d_electroInterface = dynamic_cast<MDSubcomponent*> (d_electrostatics);

    d_electroInterface->addCalculateRequirements(task, d_label);
    d_electroInterface->addCalculateComputes(task, d_label);

    task->hasSubScheduler(true);
    task->setType(Task::OncePerProc);

    LoadBalancer* loadBal = sched->getLoadBalancer();
    const PatchSet* perProcPatches = loadBal->getPerProcessorPatchSet(level);

    sched->addTask(task, perProcPatches, matls);
  }
  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }

}

void MD::scheduleElectrostaticsFinalize(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls,
                                        const LevelP& level)
{
  const std::string flowLocation = "MD::scheduleElectrostaticsFinalize | ";
  printSchedule(patches, md_cout, flowLocation);

  if (d_electrostatics->getType() != Electrostatics::NONE) {
    Task* task = scinew Task("MD::electrostaticsFinalize", this, &MD::electrostaticsFinalize);

    MDSubcomponent* d_electroInterface = dynamic_cast<MDSubcomponent*> (d_electrostatics);
    d_electroInterface->addFinalizeRequirements(task, d_label);
    d_electroInterface->addFinalizeComputes(task, d_label);

    sched->addTask(task, patches, matls);
  }

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }

}

void MD::scheduleUpdatePosition(SchedulerP& sched,
                                const PatchSet* patches,
                                const MaterialSet* matls,
                                const LevelP& level)
{
  const std::string flowLocation = "MD::scheduleUpdatePosition | ";
  printSchedule(patches, md_cout, flowLocation);

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
  task->computes(d_label->global->pV_preReloc);

//  task->computes(d_label->pXLabel_preReloc);
//  task->computes(d_label->pVelocityLabel_preReloc);
//  task->modifies(d_lb->pNonbondedForceLabel_preReloc);
//  task->modifies(d_lb->pElectrostaticsForceLabel_preReloc);
//  task->computes(d_lb->pAccelLabel_preReloc);
//  task->computes(d_lb->pMassLabel_preReloc);
//  task->computes(d_lb->pParticleIDLabel_preReloc);

  sched->addTask(task, patches, matls);
  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }

}

void MD::initialize(const ProcessorGroup*   pg,
                    const PatchSubset*      perProcPatches,
                    const MaterialSubset*   matls,
                    DataWarehouse*       /* oldDW */,
                    DataWarehouse*          newDW)
{
  const std::string location = "MD::initialize";
  const std::string flowLocation = location + " | ";
  const std::string particleLocation = location + " P ";
  printTask(perProcPatches, md_cout, location);

  Matrix3   systemInverseCell = d_coordinate->getInverseCell();
  IntVector totalSystemExtent = d_coordinate->getCellExtent();

  SCIRun::Vector inverseExtentVector;
  inverseExtentVector[0]=1.0/static_cast<double> (totalSystemExtent[0]);
  inverseExtentVector[1]=1.0/static_cast<double> (totalSystemExtent[1]);
  inverseExtentVector[2]=1.0/static_cast<double> (totalSystemExtent[2]);

  SCIRun::Vector cellDimensions =
                     d_coordinate->getUnitCell()*inverseExtentVector;

  // Loop through each patch
  size_t numPatches             =   perProcPatches->size();
  size_t numAtomTypes           =   matls->size();

  // Input coordinates from problem spec
  atomMap* parsedCoordinates    =
               atomFactory::create(d_problemSpec, d_sharedState);

  size_t numTypesParsed         =   parsedCoordinates->getNumberAtomTypes();
  size_t numMaterialTypes       =   d_sharedState->getNumMDMatls();

  if (numTypesParsed > numMaterialTypes) {
    std::stringstream errorOut;
    errorOut << " ERROR:  Expected to find " << numMaterialTypes
             << " types of materials in the coordinate file."
             << std::endl
             << "\tHowever, the coordinate file parsed " << numTypesParsed
             << std::endl;
    throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
  }

  for (size_t matlIndex = 0; matlIndex < numAtomTypes; ++matlIndex) {
    std::string materialLabel   =
                    d_sharedState->getMDMaterial(matlIndex)->getMaterialLabel();

    std::vector<atomData*>* currAtomList    =
                                parsedCoordinates->getAtomList(materialLabel);

    size_t numAtoms                         =   currAtomList->size();
    d_system->registerAtomCount(numAtoms,matlIndex);
  }
  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "  Constructed atom map."
                << std::endl;
  }

  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    // Loop over perProcPatches
    const Patch*        currPatch           =   perProcPatches->get(patchIndex);
    SCIRun::IntVector   lowCellBoundary     =   currPatch->getCellLowIndex();
    SCIRun::IntVector   highCellBoundary    =   currPatch->getCellHighIndex();

    (const_cast<Patch*> (currPatch))->getLevel(true)->setdCell(cellDimensions);

    for (size_t localType = 0; localType < numAtomTypes; ++localType) {
      // Loop over materials
      size_t        globalID    = matls->get(localType);
      MDMaterial*   atomType    = d_sharedState->getMDMaterial(globalID);
      std::string   typeLabel   = atomType->getMaterialLabel();

      // Match coordinates to material and extract coordinate list
      std::vector<atomData*>* currAtomList =
                                  parsedCoordinates->getAtomList(typeLabel);

      size_t numAtomsOfType     = parsedCoordinates->getAtomListSize(typeLabel);

      std::vector<Point>            localAtomCoordinates;
      std::vector<size_t>           localAtomID;
      std::vector<SCIRun::Vector>   localAtomVelocity;

//      std::cout << "Checking for location of atoms from pool of "
//                << numAtomsOfType << " atoms with the label "
//                << typeLabel << " with material ID: "
//                << globalID << std::endl;

      for (size_t atomIndex = 0; atomIndex < numAtomsOfType; ++atomIndex) {
        // Loop over all atoms of material
        atomData*   currAtom        = (*currAtomList)[atomIndex];
        Point       currPosition    = currAtom->getPosition();
//        IntVector   currCell        =
//                        currPatch
//                        ->getLevel()
//                          ->getCellIndex(currPosition);

        // Build local atom list for atoms of material in current patch
        bool atomInPatch = currPatch->containsPoint(currPosition);

        if (atomInPatch) { // Atom is on this patch
          size_t currID = currAtom->getID();
          SCIRun::Vector currVelocity = currAtom->getVelocity();

          localAtomCoordinates.push_back(currPosition);
          localAtomID.push_back(currID);
          localAtomVelocity.push_back(currVelocity);
        }
      }

      // Create this patch's particle set for atoms of current material
      size_t            numAtoms = localAtomCoordinates.size();
      ParticleSubset*   currPset =
                        newDW->createParticleSubset(numAtoms,
                                                    globalID,
                                                    currPatch,
                                                    lowCellBoundary,
                                                    highCellBoundary);
      if (particleDebug.active()) {
        particleDebug << particleLocation
                      << "  Created a subset with "
                      << numAtoms
                      << " of type label "
                      << "\""
                      << d_sharedState
                           ->getMDMaterial(globalID)
                             ->getMaterialLabel()
                      << "\""
                      << std::endl;
      }

    // ----> Variables from parsing the input coordinate file
    // --> Position
      ParticleVariable<Point>   pX;
      newDW->allocateAndPut(    pX, d_label->global->pX, currPset);
    // --> Velocity
      ParticleVariable<Vector>  pV;
      newDW->allocateAndPut(    pV, d_label->global->pV, currPset);
    // --> Index
      ParticleVariable<long64>  pID;
      newDW->allocateAndPut(    pID, d_label->global->pID,currPset);

      for (size_t atomIndex = 0; atomIndex < numAtoms; ++atomIndex) {
        // Transfer over currently defined atom data
        pX[atomIndex]    = localAtomCoordinates[atomIndex];
        pV[atomIndex]    = localAtomVelocity[atomIndex];
        pID[atomIndex]   = localAtomID[atomIndex];

//        if (md_dbg.active()) { // Output for debug..
//          cerrLock.lock();
//          std::cout.setf(std::ios_base::showpoint);  // print decimal and trailing zeros
//          std::cout.setf(std::ios_base::left);  // pad after the value
//          std::cout.setf(std::ios_base::uppercase);  // use upper-case scientific notation
//          std::cout << std::setw(10) << " Patch_ID: " << std::setw(4) << currPatch->getID();
//          std::cout << std::setw(14) << " Particle_ID: " << std::setw(4) << pID[atomIndex];
//          std::cout << std::setw(12) << " Position: " << pX[atomIndex];
//          std::cout << std::endl;
//          cerrLock.unlock();
//        }


      }

      CCVariable<int> subSchedulerDependency;
      newDW->allocateAndPut(subSchedulerDependency,
                            d_label->electrostatic->dSubschedulerDependency,
                            globalID,
                            currPatch,
                            Ghost::None,
                            0);
      subSchedulerDependency.initialize(0);

    } // Loop over materials
  } // Loop over patches

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }

}

void MD::computeStableTimestep(const ProcessorGroup*    pg,
                               const PatchSubset*       patches,
                               const MaterialSubset*    matls,
                               DataWarehouse*           oldDW,
                               DataWarehouse*           newDW)
{
  const std::string location = "MD::computeStableTimestep";
  const std::string flowLocation = location + " | ";
  const std::string particleLocation = location + " P ";
  const std::string electrostaticLocation = location + " E ";

  printTask(patches, md_cout, "MD::computeStableTimestep");

  sum_vartype vdwEnergy;
  sum_vartype spmeFourierEnergy;
  matrix_sum spmeFourierStress;
  matrix_sum spmeRealStress;
  matrix_sum spmeFourierStressDipole;

  // This is where we would actually map the correct timestep/taskgraph
  // for a multistep integrator

  newDW->get(vdwEnergy, d_label->nonbonded->rNonbondedEnergy);
  newDW->get(spmeFourierEnergy,
              d_label->electrostatic->rElectrostaticInverseEnergy);

  proc0cout << std::endl;
  proc0cout << "-----------------------------------------------------"           << std::endl;
  proc0cout << "Total Energy = "
            << std::setprecision(16)
            << vdwEnergy
            << std::endl;
  proc0cout << "-----------------------------------------------------"           << std::endl;
  proc0cout << "Fourier Energy = "
            << std::setprecision(16)
            << spmeFourierEnergy
            << std::endl;
  proc0cout << "-----------------------------------------------------"           << std::endl;


  if (NPT == d_system->getEnsemble()) {
    newDW->get(spmeFourierStress,
                d_label->electrostatic->rElectrostaticInverseStress);
    newDW->get(spmeRealStress,
                d_label->electrostatic->rElectrostaticRealStress);
    proc0cout << "Fourier Stress = "
              << std::setprecision(16)
              << spmeFourierStress
              << std::endl;
    proc0cout << "-----------------------------------------------------"           << std::endl;

    if (d_electrostatics->isPolarizable()) {
      newDW->get(spmeFourierStressDipole,
                  d_label->electrostatic->rElectrostaticInverseStressDipole);
    }
  }

  proc0cout << std::endl;

  newDW->put(delt_vartype(1),
              d_sharedState->get_delt_label(),
              getLevel(patches));
  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }

}

void MD::nonbondedInitialize(const ProcessorGroup*  pg,
                             const PatchSubset*     patches,
                             const MaterialSubset*  matls,
                             DataWarehouse*         oldDW,
                             DataWarehouse*         newDW)
{
  const std::string location = "MD::nonbondedInitialize";
  const std::string flowLocation = location + " | ";
  printTask(patches, md_cout, "MD::nonbondedInitialize");

  d_nonbonded->initialize(pg, patches, matls, oldDW, newDW,
                          d_sharedState, d_system, d_label, d_coordinate);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }


}

void MD::nonbondedSetup(const ProcessorGroup*   pg,
                        const PatchSubset*      patches,
                        const MaterialSubset*   matls,
                        DataWarehouse*          oldDW,
                        DataWarehouse*          newDW)
{
  const std::string location = "MD::nonbondedSetup";
  const std::string flowLocation = location + " | ";
  printTask(patches, md_cout, location);

  d_nonbonded->setup(pg, patches, matls, oldDW, newDW,
                     d_sharedState, d_system, d_label, d_coordinate);

//  if (d_coordinate->queryCellChanged()) {
//    d_nonbonded->setup(pg, patches, matls, oldDW, newDW,
//                       d_sharedState, d_system, d_label, d_coordinate);
//  }

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }


}

void MD::nonbondedCalculate(const ProcessorGroup*   pg,
                            const PatchSubset*      patches,
                            const MaterialSubset*   matls,
                            DataWarehouse*          oldDW,
                            DataWarehouse*          newDW)
{
  const std::string location = "MD::nonbondeCalculate";
  const std::string flowLocation = location + " | ";
  printTask(patches, md_cout, "MD::nonbondedCalculate");

  d_nonbonded->calculate(pg, patches, matls, oldDW, newDW,
                         d_sharedState, d_system, d_label, d_coordinate);
  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }
}

void MD::nonbondedFinalize(const ProcessorGroup*    pg,
                           const PatchSubset*       patches,
                           const MaterialSubset*    matls,
                           DataWarehouse*           oldDW,
                           DataWarehouse*           newDW)
{
  const std::string location = "MD::nonbondedFinalize";
  const std::string flowLocation = location + " | ";
  printTask(patches, md_cout, location);

  d_nonbonded->finalize(pg, patches, matls, oldDW, newDW,
                        d_sharedState, d_system, d_label, d_coordinate);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }
}

void MD::electrostaticsInitialize(const ProcessorGroup* pg,
                                  const PatchSubset*    patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse*        oldDW,
                                  DataWarehouse*        newDW)
{
  const std::string location = "MD::electrostaticsInitialize";
  const std::string flowLocation = location + " | ";
  printTask(patches, md_cout, location);

  d_electrostatics->initialize(pg, patches, matls, oldDW, newDW,
                               &d_sharedState, d_system, d_label, d_coordinate);
  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }
}

void MD::electrostaticsSetup(const ProcessorGroup*  pg,
                             const PatchSubset*     patches,
                             const MaterialSubset*  matls,
                             DataWarehouse*         oldDW,
                             DataWarehouse*         newDW)
{
  const std::string location = "MD::electrostaticsSetup";
  const std::string flowLocation = location + " | ";
  printTask(patches, md_cout, "MD::electrostaticsSetup");

  d_electrostatics->setup(pg, patches, matls, oldDW, newDW,
                          &d_sharedState, d_system, d_label, d_coordinate);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }
}

void MD::electrostaticsCalculate(const ProcessorGroup*  pg,
                                 const PatchSubset*     perProcPatches,
                                 const MaterialSubset*  matls,
                                 DataWarehouse*         parentOldDW,
                                 DataWarehouse*         parentNewDW,
                                 const LevelP           level)
{
  const std::string location = "MD::electrostaticsCalculate";
  const std::string flowLocation = location + " | ";
  const std::string electrostaticLocation = location + " E ";
  printTask(perProcPatches, md_cout, location);

  // Copy del_t to the subscheduler
  delt_vartype dt;
  DataWarehouse* subNewDW = d_electrostaticSubscheduler->get_dw(3);
  parentOldDW->get(dt,
                   d_sharedState->get_delt_label(),
                   level.get_rep());
  subNewDW->put(dt,
                d_sharedState->get_delt_label(),
                level.get_rep());

  if (electrostaticDebug.active()) {
    electrostaticDebug << electrostaticLocation
                       << "  Copied delT to the electrostatic subscheduler."
                       << std::endl;
  }

  d_electrostatics->calculate(pg,perProcPatches,matls,parentOldDW,parentNewDW,
                              &d_sharedState, d_system, d_label, d_coordinate,
                              d_electrostaticSubscheduler, level);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }
}

void MD::electrostaticsFinalize(const ProcessorGroup*   pg,
                                const PatchSubset*      patches,
                                const MaterialSubset*   matls,
                                DataWarehouse*          oldDW,
                                DataWarehouse*          newDW)
{
  const std::string location = "MD::electrostaticsFinalize";
  const std::string flowLocation = location + " | ";
  printTask(patches, md_cout, location);

  d_electrostatics->finalize(pg, patches, matls, oldDW, newDW,
                             &d_sharedState, d_system, d_label, d_coordinate);
  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }
}

void MD::updatePosition(const ProcessorGroup*   pg,
                        const PatchSubset*      patches,
                        const MaterialSubset*   matls,
                        DataWarehouse*          oldDW,
                        DataWarehouse*          newDW)
{
  const std::string location = "MD::updatePosition";
  const std::string flowLocation = location + " | ";
  const std::string particleLocation = location + " P ";
  printTask(patches, md_cout, location);

  // loop through all patches
  unsigned int numPatches = patches->size();

  for (unsigned int p = 0; p < numPatches; ++p) {
    const Patch* patch = patches->get(p);
    unsigned int numMatls = matls->size();

    for (unsigned int m = 0; m < numMatls; ++m) {
      int matl = matls->get(m);
      double massInv = 1.0/(d_sharedState->getMDMaterial(matl)->getMass());

      ParticleSubset* pset = oldDW->getParticleSubset(matl, patch);
      ParticleSubset* delset = scinew ParticleSubset(0, matl, patch);

      // Variables required in order to integrate
      // --> Position at last time step
      constParticleVariable<Point> pX;
      oldDW->get(pX,
                  d_label->global->pX, pset);

      // --> Velocity at last time step (velocity verlet algorithm)
      constParticleVariable<SCIRun::Vector> pV;
      oldDW->get(pV,
                  d_label->global->pV, pset);
      constParticleVariable<long64> pID;
      oldDW->get(pID,
                  d_label->global->pID, pset);

//      // --> Acceleration at last time step (velocity verlet algorithm)
//      constParticleVariable<SCIRun::Vector> pA;
//      old_dw->get(pA, d_lb->pAccelLabel, pset);
      // --> Forces for this time step

      constParticleVariable<SCIRun::Vector> pForceElectroReal;
      constParticleVariable<SCIRun::Vector> pForceElectroRecip;
      constParticleVariable<SCIRun::Vector> pForceNonbonded;
      newDW->get(pForceElectroReal,
                  d_label->electrostatic->pF_electroReal_preReloc,
                  pset);
      newDW->get(pForceElectroRecip,
                  d_label->electrostatic->pF_electroInverse_preReloc,
                  pset);
      newDW->get(pForceNonbonded,
                  d_label->nonbonded->pF_nonbonded_preReloc,
                  pset);

      // Variables which the integrator calculates
      // --> New position
      ParticleVariable<Point> pXNew;
      newDW->allocateAndPut(pXNew,
                             d_label->global->pX_preReloc,
                             pset);
      // --> New velocity
      ParticleVariable<SCIRun::Vector> pVNew;
      newDW->allocateAndPut(pVNew,
                             d_label->global->pV_preReloc,
                             pset);
      ParticleVariable<long64> pIDNew;
      newDW->allocateAndPut(pIDNew,
                             d_label->global->pID_preReloc,
                             pset);

      // get delT
      delt_vartype delT;
      oldDW->get(delT,
                  d_sharedState->get_delt_label(),
                  getLevel(patches));

      size_t numAtoms = pset->numParticles();

      // Loop over the atom set
      for (size_t atomIndex = 0; atomIndex < numAtoms; ++atomIndex) {
        SCIRun::Vector totalForce;
        totalForce = pForceElectroReal[atomIndex] +
                     pForceElectroRecip[atomIndex] +
                     pForceNonbonded[atomIndex];
        // pX = X_n; pV = V_n-1/2; we will now calculate A_n
        // --> Force is calculated for position X_n, therefore the acceleration is A_n
        SCIRun::Vector A_n = totalForce*massInv;

        // pV is velocity at time n - 1/2
        pVNew[atomIndex] = pV[atomIndex] + 0.5 * (A_n) * delT;
        // pVNew is therefore actually V_n

        // Calculate velocity related things here, based on pVNew;
        // NPT integration temperature determination goes here

        // --> This may eventually be the end of this routine to allow for
        //     reduction to gather the total temperature for NPT and
        //     Isokinetic integrators


        // -->  For now we simply integrate again to get to V_n+1/2
        pVNew[atomIndex] = pVNew[atomIndex] + 0.5 * (A_n) * delT;
        // pVNew = V_n+1/2

        // pXNew = X_n+1
        pXNew[atomIndex] = pX[atomIndex] + pVNew[atomIndex] * delT;

        // Simply copy over particle IDs; they never change
        pIDNew[atomIndex]= pID[atomIndex];
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

      newDW->deleteParticles(delset);

    }  // end materials loop

  }  // end patch loop

  //d_coordinate->clearCellChanged();
  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }
}

void MD::createBasePermanentParticleState() {
  // The base particle state which must be tracked when particles move across patch boundaries are
  //   the position and velocity.  Everything else bears some dependence on the form of the forcefield,
  //   electrostatic interaction, and valence interactions employed.
  //
  // Note that position is registered elsewhere since it is the position variable.

  const std::string location = "MD::createBasePermanentParticleState";
  const std::string flowLocation = location + " | ";

  d_particleState.push_back(d_label->global->pID);
  d_particleState.push_back(d_label->global->pV);

  d_particleState_preReloc.push_back(d_label->global->pID_preReloc);
  d_particleState_preReloc.push_back(d_label->global->pV_preReloc);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }
}
