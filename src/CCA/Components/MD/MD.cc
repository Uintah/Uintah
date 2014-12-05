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
#include <Core/Math/Matrix3.h>

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

#include <CCA/Components/MD/Integrators/IntegratorFactory.h>
#include <CCA/Components/MD/Integrators/velocityVerlet/velocityVerlet.h>

#include <CCA/Components/MD/atomMap.h>
#include <CCA/Components/MD/atomFactory.h>



using namespace Uintah;

//extern SCIRun::Mutex cerrLock;

//static DebugStream md_dbg("MDDebug", false);
//static DebugStream md_cout("MDCout", false);
//static DebugStream particleDebug("MDParticleVariableDebug", false);
//static DebugStream electrostaticDebug("MDElectrostaticDebug", false);
//static DebugStream mdFlowDebug("MDLogicFlowDebug", false);
//
//#define isPrincipleThread (   Uintah::Parallel::getMPIRank() == 0              \
//                           &&(                                                 \
//                                (  Uintah::Parallel::getNumThreads() > 1       \
//                                 && SCIRun::Thread::self()->myid() == 0 )      \
//                              ||(  Uintah::Parallel::getNumThreads() <= 1 )    \
//                             )                                                 \
//                          )

MD::MD(const ProcessorGroup* myworld) :
    UintahParallelComponent(myworld)
{
  d_label = scinew MDLabel();
  d_referenceStored = false;
  d_firstIntegration = true;
  d_secondIntegration = false;
  d_KineticBase = d_PotentialBase = d_referenceEnergy = 0.0;

//  d_isoKineticMult = 1.0;
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
  Integrator* tempIntegrator = IntegratorFactory::create(params,
                                                         d_system,
                                                         d_sharedState->get_delt_label());

  d_integrator = tempIntegrator;
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
  MDSubcomponent* d_integratorInterface;
  d_integratorInterface     = dynamic_cast<MDSubcomponent*> (d_integrator);

  MDSubcomponent* d_electrostaticInterface;
  d_electrostaticInterface  = dynamic_cast<MDSubcomponent*> (d_electrostatics);

  MDSubcomponent* d_nonbondedInterface;
  d_nonbondedInterface      = dynamic_cast<MDSubcomponent*> (d_nonbonded);

//  MDSubcomponent* d_integratorInterface    = dynamic_cast<MDSubcomponent*> (d_integrator);
//  MDSubcomponent* d_valenceInterface       = dynamic_cast<MDSubcomponent*> (d_valence);

// Register the general labels that all MD simulations will use
   createBasePermanentParticleState();
   // And then add the labels that each created subcomponent will require
   d_integratorInterface
    ->registerRequiredParticleStates(d_particleState,
                                     d_particleState_preReloc,
                                     d_label);
   d_electrostaticInterface
     ->registerRequiredParticleStates(d_particleState,
                                      d_particleState_preReloc,
                                      d_label);
   d_nonbondedInterface
     ->registerRequiredParticleStates(d_particleState,
                                      d_particleState_preReloc,
                                      d_label);

   // We must wait to register our atom (material) types until the
   // subcomponents have provided the per-particle labels
   //
   // For now we're assuming all atom types have the same tracked states.
   size_t stateSize, preRelocSize;
   stateSize = d_particleState.size();
   preRelocSize = d_particleState_preReloc.size();
   if (stateSize != preRelocSize)
   {
     std::cerr
       << "ERROR:  Mismatch in number of per particle variable labels."
       << std::endl;
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

void MD::scheduleInitialize(const LevelP&   level,
                            SchedulerP&     sched)
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

  task->computes(d_label->global->rKineticEnergy);
  task->computes(d_label->global->rKineticStress);
  task->computes(d_label->global->rTotalMomentum);
  task->computes(d_label->global->rTotalMass);

  // FIXME -- Original, no longer correct?
  //sched->addTask(task, level->eachPatch(), materials);
  sched->addTask(task, perProcPatches, materials);

  // Nonbonded initialization - OncePerProc, during initial (0th) timestep.
  // The required pXlabel is available to this OncePerProc task in the newDW
  // from the computes above
  scheduleNonbondedInitialize(sched, perProcPatches, materials, level);

  //   This OncePerProc task requires nothing
  scheduleElectrostaticsInitialize(sched, perProcPatches, materials, level);

  scheduleIntegratorInitialize(sched, perProcPatches, materials, level);

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

  scheduleOutputStatistics(sched, patches, matls, level);

  scheduleNonbondedSetup(sched, patches, matls, level);

  scheduleElectrostaticsSetup(sched, patches, matls, level);

  scheduleIntegratorSetup(sched, patches, matls, level);

  scheduleNonbondedCalculate(sched, patches, matls, level);

  scheduleElectrostaticsCalculate(sched, patches, matls, level);

  scheduleIntegratorCalculate(sched, patches, matls, level);

  // Should probably move the Finalizes into the appropriate clean-up step on MD.  (Destructor?)
  //   and appropriately modify the finalize routines.  !FIXME
  scheduleNonbondedFinalize(sched, patches, matls, level);

  scheduleElectrostaticsFinalize(sched, patches, matls, level);

  scheduleIntegratorFinalize(sched, patches, matls, level);

//  scheduleKineticCalculations(sched, patches, matls, level);

//  scheduleUpdatePosition(sched, patches, matls, level);

//  scheduleNewUpdatePosition(sched, patches, matls, level);

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

void MD::scheduleKineticCalculations(      SchedulerP&  baseScheduler,
                                     const PatchSet*    patches,
                                     const MaterialSet* atomTypes,
                                     const LevelP&      level)
{
  const std::string flowLocation = "MD::scheduleKineticCalculations | ";
  printSchedule(patches, md_cout, flowLocation);

  Task* task = scinew Task("MD::calculateKineticEnergy",
                           this,
                           &MD::calculateKineticEnergy);

  // Reads in old velocities
  task->requires(Task::OldDW, d_label->global->pV, Ghost::None, 0);

  // Compures kinetic energy and stress.
  task->computes(d_label->global->rKineticEnergy);
  task->computes(d_label->global->rKineticStress);
  task->computes(d_label->global->rTotalMomentum);

  baseScheduler->addTask(task, patches, atomTypes);
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

void MD::scheduleIntegratorInitialize(          SchedulerP&  sched,
                                          const PatchSet*    patches,
                                          const MaterialSet* atomTypes,
                                          const LevelP&      level)
{
  const std::string flowLocation = "MD::scheduleIntegratorInitialize | ";
  printSchedule(patches, md_cout, flowLocation);

  Task* task = scinew Task("MD::integratorInitiailize",
                           this,
                           &MD::integratorInitialize);

  MDSubcomponent* d_integratorInterface =
                      dynamic_cast<MDSubcomponent*> (d_integrator);

  d_integratorInterface->addInitializeRequirements(task, d_label);
  d_integratorInterface->addInitializeComputes(task, d_label);

  sched->addTask(task, patches, atomTypes);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }
}

void MD::scheduleIntegratorSetup(               SchedulerP&  sched,
                                          const PatchSet*    patches,
                                          const MaterialSet* atomTypes,
                                          const LevelP&      level)
{
  const std::string flowLocation = "MD::scheduleIntegratorSetup | ";
  printSchedule(patches, md_cout, flowLocation);

  Task* task = scinew Task("MD::integratorSetup",
                           this,
                           &MD::integratorSetup);

  MDSubcomponent* d_integratorInterface =
                      dynamic_cast<MDSubcomponent*> (d_integrator);

  d_integratorInterface->addSetupRequirements(task, d_label);
  d_integratorInterface->addSetupComputes(task, d_label);

  sched->addTask(task, patches, atomTypes);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }
}

void MD::scheduleIntegratorCalculate(           SchedulerP&  sched,
                                          const PatchSet*    patches,
                                          const MaterialSet* atomTypes,
                                          const LevelP&      level)
{
  const std::string flowLocation = "MD::scheduleIntegratorCalculate | ";
  printSchedule(patches, md_cout, flowLocation);

  Task* task = scinew Task("MD::integratorCalculate",
                           this,
                           &MD::integratorCalculate);

  MDSubcomponent* d_integratorInterface =
                      dynamic_cast<MDSubcomponent*> (d_integrator);

  d_integratorInterface->addCalculateRequirements(task, d_label);
  d_integratorInterface->addCalculateComputes(task, d_label);

  sched->addTask(task, patches, atomTypes);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }
}

void MD::scheduleIntegratorFinalize(            SchedulerP&  sched,
                                          const PatchSet*    patches,
                                          const MaterialSet* atomTypes,
                                          const LevelP&      level)
{
  const std::string flowLocation = "MD::scheduleIntegratorFinalize | ";
  printSchedule(patches, md_cout, flowLocation);

  Task* task = scinew Task("MD::integratorFinalize",
                           this,
                           &MD::integratorFinalize);

  MDSubcomponent* d_integratorInterface =
                      dynamic_cast<MDSubcomponent*> (d_integrator);

  d_integratorInterface->addFinalizeRequirements(task, d_label);
  d_integratorInterface->addFinalizeComputes(task, d_label);

  sched->addTask(task, patches, atomTypes);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }
}

void MD::scheduleElectrostaticsInitialize(      SchedulerP& sched,
                                          const PatchSet* perProcPatches,
                                          const MaterialSet* matls,
                                          const LevelP& level)
{
  const std::string flowLocation = "MD::scheduleElectrostaticsInitialize | ";
  printSchedule(perProcPatches, md_cout, flowLocation);

  // initialize electrostatics instance; if we're doing electrostatics
//  if (d_electrostatics->getType() != Electrostatics::NONE) {

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
//  }
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

//  if (d_electrostatics->getType() != Electrostatics::NONE) {
    Task* task = scinew Task("MD::electrostaticsSetup", this, &MD::electrostaticsSetup);

    // cast electrostatics to a subcomponent interface
    MDSubcomponent* d_electroInterface = dynamic_cast<MDSubcomponent*> (d_electrostatics);

    d_electroInterface->addSetupRequirements(task, d_label);
    d_electroInterface->addSetupComputes(task, d_label);
    sched->addTask(task, patches, matls);

//  }
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

//  if (d_electrostatics->getType() != Electrostatics::NONE) {
    Task* task = scinew Task("electrostaticsCalculate", this, &MD::electrostaticsCalculate, level);

//    task->requires(Task::OldDW,
//                   d_label->global->pX,
//                   Ghost::AroundNodes,
//                   d_electrostatics->requiredGhostCells());
//    task->requires(Task::OldDW,
//                   d_label->global->pID,
//                   Ghost::AroundNodes,
//                   d_electrostatics->requiredGhostCells());

    // Need delT for the subscheduler timestep
    task->requires(Task::OldDW, d_sharedState->get_delt_label());

    MDSubcomponent* d_electroInterface =
                        dynamic_cast<MDSubcomponent*> (d_electrostatics);

    d_electroInterface->addCalculateRequirements(task, d_label);
    d_electroInterface->addCalculateComputes(task, d_label);

    task->hasSubScheduler(true);
    task->setType(Task::OncePerProc);

    LoadBalancer* loadBal = sched->getLoadBalancer();
    const PatchSet* perProcPatches = loadBal->getPerProcessorPatchSet(level);

    sched->addTask(task, perProcPatches, matls);
//  }
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

//  if (d_electrostatics->getType() != Electrostatics::NONE) {
    Task* task = scinew Task("MD::electrostaticsFinalize", this, &MD::electrostaticsFinalize);

    MDSubcomponent* d_electroInterface = dynamic_cast<MDSubcomponent*> (d_electrostatics);
    d_electroInterface->addFinalizeRequirements(task, d_label);
    d_electroInterface->addFinalizeComputes(task, d_label);

    sched->addTask(task, patches, matls);
//  }

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }

}

//void MD::scheduleNewUpdatePosition(      SchedulerP&    sched,
//                                   const PatchSet*      patches,
//                                   const MaterialSet*   atomTypes,
//                                   const LevelP&        level)
//{
//  const std::string flowLocation = "MD::scheduleNewUpdatePosition | ";
//  printSchedule(patches, md_cout, flowLocation);
//
//  Task* task = scinew Task("newUpdatePosition", this, &MD::newUpdatePosition);
//
//  // Requirements
//  Ghost::GhostType noGhost = Ghost::None;
//  Task::WhichDW previous = Task::OldDW;
//  Task::WhichDW current = Task::NewDW;
//
//  task->requires(previous, d_label->global->pX, noGhost, 0);
//  task->requires(previous, d_label->global->pID, noGhost, 0);
//  task->requires(previous, d_label->global->pV, noGhost, 0);
//
//  task->requires(previous, d_label->global->rTotalMomentum);
//  task->requires(previous, d_label->global->rTotalMass);
//
//  task->requires(previous, d_label->nonbonded->pF_nonbonded, noGhost, 0);
//  task->requires(previous, d_label->electrostatic->pF_electroReal, noGhost, 0);
//  task->requires(previous, d_label->electrostatic->pF_electroInverse, noGhost, 0);
//
//  task->requires(current, d_label->nonbonded->pF_nonbonded_preReloc, noGhost, 0);
//  task->requires(current, d_label->electrostatic->pF_electroReal_preReloc, noGhost, 0);
//  task->requires(current, d_label->electrostatic->pF_electroInverse_preReloc, noGhost, 0);
//
//  task->requires(previous, d_sharedState->get_delt_label());
//
//  task->computes(d_label->global->pX_preReloc);
//  task->computes(d_label->global->pID_preReloc);
//  task->computes(d_label->global->pV_preReloc);
//
//  task->computes(d_label->global->rKineticEnergy);
//  task->computes(d_label->global->rKineticStress);
//  task->computes(d_label->global->rTotalMomentum);
//  task->computes(d_label->global->rTotalMass);
//
//  sched->addTask(task, patches, atomTypes);
//
//  if (mdFlowDebug.active()) {
//    mdFlowDebug << flowLocation
//                << "END"
//                << std::endl;
//  }
//
//}

//void MD::scheduleUpdatePosition(SchedulerP& sched,
//                                const PatchSet* patches,
//                                const MaterialSet* matls,
//                                const LevelP& level)
//{
//  const std::string flowLocation = "MD::scheduleUpdatePosition | ";
//  printSchedule(patches, md_cout, flowLocation);
//
//  // This should eventually schedule a call of the integrator.  Something like d_Integrator->advanceTimestep()
//  Task* task = scinew Task("updatePosition", this, &MD::updatePosition);
//
//  // Integration requires the position and particle ID from last time step
//  task->requires(Task::OldDW, d_label->global->pX, Ghost::None, 0);
//  task->requires(Task::OldDW, d_label->global->pID, Ghost::None, 0);
//  task->requires(Task::OldDW, d_label->global->pV, Ghost::None, 0);
//
//  // Need these to offset center of mass momentum
//  task->requires(Task::OldDW, d_label->global->rTotalMomentum);
//  task->requires(Task::OldDW, d_label->global->rTotalMass);
//
//  task->requires(Task::OldDW, d_label->nonbonded->pF_nonbonded, Ghost::None, 0);
//  task->requires(Task::OldDW, d_label->electrostatic->pF_electroInverse, Ghost::None, 0);
//  task->requires(Task::OldDW, d_label->electrostatic->pF_electroReal, Ghost::None, 0);
//
//  task->requires(Task::NewDW, d_label->nonbonded->pF_nonbonded_preReloc, Ghost::None, 0);
//  task->requires(Task::NewDW, d_label->electrostatic->pF_electroInverse_preReloc, Ghost::None, 0);
//  task->requires(Task::NewDW, d_label->electrostatic->pF_electroReal_preReloc, Ghost::None, 0);
//
//  // Grabs delta_t from the previous step
//  task->requires(Task::OldDW, d_sharedState->get_delt_label());
//
//  // From integration we get new positions and velocities
//  task->computes(d_label->global->pX_preReloc);
//  task->computes(d_label->global->pID_preReloc);
//  task->computes(d_label->global->pV_preReloc);
//
//  task->computes(d_label->global->rKineticEnergy);
//  task->computes(d_label->global->rKineticStress);
//  task->computes(d_label->global->rTotalMomentum);
//  task->computes(d_label->global->rTotalMass);
////  task->computes(d_label->pXLabel_preReloc);
////  task->computes(d_label->pVelocityLabel_preReloc);
////  task->modifies(d_lb->pNonbondedForceLabel_preReloc);
////  task->modifies(d_lb->pElectrostaticsForceLabel_preReloc);
////  task->computes(d_lb->pAccelLabel_preReloc);
////  task->computes(d_lb->pMassLabel_preReloc);
////  task->computes(d_lb->pParticleIDLabel_preReloc);
//
//  sched->addTask(task, patches, matls);
//  if (mdFlowDebug.active()) {
//    mdFlowDebug << flowLocation
//                << "END"
//                << std::endl;
//  }
//
//}

void MD::scheduleOutputStatistics(      SchedulerP&     sched,
                                  const PatchSet*       patches,
                                  const MaterialSet*    atomTypes,
                                  const LevelP&         level)
{
  Task* task = scinew Task("outputStatistics", this, &MD::outputStatistics);

  // Output the results from last timestep
  task->requires(Task::OldDW,
                 d_label->nonbonded->rNonbondedEnergy);
  task->requires(Task::OldDW,
                 d_label->electrostatic->rElectrostaticInverseEnergy);
  task->requires(Task::OldDW,
                 d_label->electrostatic->rElectrostaticRealEnergy);
  task->requires(Task::OldDW,
                 d_label->global->rKineticEnergy);

  // We only -need- stress tensors if we're doing NPT
  if ( NPT == d_system->getEnsemble()) {
    task->requires(Task::OldDW,
                   d_label->nonbonded->rNonbondedStress);
    task->requires(Task::OldDW,
                   d_label->electrostatic->rElectrostaticInverseStress);
    task->requires(Task::OldDW,
                   d_label->electrostatic->rElectrostaticRealStress);

    if ( d_electrostatics->isPolarizable() ) {
      task->requires(Task::OldDW,
                     d_label->electrostatic->rElectrostaticInverseStressDipole);
    }
  }

  //task->setType(Task::Output); // TODO FIXME How do I use this?
  task->setType(Task::OncePerProc);
  sched->addTask(task, patches, atomTypes);
}

void MD::outputStatistics(const ProcessorGroup* pg,
                          const PatchSubset*    patches,
                          const MaterialSubset* atomTypes,
                                DataWarehouse*  oldDW,
                                DataWarehouse*/*newDW*/)
{


  sum_vartype nonbondedEnergy;
  sum_vartype kineticEnergy;
  sum_vartype electrostaticInverseEnergy;
  sum_vartype electrostaticRealEnergy;

  matrix_sum spmeFourierStress;
  matrix_sum spmeRealStress;
  matrix_sum spmeFourierStressDipole;

  // This is where we would actually map the correct timestep/taskgraph
  // for a multistep integrator

  oldDW->get(nonbondedEnergy, d_label->nonbonded->rNonbondedEnergy);
  oldDW->get(electrostaticInverseEnergy,
              d_label->electrostatic->rElectrostaticInverseEnergy);
  oldDW->get(electrostaticRealEnergy,
              d_label->electrostatic->rElectrostaticRealEnergy);
  oldDW->get(kineticEnergy, d_label->global->rKineticEnergy);

  double totalEnergy = nonbondedEnergy + kineticEnergy
                      + electrostaticInverseEnergy + electrostaticRealEnergy;
//  proc0cout << std::endl;
//  proc0cout << "-----------------------------------------------------"
//            << std::endl
//            << " Electrostatic Energy: "
//            << std::setprecision(16)
//            << std::setw(22) << std::fixed << std::right
//            << electrostaticInverseEnergy + electrostaticRealEnergy << std::endl
//            << "\tInverse: " << std::setprecision(16)
//            << std::setw(22) << std::fixed << std::right
//            << electrostaticInverseEnergy << std::endl
//            << "\tReal:    " << std::setprecision(16)
//            << std::setw(22) << std::fixed << std::right
//            << electrostaticRealEnergy << std::endl;
//
//  proc0cout << "-----------------------------------------------------"
//            << std::endl
//            << "Nonbonded Energy:      "
//            << std::setprecision(16)
//            << std::setw(22) << std::fixed << std::right
//            << nonbondedEnergy << std::endl;
//
//  proc0cout << "-----------------------------------------------------"
//            << std::endl
//            << "Kinetic Energy:        "
//            << std::setprecision(16)
//            << std::setw(22) << std::fixed << std::right
//            << kineticEnergy << std::endl;
//
//  proc0cout << "-----------------------------------------------------"
//            << std::endl
//            << "Total Energy:          "
//            << std::setprecision(16)
//            << std::setw(22) << std::fixed << std::right
//            << totalEnergy;

  double potentialEnergy = nonbondedEnergy + electrostaticInverseEnergy + electrostaticRealEnergy;
  if (d_secondIntegration) {
    d_secondIntegration = false;
    d_PotentialBase = potentialEnergy;
    d_referenceStored = true;
    d_referenceEnergy += d_PotentialBase;
  }



  proc0cout << "  Potential:  " << std::setprecision(4) << std::setw(12) << std::right << std::fixed << potentialEnergy
            << "  Kinetic:  " << std::setprecision(4) << std::setw(12) << std::right << std::fixed << kineticEnergy
            << "  Total:  " << std::setprecision(4) << std::setw(12) << std::right << std::fixed << potentialEnergy + kineticEnergy;

  proc0cout << "  electrostaticInverse: " << std::setprecision(4) << std::setw(12) << std::right << std::fixed << electrostaticInverseEnergy
            << "  electrostaticReal: " << std::setprecision(4) << std::setw(12) << std::right << std::fixed << electrostaticRealEnergy;

  if (d_referenceStored) {
    proc0cout << "\t" << " Relative to reference: "
              << std::setprecision(3) << std::setw(6) << std::right << std::fixed
              << (totalEnergy/d_referenceEnergy)*100.0 << "%";
  }

  if (!d_referenceStored && kineticEnergy != 0.0 && nonbondedEnergy != 0.0)
  {
    d_referenceEnergy = totalEnergy;
    d_referenceStored = true;
  }

//  // FIXME TODO This is a bit of a hack for a quick up and running issue
//  double Temp = 2.0 * kineticEnergy / (3.0*(6192.0-(2.0*288.0)-1.0) * 1.98709e-3);
//  if (Temp != 0.0) {
//    d_isoKineticMult =  sqrt(11.89/Temp); // Set temp to 298.15
//    if (d_isoKineticMult > 2.0) {
//      d_isoKineticMult = 2.0;
//    }
//  }
  int timestep = d_sharedState->getCurrentTopLevelTimeStep()-1;
  if (isPrincipleThread) {
  std::string energyFileName = "EnergyOutput.txt";
  std::ofstream energyFile;
  energyFile.open(energyFileName.c_str(),std::fstream::out | std::fstream::app);
  energyFile << std::setw(10) << std::left << timestep
             << std::setprecision(2) << std::setw(13) << std::fixed << std::right << kineticEnergy
             << std::setprecision(2) << std::setw(13) << std::fixed << std::right << potentialEnergy
             << std::setprecision(2) << std::setw(13) << std::fixed << std::right << totalEnergy
             << std::endl;
  energyFile.close();
  }

//  if (timestep%25 == 0) {
//    proc0cout << "  Step: " << std::setw(10) << std::left << timestep
//              << "\t" << "KE: " << kineticEnergy << " PE:" << potentialEnergy
//              << " Total: " << totalEnergy << std::endl;
//  }
//  proc0cout << "Temperature: " << Temp << " Mult: " << d_isoKineticMult <<std::endl;
    if (NPT == d_system->getEnsemble()) {
    oldDW->get(spmeFourierStress,
                d_label->electrostatic->rElectrostaticInverseStress);
    oldDW->get(spmeRealStress,
                d_label->electrostatic->rElectrostaticRealStress);
    proc0cout << "Fourier Stress = "
              << std::setprecision(16)
              << spmeFourierStress
              << std::endl;
    proc0cout << "-----------------------------------------------------"           << std::endl;

    if (d_electrostatics->isPolarizable()) {
      oldDW->get(spmeFourierStressDipole,
                  d_label->electrostatic->rElectrostaticInverseStressDipole);
    }
  }

  proc0cout << std::endl;

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
               atomFactory::create(d_problemSpec, d_sharedState, d_forcefield);

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
  d_system->registerAtomTypes(parsedCoordinates, d_sharedState);

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


  double          kineticEnergy = 0.0;
  double          totalMass     = 0.0;
  SCIRun::Vector  totalMomentum = MDConstants::V_ZERO;
  Uintah::Matrix3 kineticStress = MDConstants::M3_0;

  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    // Loop over perProcPatches
    const Patch*        currPatch           =   perProcPatches->get(patchIndex);
    SCIRun::IntVector   lowCellBoundary     =   currPatch->getCellLowIndex();
    SCIRun::IntVector   highCellBoundary    =   currPatch->getCellHighIndex();

    std::cerr << "Current Cell Dimensions: " << cellDimensions << std::endl;
    (const_cast<Patch*> (currPatch))->getLevel(true)->setdCell(cellDimensions);

    double          atomTypeVelocitySquared     = 0.0;
    SCIRun::Vector  atomTypeCumulativeVelocity  = MDConstants::V_ZERO;
    Uintah::Matrix3 atomTypeStressTensor        = MDConstants::M3_0;
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

      double atomMass = atomType->getMass();
      for (size_t atomIndex = 0; atomIndex < numAtomsOfType; ++atomIndex) {
        // Loop over all atoms of material
        atomData*   currAtom        = (*currAtomList)[atomIndex];
        Point       currPosition    = currAtom->getPosition();

        // TODO:  This is a good location to inject initial transformations of the as-read data
//        // TODO FIXME Remove these comments once verified unbroken without VVVVV
//        double boxX = 56.0114734706852;
//        double numSteps = 20000;
//        SCIRun::Vector xDimension = d_coordinate->getUnitCell().getColumn(0);
//        xDimension -= boxX * MDConstants::V_X; // Get back to unmodified box
//        currPosition += (xDimension/2.0); // Add 10% of X dimension as constant shift
//        //d_xShift = 0.2*boxX/numSteps;
//        d_xShift = MDConstants::V_X*5e-4;

        if (currPatch->containsPoint(currPosition))
        { // Atom is on this patch
          size_t currID = currAtom->getID();
          SCIRun::Vector currVelocity = currAtom->getVelocity();

          // Use the forcefield to set the units of the read in coordinates
          // and velocity to internally consistent values
          currPosition *= d_forcefield->ffDistanceToInternal();
          currVelocity *= d_forcefield->ffVelocityToInternal();

          totalMass         += atomMass;
          atomTypeVelocitySquared       += currVelocity.length2();
          atomTypeCumulativeVelocity    += currVelocity;
          atomTypeStressTensor          += OuterProduct(currVelocity,currVelocity);

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
      }

      CCVariable<int> subSchedulerDependency;
      newDW->allocateAndPut(subSchedulerDependency,
                            d_label->electrostatic->dSubschedulerDependency,
                            globalID,
                            currPatch,
                            Ghost::None,
                            0);
      subSchedulerDependency.initialize(0);
      kineticEnergy += atomTypeVelocitySquared * atomMass;
      totalMomentum += atomTypeCumulativeVelocity * atomMass;
      kineticStress += atomTypeStressTensor * atomMass;
    } // Loop over materials
  } // Loop over patches

  kineticEnergy /= (2.0*41.84e+5); //0.5e+7;
  kineticStress *= 1.0; //1e+7;
  newDW->put(sum_vartype(kineticEnergy),d_label->global->rKineticEnergy);
  newDW->put(sumvec_vartype(totalMomentum),d_label->global->rTotalMomentum);
  newDW->put(matrix_sum(kineticStress),d_label->global->rKineticStress);
  newDW->put(sum_vartype(totalMass),d_label->global->rTotalMass);

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

  newDW->put(delt_vartype(1.0),
              d_sharedState->get_delt_label(),
              getLevel(patches));

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }

}

void MD::calculateKineticEnergy(const ProcessorGroup*   pg,
                                const PatchSubset*      patches,
                                const MaterialSubset*   localAtomTypes,
                                      DataWarehouse*    oldDW,
                                      DataWarehouse*    newDW)
{
  size_t numPatches     = patches->size();
  size_t numAtomTypes   = localAtomTypes->size();
  
  double            kineticEnergy   = 0.0;
  Uintah::Matrix3   kineticStress   = MDConstants::M3_0;
  SCIRun::Vector    currentVelocity = MDConstants::V_ZERO;
  SCIRun::Vector    cellMomentum    = MDConstants::V_ZERO;
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex)
  {
    const Patch* currPatch = patches->get(patchIndex);
    for (size_t atomIndex = 0; atomIndex < numAtomTypes; ++atomIndex)
    {
      double          atomTypeEnergy = 0.0;
      Uintah::Matrix3 atomTypeStress = MDConstants::M3_0;
      SCIRun::Vector  atomTypeCumulativeVelocity = MDConstants::V_ZERO;

      int             atomType = localAtomTypes->get(atomIndex);
      ParticleSubset* atomSet  = oldDW->getParticleSubset(atomType,currPatch);

      constParticleVariable<SCIRun::Vector> V;
      oldDW->get(V, d_label->global->pV, atomSet);

      size_t numAtoms = atomSet->numParticles();
      for (size_t atom = 0; atom < numAtoms; ++atom)
      {
        currentVelocity = V[atom];
        atomTypeCumulativeVelocity += V[atom];
        atomTypeEnergy += currentVelocity.length2();
        atomTypeStress += OuterProduct(currentVelocity,currentVelocity);
      }
      double mass    = d_sharedState->getMDMaterial(atomType)->getMass();
      kineticEnergy += atomTypeEnergy * mass;
      kineticStress += atomTypeStress * mass;
      cellMomentum   += atomTypeCumulativeVelocity *mass;
    }
  }
  kineticEnergy *= 0.5e+7;
  newDW->put(sum_vartype(kineticEnergy),d_label->global->rKineticEnergy);
  newDW->put(matrix_sum(kineticStress),d_label->global->rKineticStress);
  newDW->put(sumvec_vartype(cellMomentum),d_label->global->rTotalMomentum);
  // Fixme TODO:
  // Find appropriate place to calculate the stress contribution from
  // truncation term.  Also normalize stress tensor by degrees of freedom
  // and account for unit conversion.
}

//void MD::integratorInitialize(const ProcessorGroup*     pg,
//                              const PatchSubset*        patches,
//                              const MaterialSubset*     atomTypes,
//                                    DataWarehouse*      oldDW,
//                                    DataWarehouse*      newDW)
//{
//  const std::string location            = "MD::integratorInitialize";
//  const std::string flowLocation        = location + " | ";
//  const std::string particleLocation    = location + " P ";
//  printTask(patches, md_cout, location);
//
//  d_integrator->initialize(pg, patches, atomTypes, oldDW, newDW,
//                           &d_sharedState, d_system, d_label, d_coordinate);
//
//  if (mdFlowDebug.active()) {
//    mdFlowDebug << flowLocation
//                << "END"
//                << std::endl;
//  }
//}
//
//void MD::integratorSetup(     const ProcessorGroup*     pg,
//                              const PatchSubset*        patches,
//                              const MaterialSubset*     atomTypes,
//                                    DataWarehouse*      oldDW,
//                                    DataWarehouse*      newDW)
//{
//  const std::string location            = "MD::integratorSetup";
//  const std::string flowLocation        = location + " | ";
//  const std::string particleLocation    = location + " P ";
//  printTask(patches, md_cout, location);
//
//  d_integrator->setup(pg, patches, atomTypes, oldDW, newDW,
//                      &d_sharedState, d_system, d_label, d_coordinate);
//
//  if (mdFlowDebug.active()) {
//    mdFlowDebug << flowLocation
//                << "END"
//                << std::endl;
//  }
//
//}
//
//void MD::integratorCalculate( const ProcessorGroup*     pg,
//                              const PatchSubset*        patches,
//                              const MaterialSubset*     atomTypes,
//                                    DataWarehouse*      oldDW,
//                                    DataWarehouse*      newDW)
//{
//  const std::string location            = "MD::integratorCalculate";
//  const std::string flowLocation        = location + " | ";
//  const std::string particleLocation    = location + " P ";
//  printTask(patches, md_cout, location);
//
//  d_integrator->calculate(pg, patches, atomTypes, oldDW, newDW,
//                          &d_sharedState, d_system, d_label, d_coordinate);
//
//  if (mdFlowDebug.active()) {
//    mdFlowDebug << flowLocation
//                << "END"
//                << std::endl;
//  }
//
//}
//
//void MD::integratorFinalize(  const ProcessorGroup*     pg,
//                              const PatchSubset*        patches,
//                              const MaterialSubset*     atomTypes,
//                                    DataWarehouse*      oldDW,
//                                    DataWarehouse*      newDW)
//{
//  const std::string location            = "MD::integratorFinalize";
//  const std::string flowLocation        = location + " | ";
//  const std::string particleLocation    = location + " P ";
//  printTask(patches, md_cout, location);
//
//  d_integrator->finalize(pg, patches, atomTypes, oldDW, newDW,
//                         &d_sharedState, d_system, d_label, d_coordinate);
//
//  if (mdFlowDebug.active()) {
//    mdFlowDebug << flowLocation
//                << "END"
//                << std::endl;
//  }
//
//}

//void MD::newUpdatePosition(const ProcessorGroup* pg,
//                           const PatchSubset*    patches,
//                           const MaterialSubset* localAtomTypes,
//                                 DataWarehouse*  oldDW,
//                                 DataWarehouse*  newDW)
//{
//  // Velocity Verlet
//  const std::string location = "MD::updatePosition";
//  const std::string flowLocation = location + " | ";
//  const std::string particleLocation = location + " P ";
//  printTask(patches, md_cout, location);
//
//  // Calculate subtract CoM momentum
//  sumvec_vartype previousMomentum;
//  oldDW->get(previousMomentum, d_label->global->rTotalMomentum);
//  sum_vartype    previousMass;
//  oldDW->get(previousMass, d_label->global->rTotalMass);
//  sum_vartype    previousKE;
//  oldDW->get(previousKE, d_label->global->rKineticEnergy);
//
//  SCIRun::Vector momentumFraction  = previousMomentum;
//  double totalMass = previousMass;
//  momentumFraction /= totalMass;
//
//  totalMass = 0.0;
//  SCIRun::Vector F_n;
//  SCIRun::Vector F_nPlus1;
//  //, V_nPlusHalf;
//
//  int numPatches = patches->size();
//  int numTypes   = localAtomTypes->size();
//
//  double kineticEnergy  = 0.0;
//  Uintah::Matrix3 kineticStress  = MDConstants::M3_0;
//  SCIRun::Vector totalMomentum  = MDConstants::V_ZERO;
//
//  double forceNorm = 41.84; // kCal->J * m -> A * kg->g / s->fs
//  double velocNorm = 1.0e-5; // m->A / s->fs
//  double normKE    = 0.5*0.001*(1.0/4184.0); // (1/2) * g->kg * J->kCal;
//  delt_vartype delT;
//  oldDW->get(delT,  d_sharedState->get_delt_label(),  getLevel(patches));
//  double dT = delT;
//  int timestep = d_sharedState->getCurrentTopLevelTimeStep()-1;
//
//  // TODO FIXME:  Remove these comments
////  double appliedStressInAtm = 0.0;
////  std::stringstream xyzName;
////  xyzName << "periodicStrain_" << std::setprecision(5) << std::setw(7) << std::fixed
////          << d_xShift.length() << ".xyz";
////  bool outputXYZFile = true;
////  std::string xyzOutFileName = xyzName.str();
////  int xyzOutStep = 100;
////  std::string coordOutFilename = "coords.out";
////  int coordOutStep = 500;
////
////  std::ofstream coordOutFile;
////  if (timestep%coordOutStep == 0) { // Write coords.out;
////    coordOutFile.open(coordOutFilename.c_str(),
////                      std::fstream::out | std::fstream::trunc);
////    for (int header = 0; header < 2; ++header)
////    {
////      coordOutFile << "*" << std::endl;
////    }
////    coordOutFile << "*   Step:   " << timestep << "   Total Time:   "
////                 << std::setprecision(4) << std::setw(12) << std::right
////                 << timestep*dT << std::endl;
////    for (int header = 0; header < 2; ++header)
////    {
////      coordOutFile << "*" << std::endl;
////    }
////  }
////  if ((timestep%xyzOutStep == 0) && outputXYZFile) { // Write .xyz file header
////    d_xyzOutFile.open(xyzOutFileName.c_str(),
////                      std::fstream::out | std::fstream::app);
////
////    int numAtoms = 0;
////    for (int patchNo = 0; patchNo < numPatches; ++patchNo )
////    {
////      const Patch* currPatch = patches->get(patchNo);
////      for (int atomNo = 0; atomNo < numTypes; ++atomNo)
////      {
////        int atomType = localAtomTypes->get(atomNo);
////        ParticleSubset* atomSet = oldDW->getParticleSubset(atomType, currPatch);
////        numAtoms += static_cast<int> (atomSet->numParticles());
////      }
////    }
////    d_xyzOutFile << numAtoms << "     " << std::endl;
////    d_xyzOutFile << " Step: " << timestep << " Total Time: "
////               << std::setprecision(2) << std::setw(12) << std::right
////               << timestep*dT << std::endl;
////  }
//
////
////
////  int outputTarget = 1;
//  // FIXME TODO HACKY STUFF
//  int numXYZ = 6192;
//  std::vector<std::string> sortedAtomNames(numXYZ);
//  std::vector<Point> sortedAtomPositions(numXYZ);
//  for (int patchIndex = 0; patchIndex < numPatches; ++patchIndex)
//  {
//    const Patch* currPatch = patches->get(patchIndex);
//
//    for (int typeIndex = 0; typeIndex < numTypes; ++typeIndex)
//    {
//      int       atomType        = localAtomTypes->get(typeIndex);
//      double    atomMass        = d_sharedState->getMDMaterial(atomType)
//                                                 ->getMass();
//      double            massInv         = 1.0/atomMass;
////      double            typeKinetic     = 0.0;
////      Uintah::Matrix3   typeStress      = MDConstants::M3_0;
////      SCIRun::Vector    typeMomentum    = MDConstants::V_ZERO;
//      std::string atomName = d_sharedState->getMDMaterial(atomType)->getMapLabel();
//      std::string atomLabel = d_sharedState->getMDMaterial(atomType)->getMaterialLabel();
//
//      ParticleSubset* integrationSet = oldDW->getParticleSubset(atomType,
//                                                                currPatch);
//      constParticleVariable<long64> pID_n;
//      ParticleVariable<long64> pID_nPlus1;
//      constParticleVariable<Point> pX_n;
//      ParticleVariable<Point> pX_nPlus1;
//      constParticleVariable<SCIRun::Vector> pV_n;
//      ParticleVariable<SCIRun::Vector> pV_nPlus1;
//      constParticleVariable<SCIRun::Vector> pF_nb_n, pF_eReal_n, pF_eInv_n;
//      constParticleVariable<SCIRun::Vector> pF_nb_nPlus1, pF_eReal_nPlus1, pF_eInv_nPlus1;
//
//      oldDW->get(pID_n,d_label->global->pID,integrationSet);
//      newDW->allocateAndPut(pID_nPlus1,d_label->global->pID_preReloc,integrationSet);
//
//      oldDW->get(pX_n,d_label->global->pX,integrationSet);
//      newDW->allocateAndPut(pX_nPlus1,d_label->global->pX_preReloc,integrationSet);
//
//      oldDW->get(pV_n,d_label->global->pV,integrationSet);
//      newDW->allocateAndPut(pV_nPlus1,d_label->global->pV_preReloc,integrationSet);
//
//      oldDW->get(pF_nb_n,d_label->nonbonded->pF_nonbonded,integrationSet);
//      newDW->get(pF_nb_nPlus1,d_label->nonbonded->pF_nonbonded_preReloc,integrationSet);
//
//      oldDW->get(pF_eReal_n,d_label->electrostatic->pF_electroReal,integrationSet);
//      newDW->get(pF_eReal_nPlus1,d_label->electrostatic->pF_electroReal_preReloc, integrationSet);
//
//      oldDW->get(pF_eInv_n, d_label->electrostatic->pF_electroInverse, integrationSet);
//      newDW->get(pF_eInv_nPlus1, d_label->electrostatic->pF_electroInverse_preReloc, integrationSet);
//
//      int numAtoms = integrationSet->numParticles();
//      for (int atom = 0; atom < numAtoms; ++atom)
//      {
////        if ((timestep%xyzOutStep == 0) && outputXYZFile)
////        {
////
//////          std::cout << pID_n[atom] << "/" << numAtoms << std::endl;
////          sortedAtomNames[pID_n[atom]-1] = atomName;
////          sortedAtomPositions[pID_n[atom]-1] = pX_n[atom];
////        }
//
//
//        F_n      = pF_nb_n[atom]     + pF_eReal_n[atom]      + pF_eInv_n[atom];
//        F_nPlus1 = pF_nb_nPlus1[atom]+ pF_eReal_nPlus1[atom] + pF_eInv_nPlus1[atom];
//
//        // FIXME TODO YOU MORON isokinetic thermostat is hard coded in here!
////          if (!d_firstIntegration) { d_isoKineticMult = 1.0; } // Only rescale temp on first step
//          pV_nPlus1[atom] = (pV_n[atom] - momentumFraction)
////              * d_isoKineticMult
//                         + 0.5 * F_nPlus1 * dT * forceNorm * massInv;
////        }
//        kineticEnergy += atomMass * pV_nPlus1[atom].length2();
//        totalMomentum += atomMass * pV_nPlus1[atom];
//        totalMass += atomMass;
////        if (timestep%coordOutStep == 0) { // Write coords.out;
////          coordOutFile << std::setprecision(12) << std::setw(18) << std::right << std::fixed
////                       << pX_n[atom].x() << "      "
////                       << pX_n[atom].y() << "      "
////                       << pX_n[atom].z() << "           "
////                       << atomLabel << std::endl;
////          coordOutFile << std::setprecision(12) << std::setw(18) << std::right << std::fixed
////                       << "  "
////                       << pV_nPlus1[atom].x() << "      "
////                       << pV_nPlus1[atom].y() << "      "
////                       << pV_nPlus1[atom].z() << "      "
////                       << std::endl;
////        }
//
//
//        if (! d_firstIntegration) {
//          pV_nPlus1[atom] = pV_nPlus1[atom] + 0.5 * F_nPlus1 * dT * forceNorm * massInv;
//
//        }
////        if (pID_n[atom] < 288) {
////          pX_nPlus1[atom] = pX_n[atom] - d_xShift;
////          pV_nPlus1[atom] = MDConstants::V_ZERO;
////        }
////        else if (pID_n[atom] >= 5904)
////        {
////          pX_nPlus1[atom] = pX_n[atom] + d_xShift;
////          pV_nPlus1[atom] = MDConstants::V_ZERO;
////        }
////        else
////        {
//        pX_nPlus1[atom] = pX_n[atom] + dT * velocNorm * pV_nPlus1[atom];
//
////        }
//        pID_nPlus1[atom] = pID_n[atom];
//      }
//
////      (const_cast<Patch*> (currPatch))->getLevel(true)->setdCell(cellDimensions);
//
//      ParticleSubset* delset = scinew ParticleSubset(0, atomType, currPatch);
//      newDW->deleteParticles(delset);
//    }
//  }
//  if (d_firstIntegration) { // Done with first integration pass
//    std::cout << "First Integration!" << std::endl;
//    d_firstIntegration = false;
//    d_secondIntegration = true;
//    d_KineticBase = previousKE;
//  }
//
//  kineticEnergy *= normKE;
//  newDW->put(sum_vartype(kineticEnergy),d_label->global->rKineticEnergy);
//  newDW->put(sum_vartype(totalMass),d_label->global->rTotalMass);
//  newDW->put(sumvec_vartype(totalMomentum),d_label->global->rTotalMomentum);
////  if (outputXYZFile && (timestep%xyzOutStep == 0))
////  {
////    for (int xyzIndex = 0; xyzIndex < sortedAtomNames.size(); ++xyzIndex)
////    {
////      // Output to .xyz file
////      d_xyzOutFile << sortedAtomNames[xyzIndex] << "\t"
////                   << std::setprecision(15) << std::setw(21) << std::right
////                   << std::fixed << sortedAtomPositions[xyzIndex].x()
////                   << std::setprecision(15) << std::setw(21) << std::right
////                   << std::fixed << sortedAtomPositions[xyzIndex].y()
////                   << std::setprecision(15) << std::setw(21) << std::right
////                   << std::fixed << sortedAtomPositions[xyzIndex].z()
////                   << std::endl;
////    }
////
////    d_xyzOutFile.close();
////  }
////  if (timestep%coordOutStep == 0) {
////    coordOutFile.close();
////  }
//
//  if (mdFlowDebug.active()) {
//    mdFlowDebug << flowLocation
//                << "END"
//                << std::endl;
//  }
//
//
//}
//
//
//void MD::updatePosition(const ProcessorGroup*   pg,
//                        const PatchSubset*      patches,
//                        const MaterialSubset*   localAtomTypes,
//                              DataWarehouse*    oldDW,
//                              DataWarehouse*    newDW)
//{
//  // The generic update algorithm for a velocity verlet looks like this:
//  /*
//   *
//   * 1)  V_n+0.5 = V_n + 0.5 * dT * (1/m)*F_n
//   * 2)  X_n+1   = X_n + dT * V_n+0.5
//   * 3)  F_n+1   = f(X_n+1)
//   * 4)  V_n+1   = V_n+0.5 * 0.5 * dT * (1/m)*F_n+1
//   *
//   * So the repeated steps are:
//   * 1) Calculate half step velocity (No comm needed)
//   * 2) Calculate new positions and communicate them
//   * 3) Calculate new forces (don't need communication for forces)
//   * 4) Calculate full step velocity (Comm needed for center of mass adj.)
//   *
//   * Or:  1,2 ||COMM1|| 3,4 ||COMM2||
//   *
//   * If we build the algorithm as 3,4,1,2 we can lump all comm into the end
//   * of the algorithm.  But we have to save half-step quantities for application
//   * on the next iteration.  This is okay since we'll have old DW copies to
//   * store things in.
//   *
//   */
//
//  const std::string location = "MD::updatePosition";
//  const std::string flowLocation = location + " | ";
//  const std::string particleLocation = location + " P ";
//  printTask(patches, md_cout, location);
//
//  // loop through all patches
//  sumvec_vartype previousMomentum;
//  oldDW->get(previousMomentum, d_label->global->rTotalMomentum);
//  sum_vartype previousMass;
//  oldDW->get(previousMass, d_label->global->rTotalMass);
//  SCIRun::Vector momentumFraction = previousMomentum;
//  double totalMass = previousMass;
//  momentumFraction /= 2000.0;
//
//  totalMass = 0.0; // Reset to begin accumulation for this run
//  SCIRun::Vector F_n, A_n, V_n;
//  SCIRun::Vector F_nPlus1, A_nPlus1;
//
//  unsigned int numPatches   = patches->size();
//  unsigned int numAtomTypes = localAtomTypes->size();
//
//  double            kineticEnergy = 0.0;
//  SCIRun::Vector    totalMomentum = MDConstants::V_ZERO;
//  Uintah::Matrix3   kineticStress = MDConstants::M3_0;
//  int timestep = d_sharedState->getCurrentTopLevelTimeStep()-1;
//
//  delt_vartype delT;
//  oldDW->get(delT,  d_sharedState->get_delt_label(),  getLevel(patches));
//
//
//  for (unsigned int p = 0; p < numPatches; ++p)
//  {
//    const Patch* patch = patches->get(p);
//    // Track K.E. related quantities in the middle of the atomType loop
//    double          atomTypeKE                  = 0.0;
//    Uintah::Matrix3 atomTypeStressTensor        = MDConstants::M3_0;
//    SCIRun::Vector  atomTypeCumulativeVelocity  = MDConstants::V_ZERO;
//    for (unsigned int typeIndex = 0; typeIndex < numAtomTypes; ++typeIndex)
//    {
//      int    atomType = localAtomTypes->get(typeIndex);
//      double atomMass = d_sharedState->getMDMaterial(atomType)->getMass();
//      double massInv  = 1.0/atomMass;
//
//      std::string atomName = d_sharedState->getMDMaterial(atomType)->getMapLabel();
//
//      ParticleSubset* pset = oldDW->getParticleSubset(atomType, patch);
//      // Particle ID variables
//      constParticleVariable<long64> pID_n;
//      oldDW->get(pID_n, d_label->global->pID, pset);
//      ParticleVariable<long64> pID_nPlus1;
//      newDW->allocateAndPut(pID_nPlus1, d_label->global->pID_preReloc, pset);
//
//      // --> Position variables
//      constParticleVariable<Point> X_n;
//      oldDW->get(X_n, d_label->global->pX, pset);
//      ParticleVariable<Point> X_nPlus1;
//      newDW->allocateAndPut(X_nPlus1, d_label->global->pX_preReloc, pset);
//
//      // Velocity and force variables
//      constParticleVariable<SCIRun::Vector> V_nMinusHalf;
//      oldDW->get(V_nMinusHalf, d_label->global->pV, pset);
//      ParticleVariable<SCIRun::Vector> V_nPlusHalf;
//      newDW->allocateAndPut(V_nPlusHalf, d_label->global->pV_preReloc, pset);
//
////      // Forces from the previous time step
////      constParticleVariable<SCIRun::Vector> F_eReal_n;
////      constParticleVariable<SCIRun::Vector> F_eInv_n;
////      constParticleVariable<SCIRun::Vector> F_nb_n;
////      oldDW->get(F_eReal_n, d_label->electrostatic->pF_electroReal, pset);
////      oldDW->get(F_eInv_n, d_label->electrostatic->pF_electroInverse, pset);
////      oldDW->get(F_nb_n, d_label->nonbonded->pF_nonbonded, pset);
////
//      // Forces from this time step
//      constParticleVariable<SCIRun::Vector> F_eReal_n, F_eReal_nPlus1;
//      constParticleVariable<SCIRun::Vector> F_eInv_n, F_eInv_nPlus1;
//      constParticleVariable<SCIRun::Vector> F_nb_n, F_nb_nPlus1;
//      oldDW->get(F_eReal_n, d_label->electrostatic->pF_electroReal, pset);
//      oldDW->get(F_eInv_n, d_label->electrostatic->pF_electroInverse, pset);
//      oldDW->get(F_nb_n, d_label->nonbonded->pF_nonbonded, pset);
//      newDW->get(F_eReal_nPlus1, d_label->electrostatic->pF_electroReal_preReloc, pset);
//      newDW->get(F_eInv_nPlus1, d_label->electrostatic->pF_electroInverse_preReloc, pset);
//      newDW->get(F_nb_nPlus1, d_label->nonbonded->pF_nonbonded_preReloc, pset);
//
//      size_t numAtoms = pset->numParticles();
//      totalMass     += numAtoms*atomMass;
//
//      int tgt = 1;
//      // Loop over the atom set
//      std::cerr << " Timestep: " << delT << std::endl;
//      for (size_t atom = 0; atom < numAtoms; ++atom)
//      {
//
//        F_n = F_eReal_n[atom] + F_eInv_n[atom] + F_nb_n[atom];
//        A_n = F_n * massInv * 41.84;//1e-7;
//        if (pID_n[atom] == tgt)
//        {
//          std::cout << std::setprecision(8);
//          std::cout << " V(" << pID_n[atom] << "): before 2nd int" << V_nMinusHalf[atom] << std::endl;
//          std::cout << " F_Old(" << pID_n[atom] << "): " << F_n << std::endl;
//        }
//
//        // Integrate V(t-0.5dT) to V(t)
////        V_n = (V_nMinusHalf[atom] - momentumFraction) + 0.5 * A_n * delT;
//        V_n = (V_nMinusHalf[atom]-momentumFraction*massInv) + 0.5 * A_n * delT;
//        if (pID_n[atom] == tgt)
//        {
//          std::cout << " V(" << pID_n[atom] << "): after 2nd int" << V_nMinusHalf[atom] << std::endl;
//        }
//        // Calculate kinetic energy here!!!
//        atomTypeKE                  += V_n.length2();
//        atomTypeStressTensor        += OuterProduct(V_n,V_n);
//        atomTypeCumulativeVelocity  += V_n;
//
//        if (pID_n[atom] == tgt)
//        {
//          std::cout << " V(" << pID_n[atom] << "): before 1st int" << V_n << std::endl;
//          std::cout << " F(" << pID_n[atom] << "): at 1st int" << F_n << std::endl;
//          std::cout << " X(" << pID_n[atom] << "): before int" << X_n[atom] << std::endl;
//        }
//        // Integrate velocity up to the next half step
//        F_nPlus1= F_eReal_nPlus1[atom] + F_eInv_nPlus1[atom] + F_nb_nPlus1[atom];
//        A_nPlus1 = F_nPlus1 * massInv * 41.84;
//        V_nPlusHalf[atom] = V_n + 0.5 * delT * A_nPlus1;
//        X_nPlus1[atom] = X_n[atom] + V_nPlusHalf[atom] * delT * 1e-5;
//        if (pID_n[atom] == tgt)
//        {
//          std::cout << " V(" << pID_n[atom] << "): after 1st int" << V_nPlusHalf[atom] << std::endl;
//          std::cout << " X(" << pID_n[atom] << "): after int" << X_nPlus1[atom] << std::endl;
//        }
//
////        std::cerr << "atomType: " << atomType << " atomNumber: " << atomIndex << "\t"
////                  << " X_0: " << X_n[atomIndex] << " V: " << pVNew[atomIndex] << " A: " << A
////                  << "\t" << "X_1: " << pXNew[atom] << std::endl;
//
//        // Simply copy over particle IDs; they never change
//        pID_nPlus1[atom]= pID_n[atom];
////        if (md_dbg.active()) {
////          cerrLock.lock();
////          std::cout << "PatchID: " << std::setw(4) << patch->getID() << std::setw(6);
////          std::cout << "ParticleID: " << std::setw(6) << pidsnew[idx] << std::setw(6);
////          std::cout << "New Position: [";
////          std::cout << std::setw(10) << std::setprecision(6) << pxnew[idx].x();
////          std::cout << std::setw(10) << std::setprecision(6) << pxnew[idx].y();
////          std::cout << std::setprecision(6) << pxnew[idx].z() << std::setw(4) << "]";
////          std::cout << std::endl;
////          cerrLock.unlock();
//      } // end Atom Loop
//      kineticEnergy += atomTypeKE * atomMass;
//      kineticStress += atomTypeStressTensor * atomMass;
//      totalMomentum += atomTypeCumulativeVelocity * atomMass;
//      ParticleSubset* delset = scinew ParticleSubset(0, atomType, patch);
//      newDW->deleteParticles(delset);
//    }  // end materials loop
//  }  // end patch loop
//  kineticEnergy *= (1.19503e-7); //0.5e+7;  // Fix units for KE to be internally consistent
//  kineticStress *= 1.0; //1e+7;  // Also for stress.  TODO FIXME Verify this value!! JBH - 10/4/14
//  newDW->put(sum_vartype(kineticEnergy), d_label->global->rKineticEnergy);
//  newDW->put(matrix_sum(kineticStress), d_label->global->rKineticStress);
//  newDW->put(sumvec_vartype(totalMomentum), d_label->global->rTotalMomentum);
//  newDW->put(sum_vartype(totalMass), d_label->global->rTotalMass);
////  calculateKineticEnergy()
//  //d_coordinate->clearCellChanged();
//
//  if (mdFlowDebug.active()) {
//    mdFlowDebug << flowLocation
//                << "END"
//                << std::endl;
//  }
//}

void MD::createBasePermanentParticleState() {
  // The base particle state which must be tracked when particles move across patch boundaries are
  //   the position and velocity.  Everything else bears some dependence on the form of the forcefield,
  //   electrostatic interaction, and valence interactions employed.
  //
  // Note that position is registered elsewhere since it is the position variable.

  const std::string location = "MD::createBasePermanentParticleState";
  const std::string flowLocation = location + " | ";

//  d_particleState.push_back(d_label->global->pID);
//  d_particleState.push_back(d_label->global->pV);
//
//  d_particleState_preReloc.push_back(d_label->global->pID_preReloc);
//  d_particleState_preReloc.push_back(d_label->global->pV_preReloc);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }
}
