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

#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Matrix3.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/DebugStream.h>


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

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>


//.......0123456789..........0123456789..........0123456789..........0123456789
using namespace Uintah;

extern SCIRun::Mutex cerrLock;
SCIRun::Mutex testFileLock("Locks for test output files");

#define isPrincipleThread (        Uintah::Parallel::getMPIRank() == 0         \
                           && ( (  Uintah::Parallel::getNumThreads() > 1       \
                           &&      SCIRun::Thread::self()->myid() == 1 )       \
                           || (    Uintah::Parallel::getNumThreads() <= 1 ) )  \
                          )

#define isPrincipleProc (Uintah::Parallel::getMPIRank() == 0)

MD::MD(const ProcessorGroup* myworld) :
    UintahParallelComponent(myworld)
{
  d_label = scinew MDLabel();
  d_referenceStored = false;
  d_firstIntegration = true;
  d_secondIntegration = false;
  d_KineticBase = d_PotentialBase = d_referenceEnergy = 0.0;
  d_switchCriteria = NULL;

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

void MD::preGridProblemSetup(const ProblemSpecP&        params,
                                   GridP&               grid,
                                   SimulationStateP&    simState)
{
  // Determine the maximum cutoff radius distance of the simulation
  double cutoffRadius = -1.0;
  params->findBlock("MD")->findBlock("System")->require("cutoffRadius",cutoffRadius);
  ProblemSpecP electro_ps = params->findBlock("MD")->findBlock("Electrostatics");
  double tempCutoff=-1.0;
  if (electro_ps->get("cutoffRadius",tempCutoff))
  {
    if (tempCutoff > cutoffRadius)
    {
      cutoffRadius = tempCutoff;
    }
  }

  //grid->setPeriodicSpatialRange(SCIRun::Vector(cutoffRadius));

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

  // read in AMR flags from the main ups file
  ProblemSpecP multi_scale_ps = params->findBlock("MultiScale");
  if (multi_scale_ps) {
    ProblemSpecP md_multi_scale_ps = multi_scale_ps->findBlock("MD");
    if(!md_multi_scale_ps){
      std::ostringstream warn;
      warn<<"ERROR:MD:\n missing MD section in the MultiScale section of the input file\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    md_multi_scale_ps->getWithDefault("min_grid_level", d_minGridLevel, 0);
    md_multi_scale_ps->getWithDefault("max_grid_level", d_maxGridLevel, 1000);
  }

  // Initialize base scheduler and attach the position variable
  if (d_sharedState->d_switchState) {
    dynamic_cast<Scheduler*>(getPort("scheduler"))->setPositionVar(d_label->global->pX);
  }

//------> Set up components inherent to MD
  // create the coordinate system interface
  d_coordinate = CoordinateSystemFactory::create(params, shared_state, grid);
  d_coordinate->markCellChanged();

  // Parse the forcefield
  Forcefield* tempFF = ForcefieldFactory::create(params, shared_state);

  switch ( tempFF->getInteractionClass() ) {  // Set generic interface based on FF type
    case (TwoBody) : {
      d_forcefield = dynamic_cast<TwoBodyForcefield*>(tempFF);
      break;
    }
    case (ThreeBody) :
    case (NBody) :
    default : {
      throw InternalError("MD:  Attempted to instantiate a forcefield type which is not yet implemented", __FILE__, __LINE__);
    }
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
   d_integratorInterface->registerRequiredParticleStates(d_particleState,
                                                         d_particleState_preReloc,
                                                         d_label);

   d_electrostaticInterface->registerRequiredParticleStates(d_particleState,
                                                            d_particleState_preReloc,
                                                            d_label);

   d_nonbondedInterface->registerRequiredParticleStates(d_particleState,
                                                        d_particleState_preReloc,
                                                        d_label);

   // We must wait to register our atom (material) types until the
   // subcomponents have provided the per-particle labels
   //
   // For now we're assuming all atom types have the same tracked states.
   size_t stateSize, preRelocSize;
   stateSize = d_particleState.size();
   preRelocSize = d_particleState_preReloc.size();
  if (stateSize != preRelocSize) {
    std::cerr << "ERROR:  Mismatch in number of per particle variable labels." << std::endl;
  }

   if (particleDebug.active()) {
     particleDebug << particleLocation
                   << "  Registered particle variables: "
                   << std::endl;
     for (size_t index = 0; index < stateSize; ++index) {
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

  // Create the switching criteria port
  d_switchCriteria = dynamic_cast<SwitchingCriteria*>(getPort("switch_criteria"));

  if (d_switchCriteria) {
    d_switchCriteria->problemSetup(restart_prob_spec, restart_prob_spec, d_sharedState);
  }
}

void MD::scheduleInitialize(const LevelP&       level,
                                  SchedulerP&   sched)
{
  int currDW = 0;
  DataWarehouse* dwAddress = sched->get_dw(currDW);
  while (dwAddress)
  {
    std::cout << " Data Warehouse: " << currDW << " Address: " << std::showbase
              << std::internal << std::setfill('0') << std::hex << dwAddress << std::endl;
    ++currDW;
    dwAddress = sched->get_dw(currDW);
  }

  if (!doMDOnLevel(level->getIndex(), level->getGrid()->numLevels())) {
    return;
  }

  const std::string flowLocation = "MD::scheduleInitialize | ";
  printSchedule(level->eachPatch(), md_cout, flowLocation);

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
  const PatchSet*       perProcPatches  =   loadBal->getPerProcessorPatchSet(level);

  const PatchSubset*    patchSubset = perProcPatches->getUnion();
  std::cout << "Seeing a patch set of " << patchSubset->size() << " total patches." << std::endl;
  std::cout << "Received level " << level->getIndex() << " in MD::Initialize call." << std::endl;
  for (int pInd = 0; pInd < patchSubset->size(); ++pInd)
  {
    std::cout << "\nPatch: " << patchSubset->get(pInd)->getID() << " Level: " << patchSubset->get(pInd)->getLevel()->getIndex() << "\n\n";
  }

  Task* task = scinew Task("MD::initialize", this, &MD::initialize);

  // Initialize will load position, velocity, and ID tags
  task->computes(d_label->global->pX);
  task->computes(d_label->global->pV);
  task->computes(d_label->global->pID);

  // FIXME:  Do we still need this here?
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

void MD::scheduleRestartInitialize(const LevelP&     level,
                                         SchedulerP& sched )
{
  // Do nothing for now
}

void MD::switchInitialize(const LevelP&     level,
                                SchedulerP& sched )
{
  // Do nothing for now
}

void MD::scheduleComputeStableTimestep(const LevelP&     level,
                                             SchedulerP& sched )
{
  const std::string flowLocation = "MD::scheduleComputeStableTimestep | ";

  printSchedule(level, md_cout, flowLocation);

  Task* task = scinew Task("MD::computeStableTimestep", this, &MD::computeStableTimestep);

  task->computes(d_sharedState->get_delt_label(), level.get_rep());

  task->setType(Task::OncePerProc);

  LoadBalancer* loadBal = sched->getLoadBalancer();
  const PatchSet* perProcPatches = loadBal->getPerProcessorPatchSet(level);

  sched->addTask(task, perProcPatches, d_sharedState->allMaterials());

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }

}

void MD::scheduleTimeAdvance(const LevelP&      level,
                                   SchedulerP&  sched)
{
  if (!doMDOnLevel(level->getIndex(), level->getGrid()->numLevels())) {
    return;
  }

  const std::string flowLocation = "MD::scheduleTimeAdvance | ";
  printSchedule(level, md_cout, flowLocation);

  // Get list of MD materials for scheduling
  const MaterialSet* atomTypes = d_sharedState->allMDMaterials();
  LoadBalancer* loadBal = sched->getLoadBalancer();
  const PatchSet* perProcPatches = loadBal->getPerProcessorPatchSet(level);
  const PatchSet* patches = level->eachPatch();

  scheduleOutputStatistics(sched, perProcPatches, atomTypes, level);

  scheduleNonbondedSetup(sched, patches, atomTypes, level);

  scheduleElectrostaticsSetup(sched, patches, atomTypes, level);

  scheduleIntegratorSetup(sched, patches, atomTypes, level);

  scheduleNonbondedCalculate(sched, patches, atomTypes, level);

  scheduleElectrostaticsCalculate(sched, patches, atomTypes, level);

  scheduleIntegratorCalculate(sched, patches, atomTypes, level);

  // Should probably move the Finalizes into the appropriate clean-up step on MD.  (Destructor?)
  //   and appropriately modify the finalize routines.  !FIXME
  scheduleNonbondedFinalize(sched, patches, atomTypes, level);

  scheduleElectrostaticsFinalize(sched, patches, atomTypes, level);

  scheduleIntegratorFinalize(sched, patches, atomTypes, level);

  sched->scheduleParticleRelocation(level,
                                    d_label->global->pX_preReloc,
                                    d_sharedState->d_cohesiveZoneState_preReloc,
                                    d_label->global->pX,
                                    d_sharedState->d_cohesiveZoneState,
                                    d_label->global->pID,
                                    atomTypes,
                                    2);

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

  MDSubcomponent* d_nonbondedInterface = dynamic_cast<MDSubcomponent*> (d_nonbonded);

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

  Task* task = scinew Task("MD::integratorInitialize",
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

void MD::scheduleIntegratorFinalize(      SchedulerP&  sched,
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

void MD::scheduleSwitchTest(const LevelP&     level,
                                         SchedulerP& sched)
{
  if (d_switchCriteria) {
    d_switchCriteria->scheduleSwitchTest(level, sched);
  }
}

void MD::scheduleElectrostaticsInitialize(      SchedulerP&  sched,
                                          const PatchSet*    perProcPatches,
                                          const MaterialSet* matls,
                                          const LevelP&      level)
{
  const std::string flowLocation = "MD::scheduleElectrostaticsInitialize | ";
  printSchedule(perProcPatches, md_cout, flowLocation);

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

  Task* task = scinew Task("electrostaticsCalculate", this, &MD::electrostaticsCalculate, level);

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

  // Pair interaction debugging
  task->requires(Task::OldDW,
                 d_label->nonbonded->pNumPairsInCalc, Ghost::None, 0);
  task->requires(Task::OldDW,
                 d_label->global->pID, Ghost::None, 0);
  task->requires(Task::OldDW,
                 d_label->nonbonded->pF_nonbonded, Ghost::None, 0);
  task->requires(Task::OldDW,
                 d_label->global->pX, Ghost::None, 0);
  task->requires(Task::OldDW,
                 d_label->global->pV, Ghost::None, 0);

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
                          const PatchSubset*    perProcPatches,
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

  oldDW->get(nonbondedEnergy, d_label->nonbonded->rNonbondedEnergy);
  oldDW->get(electrostaticInverseEnergy, d_label->electrostatic->rElectrostaticInverseEnergy);
  oldDW->get(electrostaticRealEnergy, d_label->electrostatic->rElectrostaticRealEnergy);
  oldDW->get(kineticEnergy, d_label->global->rKineticEnergy);

  int timestep = d_sharedState->getCurrentTopLevelTimeStep();
  double totalEnergy = nonbondedEnergy + kineticEnergy + electrostaticInverseEnergy + electrostaticRealEnergy;
  double potentialEnergy = nonbondedEnergy + electrostaticInverseEnergy + electrostaticRealEnergy;

  if (d_secondIntegration) {
    d_secondIntegration = false;
    d_PotentialBase = potentialEnergy;
    d_referenceStored = true;
    d_referenceEnergy += d_PotentialBase;
  }

//  if (timestep >= 0)
//  {
////    cerrLock.tryLock();
//    std::vector<long64> pairCount(d_system->getNumAtoms(), 0);
//    constParticleVariable<long64> pairOutCount;
//    size_t numPatches = perProcPatches->size();
//    size_t numAtomTypes = atomTypes->size();
//
//    std::string  pairOutName = "pairCount.txt";
//    std::ofstream pairOutFile;
//    std::ostringstream intOutName;
//    std::ofstream intOutFile;
//    testFileLock.lock();
//    intOutName << "statisticOutput.txt_" << std::left << Uintah::Parallel::getMPIRank();
//    pairOutFile.open(pairOutName.c_str(),std::fstream::out | std::fstream::app);
//    intOutFile.open(intOutName.str().c_str(), std::fstream::out | std::fstream::app);

//    for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex)
//    {
//      const Patch* currPatch = perProcPatches->get(patchIndex);
//      for (size_t atomIndex = 0; atomIndex < numAtomTypes; ++atomIndex)
//      {
//        int             atomType = atomTypes->get(atomIndex);
//        ParticleSubset* atomSet  = oldDW->getParticleSubset(atomType,currPatch);
//
//        constParticleVariable<long64> pairInteractionCount, atomID;
//        constParticleVariable<SCIRun::Vector> pairForce, pairVelocity;
//        constParticleVariable<Point> pairPosition;
//        oldDW->get(pairInteractionCount, d_label->nonbonded->pNumPairsInCalc, atomSet);
//        oldDW->get(atomID, d_label->global->pID, atomSet);
//        oldDW->get(pairForce, d_label->nonbonded->pF_nonbonded, atomSet);
//        oldDW->get(pairVelocity, d_label->global->pV, atomSet);
//        oldDW->get(pairPosition, d_label->global->pX, atomSet);

//        size_t numAtoms = atomSet->numParticles();
//        for (size_t atom = 0; atom < numAtoms; ++atom)
//        {
//          pairOutFile << "Processor: " << std::setw(5) << std::right
//                      << Uintah::Parallel::getMPIRank()
//                      << "\tTimestep: " << std::setw(5) << std::right
//                      << timestep
//                      << "\tSource Atom: " << std::setw(8) << std::right
//                      << atomID[atom]
//                      << "\tTargets Seen: " << std::setw(10) << std::right
//                      << pairInteractionCount[atom]
//                    //  << "\tForce: " << pairForce[atom]
//                      << std::endl;
//          intOutFile << "t: " << std::setw(5) << std::right << std::fixed
//                     << timestep - 1
//                     << " Source Atom: " << std::setw(8) << std::right << std::fixed
//                     << atomID[atom]
//                     << " F: " << pairForce[atom]
//                     << " V: " << pairVelocity[atom]
//                     << " X: " << pairPosition[atom].asVector() << std::endl;
//        }
//      }
//    } // All pair mappings by here
//    pairOutFile.close();
//    intOutFile.close();
//    testFileLock.unlock();
//    cerrLock.unlock();
//  } // If main processor

  std::cout << "Seeing: " << d_system->getNumAtoms() << " atoms." << std::endl;

  if (isPrincipleProc)
  {
    std::cout << "  Potential:  " << std::setprecision(4) << std::setw(12) << std::right << std::fixed << potentialEnergy
              << "  Kinetic:  " << std::setprecision(4) << std::setw(12) << std::right << std::fixed << kineticEnergy
              << "  Total:  " << std::setprecision(4) << std::setw(12) << std::right << std::fixed << potentialEnergy + kineticEnergy;

    std::cout << "  electrostaticInverse: " << std::setprecision(4) << std::setw(12) << std::right << std::fixed << electrostaticInverseEnergy
              << "  electrostaticReal: " << std::setprecision(4) << std::setw(12) << std::right << std::fixed << electrostaticRealEnergy;


    if (d_referenceStored)
    {
      std::cout << "\t" << " Relative to reference: "
                << std::setprecision(3) << std::setw(6) << std::right << std::fixed
                << (totalEnergy/d_referenceEnergy)*100.0 << "%";
    }
    std::string energyFileName = "EnergyOutput.txt";
    std::ofstream energyFile;
    energyFile.open(energyFileName.c_str(),std::fstream::out | std::fstream::app);
    energyFile << std::setw(10) << std::left << timestep
               << std::setprecision(15) << std::setw(26) << std::fixed << std::right << kineticEnergy
               << std::setprecision(15) << std::setw(26) << std::fixed << std::right << potentialEnergy
               << std::setprecision(15) << std::setw(26) << std::fixed << std::right << totalEnergy
               << std::endl;
    energyFile.close();

  }

  if (!d_referenceStored && kineticEnergy != 0.0 && nonbondedEnergy != 0.0)
  {
    d_referenceEnergy = totalEnergy;
    d_referenceStored = true;
  }

  if (NPT == d_system->getEnsemble())
  {
    oldDW->get(spmeFourierStress,
               d_label->electrostatic->rElectrostaticInverseStress);
    oldDW->get(spmeRealStress,
               d_label->electrostatic->rElectrostaticRealStress);
    if (d_electrostatics->isPolarizable())
    {
      oldDW->get(spmeFourierStressDipole,
                 d_label->electrostatic->rElectrostaticInverseStressDipole);
    }
    if (isPrincipleProc)
    {

      std::cout << "Fourier Stress = " << std::setprecision(16)
                << spmeFourierStress << std::endl;
      std::cout << "-----------------------------------------------------"
                << std::endl;
    }
  }
  if (isPrincipleProc) {
      std::cout << std::endl;
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




  // Loop through each patch
  size_t numPatches             =   perProcPatches->size();
  size_t numAtomTypes           =   matls->size();

  // Input coordinates from problem spec
  atomMap* parsedCoordinates    =  atomFactory::create(d_problemSpec, d_sharedState, d_forcefield);
  size_t numTypesParsed         =  parsedCoordinates->getNumberAtomTypes();
  size_t numMaterialTypes       =  d_sharedState->getNumMDMatls();

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
    std::string materialLabel = d_sharedState->getMDMaterial(matlIndex)->getMaterialLabel();
    std::vector<atomData*>* currAtomList = parsedCoordinates->getAtomList(materialLabel);
    size_t numAtoms                      = currAtomList->size();

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


  Matrix3   systemInverseCell = d_coordinate->getInverseCell();
  IntVector totalSystemExtent = d_coordinate->getCellExtent();

  SCIRun::Vector inverseExtentVector;
  inverseExtentVector[0]=1.0/static_cast<double> (totalSystemExtent[0]);
  inverseExtentVector[1]=1.0/static_cast<double> (totalSystemExtent[1]);
  inverseExtentVector[2]=1.0/static_cast<double> (totalSystemExtent[2]);

  SCIRun::Vector cellDimensions = d_coordinate->getUnitCell()*inverseExtentVector;

  int prevLevelIndex = -1;
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex)
  {
    // Loop over perProcPatches
    const Patch*        currPatch           =   perProcPatches->get(patchIndex);

  //  int   currLevelIndex = currPatch->getLevelIndex();

    SCIRun::IntVector   lowCellBoundary     =   currPatch->getCellLowIndex();
    SCIRun::IntVector   highCellBoundary    =   currPatch->getCellHighIndex();

//    (const_cast<Patch*> (currPatch))->getLevel(true)->setdCell(cellDimensions);

    double          atomTypeVelocitySquared     = 0.0;
    SCIRun::Vector  atomTypeCumulativeVelocity  = MDConstants::V_ZERO;
    Uintah::Matrix3 atomTypeStressTensor        = MDConstants::M3_0;
    for (size_t localType = 0; localType < numAtomTypes; ++localType)
    {

      // Loop over materials
      MDMaterial*   atomType    = d_sharedState->getMDMaterial(localType);
      std::string   typeLabel   = atomType->getMaterialLabel();

      // Match coordinates to material and extract coordinate list
      std::vector<atomData*>* currAtomList = parsedCoordinates->getAtomList(typeLabel);
      size_t numAtomsOfType                = parsedCoordinates->getAtomListSize(typeLabel);

      std::vector<Point>            localAtomCoordinates;
      std::vector<size_t>           localAtomID;
      std::vector<SCIRun::Vector>   localAtomVelocity;

      double atomMass = atomType->getMass();
      for (size_t atomIndex = 0; atomIndex < numAtomsOfType; ++atomIndex)
      {

        // Loop over all atoms of material
        atomData*   currAtom        = (*currAtomList)[atomIndex];
        Point       currPosition    = currAtom->getPosition();

        // TODO:  This is a good location to inject initial transformations of the as-read data

        if (currPatch->containsPoint(currPosition))
        { // Atom is on this patch
          size_t currID = currAtom->getID();
//          std::cerr << " Atom: " << currID << " seen!" << std::endl;
//          if (currID == 2    || currID == 351  || currID == 448  ||
//              currID == 485  || currID == 524  || currID == 569  ||
//              currID == 676  || currID == 2250 || currID == 2464 ||
//              currID == 2891 || currID == 2943 || currID == 3091 ||
//              currID == 3161 || currID == 3471 || currID == 4852 ||
//              currID == 5188 || currID == 5302 || currID == 5359 ||
//              currID == 5521 || currID == 5784 || currID == 5832 ||
//              currID == 5848 || currID == 7866 || currID == 7979 ||
//              currID == 8139 || currID == 9137
//              ) {
//          if (currID == 2    || currID == 6  || currID == 10  ||
//              currID == 12  || currID == 16  || currID == 17  ||
//              currID == 22  || currID == 23 || currID == 25 ||
//              currID == 33 || currID == 34 || currID == 37 ||
//              currID == 41 || currID == 43 || currID == 47 ||
//              currID == 48 || currID == 53 || currID == 55 ||
//              currID == 76 || currID == 75 || currID == 58 ||
//              currID == 77 || currID == 88 || currID == 90 ||
//              currID == 91 || currID == 95
//              ) {
//            std::cerr << "Atom" << std::setw(5) << std::right << currID << " on patch "
//                      << std::setw(5) << std::right << patchIndex << " --> "
//                      << std::setw(5) << std::right << currPosition << " => [ "
//                      << std::setw(4) << std::right << static_cast<int> (floor(currPosition.x()/currPatch->dCell().x())) << ", "
//                      << std::setw(4) << std::right << static_cast<int> (floor(currPosition.y()/currPatch->dCell().y())) << ", "
//                      << std::setw(4) << std::right << static_cast<int> (floor(currPosition.z()/currPatch->dCell().z())) << "] -- ["
//                      << std::setw(4) << std::right << static_cast<int> (ceil(currPosition.x()/currPatch->dCell().x())) << ", "
//                      << std::setw(4) << std::right << static_cast<int> (ceil(currPosition.y()/currPatch->dCell().y())) << ", "
//                      << std::setw(4) << std::right << static_cast<int> (ceil(currPosition.z()/currPatch->dCell().z())) << "] "
//                      << currPatch->getBCType(Patch::xminus) << currPatch->getBCType(Patch::xplus) << std::endl;
//          }
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
      size_t            numAtoms    = localAtomCoordinates.size();
      size_t            globalID    = matls->get(localType);  // Map to global material type for pset creation
      std::cout << "MD::Creating particle set with: " << numAtoms << " on patch: " << currPatch->getID() << " on level: "
                << currPatch->getLevel()->getIndex() << " in DW: " << newDW->getID() << std::endl;
      ParticleSubset*   currPset    =
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
                           ->getMDMaterial(localType)
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
      double mass    = d_sharedState->getMDMaterial(atomIndex)->getMass();
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

bool MD::doMDOnLevel(int level, int numLevels) const
{
  return (level >= d_minGridLevel && level <= d_maxGridLevel) || (d_minGridLevel < 0 && level == numLevels + d_minGridLevel);
}
