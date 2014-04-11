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

#include <CCA/Components/MD/MD.h>
#include <CCA/Components/MD/Electrostatics/ElectrostaticsFactory.h>
#include <CCA/Components/MD/Electrostatics/SPME/SPME.h>
#include <CCA/Components/MD/Forcefields/Forcefield.h>
#include <CCA/Components/MD/Forcefields/ForcefieldFactory.h>
#include <CCA/Components/MD/Forcefields/TwoBodyForceField.h>
#include <CCA/Components/MD/Nonbonded/TwoBodyDeterministic.h>
#include <CCA/Components/MD/atomMap.h>
#include <CCA/Components/MD/atomFactory.h>
#include <CCA/Components/MD/NonBondedFactory.h>

using namespace Uintah;

extern SCIRun::Mutex cerrLock;

static DebugStream md_dbg("MDDebug", false);
static DebugStream md_cout("MDCout", false);

MD::MD(const ProcessorGroup* myworld) :
    UintahParallelComponent(myworld)
{
  d_lb = scinew MDLabel();
}

MD::~MD()
{
  delete d_lb;
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
  dynamic_cast<Scheduler*>(getPort("scheduler"))->setPositionVar(d_lb->pXLabel);

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

  std::cerr << "Forcefield created: " << d_forcefield->getForcefieldDescriptor() << std::endl;
// // Loop through all materials and create the local particle set
// for (size_t materialIndex = 0; materialIndex < materialEntries; ++materialIndex) {
//   std::string materialLabel = shared_state->getMDMaterial(materialIndex)->getMaterialLabel();
//   size_t maxAtomsOfMaterial = parsedCoordinates->getAtomListSize(materialLabel);
//   for (size_t atomIndex = 0; )
// }

//  // get path and name of the file with atom information
//  ProblemSpecP md_ps = params->findBlock("MD");
//  md_ps->get("coordinateFile", d_coordinateFile);

  // create and populate the MD System object
  d_system = scinew MDSystem(params, grid, shared_state, d_forcefield);
  d_system->markBoxChanged();

  std::cerr << "Created system object" << std::endl;

  // For now set the interaction model explicitly
  interactionModel deterministicModel = Deterministic;
  // create the NonBonded object via factory method

  NonBonded* tempNB = NonBondedFactory::create(params, d_system, d_lb, d_forcefield->getInteractionClass(), deterministicModel);

  if (tempNB->getNonbondedType() == "TwoBodyDeterministic") {
    d_nonbonded = dynamic_cast<TwoBodyDeterministic*> (tempNB);
  }
//  if (d_nonbonded->getType() == NonBonded::LJ12_6) {
//    dynamic_cast<AnalyticNonBonded*>(d_nonbonded)->setMDLabel(d_lb);
//  }

  std::cerr << "Created nonbonded object" << std::endl;

  // create the Electrostatics object via factory method
  d_electrostatics = ElectrostaticsFactory::create(params, d_system);
  if (d_electrostatics->getType() == Electrostatics::SPME) {
    dynamic_cast<SPME*>(d_electrostatics)->setMDLabel(d_lb);
//    std::cerr << "  Electrostatic cutoff radius: " << dynamic_cast<SPME*>(d_electrostatics)->getRealspaceCutoff() << std::endl;
//    std::cerr << "Here" << std::endl;
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
//  // create and register MD materials (this is ill defined right now)
//  d_material = scinew SimpleMaterial();
//  d_sharedState->registerSimpleMaterial(d_material);

//  // register permanent particle state; for relocation, etc
  // registerPermanentParticleState(d_material);

  // do file I/O to get atom coordinates and simulation cell size
//  extractCoordinates();
  std::cerr << "End of MD::Setup" << std::endl;
}

void MD::scheduleInitialize(const LevelP& level,
                            SchedulerP& sched)
{
  /*
   * Note there are multiple tasks scheduled here. All three need only ever happen once.
   *
   * 1.) MD::initialize
   * 2.) Nonbonded::initialize
   * 3.) SPME::initialize
   */
  std::cerr << "Enter:  Scheduled Initialization" << std::endl;

  printSchedule(level, md_cout, "MD::scheduleInitialize");

  Task* task = scinew Task("MD::initialize", this, &MD::initialize);
  std::cerr << "MD::ScheduleInitialize -> Created new task." << std::endl;

  task->computes(d_lb->pXLabel);
  task->computes(d_lb->pVelocityLabel);
  task->computes(d_lb->pParticleIDLabel);

  task->computes(d_lb->pNonbondedForceLabel);
  task->computes(d_lb->pElectrostaticsRealForce);
  task->computes(d_lb->pElectrostaticsReciprocalForce);
  task->computes(d_lb->pValenceForceLabel);
  task->computes(d_lb->pTotalDipoles);
  task->computes(d_lb->pElectrostaticsRealField);
  task->computes(d_lb->pElectrostaticsReciprocalField);

  std::cerr << "MD::Schedule computes particleID" << std::endl;

  task->computes(d_lb->subSchedulerDependencyLabel);
  std::cerr << "MD::Schedule computes subschedulerDependency" << std::endl;

  const MaterialSet* materials = d_sharedState->allMaterials();
  LoadBalancer* loadBal = sched->getLoadBalancer();
  const PatchSet* perProcPatches = loadBal->getPerProcessorPatchSet(level);

  sched->addTask(task, level->eachPatch(), materials);
  std::cerr << "MD::ScheduleInitialize -> Added MD::Initialize to task graph." << std::endl;

  // Nonbonded initialization - OncePerProc, during initial (0th) timestep.
  // The required pXlabel is available to this OncePerProc task in the new_dw from the computes above
  scheduleNonbondedInitialize(sched, perProcPatches, materials, level);
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

  task->requires(Task::NewDW, d_lb->nonbondedEnergyLabel);
  task->requires(Task::NewDW, d_lb->electrostaticReciprocalEnergyLabel);
  task->requires(Task::NewDW, d_lb->electrostaticReciprocalStressLabel);

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

  sched->scheduleParticleRelocation(level, d_lb->pXLabel_preReloc, d_sharedState->d_particleState_preReloc, d_lb->pXLabel,
                                    d_sharedState->d_particleState, d_lb->pParticleIDLabel, matls, 1);
}

void MD::scheduleNonbondedInitialize(SchedulerP& sched,
                                     const PatchSet* perProcPatches,
                                     const MaterialSet* matls,
                                     const LevelP& level)
{
  printSchedule(perProcPatches, md_cout, "MD::scheduleNonbondedInitialize");

  Task* task = scinew Task("MD::nonbondedInitialize", this, &MD::nonbondedInitialize);

  // This is during the initial timestep... no OldDW exists
  task->requires(Task::NewDW, d_lb->pXLabel, Ghost::None, 0);

   // initialize reduction variable; van der Waals energy
  task->computes(d_lb->nonbondedEnergyLabel);
  task->computes(d_lb->nonbondedDependencyLabel);

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

  task->requires(Task::OldDW, d_lb->nonbondedDependencyLabel);
  task->computes(d_lb->nonbondedDependencyLabel);

  sched->addTask(task, patches, matls);
}

void MD::scheduleNonbondedCalculate(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls,
                                    const LevelP& level)
{
  printSchedule(patches, md_cout, "MD::scheduleNonbondedCalculate");

  Task* task = scinew Task("MD::nonbondedCalculate", this, &MD::nonbondedCalculate, level);

  int CUTOFF_RADIUS = d_system->getNonbondedGhostCells();

  task->requires(Task::OldDW, d_lb->pXLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
  task->requires(Task::OldDW, d_lb->pNonbondedForceLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
//  task->requires(Task::OldDW, d_lb->pEnergyLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
  task->requires(Task::OldDW, d_lb->pParticleIDLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
  task->requires(Task::OldDW, d_lb->nonbondedDependencyLabel, Ghost::None, 0);

  task->computes(d_lb->pNonbondedForceLabel_preReloc);
//  task->computes(d_lb->pEnergyLabel_preReloc);
  task->computes(d_lb->nonbondedEnergyLabel);

  sched->addTask(task, patches, matls);
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
    Task* task = scinew Task("MD::electrostaticsInitialize", this, &MD::electrostaticsInitialize);

    if (d_electrostatics->getType() == Electrostatics::SPME) {

      // reduction variables
      task->computes(d_lb->electrostaticReciprocalEnergyLabel);
      task->computes(d_lb->electrostaticReciprocalStressLabel);

      // sole variables
      task->computes(d_lb->electrostaticsDependencyLabel);
    }

    task->setType(Task::OncePerProc);

    sched->addTask(task, perProcPatches, matls);
  }
}

void MD::scheduleElectrostaticsSetup(SchedulerP& sched,
                                     const PatchSet* patches,
                                     const MaterialSet* matls,
                                     const LevelP& level)
{
  printSchedule(patches, md_cout, "MD::scheduleElectrostaticsSetup");

  Task* task = scinew Task("MD::electrostaticsSetup", this, &MD::electrostaticsSetup);
  task->computes(d_lb->electrostaticsDependencyLabel);

  sched->addTask(task, patches, matls);
}

void MD::scheduleElectrostaticsCalculate(SchedulerP& sched,
                                         const PatchSet* patches,
                                         const MaterialSet* matls,
                                         const LevelP& level)
{
  printSchedule(patches, md_cout, "MD::scheduleElectrostaticsCalculate");

  Task* task = scinew Task("electrostaticsCalculate", this, &MD::electrostaticsCalculate, level);

  int CUTOFF_RADIUS = d_system->getElectrostaticGhostCells();

  task->requires(Task::OldDW, d_lb->pXLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
//  task->requires(Task::OldDW, d_lb->pChargeLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
  task->requires(Task::OldDW, d_lb->pParticleIDLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
  task->requires(Task::OldDW, d_lb->electrostaticsDependencyLabel);
  task->requires(Task::OldDW, d_lb->subSchedulerDependencyLabel, Ghost::None, 0);

  task->computes(d_lb->subSchedulerDependencyLabel);
  task->computes(d_lb->electrostaticReciprocalEnergyLabel);
  task->computes(d_lb->electrostaticReciprocalStressLabel);

  task->hasSubScheduler(true);
  task->setType(Task::OncePerProc);

  LoadBalancer* loadBal = sched->getLoadBalancer();
  const PatchSet* perProcPatches = loadBal->getPerProcessorPatchSet(level);

  sched->addTask(task, perProcPatches, matls);
}

void MD::scheduleElectrostaticsFinalize(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls,
                                        const LevelP& level)
{
  printSchedule(patches, md_cout, "MD::scheduleElectrostaticsFinalize");

  Task* task = scinew Task("MD::electrostaticsFinalize", this, &MD::electrostaticsFinalize);

  // particle variables
  task->requires(Task::OldDW, d_lb->pElectrostaticsReciprocalForce, Ghost::None, 0);
  task->requires(Task::OldDW, d_lb->pElectrostaticsRealForce, Ghost::None, 0);
//  task->requires(Task::OldDW, d_lb->pChargeLabel, Ghost:: Ghost::None, 0);
  task->requires(Task::NewDW, d_lb->subSchedulerDependencyLabel, Ghost:: Ghost::None, 0);

  task->computes(d_lb->pElectrostaticsReciprocalForce_preReloc);
  task->computes(d_lb->pElectrostaticsRealForce_preReloc);
//  task->computes(d_lb->pElectrostaticsForceLabel_preReloc);
//  task->computes(d_lb->pChargeLabel_preReloc);

  sched->addTask(task, patches, matls);
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
  task->requires(Task::OldDW, d_lb->pXLabel, Ghost::None, 0);
  task->requires(Task::OldDW, d_lb->pParticleIDLabel, Ghost::None, 0);

  // And the newly calculated forces
  task->requires(Task::NewDW, d_lb->pNonbondedForceLabel_preReloc, Ghost::None, 0);
  task->requires(Task::NewDW, d_lb->pElectrostaticsReciprocalForce_preReloc, Ghost::None, 0);
  task->requires(Task::NewDW, d_lb->pElectrostaticsRealForce_preReloc, Ghost::None, 0);
//  task->requires(Task::NewDW, d_lb->pElectrostaticsForceLabel_preReloc, Ghost::None, 0);
//  task->requires(Task::OldDW, d_lb->pAccelLabel, Ghost::None, 0);
//  task->requires(Task::OldDW, d_lb->pVelocityLabel, Ghost::None, 0);
//  task->requires(Task::OldDW, d_lb->pMassLabel, Ghost::None, 0);

  // Not sure what this does atm - JBH, 4-7-14
  task->requires(Task::OldDW, d_sharedState->get_delt_label());

  // From integration we get new positions and velocities
  task->computes(d_lb->pXLabel_preReloc);
  task->computes(d_lb->pVelocityLabel_preReloc);
//  task->modifies(d_lb->pNonbondedForceLabel_preReloc);
//  task->modifies(d_lb->pElectrostaticsForceLabel_preReloc);
//  task->computes(d_lb->pAccelLabel_preReloc);
//  task->computes(d_lb->pMassLabel_preReloc);
//  task->computes(d_lb->pParticleIDLabel_preReloc);

  sched->addTask(task, patches, matls);
}

void MD::initialize(const ProcessorGroup* pg,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* /* old_dw */,
                    DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::initialize");


  Matrix3 systemInverseCell = d_system->getInverseCell();
  IntVector totalSystemExtent = d_system->getCellExtent();
  SCIRun::Vector inverseExtentVector;

  inverseExtentVector[0]=1.0/static_cast<double> (totalSystemExtent[0]);
  inverseExtentVector[1]=1.0/static_cast<double> (totalSystemExtent[1]);
  inverseExtentVector[2]=1.0/static_cast<double> (totalSystemExtent[2]);

  SCIRun::Vector cellDimensions = d_system->getUnitCell()*inverseExtentVector;

  // Loop through each patch
  size_t numPatches = patches->size();
  size_t numMaterials = matls->size();

  // Input coordinates from problem spec
  atomMap* parsedCoordinates = atomFactory::create(d_problemSpec, d_sharedState);

  std::cerr << "Constructed particle map in MD::initialize" << std::endl;
  SCIRun::Vector VectorZero(0.0, 0.0, 0.0);

  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) { // Loop over patches
    const Patch* currPatch = patches->get(patchIndex);
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
        Point currPosition = (*currAtomList)[atomIndex]->getPosition();
        IntVector currCell = currPatch->getLevel()->getCellIndex(currPosition);

        // Build local atom list for atoms of material in current patch
        if (containsAtom(lowCellBoundary,highCellBoundary,currCell)) { // Atom is on this patch
          localAtomCoordinates.push_back(currPosition);
          localAtomID.push_back((*currAtomList)[atomIndex]->getID());
          localAtomVelocity.push_back((*currAtomList)[atomIndex]->getVelocity());
        }
      }

      // Create this patch's particle set for atoms of current material
      size_t numAtoms = localAtomCoordinates.size();
      ParticleSubset* currPset = new_dw->createParticleSubset(numAtoms, materialID, currPatch);

      // Allocate, link, and fill particle variables
      ParticleVariable<Point> pX;
      new_dw->allocateAndPut(pX, d_lb->pXLabel, currPset);
      ParticleVariable<Vector> pV;
      new_dw->allocateAndPut(pV, d_lb->pVelocityLabel, currPset);
      ParticleVariable<Vector> pF_nb, pF_eReal, pF_eRecip, pF_v;
      new_dw->allocateAndPut(pF_nb, d_lb->pNonbondedForceLabel, currPset);
      new_dw->allocateAndPut(pF_eReal, d_lb->pElectrostaticsRealForce, currPset);
      new_dw->allocateAndPut(pF_eRecip, d_lb->pElectrostaticsReciprocalForce, currPset);
      new_dw->allocateAndPut(pF_v, d_lb->pValenceForceLabel, currPset);

      ParticleVariable<Vector> pDipoles;
      new_dw->allocateAndPut(pDipoles, d_lb->pTotalDipoles, currPset);
      ParticleVariable<Vector> pFieldReal, pFieldReciprocal;
      new_dw->allocateAndPut(pFieldReal, d_lb->pElectrostaticsRealField, currPset);
      new_dw->allocateAndPut(pFieldReciprocal, d_lb->pElectrostaticsReciprocalField, currPset);

      ParticleVariable<long64> pID;
      new_dw->allocateAndPut(pID, d_lb->pParticleIDLabel, currPset);

      for (size_t atomIndex = 0; atomIndex < numAtoms; ++atomIndex) { // Loop over atoms in this matl in this patch
        // Transfer over currently defined atom data
        pX[atomIndex]   = localAtomCoordinates[atomIndex];
        pV[atomIndex]   = localAtomVelocity[atomIndex];
        pID[atomIndex]  = localAtomID[atomIndex];

        // Initialize the rest
        pF_nb[atomIndex]            = VectorZero;
        pF_eReal[atomIndex]         = VectorZero;
        pF_eRecip[atomIndex]        = VectorZero;
        pF_v[atomIndex]             = VectorZero;
        pDipoles[atomIndex]         = VectorZero;
        pFieldReal[atomIndex]       = VectorZero;
        pFieldReciprocal[atomIndex] = VectorZero;

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
      new_dw->allocateAndPut(subSchedulerDependency, d_lb->subSchedulerDependencyLabel, materialID, currPatch, Ghost::None, 0);
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

  new_dw->get(vdwEnergy, d_lb->nonbondedEnergyLabel);
  new_dw->get(spmeFourierEnergy, d_lb->electrostaticReciprocalEnergyLabel);
  new_dw->get(spmeFourierStress, d_lb->electrostaticReciprocalStressLabel);

  proc0cout << std::endl;
  proc0cout << "-----------------------------------------------------"           << std::endl;
  proc0cout << "Total Energy   = " << std::setprecision(16) << vdwEnergy         << std::endl;
  proc0cout << "-----------------------------------------------------"           << std::endl;
  proc0cout << "Fourier Energy = " << std::setprecision(16) << spmeFourierEnergy << std::endl;
  proc0cout << "-----------------------------------------------------"           << std::endl;
  proc0cout << "Fourier Stress = " << std::setprecision(16) << spmeFourierStress << std::endl;
  proc0cout << "-----------------------------------------------------"           << std::endl;
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

  d_nonbonded->initialize(pg, patches, matls, old_dw, new_dw);
}

void MD::nonbondedSetup(const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::nonbondedSetup");

  if (d_system->queryBoxChanged()) {
    d_nonbonded->setup(pg, patches, matls, old_dw, new_dw);
  }
}

void MD::nonbondedCalculate(const ProcessorGroup* pg,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* parentOldDW,
                            DataWarehouse* parentNewDW,
                            const LevelP level)
{
  printTask(patches, md_cout, "MD::nonbondedCalculate");

  d_nonbonded->calculate(pg, patches, matls, parentOldDW, parentNewDW, d_electrostaticSubscheduler, level);
}

void MD::nonbondedFinalize(const ProcessorGroup* pg,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::nonbondedFinalize");

  d_nonbonded->finalize(pg, patches, matls, old_dw, new_dw);
}

void MD::electrostaticsInitialize(const ProcessorGroup* pg,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::electrostaticsInitialize");

  d_electrostatics->initialize(pg, patches, matls, old_dw, new_dw);
}

void MD::electrostaticsSetup(const ProcessorGroup* pg,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::electrostaticsSetup");

  d_electrostatics->setup(pg, patches, matls, old_dw, new_dw);
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

  d_electrostatics->calculate(pg, perProcPatches, matls, parentOldDW, parentNewDW, d_electrostaticSubscheduler, level, d_sharedState);
}

void MD::electrostaticsFinalize(const ProcessorGroup* pg,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::electrostaticsFinalize");

  d_electrostatics->finalize(pg, patches, matls, old_dw, new_dw);
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
      old_dw->get(pX, d_lb->pXLabel, pset);
      // --> Velocity at last time step (velocity verlet algorithm)
      constParticleVariable<SCIRun::Vector> pV;
      old_dw->get(pV, d_lb->pVelocityLabel, pset);
//      // --> Acceleration at last time step (velocity verlet algorithm)
//      constParticleVariable<SCIRun::Vector> pA;
//      old_dw->get(pA, d_lb->pAccelLabel, pset);
      // --> Forces for this time step
      constParticleVariable<SCIRun::Vector> pForceElectroReal;
      constParticleVariable<SCIRun::Vector> pForceElectroRecip;
      constParticleVariable<SCIRun::Vector> pForceNonbonded;
      new_dw->get(pForceElectroReal,d_lb->pElectrostaticsRealForce_preReloc, pset);
      new_dw->get(pForceElectroRecip, d_lb->pElectrostaticsReciprocalForce_preReloc, pset);
      new_dw->get(pForceNonbonded, d_lb->pNonbondedForceLabel_preReloc, pset);

      // Variables which the integrator calculates
      // --> New position
      ParticleVariable<Point> pXNew;
      new_dw->allocateAndPut(pXNew, d_lb->pXLabel_preReloc, pset);
      // --> New velocity
      ParticleVariable<SCIRun::Vector> pVNew;
      new_dw->allocateAndPut(pVNew, d_lb->pVelocityLabel_preReloc, pset);
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

  d_system->clearBoxChanged();
}

void MD::registerPermanentParticleState(SimpleMaterial* matl)
{
  // Register the particle variables that should be tracked with position changes
  d_particleState_preReloc.push_back(d_lb->pParticleIDLabel_preReloc);
  d_particleState.push_back(d_lb->pParticleIDLabel);

  //Electrostatics
  // --> Realspace
  d_particleState_preReloc.push_back(d_lb->pElectrostaticsRealForce_preReloc);
  d_particleState.push_back(d_lb->pElectrostaticsRealForce);
  d_particleState_preReloc.push_back(d_lb->pElectrostaticsRealField_preReloc);
  d_particleState.push_back(d_lb->pElectrostaticsRealField);
  // --> Reciprocal
  d_particleState_preReloc.push_back(d_lb->pElectrostaticsReciprocalForce_preReloc);
  d_particleState.push_back(d_lb->pElectrostaticsReciprocalForce);
  d_particleState_preReloc.push_back(d_lb->pElectrostaticsReciprocalField_preReloc);
  d_particleState.push_back(d_lb->pElectrostaticsReciprocalField);
  // --> Dipoles
  d_particleState_preReloc.push_back(d_lb->pTotalDipoles_preReloc);
  d_particleState.push_back(d_lb->pTotalDipoles);

  //Nonbonded
  d_particleState_preReloc.push_back(d_lb->pNonbondedForceLabel_preReloc);
  d_particleState.push_back(d_lb->pNonbondedForceLabel);

  //Valence (future)
  d_particleState_preReloc.push_back(d_lb->pValenceForceLabel_preReloc);
  d_particleState.push_back(d_lb->pValenceForceLabel);

  //Integrator
  d_particleState_preReloc.push_back(d_lb->pVelocityLabel_preReloc);
  d_particleState.push_back(d_lb->pVelocityLabel);

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
