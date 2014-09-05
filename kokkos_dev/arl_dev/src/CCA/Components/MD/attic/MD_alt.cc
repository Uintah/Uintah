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

#include <CCA/Components/MD/MD.h>
#include <CCA/Components/MD/SPME.h>
#include <CCA/Components/MD/NonbondedFactory.h>
#include <CCA/Components/MD/AnalyticNonbonded.h>

#include <CCA/Components/MD/ElectrostaticsFactory.h>
#include <CCA/Components/MD/Forcefields/ForcefieldFactory.h>
#include <CCA/Components/MD/Integrators/IntegratorFactory.h>

#include <CCA/Ports/Scheduler.h>

#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>

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

#include <Core/Malloc/Allocator.h>

#include <Core/Util/DebugStream.h>

#include <Core/Thread/Mutex.h>

#include <iostream>
#include <iomanip>
#include <fstream>

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
  // Inherit shared state from the framework
  d_sharedState = shared_state;

  // Construct handle to output stream
  d_dataArchiver = dynamic_cast<Output*>(getPort("output"));
  if (!d_dataArchiver) {
    throw InternalError("MD: couldn't get output port", __FILE__, __LINE__);
  }

  // Register the particle position variable with the scheduler
  dynamic_cast<Scheduler*>(getPort("scheduler"))->setPositionVar(d_label->pXLabel);

  // Extract the MD specific information from the input file
  ProblemSpecP md_ps = params->findBlock("MD");

  // Load/parse the forcefield we'll be using
  d_forcefield = ForcefieldFactory::create(params, d_system);

  // Determine the forcefield model we'll be using

  md_ps->get("coordinateFile", d_coordinateFile);

  // create and populate the MD System object
  d_system = scinew MDSystem(md_ps, grid, shared_state);
  d_system->markBoxChanged();

  // create the Nonbonded object via factory method
  d_nonbonded = NonbondedFactory::create(params, d_system);
  if (d_nonbonded->getType() == Nonbonded::LJ12_6) {
    dynamic_cast<AnalyticNonbonded*>(d_nonbonded)->setMDLabel(d_label);
  }

  // create the Electrostatics object via factory method
  d_electrostatics = ElectrostaticsFactory::create(params, d_system);
  if (d_electrostatics->getType() == Electrostatics::SPME) {
    dynamic_cast<SPME*>(d_electrostatics)->setMDLabel(d_label);

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

  // create and register MD materials (this is ill defined right now)
  d_material = scinew SimpleMaterial();
  d_sharedState->registerSimpleMaterial(d_material);

  // register permanent particle state; for relocation, etc
  registerPermanentParticleState(d_material);

  // do file I/O to get atom coordinates and simulation cell size
  extractCoordinates();
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

  printSchedule(level, md_cout, "MD::scheduleInitialize");

  Task* task = scinew Task("MD::initialize", this, &MD::initialize);

  task->computes(d_label->pXLabel);
  task->computes(d_label->pNonbondedForceLabel);
  task->computes(d_label->pElectrostaticsForceLabel);
  task->computes(d_label->pAccelLabel);
  task->computes(d_label->pVelocityLabel);
  task->computes(d_label->pEnergyLabel);
  task->computes(d_label->pMassLabel);
  task->computes(d_label->pChargeLabel);
  task->computes(d_label->pParticleIDLabel);
  task->computes(d_label->subSchedulerDependencyLabel);

  const MaterialSet* materials = d_sharedState->allMaterials();
  LoadBalancer* loadBal = sched->getLoadBalancer();
  const PatchSet* perProcPatches = loadBal->getPerProcessorPatchSet(level);

  sched->addTask(task, level->eachPatch(), materials);

  // Nonbonded initialization - OncePerProc, during initial (0th) timestep.
  // The required pXlabel is available to this OncePerProc task in the new_dw from the computes above
  scheduleNonbondedInitialize(sched, perProcPatches, materials, level);

  // Nonbonded initialization - OncePerProc, during initial (0th) timestep.
  //   This OncePerProc task requires nothing
  scheduleElectrostaticsInitialize(sched, perProcPatches, materials, level);
}

void MD::scheduleComputeStableTimestep(const LevelP& level,
                                       SchedulerP& sched)
{
  printSchedule(level, md_cout, "MD::scheduleComputeStableTimestep");

  Task* task = scinew Task("MD::computeStableTimestep", this, &MD::computeStableTimestep);

  task->requires(Task::NewDW, d_label->nonbondedEnergyLabel);
  task->requires(Task::NewDW, d_label->electrostaticReciprocalEnergyLabel);
  task->requires(Task::NewDW, d_label->electrostaticReciprocalStressLabel);

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

  sched->scheduleParticleRelocation(level, d_label->pXLabel_preReloc, d_sharedState->d_particleState_preReloc, d_label->pXLabel,
                                    d_sharedState->d_particleState, d_label->pParticleIDLabel, matls, 1);
}

void MD::scheduleNonbondedInitialize(SchedulerP& sched,
                                     const PatchSet* perProcPatches,
                                     const MaterialSet* matls,
                                     const LevelP& level)
{
  printSchedule(perProcPatches, md_cout, "MD::scheduleNonbondedInitialize");

  Task* task = scinew Task("MD::nonbondedInitialize", this, &MD::nonbondedInitialize);

  // This is during the initial timestep... no OldDW exists
  task->requires(Task::NewDW, d_label->pXLabel, Ghost::None, 0);

   // initialize reduction variable; van der Waals energy
  task->computes(d_label->nonbondedEnergyLabel);
  task->computes(d_label->nonbondedDependencyLabel);

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

  task->requires(Task::OldDW, d_label->nonbondedDependencyLabel);
  task->computes(d_label->nonbondedDependencyLabel);

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

  task->requires(Task::OldDW, d_label->pXLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
  task->requires(Task::OldDW, d_label->pNonbondedForceLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
  task->requires(Task::OldDW, d_label->pEnergyLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
  task->requires(Task::OldDW, d_label->pParticleIDLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
  task->requires(Task::OldDW, d_label->nonbondedDependencyLabel, Ghost::None, 0);

  task->computes(d_label->pNonbondedForceLabel_preReloc);
  task->computes(d_label->pEnergyLabel_preReloc);
  task->computes(d_label->nonbondedEnergyLabel);

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
      task->computes(d_label->electrostaticReciprocalEnergyLabel);
      task->computes(d_label->electrostaticReciprocalStressLabel);

      // sole variables
      task->computes(d_label->electrostaticsDependencyLabel);
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
  task->computes(d_label->electrostaticsDependencyLabel);

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

  task->requires(Task::OldDW, d_label->pXLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
  task->requires(Task::OldDW, d_label->pChargeLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
  task->requires(Task::OldDW, d_label->pParticleIDLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
  task->requires(Task::OldDW, d_label->electrostaticsDependencyLabel);
  task->requires(Task::OldDW, d_label->subSchedulerDependencyLabel, Ghost::None, 0);

  task->computes(d_label->subSchedulerDependencyLabel);
  task->computes(d_label->electrostaticReciprocalEnergyLabel);
  task->computes(d_label->electrostaticReciprocalStressLabel);

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
  task->requires(Task::OldDW, d_label->pElectrostaticsForceLabel, Ghost::None, 0);
  task->requires(Task::OldDW, d_label->pChargeLabel, Ghost:: Ghost::None, 0);
  task->requires(Task::NewDW, d_label->subSchedulerDependencyLabel, Ghost:: Ghost::None, 0);

  task->computes(d_label->pElectrostaticsForceLabel_preReloc);
  task->computes(d_label->pChargeLabel_preReloc);

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

  task->requires(Task::OldDW, d_label->pXLabel, Ghost::None, 0);
  task->requires(Task::NewDW, d_label->pNonbondedForceLabel_preReloc, Ghost::None, 0);
  task->requires(Task::NewDW, d_label->pElectrostaticsForceLabel_preReloc, Ghost::None, 0);
  task->requires(Task::OldDW, d_label->pAccelLabel, Ghost::None, 0);
  task->requires(Task::OldDW, d_label->pVelocityLabel, Ghost::None, 0);
  task->requires(Task::OldDW, d_label->pMassLabel, Ghost::None, 0);
  task->requires(Task::OldDW, d_label->pParticleIDLabel, Ghost::None, 0);
  task->requires(Task::OldDW, d_sharedState->get_delt_label());

  task->computes(d_label->pXLabel_preReloc);
  task->modifies(d_label->pNonbondedForceLabel_preReloc);
  task->modifies(d_label->pElectrostaticsForceLabel_preReloc);
  task->computes(d_label->pAccelLabel_preReloc);
  task->computes(d_label->pVelocityLabel_preReloc);
  task->computes(d_label->pMassLabel_preReloc);
  task->computes(d_label->pParticleIDLabel_preReloc);

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

  // loop through all patches
  unsigned int numAtoms = d_system->getNumAtoms();
  unsigned int numPatches = patches->size();
  for (unsigned int p = 0; p < numPatches; ++p) {
    const Patch* patch = patches->get(p);

    // get bounds of current patch to correctly initialize particles (atoms)
    IntVector low = patch->getCellLowIndex();
    IntVector high = patch->getCellHighIndex();

    // do this for each material
    unsigned int numMatls = matls->size();
    for (unsigned int m = 0; m < numMatls; ++m) {
      int matl = matls->get(m);

      ParticleVariable<Point> px;
      ParticleVariable<Vector> pforceNonbonded;
      ParticleVariable<Vector> pforceElectrostatics;
      ParticleVariable<Vector> paccel;
      ParticleVariable<Vector> pvelocity;
      ParticleVariable<double> penergy;
      ParticleVariable<double> pmass;
      ParticleVariable<double> pcharge;
      ParticleVariable<long64> pids;
      CCVariable<int> subSchedulerDependency;

      // eventually we'll need to use PFS for this
      std::vector<Atom> localAtoms;
      for (unsigned int i = 0; i < numAtoms; ++i) {
//        Vector reducedCoordinates = ((d_atomList[i].coords).asVector() * systemInverseCell);
//        IntVector cellCoordinates((reducedCoordinates * totalSystemExtent.asVector()).asPoint());
        // TODO make sure this is correct before deleting the above lines
        Level* level = ((Patch*)patch)->getLevel(0);
        level->setdCell(d_system->getBox() / totalSystemExtent.asVector());
        IntVector ptIndex = patch->getLevel()->getCellIndex(d_atomList[i].coords);
        if (containsAtom(low, high, ptIndex)) {
          localAtoms.push_back(d_atomList[i]);
        }
      }

      // insert particle type counting loop here

      ParticleSubset* pset = new_dw->createParticleSubset(localAtoms.size(), matl, patch);
      new_dw->allocateAndPut(px, d_label->pXLabel, pset);
      new_dw->allocateAndPut(pforceNonbonded, d_label->pNonbondedForceLabel, pset);
      new_dw->allocateAndPut(pforceElectrostatics, d_label->pElectrostaticsForceLabel, pset);
      new_dw->allocateAndPut(paccel, d_label->pAccelLabel, pset);
      new_dw->allocateAndPut(pvelocity, d_label->pVelocityLabel, pset);
      new_dw->allocateAndPut(penergy, d_label->pEnergyLabel, pset);
      new_dw->allocateAndPut(pmass, d_label->pMassLabel, pset);
      new_dw->allocateAndPut(pcharge, d_label->pChargeLabel, pset);
      new_dw->allocateAndPut(pids, d_label->pParticleIDLabel, pset);
      new_dw->allocateAndPut(subSchedulerDependency, d_label->subSchedulerDependencyLabel, matl, patch, Ghost::None, 0);
      subSchedulerDependency.initialize(0);

      int numParticles = pset->numParticles();
      for (int i = 0; i < numParticles; ++i) {
        Point pos = localAtoms[i].coords;
        px[i] = pos;
        pforceNonbonded[i] = Vector(0.0, 0.0, 0.0);
        pforceElectrostatics[i] = Vector(0.0, 0.0, 0.0);
        paccel[i] = Vector(0.0, 0.0, 0.0);
        pvelocity[i] = Vector(0.0, 0.0, 0.0);
        penergy[i] = 1.1;
        pmass[i] = 2.5;
        pcharge[i] = localAtoms[i].charge;
        pids[i] = patch->getID() * numAtoms + i;

        if (md_dbg.active()) {
          cerrLock.lock();
          std::cout.setf(std::ios_base::showpoint);  // print decimal and trailing zeros
          std::cout.setf(std::ios_base::left);  // pad after the value
          std::cout.setf(std::ios_base::uppercase);  // use upper-case scientific notation
          std::cout << std::setw(10) << " Patch_ID: " << std::setw(4) << patch->getID();
          std::cout << std::setw(14) << " Particle_ID: " << std::setw(4) << pids[i];
          std::cout << std::setw(12) << " Position: " << pos;
          std::cout << std::endl;
          cerrLock.unlock();
        }
      }
    }
  }
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

  new_dw->get(vdwEnergy, d_label->nonbondedEnergyLabel);
  new_dw->get(spmeFourierEnergy, d_label->electrostaticReciprocalEnergyLabel);
  new_dw->get(spmeFourierStress, d_label->electrostaticReciprocalStressLabel);

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
  for (unsigned int p = 0; p < numPatches; ++p) {
    const Patch* patch = patches->get(p);

    // do this for each material; for this example, there is only a single material, material "0"
    unsigned int numMatls = matls->size();
    for (unsigned int m = 0; m < numMatls; ++m) {
      int matl = matls->get(m);

      ParticleSubset* pset = old_dw->getParticleSubset(matl, patch);
      ParticleSubset* delset = scinew ParticleSubset(0, matl, patch);

      // requires variables
      constParticleVariable<Point> px;
      constParticleVariable<Vector> paccel;
      constParticleVariable<Vector> pvelocity;
      constParticleVariable<double> pmass;
      constParticleVariable<long64> pids;
      old_dw->get(px, d_label->pXLabel, pset);
      old_dw->get(paccel, d_label->pAccelLabel, pset);
      old_dw->get(pvelocity, d_label->pVelocityLabel, pset);
      old_dw->get(pmass, d_label->pMassLabel, pset);
      old_dw->get(pids, d_label->pParticleIDLabel, pset);

      // computes variables
      ParticleVariable<Point> pxnew;
      ParticleVariable<Vector> pforceNonbonded;
      ParticleVariable<Vector> pforceElectrostatics;
      ParticleVariable<Vector> paccelnew;
      ParticleVariable<Vector> pvelocitynew;
      ParticleVariable<double> pmassnew;
      ParticleVariable<long64> pidsnew;
      new_dw->allocateAndPut(pxnew, d_label->pXLabel_preReloc, pset);
      new_dw->getModifiable(pforceNonbonded, d_label->pNonbondedForceLabel_preReloc, pset);
      new_dw->getModifiable(pforceElectrostatics, d_label->pElectrostaticsForceLabel_preReloc, pset);
      new_dw->allocateAndPut(paccelnew, d_label->pAccelLabel_preReloc, pset);
      new_dw->allocateAndPut(pvelocitynew, d_label->pVelocityLabel_preReloc, pset);
      new_dw->allocateAndPut(pmassnew, d_label->pMassLabel_preReloc, pset);
      new_dw->allocateAndPut(pidsnew, d_label->pParticleIDLabel_preReloc, pset);

      // get delT
      delt_vartype delT;
      old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches));

      // loop over the local atoms
      for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); ++iter) {
        particleIndex idx = *iter;

        // carry these values over for now
        pmassnew[idx] = pmass[idx];
        pidsnew[idx] = pids[idx];

        // update position (Velocity Verlet)
        paccelnew[idx] = pforceNonbonded[idx] + pforceElectrostatics[idx] / pmass[idx];
        pvelocitynew[idx] = pvelocity[idx] + paccel[idx] * delT;
        pxnew[idx] = px[idx] + pvelocity[idx] + pvelocitynew[idx] * 0.5 * delT;

        pxnew[idx] = px[idx];

        if (md_dbg.active()) {
          cerrLock.lock();
          std::cout << "PatchID: " << std::setw(4) << patch->getID() << std::setw(6);
          std::cout << "ParticleID: " << std::setw(6) << pidsnew[idx] << std::setw(6);
          std::cout << "New Position: [";
          std::cout << std::setw(10) << std::setprecision(6) << pxnew[idx].x();
          std::cout << std::setw(10) << std::setprecision(6) << pxnew[idx].y();
          std::cout << std::setprecision(6) << pxnew[idx].z() << std::setw(4) << "]";
          std::cout << std::endl;
          cerrLock.unlock();
        }
      }  // end atom loop

      new_dw->deleteParticles(delset);

    }  // end materials loop

  }  // end patch loop

  d_system->clearBoxChanged();
}

void MD::registerPermanentParticleState(SimpleMaterial* matl)
{
  // load up the ParticleVariables we want to register for relocation
  d_particleState_preReloc.push_back(d_label->pNonbondedForceLabel_preReloc);
  d_particleState.push_back(d_label->pNonbondedForceLabel);

  d_particleState_preReloc.push_back(d_label->pElectrostaticsForceLabel_preReloc);
  d_particleState.push_back(d_label->pElectrostaticsForceLabel);

  d_particleState_preReloc.push_back(d_label->pAccelLabel_preReloc);
  d_particleState.push_back(d_label->pAccelLabel);

  d_particleState_preReloc.push_back(d_label->pVelocityLabel_preReloc);
  d_particleState.push_back(d_label->pVelocityLabel);

  d_particleState_preReloc.push_back(d_label->pEnergyLabel_preReloc);
  d_particleState.push_back(d_label->pEnergyLabel);

  d_particleState_preReloc.push_back(d_label->pMassLabel_preReloc);
  d_particleState.push_back(d_label->pMassLabel);

  d_particleState_preReloc.push_back(d_label->pChargeLabel_preReloc);
  d_particleState.push_back(d_label->pChargeLabel);

  d_particleState_preReloc.push_back(d_label->pParticleIDLabel_preReloc);
  d_particleState.push_back(d_label->pParticleIDLabel);

  // register the particle states with the shared SimulationState for persistence across timesteps
  d_sharedState->d_particleState_preReloc.push_back(d_particleState_preReloc);
  d_sharedState->d_particleState.push_back(d_particleState);
}

void MD::extractCoordinates()
{
  std::ifstream inputFile;
  inputFile.open(d_coordinateFile.c_str());
  if (!inputFile.is_open()) {
    std::string message = "\tCannot open input file: " + d_coordinateFile;
    throw ProblemSetupException(message, __FILE__, __LINE__);
  }

  // do file IO to extract atom coordinates and charge
  std::string line;
  unsigned int numRead;
  unsigned int numAtoms = d_system->getNumAtoms();
  for (unsigned int i = 0; i < numAtoms; ++i) {
    // get the atom coordinates
    getline(inputFile, line);
    double x, y, z;
    double charge;
    numRead = sscanf(line.c_str(), "%lf %lf %lf %lf", &x, &y, &z, &charge);
    if (numRead != 4) {
      std::string message = "\tMalformed input file. Should have [x,y,z] coordinates and [charge] per line: ";
      throw ProblemSetupException(message, __FILE__, __LINE__);
    }

    //FIXME This is hacky!! Fix for generic case of wrapping arbitrary coordinates into arbitrary unit cells using
    //  reduced coordinate transformation!  -- JBH 5/9/13
    Vector box = d_system->getBox();
    if (x < 0) { x += box.x(); }
    if (y < 0) { y += box.y(); }
    if (z < 0) { z += box.z(); }

    if (x >= box.x()) { x -= box.x(); }
    if (y >= box.y()) { y -= box.y(); }
    if (z >= box.z()) { z -= box.z(); }

    Atom atom(Point(x, y, z), charge);

    d_atomList.push_back(atom);
  }
  inputFile.close();
}
