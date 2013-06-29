/*

 The MIT License

 Copyright (c) 1997-2013 The University of Utah

 License for the specific language governing rights and limitations under
 Permission is hereby granted, free of charge, to any person obtaining a
 copy of this software and associated documentation files (the "Software"),
 to deal in the Software without restriction, including without limitation
 the rights to use, copy, modify, merge, publish, distribute, sublicense,
 and/or sell copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included
 in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 DEALINGS IN THE SOFTWARE.

 */

#include <CCA/Components/MD/MD.h>
#include <CCA/Components/MD/ElectrostaticsFactory.h>
#include <CCA/Components/MD/SPME.h>
#include <CCA/Components/MD/NonBondedFactory.h>
#include <CCA/Components/MD/AnalyticNonBonded.h>
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

  d_sharedState = shared_state;

  d_dataArchiver = dynamic_cast<Output*>(getPort("output"));
  if (!d_dataArchiver) {
    throw InternalError("MD: couldn't get output port", __FILE__, __LINE__);
  }

  dynamic_cast<Scheduler*>(getPort("scheduler"))->setPositionVar(d_lb->pXLabel);

  // get path and name of the file with atom information
  ProblemSpecP md_ps = params->findBlock("MD");
  md_ps->get("coordinateFile", d_coordinateFile);

  // create and populate the MD System object
  d_system = scinew MDSystem(md_ps, grid, shared_state);
  d_system->markBoxChanged();

  // create the NonBonded object via factory method
  d_nonbonded = NonBondedFactory::create(params, d_system);
  if (d_nonbonded->getType() == NonBonded::LJ12_6) {
    dynamic_cast<AnalyticNonBonded*>(d_nonbonded)->setMDLabel(d_lb);
  }

  // create the Electrostatics object via factory method
  d_electrostatics = ElectrostaticsFactory::create(params, d_system);
  if (d_electrostatics->getType() == Electrostatics::SPME) {
    dynamic_cast<SPME*>(d_electrostatics)->setMDLabel(d_lb);
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
  printSchedule(level, md_cout, "MD::scheduleInitialize");

  // this task creates all the ParticleVariables and places atoms on correct patches spatially
  Task* mdInitTask = scinew Task("MD::initialize", this, &MD::initialize);
  mdInitTask->computes(d_lb->pXLabel);
  mdInitTask->computes(d_lb->pNonbondedForceLabel);
  mdInitTask->computes(d_lb->pElectrostaticsForceLabel);
  mdInitTask->computes(d_lb->pAccelLabel);
  mdInitTask->computes(d_lb->pVelocityLabel);
  mdInitTask->computes(d_lb->pEnergyLabel);
  mdInitTask->computes(d_lb->pMassLabel);
  mdInitTask->computes(d_lb->pChargeLabel);
  mdInitTask->computes(d_lb->pParticleIDLabel);
  sched->addTask(mdInitTask, level->eachPatch(), d_sharedState->allMaterials());

  // initialize nonbonded instance
  Task* nonbondedInitTask = scinew Task("MD::nonbondedInitialize", this, &MD::nonbondedInitialize);
  nonbondedInitTask->requires(Task::NewDW, d_lb->pXLabel, Ghost::None);
  nonbondedInitTask->computes(d_lb->vdwEnergyLabel);

  nonbondedInitTask->setType(Task::OncePerProc);
  LoadBalancer* loadBal = sched->getLoadBalancer();
  GridP grid = level->getGrid();
  const PatchSet* perprocPatches = loadBal->getPerProcessorPatchSet(grid);
  sched->addTask(nonbondedInitTask, perprocPatches, d_sharedState->allMaterials());

  // initialize electrostatics instance; if we're doing electrostatics
  if (d_electrostatics->getType() != Electrostatics::NONE) {
    Task* electrostaticsInitTask = scinew Task("MD::electrostaticsInitialize", this, &MD::electrostaticsInitialize);
    if (d_electrostatics->getType() == Electrostatics::SPME) {
      electrostaticsInitTask->computes(d_lb->spmeFourierEnergyLabel);
      electrostaticsInitTask->computes(d_lb->spmeFourierStressLabel);
      electrostaticsInitTask->computes(d_lb->forwardTransformPlanLabel);
      electrostaticsInitTask->computes(d_lb->backwardTransformPlanLabel);
      electrostaticsInitTask->computes(d_lb->globalQLabel);
    }
    electrostaticsInitTask->setType(Task::OncePerProc);
    LoadBalancer* loadBal = sched->getLoadBalancer();
    GridP grid = level->getGrid();
    const PatchSet* perprocPatches = loadBal->getPerProcessorPatchSet(grid);
    sched->addTask(electrostaticsInitTask, perprocPatches, d_sharedState->allMaterials());
  }
}

void MD::scheduleComputeStableTimestep(const LevelP& level,
                                       SchedulerP& sched)
{
  printSchedule(level, md_cout, "MD::scheduleComputeStableTimestep");

  Task* task = scinew Task("MD::computeStableTimestep", this, &MD::computeStableTimestep);

  task->requires(Task::NewDW, d_lb->vdwEnergyLabel);
  task->requires(Task::NewDW, d_lb->spmeFourierEnergyLabel);
  task->requires(Task::NewDW, d_lb->spmeFourierStressLabel);

  task->computes(d_sharedState->get_delt_label(), level.get_rep());

  sched->addTask(task, level->eachPatch(), d_sharedState->allMaterials());
}

void MD::scheduleTimeAdvance(const LevelP& level,
                             SchedulerP& sched)
{
  printSchedule(level, md_cout, "MD::scheduleTimeAdvance");

  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_sharedState->allMaterials();

  scheduleNonbondedCalculate(sched, patches, matls, level);

  scheduleElectrostaticsCalculate(sched, patches, matls, level);

  scheduleUpdatePosition(sched, patches, matls, level);

  sched->scheduleParticleRelocation(level, d_lb->pXLabel_preReloc, d_sharedState->d_particleState_preReloc, d_lb->pXLabel,
                                    d_sharedState->d_particleState, d_lb->pParticleIDLabel, matls, 1);
}

void MD::scheduleNonbondedCalculate(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls,
                                    const LevelP& level)
{
  printSchedule(patches, md_cout, "MD::scheduleNonbondedCalculate");

  Task* task = scinew Task("MD::nonbondedCalculate", this, &MD::nonbondedCalculate, d_subScheduler, level);

  task->requires(Task::OldDW, d_lb->pXLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, d_lb->pNonbondedForceLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, d_lb->pEnergyLabel, Ghost::AroundNodes, SHRT_MAX);

  task->computes(d_lb->pNonbondedForceLabel_preReloc);
  task->computes(d_lb->pEnergyLabel_preReloc);
  task->computes(d_lb->vdwEnergyLabel);

  sched->addTask(task, patches, matls);
}

void MD::scheduleElectrostaticsCalculate(SchedulerP& sched,
                                         const PatchSet* patches,
                                         const MaterialSet* matls,
                                         const LevelP& level)
{
  printSchedule(patches, md_cout, "MD::scheduleElectrostaticsCalculate");

  Task* task = scinew Task("electrostaticsCalculate", this, &MD::electrostaticsCalculate, d_subScheduler, level);

  task->requires(Task::OldDW, d_lb->pXLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, d_lb->pElectrostaticsForceLabel_preReloc, Ghost::None, 0);
  task->requires(Task::OldDW, d_lb->pChargeLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, d_lb->pParticleIDLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, d_lb->pParticleIDLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, d_lb->pParticleIDLabel, Ghost::AroundNodes, SHRT_MAX);

  task->requires(Task::OldDW, d_lb->forwardTransformPlanLabel);
  task->requires(Task::OldDW, d_lb->backwardTransformPlanLabel);
  task->requires(Task::OldDW, d_lb->globalQLabel);

  task->computes(d_lb->pElectrostaticsForceLabel_preReloc);
  task->computes(d_lb->pChargeLabel_preReloc);
  task->computes(d_lb->forwardTransformPlanLabel);
  task->computes(d_lb->backwardTransformPlanLabel);
  task->computes(d_lb->globalQLabel);

  // reduction variables
  task->computes(d_lb->spmeFourierEnergyLabel);
  task->computes(d_lb->spmeFourierStressLabel);

  sched->addTask(task, patches, matls);
}

void MD::scheduleInterpolateParticlesToGrid(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  printSchedule(patches, md_cout, "MD::scheduleInterpolateParticlesToGrid");
}

void MD::scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
                                                 const PatchSet* patches,
                                                 const MaterialSet* matls)
{
  printSchedule(patches, md_cout, "MD::scheduleInterpolateToParticlesAndUpdate");
}

void MD::scheduleUpdatePosition(SchedulerP& sched,
                                const PatchSet* patches,
                                const MaterialSet* matls,
                                const LevelP& level)
{
  printSchedule(patches, md_cout, "MD::scheduleUpdatePosition");

  Task* task = scinew Task("updatePosition", this, &MD::updatePosition);

  task->requires(Task::OldDW, d_lb->pXLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::NewDW, d_lb->pNonbondedForceLabel_preReloc, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::NewDW, d_lb->pElectrostaticsForceLabel_preReloc, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, d_lb->pAccelLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, d_lb->pVelocityLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, d_lb->pMassLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, d_lb->pParticleIDLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, d_sharedState->get_delt_label());

  task->computes(d_lb->pXLabel_preReloc);
  task->modifies(d_lb->pNonbondedForceLabel_preReloc);
  task->modifies(d_lb->pElectrostaticsForceLabel_preReloc);
  task->computes(d_lb->pAccelLabel_preReloc);
  task->computes(d_lb->pVelocityLabel_preReloc);
  task->computes(d_lb->pMassLabel_preReloc);
  task->computes(d_lb->pParticleIDLabel_preReloc);

  sched->addTask(task, patches, matls);
}

void MD::initialize(const ProcessorGroup* pg,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::initialize");

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

      // eventually we'll need to use PFS for this
      vector<Atom> localAtoms;
      for (unsigned int i = 0; i < numAtoms; ++i) {
        if (containsAtom(low, high, d_atomList[i].coords)) {
          localAtoms.push_back(d_atomList[i]);
        }
      }
      // insert particle type counting loop here

      ParticleSubset* pset = new_dw->createParticleSubset(localAtoms.size(), matl, patch);
      new_dw->allocateAndPut(px, d_lb->pXLabel, pset);
      new_dw->allocateAndPut(pforceNonbonded, d_lb->pNonbondedForceLabel, pset);
      new_dw->allocateAndPut(pforceElectrostatics, d_lb->pElectrostaticsForceLabel, pset);
      new_dw->allocateAndPut(paccel, d_lb->pAccelLabel, pset);
      new_dw->allocateAndPut(pvelocity, d_lb->pVelocityLabel, pset);
      new_dw->allocateAndPut(penergy, d_lb->pEnergyLabel, pset);
      new_dw->allocateAndPut(pmass, d_lb->pMassLabel, pset);
      new_dw->allocateAndPut(pcharge, d_lb->pChargeLabel, pset);
      new_dw->allocateAndPut(pids, d_lb->pParticleIDLabel, pset);

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

void MD::registerPermanentParticleState(SimpleMaterial* matl)
{
  // load up the ParticleVariables we want to register for relocation
  d_particleState_preReloc.push_back(d_lb->pNonbondedForceLabel_preReloc);
  d_particleState.push_back(d_lb->pNonbondedForceLabel);

  d_particleState_preReloc.push_back(d_lb->pElectrostaticsForceLabel_preReloc);
  d_particleState.push_back(d_lb->pElectrostaticsForceLabel);

  d_particleState_preReloc.push_back(d_lb->pAccelLabel_preReloc);
  d_particleState.push_back(d_lb->pAccelLabel);

  d_particleState_preReloc.push_back(d_lb->pVelocityLabel_preReloc);
  d_particleState.push_back(d_lb->pVelocityLabel);

  d_particleState_preReloc.push_back(d_lb->pEnergyLabel_preReloc);
  d_particleState.push_back(d_lb->pEnergyLabel);

  d_particleState_preReloc.push_back(d_lb->pMassLabel_preReloc);
  d_particleState.push_back(d_lb->pMassLabel);

  d_particleState_preReloc.push_back(d_lb->pChargeLabel_preReloc);
  d_particleState.push_back(d_lb->pChargeLabel);

  d_particleState_preReloc.push_back(d_lb->pParticleIDLabel_preReloc);
  d_particleState.push_back(d_lb->pParticleIDLabel);

  // register the particle states with the shared SimulationState for persistence across timesteps
  d_sharedState->d_particleState_preReloc.push_back(d_particleState_preReloc);
  d_sharedState->d_particleState.push_back(d_particleState);
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

  new_dw->get(vdwEnergy, d_lb->vdwEnergyLabel);
  new_dw->get(spmeFourierEnergy, d_lb->spmeFourierEnergyLabel);
  new_dw->get(spmeFourierStress, d_lb->spmeFourierStressLabel);

  proc0thread0cout << std::endl;
  proc0thread0cout << "-----------------------------------------------------"           << std::endl;
  proc0thread0cout << "Total Energy   = " << std::setprecision(16) << vdwEnergy         << std::endl;
  proc0thread0cout << "-----------------------------------------------------"           << std::endl;
  proc0thread0cout << "Fourier Energy = " << std::setprecision(16) << spmeFourierEnergy << std::endl;
  proc0thread0cout << "-----------------------------------------------------"           << std::endl;
  proc0thread0cout << "Fourier Stress = " << std::setprecision(16) << spmeFourierStress << std::endl;
  proc0thread0cout << "-----------------------------------------------------"           << std::endl;
  proc0thread0cout << std::endl;

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

  d_nonbonded->setup(pg, patches, matls, old_dw, new_dw);
}

void MD::nonbondedCalculate(const ProcessorGroup* pg,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* parentOldDW,
                            DataWarehouse* parentNewDW,
                            SchedulerP subscheduler,
                            const LevelP level)
{
  printTask(patches, md_cout, "MD::nonbondedCalculate");

  d_nonbonded->calculate(pg, patches, matls, parentOldDW, parentNewDW, subscheduler, level);

  d_nonbonded->finalize(pg, patches, matls, parentOldDW, parentNewDW);
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
                                 const PatchSubset* perprocPatches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* parentOldDW,
                                 DataWarehouse* parentNewDW,
                                 SchedulerP subscheduler,
                                 const LevelP level)
{
  printTask(perprocPatches, md_cout, "MD::electrostaticsCalculate");

  d_electrostatics->calculate(pg, perprocPatches, matls, parentOldDW, parentNewDW, subscheduler, level);

  d_electrostatics->finalize(pg, perprocPatches, matls, parentOldDW, parentNewDW);
}

void MD::interpolateParticlesToGrid(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* matls,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::interpolateChargesToGrid");
}

void MD::interpolateToParticlesAndUpdate(const ProcessorGroup* pg,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::interpolateToParticlesAndUpdate");
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
      old_dw->get(px, d_lb->pXLabel, pset);
      old_dw->get(paccel, d_lb->pAccelLabel, pset);
      old_dw->get(pvelocity, d_lb->pVelocityLabel, pset);
      old_dw->get(pmass, d_lb->pMassLabel, pset);
      old_dw->get(pids, d_lb->pParticleIDLabel, pset);

      // computes variables
      ParticleVariable<Point> pxnew;
      ParticleVariable<Vector> pforceNonbonded;
      ParticleVariable<Vector> pforceElectrostatics;
      ParticleVariable<Vector> paccelnew;
      ParticleVariable<Vector> pvelocitynew;
      ParticleVariable<double> pmassnew;
      ParticleVariable<long64> pidsnew;
      new_dw->allocateAndPut(pxnew, d_lb->pXLabel_preReloc, pset);
      new_dw->getModifiable(pforceNonbonded, d_lb->pNonbondedForceLabel_preReloc, pset);
      new_dw->getModifiable(pforceElectrostatics, d_lb->pElectrostaticsForceLabel_preReloc, pset);
      new_dw->allocateAndPut(paccelnew, d_lb->pAccelLabel_preReloc, pset);
      new_dw->allocateAndPut(pvelocitynew, d_lb->pVelocityLabel_preReloc, pset);
      new_dw->allocateAndPut(pmassnew, d_lb->pMassLabel_preReloc, pset);
      new_dw->allocateAndPut(pidsnew, d_lb->pParticleIDLabel_preReloc, pset);

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

void MD::extractCoordinates()
{
  std::ifstream inputFile;
  inputFile.open(d_coordinateFile.c_str());
  if (!inputFile.is_open()) {
    string message = "\tCannot open input file: " + d_coordinateFile;
    throw ProblemSetupException(message, __FILE__, __LINE__);
  }

  // do file IO to extract atom coordinates and charge
  string line;
  unsigned int numRead;
  unsigned int numAtoms = d_system->getNumAtoms();
  for (unsigned int i = 0; i < numAtoms; ++i) {
    // get the atom coordinates
    getline(inputFile, line);
    double x, y, z;
    double charge;
    numRead = sscanf(line.c_str(), "%lf %lf %lf %lf", &x, &y, &z, &charge);
    if (numRead != 4) {
      string message = "\tMalformed input file. Should have [x,y,z] coordinates and [charge] per line: ";
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
