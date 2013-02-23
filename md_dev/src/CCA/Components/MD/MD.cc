/*

 The MIT License

 Copyright (c) 1997-2012 The University of Utah

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
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
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
#include <CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Mutex.h>

#include <iostream>
#include <iomanip>
#include <fstream>

#include <sci_defs/fftw_defs.h>

using namespace Uintah;

extern SCIRun::Mutex cerrLock;

static DebugStream md_dbg("MDDebug", false);
static DebugStream md_cout("MDCout", false);
static DebugStream md_spme("MDSPME", false);

MD::MD(const ProcessorGroup* myworld) :
    UintahParallelComponent(myworld)
{
  lb = scinew MDLabel();
}

MD::~MD()
{
  delete lb;
  delete system;
  delete electrostatics;
}

void MD::problemSetup(const ProblemSpecP& params,
                      const ProblemSpecP& restart_prob_spec,
                      GridP& grid,
                      SimulationStateP& shared_state)
{
  printTask(md_cout, "MD::problemSetup");

  sharedState = shared_state;
  dynamic_cast<Scheduler*>(getPort("scheduler"))->setPositionVar(lb->pXLabel);
  ProblemSpecP md_ps = params->findBlock("MD");

  md_ps->get("coordinateFile", coordinateFile);
  md_ps->get("numAtoms", numAtoms);
  md_ps->get("boxSize", box);
  md_ps->get("cutoffRadius", cutoffRadius);
  md_ps->get("R12", R12);
  md_ps->get("R6", R6);

  // create and populate the MD System object
  system = scinew MDSystem(md_ps);

  // create the Electrostatics object via factory method
  electrostatics = ElectrostaticsFactory::create(params, system);

  // create and register MD materials (this is ill defined right now)
  material = scinew SimpleMaterial();
  sharedState->registerSimpleMaterial(material);

  // register permanent particle state; for relocation, etc
  registerPermanentParticleState(material);

  // do file I/O to get atom coordinates and simulation cell size
  extractCoordinates();

  // for neighbor indices; one list for each atom
  for (unsigned int i = 0; i < numAtoms; i++) {
    neighborList.push_back(vector<int>(0));
  }

  // create neighbor list for each atom in the system
  generateNeighborList();
}

void MD::scheduleInitialize(const LevelP& level,
                            SchedulerP& sched)
{
  printSchedule(level, md_cout, "MD::scheduleInitialize");

  Task* task = scinew Task("MD::initialize", this, &MD::initialize);
  task->computes(lb->pXLabel);
  task->computes(lb->pForceLabel);
  task->computes(lb->pAccelLabel);
  task->computes(lb->pVelocityLabel);
  task->computes(lb->pEnergyLabel);
  task->computes(lb->pMassLabel);
  task->computes(lb->pChargeLabel);
  task->computes(lb->pParticleIDLabel);
  task->computes(lb->vdwEnergyLabel);
  sched->addTask(task, level->eachPatch(), sharedState->allMaterials());
}

void MD::scheduleComputeStableTimestep(const LevelP& level,
                                       SchedulerP& sched)
{
  printSchedule(level, md_cout, "MD::scheduleComputeStableTimestep");

  Task* task = scinew Task("MD::computeStableTimestep", this, &MD::computeStableTimestep);
  task->requires(Task::NewDW, lb->vdwEnergyLabel);
  task->computes(sharedState->get_delt_label(), level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState->allMaterials());
}

void MD::scheduleTimeAdvance(const LevelP& level,
                             SchedulerP& sched)
{
  printSchedule(level, md_cout, "MD::scheduleTimeAdvance");

  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = sharedState->allMaterials();

  scheduleCalculateNonBondedForces(sched, patches, matls);
  scheduleInterpolateParticlesToGrid(sched, patches, matls);
  schedulePerformSPME(sched, patches, matls);
  scheduleInterpolateToParticlesAndUpdate(sched, patches, matls);

  sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc, sharedState->d_particleState_preReloc, lb->pXLabel,
                                    sharedState->d_particleState, lb->pParticleIDLabel, matls, 1);
}

void MD::computeStableTimestep(const ProcessorGroup* pg,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::computeStableTimestep");

  if (pg->myrank() == 0) {
    sum_vartype vdwEnergy;
    new_dw->get(vdwEnergy, lb->vdwEnergyLabel);
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << "Total Energy = " << std::setprecision(16) << vdwEnergy << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << std::endl;
  }
  new_dw->put(delt_vartype(1), sharedState->get_delt_label(), getLevel(patches));
}

void MD::scheduleCalculateNonBondedForces(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSet* matls)
{
  printSchedule(patches, md_cout, "MD::scheduleCalculateNonBondedForces");

  Task* task = scinew Task("MD::calculateNonBondedForces", this, &MD::calculateNonBondedForces);

  task->requires(Task::OldDW, lb->pXLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, lb->pForceLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, lb->pEnergyLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, lb->pParticleIDLabel, Ghost::AroundNodes, SHRT_MAX);

  task->computes(lb->pForceLabel_preReloc);
  task->computes(lb->pEnergyLabel_preReloc);
  task->computes(lb->vdwEnergyLabel);

  sched->addTask(task, patches, matls);
}

void MD::scheduleInterpolateParticlesToGrid(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  printSchedule(patches, md_cout, "MD::scheduleInterpolateParticlesToGrid");

  Task* task = scinew Task("MD::interpolateParticlesToGrid", this, &MD::interpolateParticlesToGrid);

  sched->addTask(task, patches, matls);
}

void MD::schedulePerformSPME(SchedulerP& sched,
                             const PatchSet* patches,
                             const MaterialSet* matls)
{
  printSchedule(patches, md_cout, "MD::schedulePerformSPME");

  Task* task = scinew Task("performSPME", this, &MD::performSPME);

  task->requires(Task::OldDW, lb->pXLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, lb->pForceLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, lb->pChargeLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, lb->pParticleIDLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, sharedState->get_delt_label());

  task->computes(lb->pXLabel_preReloc);
  task->computes(lb->pForceLabel_preReloc);
  task->computes(lb->pChargeLabel_preReloc);
  task->computes(lb->pParticleIDLabel_preReloc);

  sched->addTask(task, patches, matls);
}

void MD::scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
                                                 const PatchSet* patches,
                                                 const MaterialSet* matls)
{
  printSchedule(patches, md_cout, "MD::scheduleInterpolateToParticlesAndUpdate");

  Task* task = scinew Task("interpolateToParticlesAndUpdate", this, &MD::interpolateToParticlesAndUpdate);

  task->requires(Task::OldDW, lb->pXLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, lb->pForceLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, lb->pAccelLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, lb->pVelocityLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, lb->pMassLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, lb->pChargeLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, lb->pParticleIDLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, sharedState->get_delt_label());

  task->computes(lb->pXLabel_preReloc);
  task->computes(lb->pAccelLabel_preReloc);
  task->computes(lb->pVelocityLabel_preReloc);
  task->computes(lb->pMassLabel_preReloc);
  task->computes(lb->pChargeLabel_preReloc);
  task->computes(lb->pParticleIDLabel_preReloc);

  sched->addTask(task, patches, matls);
}

void MD::extractCoordinates()
{
  std::ifstream inputFile;
  inputFile.open(coordinateFile.c_str());
  if (!inputFile.is_open()) {
    string message = "\tCannot open input file: " + coordinateFile;
    throw ProblemSetupException(message, __FILE__, __LINE__);
  }

  // do file IO to extract atom coordinates
  string line;
  unsigned int numRead;
  for (unsigned int i = 0; i < numAtoms; i++) {
    // get the atom coordinates
    getline(inputFile, line);
    double x, y, z;
    numRead = sscanf(line.c_str(), "%lf %lf %lf", &x, &y, &z);
    if (numRead != 3) {
      string message = "\tMalformed input file. Should have [x,y,z] coordinates per line: ";
      throw ProblemSetupException(message, __FILE__, __LINE__);
    }
    Point pnt(x, y, z);
    atomList.push_back(pnt);
  }
  inputFile.close();
}

void MD::generateNeighborList()
{
  double r2;
  Vector reducedCoordinates;
  double cut_sq = cutoffRadius * cutoffRadius;
  for (unsigned int i = 0; i < numAtoms; i++) {
    for (unsigned int j = 0; j < numAtoms; j++) {
      if (i != j) {
        // the vector distance between atom i and j
        reducedCoordinates = atomList[i] - atomList[j];

        // this is required for periodic boundary conditions
        reducedCoordinates -= (reducedCoordinates / box).vec_rint() * box;

        // eliminate atoms outside of cutoff radius, add those within as neighbors
        if ((fabs(reducedCoordinates[0]) < cutoffRadius) && (fabs(reducedCoordinates[1]) < cutoffRadius)
            && (fabs(reducedCoordinates[2]) < cutoffRadius)) {
          double reducedX = reducedCoordinates[0] * reducedCoordinates[0];
          double reducedY = reducedCoordinates[1] * reducedCoordinates[1];
          double reducedZ = reducedCoordinates[2] * reducedCoordinates[2];
          r2 = sqrt(reducedX + reducedY + reducedZ);
          // only add neighbor atoms within spherical cut-off around atom "i"
          if (r2 < cut_sq) {
            neighborList[i].push_back(j);
          }
        }
      }
    }
  }
}

bool MD::isNeighbor(const Point* atom1,
                    const Point* atom2)
{
  double r2;
  Vector reducedCoordinates;
  double cut_sq = cutoffRadius * cutoffRadius;

  // the vector distance between atom 1 and 2
  reducedCoordinates = *atom1 - *atom2;

  // this is required for periodic boundary conditions
  reducedCoordinates -= (reducedCoordinates / box).vec_rint() * box;

  // check if outside of cutoff radius
  if ((fabs(reducedCoordinates[0]) < cutoffRadius) && (fabs(reducedCoordinates[1]) < cutoffRadius)
      && (fabs(reducedCoordinates[2]) < cutoffRadius)) {
    r2 = sqrt(pow(reducedCoordinates[0], 2.0) + pow(reducedCoordinates[1], 2.0) + pow(reducedCoordinates[2], 2.0));
    return r2 < cut_sq;
  }
  return false;
}

void MD::initialize(const ProcessorGroup* pg,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::initialize");

  // initialize electrostatics object
  electrostatics->initialize(system, patches, matls);

  // loop through all patches
  unsigned int numPatches = patches->size();
  for (unsigned int p = 0; p < numPatches; p++) {
    const Patch* patch = patches->get(p);

    // get bounds of current patch to correctly initialize particles (atoms)
    IntVector low = patch->getExtraCellLowIndex();
    IntVector high = patch->getExtraCellHighIndex();

    // do this for each material
    unsigned int numMatls = matls->size();
    for (unsigned int m = 0; m < numMatls; m++) {
      int matl = matls->get(m);

      ParticleVariable<Point> px;
      ParticleVariable<Vector> pforce;
      ParticleVariable<Vector> paccel;
      ParticleVariable<Vector> pvelocity;
      ParticleVariable<double> penergy;
      ParticleVariable<double> pmass;
      ParticleVariable<double> pcharge;
      ParticleVariable<long64> pids;

      // eventually we'll need to use PFS for this
      vector<Point> localAtoms;
      for (unsigned int i = 0; i < numAtoms; i++) {
        if (containsAtom(low, high, atomList[i])) {
          localAtoms.push_back(atomList[i]);
        }
      }

      ParticleSubset* pset = new_dw->createParticleSubset(localAtoms.size(), matl, patch);
      new_dw->allocateAndPut(px, lb->pXLabel, pset);
      new_dw->allocateAndPut(pforce, lb->pForceLabel, pset);
      new_dw->allocateAndPut(paccel, lb->pAccelLabel, pset);
      new_dw->allocateAndPut(pvelocity, lb->pVelocityLabel, pset);
      new_dw->allocateAndPut(penergy, lb->pEnergyLabel, pset);
      new_dw->allocateAndPut(pmass, lb->pMassLabel, pset);
      new_dw->allocateAndPut(pcharge, lb->pChargeLabel, pset);
      new_dw->allocateAndPut(pids, lb->pParticleIDLabel, pset);

      int numParticles = pset->numParticles();
      for (int i = 0; i < numParticles; i++) {
        Point pos = localAtoms[i];
        px[i] = pos;
        pforce[i] = Vector(0.0, 0.0, 0.0);
        paccel[i] = Vector(0.0, 0.0, 0.0);
        pvelocity[i] = Vector(0.0, 0.0, 0.0);
        penergy[i] = 0.0;
        pmass[i] = 2.5;
        pcharge[i] = 0.0;
        pids[i] = patch->getID() * numAtoms + i;

        // TODO update this with new VarLabels
        if (md_dbg.active()) {
          cerrLock.unlock();
          std::cout.setf(std::ios_base::showpoint);  // print decimal and trailing zeros
          std::cout.setf(std::ios_base::left);  // pad after the value
          std::cout.setf(std::ios_base::uppercase);  // use upper-case scientific notation
          std::cout << std::setw(10) << "Patch_ID: " << std::setw(4) << patch->getID();
          std::cout << std::setw(14) << " Particle_ID: " << std::setw(4) << pids[i];
          std::cout << std::setw(12) << " Position: " << pos;
          std::cout << std::endl;
          cerrLock.unlock();
        }
      }
    }
    new_dw->put(sum_vartype(0.0), lb->vdwEnergyLabel);
  }
}

void MD::registerPermanentParticleState(SimpleMaterial* matl)
{
  // load up the ParticleVariables we want to register for relocation
  particleState_preReloc.push_back(lb->pForceLabel_preReloc);
  particleState.push_back(lb->pForceLabel);

  particleState_preReloc.push_back(lb->pAccelLabel_preReloc);
  particleState.push_back(lb->pAccelLabel);

  particleState_preReloc.push_back(lb->pVelocityLabel_preReloc);
  particleState.push_back(lb->pVelocityLabel);

  particleState_preReloc.push_back(lb->pEnergyLabel_preReloc);
  particleState.push_back(lb->pEnergyLabel);

  particleState_preReloc.push_back(lb->pMassLabel_preReloc);
  particleState.push_back(lb->pMassLabel);

  particleState_preReloc.push_back(lb->pChargeLabel_preReloc);
  particleState.push_back(lb->pChargeLabel);

  particleState_preReloc.push_back(lb->pParticleIDLabel_preReloc);
  particleState.push_back(lb->pParticleIDLabel);

  // register the particle states with the shared SimulationState for persistence across timesteps
  sharedState->d_particleState_preReloc.push_back(particleState_preReloc);
  sharedState->d_particleState.push_back(particleState);
}

void MD::interpolateParticlesToGrid(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* matls,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::interpolateChargesToGrid");
}

void MD::performSPME(const ProcessorGroup* pg,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::performSPME");

  if (system->newBox()) {
    electrostatics->setup();
    system->changeBox(false);
  }

  electrostatics->calculate();

  electrostatics->finalize();
}

void MD::calculateNonBondedForces(const ProcessorGroup* pg,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::calculateNonBondedForces");

  // loop through all patches
  unsigned int numPatches = patches->size();
  for (unsigned int p = 0; p < numPatches; p++) {
    const Patch* patch = patches->get(p);

    // do this for each material; for this example, there is only a single material, material "0"
    unsigned int numMatls = matls->size();
    double vdwEnergy = 0;
    for (unsigned int m = 0; m < numMatls; m++) {
      int matl = matls->get(m);

      ParticleSubset* pset = old_dw->getParticleSubset(matl, patch);
      ParticleSubset* delset = scinew ParticleSubset(0, matl, patch);

      // requires variables
      constParticleVariable<Point> px;
      constParticleVariable<Vector> pforce;
      constParticleVariable<double> penergy;
      constParticleVariable<long64> pids;
      old_dw->get(px, lb->pXLabel, pset);
      old_dw->get(penergy, lb->pEnergyLabel, pset);
      old_dw->get(pforce, lb->pForceLabel, pset);
      old_dw->get(pids, lb->pParticleIDLabel, pset);

      // computes variables
      ParticleVariable<Vector> pforcenew;
      ParticleVariable<double> penergynew;
      new_dw->allocateAndPut(penergynew, lb->pEnergyLabel_preReloc, pset);
      new_dw->allocateAndPut(pforcenew, lb->pForceLabel_preReloc, pset);

      unsigned int numParticles = pset->numParticles();
      for (unsigned int i = 0; i < numParticles; i++) {
        pforcenew[i] = pforce[i];
        penergynew[i] = penergy[i];
      }

      // loop over all atoms in system, calculate the forces
      double r2, ir2, ir6, ir12, T6, T12;
      double forceTerm;
      Vector totalForce, atomForce;
      Vector reducedCoordinates;
      unsigned int totalAtoms = pset->numParticles();
      for (unsigned int i = 0; i < totalAtoms; i++) {
        atomForce = Vector(0.0, 0.0, 0.0);

        // loop over the neighbors of atom "i"
        unsigned int idx;
        unsigned int numNeighbors = neighborList[i].size();
        for (unsigned int j = 0; j < numNeighbors; j++) {
          idx = neighborList[i][j];

          // the vector distance between atom i and j
          reducedCoordinates = px[i] - px[idx];

          // this is required for periodic boundary conditions
          reducedCoordinates -= (reducedCoordinates / box).vec_rint() * box;
          double reducedX = reducedCoordinates[0] * reducedCoordinates[0];
          double reducedY = reducedCoordinates[1] * reducedCoordinates[1];
          double reducedZ = reducedCoordinates[2] * reducedCoordinates[2];
          r2 = reducedX + reducedY + reducedZ;
          ir2 = 1.0 / r2;  // 1/r^2
          ir6 = ir2 * ir2 * ir2;  // 1/r^6
          ir12 = ir6 * ir6;  // 1/r^12
          T12 = R12 * ir12;
          T6 = R6 * ir6;
          penergynew[idx] = T12 - T6;  // energy
          vdwEnergy += penergynew[idx];  // count the energy
          forceTerm = (12.0 * T12 - 6.0 * T6) * ir2;  // the force term
          totalForce = forceTerm * reducedCoordinates;

          // the contribution of force on atom i
          atomForce += totalForce;
        }  // end neighbor loop for atom "i"

        // sum up contributions to force for atom i
        pforcenew[i] += atomForce;

        if (md_dbg.active()) {
          cerrLock.lock();
          std::cout << "PatchID: " << std::setw(4) << patch->getID() << std::setw(6);
          std::cout << "ParticleID: " << std::setw(6) << pids[i] << std::setw(6);
          std::cout << "Prev Position: [";
          std::cout << std::setw(10) << std::setprecision(4) << px[i].x();
          std::cout << std::setw(10) << std::setprecision(4) << px[i].y();
          std::cout << std::setprecision(10) << px[i].z() << std::setw(4) << "]";
          std::cout << "Energy: ";
          std::cout << std::setw(14) << std::setprecision(6) << penergynew[i];
          std::cout << "Force: [";
          std::cout << std::setw(14) << std::setprecision(6) << pforcenew[i].x();
          std::cout << std::setw(14) << std::setprecision(6) << pforcenew[i].y();
          std::cout << std::setprecision(6) << pforcenew[i].z() << std::setw(4) << "]";
          std::cout << std::endl;
          cerrLock.unlock();
        }
      }  // end atom loop

      // this accounts for double energy with Aij and Aji
      vdwEnergy *= 0.50;

      if (md_dbg.active()) {
        cerrLock.lock();
        Vector forces(0.0, 0.0, 0.0);
        for (unsigned int i = 0; i < numParticles; i++) {
          forces += pforcenew[i];
        }
        std::cout.setf(std::ios_base::scientific);
        std::cout << "Total Local Energy: " << std::setprecision(16) << vdwEnergy << std::endl;
        std::cout << "Local Force: [";
        std::cout << std::setw(16) << std::setprecision(8) << forces.x();
        std::cout << std::setw(16) << std::setprecision(8) << forces.y();
        std::cout << std::setprecision(8) << forces.z() << std::setw(4) << "]";
        std::cout << std::endl;
        std::cout.unsetf(std::ios_base::scientific);
        cerrLock.unlock();
      }

      new_dw->deleteParticles(delset);

    }  // end materials loop

    // global reduction on
    new_dw->put(sum_vartype(vdwEnergy), lb->vdwEnergyLabel);

  }  // end patch loop

}

void MD::interpolateToParticlesAndUpdate(const ProcessorGroup* pg,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  printTask(patches, md_cout, "MD::interpolateToParticlesAndUpdate");

  // loop through all patches
  unsigned int numPatches = patches->size();
  for (unsigned int p = 0; p < numPatches; p++) {
    const Patch* patch = patches->get(p);

    // do this for each material; for this example, there is only a single material, material "0"
    unsigned int numMatls = matls->size();
    for (unsigned int m = 0; m < numMatls; m++) {
      int matl = matls->get(m);

      ParticleSubset* lpset = old_dw->getParticleSubset(matl, patch);
      ParticleSubset* delset = scinew ParticleSubset(0, matl, patch);

      // requires variables
      constParticleVariable<Point> px;
      constParticleVariable<Vector> pforce;
      constParticleVariable<Vector> paccel;
      constParticleVariable<Vector> pvelocity;
      constParticleVariable<double> pmass;
      constParticleVariable<double> pcharge;
      constParticleVariable<long64> pids;
      old_dw->get(px, lb->pXLabel, lpset);
      old_dw->get(pforce, lb->pForceLabel, lpset);
      old_dw->get(paccel, lb->pAccelLabel, lpset);
      old_dw->get(pvelocity, lb->pVelocityLabel, lpset);
      old_dw->get(pmass, lb->pMassLabel, lpset);
      old_dw->get(pcharge, lb->pChargeLabel, lpset);
      old_dw->get(pids, lb->pParticleIDLabel, lpset);

      // computes variables
      ParticleVariable<Point> pxnew;
      ParticleVariable<Vector> paccelnew;
      ParticleVariable<Vector> pvelocitynew;
      ParticleVariable<double> pmassnew;
      ParticleVariable<double> pchargenew;
      ParticleVariable<long64> pidsnew;
      new_dw->allocateAndPut(pxnew, lb->pXLabel_preReloc, lpset);
      new_dw->allocateAndPut(paccelnew, lb->pAccelLabel_preReloc, lpset);
      new_dw->allocateAndPut(pvelocitynew, lb->pVelocityLabel_preReloc, lpset);
      new_dw->allocateAndPut(pmassnew, lb->pMassLabel_preReloc, lpset);
      new_dw->allocateAndPut(pchargenew, lb->pChargeLabel_preReloc, lpset);
      new_dw->allocateAndPut(pidsnew, lb->pParticleIDLabel_preReloc, lpset);

      // get delT
      delt_vartype delT;
      old_dw->get(delT, sharedState->get_delt_label(), getLevel(patches));

      // loop over the local atoms
      unsigned int localNumParticles = lpset->numParticles();
      for (unsigned int i = 0; i < localNumParticles; i++) {

        // carry these values over for now
        pmassnew[i] = pmass[i];
        pchargenew[i] = pcharge[i];
        pidsnew[i] = pids[i];

        // update position
        paccelnew[i] = pforce[i] / pmass[i];
        pvelocitynew[i] = pvelocity[i] + paccel[i] * delT;
        pxnew[i] = px[i] + pvelocity[i] + pvelocitynew[i] * 0.5 * delT;

        if (md_dbg.active()) {
          cerrLock.lock();
          std::cout << "PatchID: " << std::setw(4) << patch->getID() << std::setw(6);
          std::cout << "ParticleID: " << std::setw(6) << pidsnew[i] << std::setw(6);
          std::cout << "New Position: [";
          std::cout << std::setw(10) << std::setprecision(6) << pxnew[i].x();
          std::cout << std::setw(10) << std::setprecision(6) << pxnew[i].y();
          std::cout << std::setprecision(6) << pxnew[i].z() << std::setw(4) << "]";
          std::cout << std::endl;
          cerrLock.unlock();
        }
      }  // end atom loop

      new_dw->deleteParticles(delset);

    }  // end materials loop

  }  // end patch loop
}
