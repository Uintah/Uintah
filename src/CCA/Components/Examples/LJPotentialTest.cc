/*

 The MIT License

 Copyright (c) 2012 The University of Utah

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

#include <CCA/Components/Examples/LJPotentialTest.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
#include <iomanip>
#include <fstream>

using namespace Uintah;

static DebugStream ljdbg("LJDebug", false);

LJPotentialTest::LJPotentialTest(const ProcessorGroup* myworld) :
    UintahParallelComponent(myworld)
{
  // create labels to be used in the simulation
  pXLabel = VarLabel::create("p.x", ParticleVariable<Point>::getTypeDescription());
  pXLabel_preReloc = VarLabel::create("p.x+", ParticleVariable<Point>::getTypeDescription(), IntVector(0, 0, 0),
                                      VarLabel::PositionVariable);

  pForceLabel = VarLabel::create("p.force", ParticleVariable<Vector>::getTypeDescription());
  pForceLabel_preReloc = VarLabel::create("p.force+", ParticleVariable<Vector>::getTypeDescription());

  pAccelLabel = VarLabel::create("p.accel", ParticleVariable<Vector>::getTypeDescription());
  pAccelLabel_preReloc = VarLabel::create("p.accel+", ParticleVariable<Vector>::getTypeDescription());

  pEnergyLabel = VarLabel::create("p.energy", ParticleVariable<double>::getTypeDescription());
  pEnergyLabel_preReloc = VarLabel::create("p.energy+", ParticleVariable<double>::getTypeDescription());

  pMassLabel = VarLabel::create("p.mass", ParticleVariable<Vector>::getTypeDescription());
  pMassLabel_preReloc = VarLabel::create("p.mass+", ParticleVariable<Vector>::getTypeDescription());

  pChargeLabel = VarLabel::create("charge", ParticleVariable<Vector>::getTypeDescription());
  pChargeLabel_preReloc = VarLabel::create("p.charge+", ParticleVariable<Vector>::getTypeDescription());

  pParticleIDLabel = VarLabel::create("p.particleID", ParticleVariable<long64>::getTypeDescription());
  pParticleIDLabel_preReloc = VarLabel::create("p.particleID+", ParticleVariable<long64>::getTypeDescription());

  vdwEnergyLabel = VarLabel::create("vdwEnergy", sum_vartype::getTypeDescription());
}

LJPotentialTest::~LJPotentialTest()
{
  VarLabel::destroy(pXLabel);
  VarLabel::destroy(pXLabel_preReloc);
  VarLabel::destroy(pForceLabel);
  VarLabel::destroy(pForceLabel_preReloc);
  VarLabel::destroy(pAccelLabel);
  VarLabel::destroy(pAccelLabel_preReloc);
  VarLabel::destroy(pEnergyLabel);
  VarLabel::destroy(pEnergyLabel_preReloc);
  VarLabel::destroy(pMassLabel);
  VarLabel::destroy(pMassLabel_preReloc);
  VarLabel::destroy(pChargeLabel);
  VarLabel::destroy(pChargeLabel_preReloc);
  VarLabel::destroy(pParticleIDLabel);
  VarLabel::destroy(pParticleIDLabel_preReloc);
  VarLabel::destroy(vdwEnergyLabel);
}

void LJPotentialTest::problemSetup(const ProblemSpecP& params,
                                   const ProblemSpecP& restart_prob_spec,
                                   GridP& /*grid*/,
                                   SimulationStateP& sharedState)
{
  d_sharedState_ = sharedState;
  dynamic_cast<Scheduler*>(getPort("scheduler"))->setPositionVar(pXLabel);
  ProblemSpecP ps = params->findBlock("LJPotentialTest");

  ps->get("coordinateFile", coordinateFile_);
  ps->get("numAtoms", numAtoms_);
  ps->get("boxSize", box_);
  ps->get("cutoffRadius", cutoffRadius_);
  ps->get("R12", R12_);
  ps->get("R6", R6_);

  mymat_ = scinew SimpleMaterial();
  d_sharedState_->registerSimpleMaterial(mymat_);

  // do file I/O to get atom coordinates and simulation cell size
  extractCoordinates();

  // for neighbor indices
  for (unsigned int i = 0; i < numAtoms_; i++) {
    neighborList.push_back(vector<int>(0));
  }

  // create neighbor list for each atom in the system
  generateNeighborList();
}

void LJPotentialTest::scheduleInitialize(const LevelP& level,
                                         SchedulerP& sched)
{
  Task* task = scinew Task("initialize", this, &LJPotentialTest::initialize);
  task->computes(pXLabel);
  task->computes(pForceLabel);
  task->computes(pAccelLabel);
  task->computes(pEnergyLabel);
  task->computes(pMassLabel);
  task->computes(pChargeLabel);
  task->computes(pParticleIDLabel);
  task->computes(vdwEnergyLabel);
  sched->addTask(task, level->eachPatch(), d_sharedState_->allMaterials());
}

void LJPotentialTest::scheduleComputeStableTimestep(const LevelP& level,
                                                    SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimestep", this, &LJPotentialTest::computeStableTimestep);
  task->requires(Task::NewDW, vdwEnergyLabel);
  task->computes(d_sharedState_->get_delt_label(), level.get_rep());
  sched->addTask(task, level->eachPatch(), d_sharedState_->allMaterials());
}

void LJPotentialTest::scheduleTimeAdvance(const LevelP& level,
                                          SchedulerP& sched)
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_sharedState_->allMaterials();

  d_particleState.clear();
  d_particleState_preReloc.clear();

  d_particleState.resize(matls->size());
  d_particleState_preReloc.resize(matls->size());

  scheduleCalculateNonBondedForces(sched, patches, matls);
  scheduleIntegrateVelocity(sched, patches, matls);

  sched->scheduleParticleRelocation(level, pXLabel_preReloc, d_particleState_preReloc, pXLabel, d_particleState, pParticleIDLabel,
                                    matls);
}

void LJPotentialTest::computeStableTimestep(const ProcessorGroup* pg,
                                            const PatchSubset* patches,
                                            const MaterialSubset* /*matls*/,
                                            DataWarehouse*,
                                            DataWarehouse* new_dw)
{
  if (pg->myrank() == 0) {
    sum_vartype vdwEnergy;
    new_dw->get(vdwEnergy, vdwEnergyLabel);
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << "Total Energy = " << std::setprecision(16) << vdwEnergy << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << std::endl;
  }
  new_dw->put(delt_vartype(1), d_sharedState_->get_delt_label(), getLevel(patches));
}

void LJPotentialTest::scheduleCalculateNonBondedForces(SchedulerP& sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* matls)
{
  Task* task = scinew Task("calculateNonBondedForces", this, &LJPotentialTest::calculateNonBondedForces);

  task->requires(Task::OldDW, pXLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, pForceLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, pEnergyLabel, Ghost::AroundNodes, SHRT_MAX);

  task->computes(pForceLabel_preReloc);
  task->computes(pEnergyLabel_preReloc);
  task->computes(vdwEnergyLabel);

  sched->addTask(task, patches, matls);

  // for particle relocation
  for (int m = 0; m < matls->size(); m++) {
    d_particleState_preReloc[m].push_back(pForceLabel_preReloc);
    d_particleState_preReloc[m].push_back(pEnergyLabel_preReloc);
    d_particleState[m].push_back(pForceLabel);
    d_particleState[m].push_back(pEnergyLabel);
  }
}

void LJPotentialTest::scheduleIntegrateVelocity(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* matls)
{
  Task* task = scinew Task("integrateVelocity", this, &LJPotentialTest::integrateVelocity);

  task->requires(Task::OldDW, pXLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, pForceLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, pAccelLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, pMassLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, pChargeLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, pParticleIDLabel, Ghost::AroundNodes, SHRT_MAX);

  task->computes(pXLabel_preReloc);
  task->computes(pAccelLabel_preReloc);
  task->computes(pMassLabel_preReloc);
  task->computes(pChargeLabel_preReloc);
  task->computes(pParticleIDLabel_preReloc);

  sched->addTask(task, patches, matls);

  // for particle relocation
  for (int m = 0; m < matls->size(); m++) {
    d_particleState_preReloc[m].push_back(pAccelLabel_preReloc);
    d_particleState_preReloc[m].push_back(pMassLabel_preReloc);
    d_particleState_preReloc[m].push_back(pChargeLabel_preReloc);
    d_particleState_preReloc[m].push_back(pParticleIDLabel_preReloc);
    d_particleState[m].push_back(pAccelLabel);
    d_particleState[m].push_back(pMassLabel);
    d_particleState[m].push_back(pChargeLabel);
    d_particleState[m].push_back(pParticleIDLabel);
  }

}

void LJPotentialTest::extractCoordinates()
{
  std::ifstream inputFile;
  inputFile.open(coordinateFile_.c_str());
  if (!inputFile.is_open()) {
    string message = "\tCannot open input file: " + coordinateFile_;
    throw ProblemSetupException(message, __FILE__, __LINE__);
  }

  // do file IO to extract atom coordinates
  string line;
  unsigned int numRead;
  for (unsigned int i = 0; i < numAtoms_; i++) {
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

void LJPotentialTest::generateNeighborList()
{
  double r2;
  Vector reducedCoordinates;
  double cut_sq = cutoffRadius_ * cutoffRadius_;
  for (unsigned int i = 0; i < numAtoms_; i++) {
    for (unsigned int j = 0; j < numAtoms_; j++) {
      if (i != j) {
        // the vector distance between atom i and j
        reducedCoordinates = atomList[i] - atomList[j];

        // this is required for periodic boundary conditions
        reducedCoordinates -= (reducedCoordinates / box_).vec_rint() * box_;

        // eliminate atoms outside of cutoff radius, add those within as neighbors
        if ((fabs(reducedCoordinates[0]) < cutoffRadius_) && (fabs(reducedCoordinates[1]) < cutoffRadius_)
            && (fabs(reducedCoordinates[2]) < cutoffRadius_)) {
          r2 = sqrt(pow(reducedCoordinates[0], 2.0) + pow(reducedCoordinates[1], 2.0) + pow(reducedCoordinates[2], 2.0));
          // only add neighbor atoms within spherical cut-off around atom "i"
          if (r2 < cut_sq) {
            neighborList[i].push_back(j);
          }
        }
      }
    }
  }
}

bool LJPotentialTest::isNeighbor(const Point* atom1,
                                 const Point* atom2)
{
  double r2;
  Vector reducedCoordinates;
  double cut_sq = cutoffRadius_ * cutoffRadius_;

  // the vector distance between atom 1 and 2
  reducedCoordinates = *atom1 - *atom2;

  // this is required for periodic boundary conditions
  reducedCoordinates -= (reducedCoordinates / box_).vec_rint() * box_;

  // check if outside of cutoff radius
  if ((fabs(reducedCoordinates[0]) < cutoffRadius_) && (fabs(reducedCoordinates[1]) < cutoffRadius_)
      && (fabs(reducedCoordinates[2]) < cutoffRadius_)) {
    r2 = sqrt(pow(reducedCoordinates[0], 2.0) + pow(reducedCoordinates[1], 2.0) + pow(reducedCoordinates[2], 2.0));
    return r2 < cut_sq;
  }
  return false;
}

void LJPotentialTest::initialize(const ProcessorGroup* /* pg */,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* /*old_dw*/,
                                 DataWarehouse* new_dw)
{
  // loop through all patches
  unsigned int numPatches = patches->size();
  for (unsigned int p = 0; p < numPatches; p++) {
    const Patch* patch = patches->get(p);

    // get bounds of current patch to correctly initialize particles (atoms)
    IntVector low = patch->getExtraCellLowIndex();
    IntVector high = patch->getExtraCellHighIndex();

    // do this for each material; for this example, there is only a single material, material "0"
    unsigned int numMatls = matls->size();
    for (unsigned int m = 0; m < numMatls; m++) {
      int matl = matls->get(m);

      ParticleVariable<Point> px;
      ParticleVariable<Vector> pforce;
      ParticleVariable<Vector> paccel;
      ParticleVariable<double> penergy;
      ParticleVariable<double> pmass;
      ParticleVariable<double> pcharge;
      ParticleVariable<long64> pids;

      // eventually we'll need to use PFS for this
      vector<Point> localAtoms;
      for (unsigned int i = 0; i < numAtoms_; i++) {
        if (containsAtom(low, high, atomList[i])) {
          localAtoms.push_back(atomList[i]);
        }
      }

      ParticleSubset* pset = new_dw->createParticleSubset(localAtoms.size(), matl, patch);
      new_dw->allocateAndPut(px, pXLabel, pset);
      new_dw->allocateAndPut(pforce, pForceLabel, pset);
      new_dw->allocateAndPut(paccel, pAccelLabel, pset);
      new_dw->allocateAndPut(penergy, pEnergyLabel, pset);
      new_dw->allocateAndPut(pmass, pMassLabel, pset);
      new_dw->allocateAndPut(pcharge, pChargeLabel, pset);
      new_dw->allocateAndPut(pids, pParticleIDLabel, pset);

      int numParticles = pset->numParticles();
      for (int i = 0; i < numParticles; i++) {
        Point pos = localAtoms[i];
        px[i] = pos;
        pforce[i] = Vector(0.0, 0.0, 0.0);
        paccel[i] = Vector(0.0, 0.0, 0.0);
        penergy[i] = 0.0;
        pmass[i] = 0.1;
        pcharge[i] = 0.0;
        pids[i] = patch->getID() * numAtoms_ + i;

        // TODO update this with new VarLabels
        if (ljdbg.active()) {
          std::cout.setf(std::ios_base::showpoint);  // print decimal and trailing zeros
          std::cout.setf(std::ios_base::left);  // pad after the value
          std::cout.setf(std::ios_base::uppercase);  // use upper-case scientific notation
          std::cout << std::setw(10) << "Patch_ID: " << std::setw(4) << patch->getID();
          std::cout << std::setw(14) << " Particle_ID: " << std::setw(4) << pids[i];
          std::cout << std::setw(12) << " Position: " << pos;
          std::cout << std::endl;
        }
      }
    }
    new_dw->put(sum_vartype(0.0), vdwEnergyLabel);
  }
}

void LJPotentialTest::calculateNonBondedForces(const ProcessorGroup* pg,
                                               const PatchSubset* patches,
                                               const MaterialSubset* matls,
                                               DataWarehouse* old_dw,
                                               DataWarehouse* new_dw)
{
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
      old_dw->get(px, pXLabel, pset);
      old_dw->get(penergy, pEnergyLabel, pset);
      old_dw->get(pforce, pForceLabel, pset);

      // computes variables
      ParticleVariable<Vector> pforcenew;
      ParticleVariable<double> penergynew;
      new_dw->allocateAndPut(penergynew, pEnergyLabel_preReloc, pset);
      new_dw->allocateAndPut(pforcenew, pForceLabel_preReloc, pset);

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
        register unsigned idx;
        unsigned int numNeighbors = neighborList[i].size();
        for (unsigned int j = 0; j < numNeighbors; j++) {
          idx = neighborList[i][j];

          // the vector distance between atom i and j
          reducedCoordinates = px[i] - px[idx];

          // this is required for periodic boundary conditions
          reducedCoordinates -= (reducedCoordinates / box_).vec_rint() * box_;

          r2 = pow(reducedCoordinates[0], 2.0) + pow(reducedCoordinates[1], 2.0) + pow(reducedCoordinates[2], 2.0);
          ir2 = 1.0 / r2;  // 1/r^2
          ir6 = pow(ir2, 3.0);  // 1/r^6
          ir12 = pow(ir6, 2.0);  // 1/r^12
          T12 = R12_ * ir12;
          T6 = R6_ * ir6;
          penergynew[idx] = T12 - T6;  // energy
          vdwEnergy += penergynew[idx];  // count the energy
          forceTerm = (12.0 * T12 - 6.0 * T6) * ir2;  // the force term
          totalForce = forceTerm * reducedCoordinates;

          // the contribution of force on atom i
          atomForce += totalForce;
        }  // end neighbor loop for atom "i"

        // sum up contributions to force for atom i
        pforcenew[i] += atomForce;

        if (ljdbg.active()) {
          std::cout << "Patch: " << std::setw(4) << patch->getID() << std::setw(6);
          std::cout << "Prev Position: [";
          std::cout << std::setw(10) << std::setprecision(6) << px[i].x();
          std::cout << std::setw(10) << std::setprecision(6) << px[i].y();
          std::cout << std::setprecision(6) << px[i].z() << std::setw(4) << "]";
          std::cout << "Energy: ";
          std::cout << std::setw(14) << std::setprecision(6) << penergynew[i];
          std::cout << "Force: [";
          std::cout << std::setw(12) << std::setprecision(6) << pforcenew[i].x();
          std::cout << std::setw(12) << std::setprecision(6) << pforcenew[i].y();
          std::cout << std::setprecision(6) << pforcenew[i].z() << std::setw(4) << "]";
          std::cout << std::endl;
        }
      }  // end atom loop

      // this acounts for double energy with Aij and Aji
      vdwEnergy *= 0.50;

      if (ljdbg.active()) {
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
      }

      new_dw->deleteParticles(delset);

    }  // end materials loop

    // global reduction on
    new_dw->put(sum_vartype(vdwEnergy), vdwEnergyLabel);

  }  // end patch loop

}

void LJPotentialTest::integrateVelocity(const ProcessorGroup* pg,
                                        const PatchSubset* patches,
                                        const MaterialSubset* matls,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
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
      constParticleVariable<double> pmass;
      constParticleVariable<double> pcharge;
      constParticleVariable<long64> pids;
      old_dw->get(px, pXLabel, lpset);
      old_dw->get(pforce, pForceLabel, lpset);
      old_dw->get(paccel, pAccelLabel, lpset);
      old_dw->get(pmass, pMassLabel, lpset);
      old_dw->get(pcharge, pChargeLabel, lpset);
      old_dw->get(pids, pParticleIDLabel, lpset);

      // computes variables
      ParticleVariable<Point> pxnew;
      ParticleVariable<Vector> paccelnew;
      ParticleVariable<double> pmassnew;
      ParticleVariable<double> pchargenew;
      ParticleVariable<long64> pidsnew;
      new_dw->allocateAndPut(pxnew, pXLabel_preReloc, lpset);
      new_dw->allocateAndPut(paccelnew, pAccelLabel_preReloc, lpset);
      new_dw->allocateAndPut(pmassnew, pMassLabel_preReloc, lpset);
      new_dw->allocateAndPut(pchargenew, pChargeLabel_preReloc, lpset);
      new_dw->allocateAndPut(pidsnew, pParticleIDLabel_preReloc, lpset);

      // loop over the local atoms
      unsigned int localNumParticles = lpset->numParticles();
      for (unsigned int i = 0; i < localNumParticles; i++) {

        // no time integration here yet... just testing
        pxnew[i] = px[i];
        paccelnew[i] = paccel[i];
        pmassnew[i] = pmass[i];
        pchargenew[i] = pcharge[i];
        pidsnew[i] = pids[i];

      }  // end atom loop

      new_dw->deleteParticles(delset);

    }  // end materials loop

  }  // end patch loop
}
