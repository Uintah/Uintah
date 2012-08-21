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

#include <iostream>
#include <fstream>

using namespace std;
using namespace Uintah;

LJPotentialTest::LJPotentialTest(const ProcessorGroup* myworld) :
    UintahParallelComponent(myworld)
{
  // create labels to be used in the simulation
  pXLabel = VarLabel::create("p.x", ParticleVariable<Point>::getTypeDescription());
  pXLabel_preReloc = VarLabel::create("p.x+", ParticleVariable<Point>::getTypeDescription(), IntVector(0, 0, 0),
                                      VarLabel::PositionVariable);

  pEnergyLabel = VarLabel::create("p.energy", ParticleVariable<double>::getTypeDescription());
  pEnergyLabel_preReloc = VarLabel::create("p.energy+", ParticleVariable<double>::getTypeDescription());

  pForceLabel = VarLabel::create("p.force", ParticleVariable<Vector>::getTypeDescription());
  pForceLabel_preReloc = VarLabel::create("p.force+", ParticleVariable<Vector>::getTypeDescription());

  pParticleIDLabel = VarLabel::create("p.particleID", ParticleVariable<long64>::getTypeDescription());
  pParticleIDLabel_preReloc = VarLabel::create("p.particleID+", ParticleVariable<long64>::getTypeDescription());
}

LJPotentialTest::~LJPotentialTest()
{
  VarLabel::destroy(pXLabel);
  VarLabel::destroy(pXLabel_preReloc);
  VarLabel::destroy(pEnergyLabel);
  VarLabel::destroy(pEnergyLabel_preReloc);
  VarLabel::destroy(pForceLabel);
  VarLabel::destroy(pForceLabel_preReloc);
  VarLabel::destroy(pParticleIDLabel);
  VarLabel::destroy(pParticleIDLabel_preReloc);
}

void LJPotentialTest::problemSetup(const ProblemSpecP& params,
                                   const ProblemSpecP& restart_prob_spec,
                                   GridP& /*grid*/,
                                   SimulationStateP& sharedState)
{
  sharedState_ = sharedState;
  dynamic_cast<Scheduler*>(getPort("scheduler"))->setPositionVar(pXLabel);
  ProblemSpecP ps = params->findBlock("LJPotentialTest");

  ps->getWithDefault("doOutput", doOutput_, 0);
  ps->getWithDefault("doGhostCells", doGhostCells_, 0);
  ps->get("cutoffDistance", cutoffDistance);
  ps->get("numAtoms", numAtoms);
  ps->get("R12", R12);
  ps->get("R6", R6);

  mymat_ = scinew SimpleMaterial();
  sharedState_->registerSimpleMaterial(mymat_);
}

void LJPotentialTest::scheduleInitialize(const LevelP& level,
                                         SchedulerP& sched)
{
  Task* task = scinew Task("initialize", this, &LJPotentialTest::initialize);
  task->computes(pXLabel);
  task->computes(pEnergyLabel);
  task->computes(pForceLabel);
  task->computes(pParticleIDLabel);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

void LJPotentialTest::scheduleComputeStableTimestep(const LevelP& level,
                                                    SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimestep", this, &LJPotentialTest::computeStableTimestep);
  task->computes(sharedState_->get_delt_label(), level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

void LJPotentialTest::scheduleTimeAdvance(const LevelP& level,
                                          SchedulerP& sched)
{
  const MaterialSet* matls = sharedState_->allMaterials();
  Task* task = scinew Task("timeAdvance", this, &LJPotentialTest::timeAdvance);

  // set this in problemSetup.  0 is no ghost cells, 1 is all with 1 ghost around-node, and 2 mixes them
  if (doGhostCells_ == 0) {
    task->requires(Task::OldDW, pXLabel, Ghost::None, 0);
    task->requires(Task::OldDW, pEnergyLabel, Ghost::None, 0);
    task->requires(Task::OldDW, pForceLabel, Ghost::None, 0);
    task->requires(Task::OldDW, pParticleIDLabel, Ghost::None, 0);
  } else if (doGhostCells_ == 1) {
    task->requires(Task::OldDW, pXLabel, Ghost::AroundNodes, 1);
    task->requires(Task::OldDW, pEnergyLabel, Ghost::None, 1);
    task->requires(Task::OldDW, pForceLabel, Ghost::None, 1);
    task->requires(Task::OldDW, pParticleIDLabel, Ghost::AroundNodes, 1);
  } else if (doGhostCells_ == 2) {
    task->requires(Task::OldDW, pXLabel, Ghost::None, 0);
    task->requires(Task::OldDW, pEnergyLabel, Ghost::None, 0);
    task->requires(Task::OldDW, pForceLabel, Ghost::None, 0);
    task->requires(Task::OldDW, pParticleIDLabel, Ghost::None, 0);
  }

  task->computes(pXLabel_preReloc);
  task->computes(pEnergyLabel_preReloc);
  task->computes(pForceLabel_preReloc);
  task->computes(pParticleIDLabel_preReloc);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

  d_particleState.clear();
  d_particleState_preReloc.clear();

  for (int m = 0; m < matls->size(); m++) {
    vector<const VarLabel*> vars;
    vector<const VarLabel*> vars_preReloc;

    vars.push_back(pEnergyLabel);
    vars.push_back(pForceLabel);
    vars.push_back(pParticleIDLabel);

    vars_preReloc.push_back(pEnergyLabel_preReloc);
    vars_preReloc.push_back(pForceLabel_preReloc);
    vars_preReloc.push_back(pParticleIDLabel_preReloc);
    d_particleState.push_back(vars);
    d_particleState_preReloc.push_back(vars_preReloc);
  }

  sched->scheduleParticleRelocation(level, pXLabel_preReloc, d_particleState_preReloc, pXLabel, d_particleState, pParticleIDLabel,
                                    matls);
}

void LJPotentialTest::computeStableTimestep(const ProcessorGroup* /*pg*/,
                                            const PatchSubset* patches,
                                            const MaterialSubset* /*matls*/,
                                            DataWarehouse*,
                                            DataWarehouse* new_dw)
{
  new_dw->put(delt_vartype(1), sharedState_->get_delt_label(), getLevel(patches));
}

void LJPotentialTest::generateNeighborList(constParticleVariable<Point> px)
{
  double r2;
  Vector reducedCoordinates;
  double cut_sq = cutoffDistance * cutoffDistance;
  for (unsigned int i = 0; i < numAtoms - 1; i++) {
    for (unsigned int j = i + 1; j < numAtoms; j++) {
      if (i != j) {
        // the vector distance between atom i and j
        reducedCoordinates = px[i] - px[j];

        // this is required for periodic boundary conditions
        reducedCoordinates -= (reducedCoordinates / box).vec_rint() * box;

        // eliminate atoms outside of cutoff radius, add those within as neighbors
        if ((fabs(reducedCoordinates[0]) < cutoffDistance) && (fabs(reducedCoordinates[1]) < cutoffDistance)
            && (fabs(reducedCoordinates[2]) < cutoffDistance)) {
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

void LJPotentialTest::initialize(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* /*old_dw*/,
                                 DataWarehouse* new_dw)
{
  unsigned int numPatches = patches->size();
  for (unsigned int p = 0; p < numPatches; p++) {
    const Patch* patch = patches->get(p);
    unsigned int numMatls = matls->size();
    for (unsigned int m = 0; m < numMatls; m++) {
      int matl = matls->get(m);

      // do the file I/O to get number of atoms and box size
      ifstream inputFile;
      string filePath = "ljpotential_input.medium";
      inputFile.open(filePath.c_str());
      if (!inputFile.is_open()) {
        string message = "\tCannot open input file: ";
        message += filePath;
        throw ProblemSetupException(message, __FILE__, __LINE__);
      }
      string line;
      unsigned int numRead;
      getline(inputFile, line);
      numRead = sscanf(line.c_str(), "%d", &numAtoms);
      getline(inputFile, line);
      numRead = sscanf(line.c_str(), "%lf %lf %lf", &box[0], &box[1], &box[2]);

      ParticleVariable<Point> px;
      ParticleVariable<double> penergy;
      ParticleVariable<Vector> pforce;
      ParticleVariable<long64> pids;

      ParticleSubset* subset = new_dw->createParticleSubset(numAtoms, matl, patch);
      new_dw->allocateAndPut(px, pXLabel, subset);
      new_dw->allocateAndPut(penergy, pEnergyLabel, subset);
      new_dw->allocateAndPut(pforce, pForceLabel, subset);
      new_dw->allocateAndPut(pids, pParticleIDLabel, subset);

      // initialize requisite ParticleVariables
      for (unsigned int i = 0; i < numAtoms; i++) {
        // get the atom coordinates
        getline(inputFile, line);
        double x, y, z;
        numRead = sscanf(line.c_str(), "%lf %lf %lf", &x, &y, &z);
        px[i] = Point(x, y, z);
        pforce[i] = Vector(0.0, 0.0, 0.0);
        penergy[i] = 0.0;
        pids[i] = patch->getID() * numAtoms + i;
      }
      inputFile.close();
    }
  }

  // for neighbor indices
  for (unsigned int i = 0; i < numAtoms; i++) {
    neighborList.push_back(vector<int>(0));
  }
}

void LJPotentialTest::timeAdvance(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  double vdwEnergy;
  unsigned int numPatches = patches->size();
  for (unsigned int p = 0; p < numPatches; p++) {
    const Patch* patch = patches->get(p);
    unsigned int numMatls = matls->size();
    for (unsigned int m = 0; m < numMatls; m++) {
      int matl = matls->get(m);
      ParticleSubset* pset = old_dw->getParticleSubset(matl, patch);
      ParticleSubset* delset = scinew ParticleSubset(0, matl, patch);

      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      ParticleVariable<Point> pxnew;
      constParticleVariable<double> penergy;
      ParticleVariable<double> penergynew;
      constParticleVariable<Vector> pforce;
      ParticleVariable<Vector> pforcenew;
      constParticleVariable<long64> pids;
      ParticleVariable<long64> pidsnew;

      old_dw->get(px, pXLabel, pset);
      old_dw->get(penergy, pEnergyLabel, pset);
      old_dw->get(pforce, pForceLabel, pset);
      old_dw->get(pids, pParticleIDLabel, pset);

      new_dw->allocateAndPut(pxnew, pXLabel_preReloc, pset);
      new_dw->allocateAndPut(penergynew, pEnergyLabel_preReloc, pset);
      new_dw->allocateAndPut(pforcenew, pForceLabel_preReloc, pset);
      new_dw->allocateAndPut(pidsnew, pParticleIDLabel_preReloc, pset);

      unsigned int numParticles = pset->numParticles();
      for (unsigned int i = 0; i < numParticles; i++) {
        pforcenew[i] = pforce[i];
        penergynew[i] = penergy[i];
      }

      // create neighbor list for each atom in the system
      generateNeighborList(px);

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
          reducedCoordinates -= (reducedCoordinates / box).vec_rint() * box;

          r2 = pow(reducedCoordinates[0], 2.0) + pow(reducedCoordinates[1], 2.0) + pow(reducedCoordinates[2], 2.0);
          ir2 = 1.0 / r2;  // 1/r^2
          ir6 = pow(ir2, 3.0);  // 1/r^6
          ir12 = pow(ir6, 2.0);  // 1/r^12
          T12 = R12 * ir12;
          T6 = R6 * ir6;
          penergynew[idx] = T12 - T6;  // energy
          vdwEnergy += penergynew[idx];  // count the energy
          forceTerm = (12.0 * T12 - 6.0 * T6) * ir2;  // the force term
          totalForce = forceTerm * reducedCoordinates;

          // the force on atom neighborList[i][j]
          pforcenew[idx] -= totalForce;

          // the contribution of force on atom i
          atomForce += totalForce;
        }  // end neighbor loop for atom "i"

        // sum up contributions to force for atom i
        pforcenew[i] += atomForce;

        // carry same position over until we get integrator implemented
        pxnew[i] = px[i];

        // keep same ID
        pidsnew[i] = pids[i];

        if (0) {
          cout << " Patch " << patch->getID() << ": Particle_ID " << pidsnew[i] << ", pos " << pxnew[i] << ", energy "
               << penergynew[i] << ", forces " << pforcenew[i] << endl;
        }
      }  // end atom loop

      if (doOutput_) {
        Vector forces(0.0, 0.0, 0.0);
        for (unsigned int i = 0; i < totalAtoms; i++) {
          forces += pforcenew[i];
        }
        printf("Total Energy: %E\n", vdwEnergy);
        printf("Forces: [%E\t%E\t%E]\n\n", forces.x(), forces.y(), forces.z());
      }

      new_dw->deleteParticles(delset);

    }  // end materials loop
  }  // end patch loop

}
