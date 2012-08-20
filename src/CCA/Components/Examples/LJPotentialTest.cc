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

  pForceLabel = VarLabel::create("p.force", ParticleVariable<double>::getTypeDescription());
  pForceLabel_preReloc = VarLabel::create("p.force+", ParticleVariable<double>::getTypeDescription());

  pParticleIDLabel = VarLabel::create("p.particleID", ParticleVariable<long64>::getTypeDescription());
  pParticleIDLabel_preReloc = VarLabel::create("p.particleID+", ParticleVariable<long64>::getTypeDescription());

  // initialize fields related to non-bonded interactions
  cut = 10.0;
  ff_A12 = 1.0E5;
  ff_B6 = 1.0E3;
  vdwEnergy = 0.0;
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
  // for neighbor indices
  for (int i = 0; i < numAtoms; i++) {
    neighborList.push_back(vector<int>(0));
  }

  double r2, t[3];
  double cut_sq = cut * cut;
  for (int i = 0; i < numAtoms - 1; i++) {
    for (int j = i + 1; j < numAtoms; j++) {
      if (i != j) {
        // the vector distance between atom i and j
        t[0] = px[i].x() - px[j].x();
        t[1] = px[i].y() - px[j].y();
        t[2] = px[i].z() - px[j].z();

        // this is required for periodic boundary conditions
        t[0] -= rint(t[0] / box[0]) * box[0];
        t[1] -= rint(t[1] / box[1]) * box[1];
        t[2] -= rint(t[2] / box[2]) * box[2];

        // eliminate atoms outside of cutoff radius, add those within as neighbors
        if ((fabs(t[0]) < cut) && (fabs(t[1]) < cut) && (fabs(t[2]) < cut)) {
          r2 = sqrt(pow(t[0], 2.0) + pow(t[1], 2.0) + pow(t[2], 2.0));
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
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    Point low = patch->cellPosition(patch->getCellLowIndex());
    Point high = patch->cellPosition(patch->getCellHighIndex());
    for (int m = 0; m < matls->size(); m++) {
      int matl = matls->get(m);

      // do the file I/O to get number of atoms and box size
      FILE* fp;
      const int LINE_SIZE = 128;
      char line[LINE_SIZE];
      const char* filename = "ljpotential_input.medium";
      int numRead = 0;

      memset(line, '\0', sizeof(line));
      if ((fp = fopen(filename, "r")) == NULL) {
        throw ProblemSetupException("Cannot open input file.", __FILE__, __LINE__);
      }
      fgets(line, LINE_SIZE, fp);
      numRead = sscanf(line, "%d", &numAtoms);
      fgets(line, LINE_SIZE, fp);
      numRead = sscanf(line, "%lf %lf %lf", &box[0], &box[1], &box[2]);

      ParticleVariable<Point> px;
      ParticleVariable<double> penergy;
      ParticleVariable<Point> pforce;
      ParticleVariable<long64> pids;

      ParticleSubset* subset = new_dw->createParticleSubset(numAtoms, matl, patch);
      new_dw->allocateAndPut(px, pXLabel, subset);
      new_dw->allocateAndPut(penergy, pEnergyLabel, subset);
      new_dw->allocateAndPut(pforce, pForceLabel, subset);
      new_dw->allocateAndPut(pids, pParticleIDLabel, subset);

      // initialize requisite ParticleVariables
      for (int i = 0; i < numAtoms; i++) {
        // get the atom coordinates
        fgets(line, LINE_SIZE, fp);
        double x, y, z;
        numRead = sscanf(line, "%lf %lf %lf", &x, &y, &z);
        px[i] = Point(x, y, z);
        pforce[i] = Point(0.0, 0.0, 0.0);
        penergy[i] = 0.0;
        pids[i] = patch->getID() * numAtoms + i;
      }
      fclose(fp);
    }
  }
}

void LJPotentialTest::timeAdvance(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    for (int m = 0; m < matls->size(); m++) {
      int matl = matls->get(m);
      ParticleSubset* pset = old_dw->getParticleSubset(matl, patch);
      ParticleSubset* delset = scinew ParticleSubset(0, matl, patch);

      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      ParticleVariable<Point> pxnew;
      constParticleVariable<double> penergy;
      ParticleVariable<double> penergynew;
      constParticleVariable<Point> pforce;
      ParticleVariable<Point> pforcenew;
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

      for (int i = 0; i < pset->numParticles(); i++) {
        pforcenew[i] = pforce[i];
        penergynew[i] = penergy[i];
      }

      // create neighbor list for each atom in the system
      generateNeighborList(px);

      // loop over all atoms in system, calculate the forces
      double r2, ir2, ir6, ir12, T6, T12;
      double ffxx, ffyy, ffzz, fxx_i, fyy_i, fzz_i, forceTerm;
      double t[3];
      register int totalAtoms = pset->numParticles();
      for (int i = 0; i < totalAtoms; i++) {
        fxx_i = 0;
        fyy_i = 0;
        fzz_i = 0;

        // loop over the neighbors of atom "i"
        register int idx;
        register int numNeighbors = neighborList[i].size();
        for (int j = 0; j < numNeighbors; j++) {
          idx = neighborList[i][j];

          // the vector distance between atom i and j
          t[0] = px[i].x() - px[idx].x();
          t[1] = px[i].y() - px[idx].y();
          t[2] = px[i].z() - px[idx].z();

          // this is required for periodic boundary conditions
          t[0] -= rint(t[0] / box[0]) * box[0];
          t[1] -= rint(t[1] / box[1]) * box[1];
          t[2] -= rint(t[2] / box[2]) * box[2];

          r2 = pow(t[0], 2.0) + pow(t[1], 2.0) + pow(t[2], 2.0);
          ir2 = 1.0 / r2;  // 1/r^2
          ir6 = pow(ir2, 3.0);  // 1/r^6
          ir12 = pow(ir6, 2.0);  // 1/r^12
          T12 = ff_A12 * ir12;
          T6 = ff_B6 * ir6;
          penergynew[idx] = T12 - T6;  // energy
          vdwEnergy += penergynew[idx];  // count the energy
          forceTerm = (12.0 * T12 - 6.0 * T6) * ir2;  // the force term
          ffxx = forceTerm * t[0];
          ffyy = forceTerm * t[1];
          ffzz = forceTerm * t[2];

          // the force on atom neighborList[i][j]
          pforcenew[idx] = Point(pforcenew[idx].x() - ffxx, pforcenew[idx].y() - ffyy, pforcenew[idx].z() - ffzz);

          // the contribution of force on atom i
          fxx_i += ffxx;
          fyy_i += ffyy;
          fzz_i += ffzz;
        }  // end neighbor loop for atom "i"

        // sum up contributions to force for atom i
        pforcenew[i] = Point(pforcenew[i].x() + fxx_i, pforcenew[i].y() + fyy_i, pforcenew[i].z() + fzz_i);
        pxnew[i] = px[i];  // carry same position over until we get integrator
        pidsnew[i] = pids[i];
        if (doOutput_) {
          cout << " Patch " << patch->getID() << ": Particle_ID " << pidsnew[i] << ", pos " << pxnew[i] << ", energy "
               << penergynew[i] << ", forces " << pforcenew[i] << endl;
        }
      }  // end atom loop

      if (doOutput_) {
        double pfx = 0;
        double pfy = 0;
        double pfz = 0;
        for (int i = 0; i < totalAtoms; i++) {
          pfx += pforcenew[i].x();
          pfy += pforcenew[i].y();
          pfz += pforcenew[i].z();
        }
        printf("Total Energy: %E\n", vdwEnergy);
        printf("Forces: [%E\t%E\t%E]\n\n", pfx, pfy, pfz);
      }

      new_dw->deleteParticles(delset);

    }  // end materials loop
  }  // end patch loop

}
