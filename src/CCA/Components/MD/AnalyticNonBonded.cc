/*
 * The MIT License
 *
 * Copyright (c) 1997-2013 The University of Utah
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

#include <CCA/Components/MD/AnalyticNonBonded.h>
#include <CCA/Components/MD/MDSystem.h>
#include <CCA/Components/MD/MDLabel.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Box.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
#include <iomanip>

#ifdef DEBUG
#include <Core/Util/FancyAssert.h>
#endif

using namespace Uintah;

extern SCIRun::Mutex cerrLock;

static DebugStream lj12_6_dbg("LJ12_6", false);
static DebugStream lj12_6_cout("LJ12_6Cout", false);

AnalyticNonBonded::AnalyticNonBonded()
{

}

AnalyticNonBonded::~AnalyticNonBonded()
{

}

AnalyticNonBonded::AnalyticNonBonded(MDSystem* system,
                         const double r12,
                         const double r6,
                         const double cutoffRadius) :
    d_system(system), d_r12(r12), d_r6(r6), d_cutoffRadius(cutoffRadius)
{
  d_nonBondedInteractionType = NonBonded::LJ12_6;
}

//-----------------------------------------------------------------------------
// Interface implementations
void AnalyticNonBonded::initialize(const ProcessorGroup* pg,
                             const PatchSubset* patches,
                             const MaterialSubset* materials,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{
  // for neighbor indices; one list for each atom
  unsigned int numAtoms = d_system->getNumAtoms();
  for (unsigned int i = 0; i < numAtoms; i++) {
    d_neighborList.push_back(vector<int>());
  }
}

void AnalyticNonBonded::setup(const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* materials,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw)
{
  // create neighbor list for each atom in the system
  generateNeighborList();
}

void AnalyticNonBonded::calculate(const ProcessorGroup* pg,
                            const PatchSubset* patches,
                            const MaterialSubset* materials,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw)
{
  Vector box = d_system->getBox();
  // loop through all patches
  unsigned int numPatches = patches->size();
  for (unsigned int p = 0; p < numPatches; p++) {
    const Patch* patch = patches->get(p);

    // do this for each material; currently only using material "0"
    unsigned int numMatls = materials->size();
    double vdwEnergy = 0;
    for (unsigned int m = 0; m < numMatls; m++) {
      int matl = materials->get(m);

      ParticleSubset* pset = old_dw->getParticleSubset(matl, patch);
      ParticleSubset* delset = scinew ParticleSubset(0, matl, patch);

      // requires variables
      constParticleVariable<Point> px;
      constParticleVariable<Vector> pforce;
      constParticleVariable<double> penergy;
      old_dw->get(px, d_lb->pXLabel, pset);
      old_dw->get(penergy, d_lb->pEnergyLabel, pset);
      old_dw->get(pforce, d_lb->pForceLabel, pset);

      // computes variables
      ParticleVariable<Vector> pforcenew;
      ParticleVariable<double> penergynew;
      new_dw->allocateAndPut(pforcenew, d_lb->pForceLabel_preReloc, pset);
      new_dw->allocateAndPut(penergynew, d_lb->pEnergyLabel_preReloc, pset);

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
        unsigned int numNeighbors = d_neighborList[i].size();
        for (unsigned int j = 0; j < numNeighbors; j++) {
          idx = d_neighborList[i][j];

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
          T12 = d_r12 * ir12;
          T6 = d_r6 * ir6;
          penergynew[idx] = T12 - T6;  // energy
          vdwEnergy += penergynew[idx];  // count the energy
          forceTerm = (12.0 * T12 - 6.0 * T6) * ir2;  // the force term
          totalForce = forceTerm * reducedCoordinates;

          // the contribution of force on atom i
          atomForce += totalForce;
        }  // end neighbor loop for atom "i"

        // sum up contributions to force for atom i
        pforcenew[i] += atomForce;

        if (lj12_6_dbg.active()) {
          cerrLock.lock();
          std::cout << "PatchID: " << std::setw(4) << patch->getID() << std::setw(6);
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

      if (lj12_6_dbg.active()) {
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

    // global reduction on vdwEnergy
    new_dw->put(sum_vartype(vdwEnergy), d_lb->vdwEnergyLabel);

  }  // end patch loop
}

void AnalyticNonBonded::finalize(const ProcessorGroup* pg,
                           const PatchSubset* patches,
                           const MaterialSubset* materials,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw)
{

}

void AnalyticNonBonded::generateNeighborList()
{
  double r2;
  Vector box = d_system->getBox();

  SCIRun::Vector reducedCoordinates;
  double cut_sq = d_cutoffRadius * d_cutoffRadius;
  unsigned int numAtoms = d_system->getNumAtoms();

  for (unsigned int i = 0; i < numAtoms; i++) {
    for (unsigned int j = 0; j < numAtoms; j++) {
      if (i != j) {
        // the vector distance between atom i and j
        reducedCoordinates = d_atomList->data()[i].coords - d_atomList->data()[j].coords;

        // this is required for periodic boundary conditions
        reducedCoordinates -= (reducedCoordinates / box).vec_rint() * box;

        // eliminate atoms outside of cutoff radius, add those within as neighbors
        if ((fabs(reducedCoordinates[0]) < d_cutoffRadius) && (fabs(reducedCoordinates[1]) < d_cutoffRadius)
            && (fabs(reducedCoordinates[2]) < d_cutoffRadius)) {
          double reducedX = reducedCoordinates[0] * reducedCoordinates[0];
          double reducedY = reducedCoordinates[1] * reducedCoordinates[1];
          double reducedZ = reducedCoordinates[2] * reducedCoordinates[2];
          r2 = sqrt(reducedX + reducedY + reducedZ);
          // only add neighbor atoms within spherical cut-off around atom "i"
          if (r2 < cut_sq) {
            d_neighborList[i].push_back(j);
          }
        }
      }
    }
  }
}

bool AnalyticNonBonded::isNeighbor(const Point* atom1,
                             const Point* atom2)
{
  // get the simulation box size
  Vector box = d_system->getBox();

  double r2;
  Vector reducedCoordinates;
  double cut_sq = d_cutoffRadius * d_cutoffRadius;

  // the vector distance between atom 1 and 2
  reducedCoordinates = *atom1 - *atom2;

  // this is required for periodic boundary conditions
  reducedCoordinates -= (reducedCoordinates / box).vec_rint() * box;

  // check if outside of cutoff radius
  if ((fabs(reducedCoordinates[0]) < d_cutoffRadius) && (fabs(reducedCoordinates[1]) < d_cutoffRadius)
      && (fabs(reducedCoordinates[2]) < d_cutoffRadius)) {
    r2 = sqrt(pow(reducedCoordinates[0], 2.0) + pow(reducedCoordinates[1], 2.0) + pow(reducedCoordinates[2], 2.0));
    return r2 < cut_sq;
  }
  return false;
}

