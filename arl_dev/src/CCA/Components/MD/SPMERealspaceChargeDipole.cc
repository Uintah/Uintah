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

#include <CCA/Components/MD/SPME.h>
#include <CCA/Components/MD/ShiftedCardinalBSpline.h>
#include <CCA/Components/MD/SPMEMapPoint.h>
#include <CCA/Components/MD/MDSystem.h>
#include <CCA/Components/MD/MDLabel.h>
#include <CCA/Components/MD/SimpleGrid.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Patch.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Thread/Thread.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>

#include <sci_values.h>
#include <sci_defs/fftw_defs.h>

#ifdef DEBUG
#include <Core/Util/FancyAssert.h>
#endif

#define IV_ZERO IntVector(0,0,0)

using namespace Uintah;

void SPME::calculateRealspace(const ProcessorGroup* pg,
                              const PatchSubset* patches,
                              const MaterialSubset* materials,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw)
{
  size_t numPatches = patches->size();
  size_t numMaterials = materials->size();
  double Pi = acos(-1.0);
  double rootPi = sqrt(Pi);
  SCIRun::Vector ZERO_VECTOR(0.0, 0.0, 0.0);

  double cutoff2 = d_electrostaticRadius * d_electrostaticRadius;
  double realElectrostaticEnergy = 0;
  Matrix3 realElectrostaticStress = Matrix3(0.0,0.0,0.0,
                                            0.0,0.0,0.0,
                                            0.0,0.0,0.0);

  int ELECTROSTATIC_RADIUS = d_system->getElectrostaticGhostCells();
  SCIRun::Vector box = d_system->getBox();

  // Step through all the patches on this thread
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch* patch = patches->get(patchIndex);

    // step through the materials for the reference sites
    for (size_t localIndex = 0; localIndex < numMaterials; ++localIndex) {
      int atomType = materials->get(localIndex);
      double atomCharge = d_system->getAtomicCharge(atomType);
      ParticleSubset* atomSubset = old_dw->getParticleSubset(atomType, patch);
      constParticleVariable<Point> localX;
      old_dw->get(localX, d_lb->pXLabel, atomSubset);
      constParticleVariable<long64> localID;
      old_dw->get(localID, d_lb->pParticleIDLabel, atomSubset);
      ParticleVariable<Vector> dipolesLocal;
      new_dw->getModifiable(dipolesLocal, d_lb->pRealDipoles, atomSubset);
      size_t localAtoms = atomSubset->numParticles();
      ParticleVariable<SCIRun::Vector> localForce;
      new_dw->allocateAndPut(localForce, d_lb->pElectrostaticsRealForce_preReloc, atomSubset);
      ParticleVariable<SCIRun::Vector> localField;
      new_dw->allocateAndPut(localField, d_lb->pElectrostaticsRealField_preReloc, atomSubset);
      for (size_t Index = 0; Index < localAtoms; ++ Index) {
        localForce[Index] = ZERO_VECTOR;
        localField[Index] = ZERO_VECTOR;
      }

      for (size_t neighborIndex = 0; neighborIndex < numMaterials; ++neighborIndex) {
        int neighborType = materials->get(neighborIndex);
        double neighborCharge = d_system->getAtomicCharge(atomType);
        ParticleSubset* neighborSubset = old_dw->getParticleSubset(neighborType, patch, Ghost::AroundNodes, ELECTROSTATIC_RADIUS, d_lb->pXLabel);
        // Map neighbor atoms to their positions
        constParticleVariable<Point> neighborX;
        old_dw->get(neighborX, d_lb->pXLabel, neighborSubset);
        // Map neighbor atoms to their IDs
        constParticleVariable<long64> neighborID;
        old_dw->get(neighborID, d_lb->pParticleIDLabel, neighborSubset);
        constParticleVariable<Vector> dipolesNeighbor;
        new_dw->get(dipolesNeighbor, d_lb->pRealDipoles, neighborSubset);

        size_t neighborAtoms = neighborSubset->numParticles();

        // loop over the local atoms
        for (size_t localIdx=0; localIdx < localAtoms; ++localIdx) {
          SCIRun::Vector atomDipole = dipolesLocal[localIdx];
          SCIRun::Vector localFieldAtAtom(0.0, 0.0, 0.0);
          Vector realElectrostaticForce = SCIRun::Vector(0.0, 0.0, 0.0);
          // loop over the neighbors
          for (size_t neighborIdx=0; neighborIdx < neighborAtoms; ++neighborIdx) {
            // Ensure i != j
            if (localID[localIdx] != neighborID[neighborIdx]) {
              SCIRun::Vector atomicDistanceVector = neighborX[neighborIdx]-localX[localIdx];
              // Periodic boundary condition; should eventually check against actual BC of system
              atomicDistanceVector -= (atomicDistanceVector / box).vec_rint() * box; // For orthorhombic only
              double radius2 = atomicDistanceVector.length2();

              // only calculate if neighbor within spherical cutoff around local atom
              if (radius2 < cutoff2 ) {
                SCIRun::Vector neighborDipole = dipolesNeighbor[neighborIdx];
                double radius = sqrt(radius2);
                double betar = d_ewaldBeta*radius;
                double twobeta2 = 2.0 * d_ewaldBeta * d_ewaldBeta;
                double expnegbeta2r2_over_betarootpi = exp(-(d_ewaldBeta*d_ewaldBeta)*radius2)/(d_ewaldBeta*rootPi);
                double B0 = erfc(betar)/radius;
                double B1 = (B0 + twobeta2*expnegbeta2r2_over_betarootpi);
                double B2 = (3*B1 + twobeta2*twobeta2*expnegbeta2r2_over_betarootpi)/radius2;
                double B3 = (5*B2 + twobeta2*twobeta2*twobeta2*expnegbeta2r2_over_betarootpi)/radius2;
                double G0 = atomCharge*neighborCharge;
                double pa_dot_rna = Dot(atomDipole,atomicDistanceVector);
                double pn_dot_rna = Dot(neighborDipole,atomicDistanceVector);
                double G1 = pa_dot_rna*neighborCharge - pn_dot_rna*atomCharge + Dot(atomDipole,neighborDipole);
                double G2 = -pa_dot_rna*pn_dot_rna;
                double delG0 = 0.0;
                SCIRun::Vector delG1 = atomCharge*neighborDipole - neighborCharge*atomDipole;
                SCIRun::Vector delG2 = pn_dot_rna*atomDipole + pa_dot_rna*neighborDipole;

                realElectrostaticEnergy += (B0*G0 + B1*G1 + B2*G2);
                SCIRun::Vector localForceVector = atomicDistanceVector*(G0*B1+G1*B2+G2*B3) + (B1*delG1 + B2*delG2);
                localForce[localIdx] += localForceVector;
                localField[localIdx] += (atomCharge*B1-pn_dot_rna*B2)*atomicDistanceVector + B1*neighborDipole;
                realElectrostaticStress += OuterProduct(atomicDistanceVector, localForceVector);
              } // Interaction within cutoff
            } // If atoms are different
          } // Loop over neighbors
        } // Loop over local atoms
      } // Loop over neighbor materials
    } // Loop over local materials
  } // Loop over patches
  //!FIXME  Store energy and stress tensor here
  // put updated values for reduction variables into the DW
  new_dw->put(sum_vartype(0.5 * realElectrostaticEnergy), d_lb->electrostaticRealEnergyLabel);
  new_dw->put(matrix_sum(0.5 * realElectrostaticStress), d_lb->electrostaticRealStressLabel);
  return;
} // End method

void SPME::calculateNewDipoles(const ProcessorGroup* pg,
                         const PatchSubset* patches,
                         const MaterialSubset* materials,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw) {

  size_t numPatches = patches->size();
  size_t numMaterials = materials->size();

  // Step through all the patches on this thread
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch* patch = patches->get(patchIndex);

    // step through the materials for the reference sites
    for (size_t localIndex = 0; localIndex < numMaterials; ++localIndex) {
      int atomType = materials->get(localIndex);
      ParticleSubset* localSet = old_dw->getParticleSubset(atomType, patch);
      constParticleVariable<Vector> oldDipoles;
      old_dw->get(oldDipoles, d_lb->pRealDipoles, localSet);
      ParticleVariable<Vector> newDipoles;
      new_dw->getModifiable(newDipoles, d_lb->pRealDipoles_preReloc, localSet);
      size_t localAtoms = localSet->numParticles();
      for (size_t Index = 0; Index < localAtoms; ++ Index) {
        newDipoles[Index] *= (1.0 - d_dipoleMixRatio);
        newDipoles[Index] += d_dipoleMixRatio * oldDipoles[Index];
      }
    }
  }

  // TODO fixme [APH]
}

bool SPME::checkConvergence() const
{
  // Subroutine determines if polarizable component has converged
  if (!d_polarizable) {
    return true;
  } else {
    // throw an exception for now, but eventually will check convergence here.
    throw InternalError("Error: Polarizable force field not yet implemented!", __FILE__, __LINE__);
  }

  // TODO keep an eye on this to make sure it works like we think it should
  if (Thread::self()->myid() == 0) {
    d_Q_nodeLocal->initialize(dblcomplex(0.0, 0.0));
  }
}

