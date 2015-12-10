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

#include <CCA/Components/MD/MDSystem.h>
#include <CCA/Components/MD/MDLabel.h>
#include <CCA/Components/MD/SimpleGrid.h>
#include <CCA/Components/MD/Electrostatics/SPME/SPME_NewHeader.h>
#include <CCA/Components/MD/CoordinateSystems/CoordinateSystem.h>
#include <CCA/Components/MD/Electrostatics/SPME-busted/ShiftedCardinalBSpline.h>
#include <CCA/Components/MD/Electrostatics/SPME-busted/SPMEMapPoint.h>


#ifdef DEBUG
#include <Core/Util/FancyAssert.h>
#endif

#define IV_ZERO IntVector(0,0,0)

using namespace OldSPME;

void SPME::calculateRealspace(const ProcessorGroup*     pg,
                              const PatchSubset*        patches,
                              const MaterialSubset*     materials,
                              DataWarehouse*            old_dw,
                              DataWarehouse*            new_dw,
                              SimulationStateP&         sharedState,
                              MDLabel*                  label,
                              coordinateSystem*         coordinateSystem)
{
  size_t numPatches = patches->size();
  size_t numMaterials = materials->size();

  // Static constants we'll use but that are calculated once
  double twobeta2 = 2.0 * d_ewaldBeta * d_ewaldBeta;
  double Pi = acos(-1.0);
  double rootPi = sqrt(Pi);
  double one_over_betarootPi = 1.0/(d_ewaldBeta*rootPi);
  SCIRun::Vector ZERO_VECTOR(0.0, 0.0, 0.0);

  double cutoff2 = d_electrostaticRadius * d_electrostaticRadius;
  double realElectrostaticEnergy = 0;
  Matrix3 realElectrostaticStress = Matrix3(0.0,0.0,0.0,
                                            0.0,0.0,0.0,
                                            0.0,0.0,0.0);

  // Method global vector to catch the distance offset and avoid lots of
  // spurious temporary vector creations.
  SCIRun::Vector offset;

  // Step through all the patches on this thread
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch* patch = patches->get(patchIndex);

    // step through the materials for the reference sites
    for (size_t localIndex = 0; localIndex < numMaterials; ++localIndex) {
      int atomType = materials->get(localIndex);
      double atomCharge = sharedState->getMDMaterial(atomType)->getCharge();
      ParticleSubset* atomSubset = old_dw->getParticleSubset(atomType, patch);

      constParticleVariable<Point>  localX;
      constParticleVariable<long64> localID;
      old_dw->get(localX, label->global->pX, atomSubset);
      old_dw->get(localID, label->global->pID, atomSubset);

      size_t numLocalAtoms = atomSubset->numParticles();

      ParticleVariable<SCIRun::Vector> localForce;
      new_dw->allocateAndPut(localForce, label->electrostatic->pF_electroReal_preReloc, atomSubset);

      for (size_t Index = 0; Index < numLocalAtoms; ++ Index) {
        localForce[Index] = ZERO_VECTOR;
      }

      for (size_t neighborIndex = 0; neighborIndex < numMaterials; ++neighborIndex) {
        int neighborType = materials->get(neighborIndex);
        double neighborCharge = sharedState->getMDMaterial(neighborType)->getCharge();
        ParticleSubset* neighborSubset = old_dw->getParticleSubset(neighborType, patch, Ghost::AroundNodes, d_electrostaticGhostCells, label->global->pX);

        constParticleVariable<Point>  neighborX;
        constParticleVariable<long64> neighborID;
        old_dw->get(neighborX, label->global->pX, neighborSubset);
        old_dw->get(neighborID, label->global->pID, neighborSubset);

        size_t numNeighborAtoms = neighborSubset->numParticles();

        // loop over the local atoms
        for (size_t patchAtom=0; patchAtom < numLocalAtoms; ++patchAtom) {
          localForce[patchAtom]=ZERO_VECTOR;
          // loop over the neighbors
          for (size_t neighborAtom=0; neighborAtom < numNeighborAtoms; ++neighborAtom) {
            // Ensure i != j
            if (localID[patchAtom] != neighborID[neighborAtom]) {
              coordinateSystem->minimumImageDistance(neighborX[neighborAtom],localX[patchAtom],offset);
              // Periodic boundary condition; should eventually check against actual BC of system
              double radius2 = offset.length2();

              // only calculate if neighbor within spherical cutoff around local atom
              if (radius2 < cutoff2 ) {
                double rad2inv = 1.0/radius2;
                double radius = sqrt(radius2);
                double radinv = 1.0/radius;
                double betar = d_ewaldBeta*radius;
                double expnegbeta2r2_over_betarootpi = exp(-(d_ewaldBeta*d_ewaldBeta)*radius2)*one_over_betarootPi;
                double B0 = erfc(betar)*radinv;
                double B1 = (B0 + twobeta2*expnegbeta2r2_over_betarootpi);
                double G0 = atomCharge*neighborCharge;

                realElectrostaticEnergy += (B0*G0);
                SCIRun::Vector localForceVector = offset*(G0*B1);
                localForce[patchAtom] += localForceVector;
                realElectrostaticStress += OuterProduct(offset, localForceVector);
              } // Interaction within cutoff
            } // If atoms are different
          } // Loop over neighbors
        } // Loop over local atoms
      } // Loop over neighbor materials
    } // Loop over local materials
  } // Loop over patches
  // put updated values for reduction variables into the DW
  new_dw->put(sum_vartype(0.5 * realElectrostaticEnergy), label->electrostatic->rElectrostaticRealEnergy);
  new_dw->put(matrix_sum(0.5 * realElectrostaticStress), label->electrostatic->rElectrostaticRealStress);
  return;
} // End method

void SPME::calculateRealspaceDipole(const ProcessorGroup*     pg,
                                    const PatchSubset*        patches,
                                    const MaterialSubset*     materials,
                                    DataWarehouse*            old_dw,
                                    DataWarehouse*            new_dw,
                                    SimulationStateP&         sharedState,
                                    MDLabel*                  label,
                                    coordinateSystem*         coordinateSystem)
{
  size_t numPatches = patches->size();
  size_t numMaterials = materials->size();

  // Static constants we'll use but that are calculated once
  double twobeta2 = 2.0 * d_ewaldBeta * d_ewaldBeta;
  double Pi = acos(-1.0);
  double rootPi = sqrt(Pi);
  double one_over_betarootPi = 1.0/(d_ewaldBeta*rootPi);
  SCIRun::Vector ZERO_VECTOR(0.0, 0.0, 0.0);

  double cutoff2 = d_electrostaticRadius * d_electrostaticRadius;
  double realElectrostaticEnergy = 0;
  Matrix3 realElectrostaticStress = Matrix3(0.0,0.0,0.0,
                                            0.0,0.0,0.0,
                                            0.0,0.0,0.0);

  // Method global vector to catch the distance offset and avoid lots of
  // spurious temporary vector creations.
  SCIRun::Vector offset;

  // Step through all the patches on this thread
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch* patch = patches->get(patchIndex);

    // step through the materials for the reference sites
    for (size_t localIndex = 0; localIndex < numMaterials; ++localIndex) {
      int atomType = materials->get(localIndex);
      double atomCharge = sharedState->getMDMaterial(atomType)->getCharge();
      ParticleSubset* atomSubset = old_dw->getParticleSubset(atomType, patch);

      constParticleVariable<Point>  localX;
      constParticleVariable<long64> localID;
      constParticleVariable<Vector> localMu;
      old_dw->get(localX, label->global->pX, atomSubset);
      old_dw->get(localID, label->global->pID, atomSubset);
      old_dw->get(localMu, label->electrostatic->pMu, atomSubset);

      size_t numLocalAtoms = atomSubset->numParticles();

      ParticleVariable<SCIRun::Vector> localForce, localField;
      new_dw->allocateAndPut(localForce, label->electrostatic->pF_electroReal_preReloc, atomSubset);
      new_dw->allocateAndPut(localField, label->electrostatic->pE_electroReal_preReloc, atomSubset);

      for (size_t Index = 0; Index < numLocalAtoms; ++ Index) {
        localForce[Index] = ZERO_VECTOR;
        localField[Index] = ZERO_VECTOR;
      }

      for (size_t neighborIndex = 0; neighborIndex < numMaterials; ++neighborIndex) {
        int neighborType = materials->get(neighborIndex);
        double neighborCharge = sharedState->getMDMaterial(neighborType)->getCharge();
        ParticleSubset* neighborSubset = old_dw->getParticleSubset(neighborType, patch, Ghost::AroundNodes, d_electrostaticGhostCells, label->global->pX);

        constParticleVariable<Point>  neighborX;
        constParticleVariable<long64> neighborID;
        constParticleVariable<Vector> neighborMu;
        old_dw->get(neighborX, label->global->pX, neighborSubset);
        old_dw->get(neighborID, label->global->pID, neighborSubset);
        old_dw->get(neighborMu, label->electrostatic->pMu, neighborSubset);

        size_t numNeighborAtoms = neighborSubset->numParticles();

        // loop over the local atoms
        for (size_t patchAtom=0; patchAtom < numLocalAtoms; ++patchAtom) {
          SCIRun::Vector atomDipole = localMu[patchAtom];
          localForce[patchAtom]=ZERO_VECTOR;
          localField[patchAtom]=ZERO_VECTOR;
          // loop over the neighbors
          for (size_t neighborAtom=0; neighborAtom < numNeighborAtoms; ++neighborAtom) {
            // Ensure i != j
            if (localID[patchAtom] != neighborID[neighborAtom]) {
              // d_offsetProxy contains the proper implementation for distance offsets based on
              // orthorhombic nature of the cell and the current inverse cell
              coordinateSystem->minimumImageDistance(neighborX[neighborAtom],localX[patchAtom],offset);
//              SCIRun::Vector atomicDistanceVector = neighborX[neighborIdx]-localX[localIdx];
              // Periodic boundary condition; should eventually check against actual BC of system
//              atomicDistanceVector -= (atomicDistanceVector / box).vec_rint() * box; // For orthorhombic only
              double radius2 = offset.length2();

              // only calculate if neighbor within spherical cutoff around local atom
              if (radius2 < cutoff2 ) {
                SCIRun::Vector neighborDipole = neighborMu[neighborAtom];
                double rad2inv = 1.0/radius2;
                double radius = sqrt(radius2);
                double radinv = 1.0/radius;
                double betar = d_ewaldBeta*radius;
                double expnegbeta2r2_over_betarootpi = exp(-(d_ewaldBeta*d_ewaldBeta)*radius2)*one_over_betarootPi;
                double B0 = erfc(betar)*radinv;
                double B1 = (B0 + twobeta2*expnegbeta2r2_over_betarootpi);
                double B2 = (3*B1 + twobeta2*twobeta2*expnegbeta2r2_over_betarootpi)*rad2inv;
                double B3 = (5*B2 + twobeta2*twobeta2*twobeta2*expnegbeta2r2_over_betarootpi)*rad2inv;
                double G0 = atomCharge*neighborCharge;
                double pa_dot_rna = Dot(atomDipole,offset);
                double pn_dot_rna = Dot(neighborDipole,offset);
                double G1 = pa_dot_rna*neighborCharge - pn_dot_rna*atomCharge + Dot(atomDipole,neighborDipole);
                double G2 = -pa_dot_rna*pn_dot_rna;
                double delG0 = 0.0;
                SCIRun::Vector delG1 = atomCharge*neighborDipole - neighborCharge*atomDipole;
                SCIRun::Vector delG2 = pn_dot_rna*atomDipole + pa_dot_rna*neighborDipole;

                realElectrostaticEnergy += (B0*G0 + B1*G1 + B2*G2);
                SCIRun::Vector localForceVector = offset*(G0*B1+G1*B2+G2*B3) + (B1*delG1 + B2*delG2);
                localForce[patchAtom] += localForceVector;
                localField[patchAtom] += (atomCharge*B1-pn_dot_rna*B2)*offset + B1*neighborDipole;
                realElectrostaticStress += OuterProduct(offset, localForceVector);
              } // Interaction within cutoff
            } // If atoms are different
          } // Loop over neighbors
        } // Loop over local atoms
      } // Loop over neighbor materials
    } // Loop over local materials
  } // Loop over patches
  // put updated values for reduction variables into the DW
  new_dw->put(sum_vartype(0.5 * realElectrostaticEnergy), label->electrostatic->rElectrostaticRealEnergy);
  new_dw->put(matrix_sum(0.5 * realElectrostaticStress), label->electrostatic->rElectrostaticRealStress);
  return;
} // End method

void SPME::calculateNewDipoles(const ProcessorGroup* pg,
                         const PatchSubset* patches,
                         const MaterialSubset* materials,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw) {

  size_t numPatches = patches->size();
  size_t numMaterials = materials->size();

  double newMix = 1.0 - d_dipoleMixRatio;
  // Step through all the patches on this thread
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch* patch = patches->get(patchIndex);

    // step through the materials for the reference sites
    for (size_t localIndex = 0; localIndex < numMaterials; ++localIndex) {
      int atomType = materials->get(localIndex);
      ParticleSubset* localSet = old_dw->getParticleSubset(atomType, patch);
      constParticleVariable<Vector> oldDipoles;
      // We may want to mix with the old dipoles instead of replacing
      old_dw->get(oldDipoles, d_label->electrostatic->pMu, localSet);
//      old_dw->get(oldDipoles, d_label->pTotalDipoles, localSet);
      // We need the field estimation from the current iteration
      constParticleVariable<SCIRun::Vector> reciprocalField, realField;
      old_dw->get(reciprocalField, d_label->electrostatic->pE_electroInverse_preReloc, localSet);
      old_dw->get(realField, d_label->electrostatic->pE_electroReal_preReloc, localSet);
//      old_dw->get(reciprocalField, d_label->pElectrostaticsReciprocalField, localSet);
//      old_dw->get(realField, d_label->pElectrostaticsRealField, localSet);
      ParticleVariable<SCIRun::Vector> newDipoles;
      // And we'll be calculating into the new dipole container
      new_dw->allocateAndPut(newDipoles, d_label->electrostatic->pMu_preReloc, localSet);
//      new_dw->allocateAndPut(newDipoles, d_label->pTotalDipoles_preReloc, localSet);
      size_t localAtoms = localSet->numParticles();
      for (size_t Index = 0; Index < localAtoms; ++ Index) {
        //FIXME JBH -->  Put proper terms in here to actually calculate the new dipole contribution
        // Thole damping from Lucretius_Serial.f lines 2192-2202, 2291,
        newDipoles[Index] = reciprocalField[Index]+realField[Index];
        newDipoles[Index] += d_dipoleMixRatio * oldDipoles[Index];
      }
    }
  }

  // TODO fixme [APH]
}

bool SPME::checkConvergence() const
{
  // Subroutine determines if polarizable component has converged
  if (!f_polarizable) {
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

