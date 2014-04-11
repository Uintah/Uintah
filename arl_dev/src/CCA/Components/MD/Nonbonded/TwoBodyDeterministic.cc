/*
 * TwoBodyDeterministic.cc
 *
 *  Created on: Apr 6, 2014
 *      Author: jbhooper
 */

#include <CCA/Components/MD/MDLabel.h>
#include <CCA/Components/MD/MDSystem.h>
#include <CCA/Components/MD/Nonbonded/TwoBodyDeterministic.h>
#include <CCA/Components/MD/Forcefields/TwoBodyForceField.h>

#include <Core/Grid/Patch.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>

#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>


using namespace Uintah;

const std::string TwoBodyDeterministic::nonbondedType = "TwoBodyDeterministic";

TwoBodyDeterministic::TwoBodyDeterministic(MDSystem* _system,
                                           MDLabel* _label,
                                           double _nbRadius)
                                          :d_System(_system),
                                           d_Label(_label),
                                           d_nonbondedRadius(_nbRadius) { }

void TwoBodyDeterministic::initialize(const ProcessorGroup* pg,
                                      const PatchSubset* patches,
                                      const MaterialSubset* materials,
                                      DataWarehouse* oldDW,
                                      DataWarehouse* newDW) {

  // global sum reduction of nonbonded energy
  newDW->put(sum_vartype(0.0), d_Label->nonbondedEnergyLabel);

  SoleVariable<double> dependency;
  newDW->put(dependency, d_Label->nonbondedDependencyLabel);

}

void TwoBodyDeterministic::setup(const ProcessorGroup* pg,
                                 const PatchSubset* patches,
                                 const MaterialSubset* materials,
                                 DataWarehouse* oldDW,
                                 DataWarehouse* newDW) {

}

void TwoBodyDeterministic::calculate(const ProcessorGroup* pg,
                                     const PatchSubset* patches,
                                     const MaterialSubset* materials,
                                     DataWarehouse* oldDW,
                                     DataWarehouse* newDW,
                                     SchedulerP& subscheduler,
                                     const LevelP& level) {

  double cutoff2 = d_nonbondedRadius * d_nonbondedRadius;
  SimulationStateP simState = d_System->getStatePointer();
  TwoBodyForcefield* forcefield = dynamic_cast<TwoBodyForcefield*> (d_System->getForcefieldPointer());
  size_t nbGhostCells = d_System->getNonbondedGhostCells();

  // Initialize local accumulators
  double nbEnergy_patchLocal = 0;
  Matrix3 stressTensor_patchLocal( 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0);

  size_t numPatches = patches->size();
  size_t numMaterials = materials->size();

  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) { // Loop over all patches
    const Patch* currPatch = patches->get(patchIndex);
    for (size_t localMaterialIndex = 0; localMaterialIndex < numMaterials; ++localMaterialIndex) { // Internal patch material loop

      // Build particle set for on-patch atom/material set
      int localMaterialID = materials->get(localMaterialIndex);
      ParticleSubset* localAtoms = oldDW->getParticleSubset(localMaterialID, currPatch);
      size_t localAtomCount = localAtoms->numParticles();

      // Get atom ID and positions
      constParticleVariable<long64> localParticleID;
      oldDW->get(localParticleID, d_Label->pParticleIDLabel, localAtoms);
      constParticleVariable<Point> localPositions;
      oldDW->get(localPositions, d_Label->pXLabel, localAtoms);

      // Get material map label for source atom type
      std::string localMaterialLabel = simState->getMDMaterial(localMaterialID)->getMapLabel();

      // Initialize force variable
      ParticleVariable<Vector> pForce;
      newDW->allocateAndPut(pForce, d_Label->pNonbondedForceLabel_preReloc, localAtoms);
      for (size_t localAtomIndex = 0; localAtomIndex < localAtomCount; ++localAtomIndex) { // Initialize force array
        pForce[localAtomIndex] = 0.0;
      }

      for (size_t neighborMaterialIndex = 0; neighborMaterialIndex < numMaterials; ++neighborMaterialIndex) { // Internal + ghost patch material loop

        // Build particle set for on patch + nearby atoms
        int neighborMaterialID = materials->get(neighborMaterialIndex);
        ParticleSubset* neighborAtoms = oldDW->getParticleSubset(neighborMaterialID, currPatch, Ghost::AroundNodes, nbGhostCells, d_Label->pXLabel);
        size_t neighborAtomCount = neighborAtoms->numParticles();

        // Get atom ID and positions
        constParticleVariable<long64> neighborParticleID;
        oldDW->get(neighborParticleID, d_Label->pParticleIDLabel, neighborAtoms);
        constParticleVariable<Point> neighborPositions;
        oldDW->get(neighborPositions, d_Label->pXLabel, neighborAtoms);

        // Get material map label for source atom type
        std::string neighborMaterialLabel = simState->getMDMaterial(neighborMaterialID)->getMapLabel();

        // All the local and neighbor related variables have been set up, get a potential and begin calculation
        NonbondedTwoBodyPotential* currentPotential = forcefield->getNonbondedPotential(localMaterialLabel, neighborMaterialLabel);

        for (size_t localAtomIndex = 0; localAtomIndex < localAtomCount; ++localAtomIndex) { // Loop over atoms in local patch
          for (size_t neighborAtomIndex = 0; neighborAtomIndex < neighborAtomCount; ++neighborAtomIndex) { // Loop over local plus nearby atoms
            if (localParticleID[localAtomIndex] != neighborParticleID[neighborAtomIndex]) { // Ensure we're not working with the same particle
              SCIRun::Vector atomOffsetVector = neighborPositions[neighborAtomIndex] - localPositions[localAtomIndex];
              // find minimum image of offset vector

              if (atomOffsetVector.length2() <= cutoff2) { // Interaction is within range
                SCIRun::Vector tempForce;
                double tempEnergy;
                currentPotential->fillEnergyAndForce(tempForce, tempEnergy, atomOffsetVector);
                nbEnergy_patchLocal += tempEnergy;
                stressTensor_patchLocal += OuterProduct(atomOffsetVector,tempForce);
                pForce[localAtomIndex] += tempForce;
              }  // Within cutoff
            }  // IDs not the same
          } // loop over neighbor Atoms
        }  // Loop over local Atoms
      }  // Loop over neighbor materials
    }  // Loop over local materials
  }  // Loop over patches
  newDW->put(sum_vartype(0.5 * nbEnergy_patchLocal), d_Label->nonbondedEnergyLabel);
  newDW->put(matrix_sum(0.5 * stressTensor_patchLocal), d_Label->nonbondedStressLabel);
} // TwoBodyDeterministic::calculate


void TwoBodyDeterministic::finalize(const ProcessorGroup* pg,
                                    const PatchSubset* patches,
                                    const MaterialSubset* materials,
                                    DataWarehouse* oldDW,
                                    DataWarehouse* newDW) {
  // Nothing to put here now
}
