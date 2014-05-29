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
#include <CCA/Components/MD/CoordinateSystems/coordinateSystem.h>

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

TwoBodyDeterministic::TwoBodyDeterministic(double _nbRadius,
                                           int _nbGhost)
                                          :d_nonbondedRadius(_nbRadius),
                                           d_nonbondedGhostCells(_nbGhost) { }

void TwoBodyDeterministic::addInitializeRequirements(Task* task,
                                                     MDLabel* label) const {

}

void TwoBodyDeterministic::addInitializeComputes(Task* task,
                                                 MDLabel* label) const {
  // None of this probably really needs to be here, except the dependency
  task->computes(label->nonbonded->rNonbondedEnergy);
  task->computes(label->nonbonded->rNonbondedStress);
  task->computes(label->nonbonded->dNonbondedDependency);

}

void TwoBodyDeterministic::initialize(const ProcessorGroup*     pg,
                                      const PatchSubset*        patches,
                                      const MaterialSubset*     materials,
                                      DataWarehouse*          /*oldDW*/,
                                      DataWarehouse*            newDW,
                                      SimulationStateP&         simState,
                                      MDSystem*                 systemInfo,
                                      const MDLabel*            label,
                                      coordinateSystem*         coordSys) {

  // global sum reduction of nonbonded energy
  newDW->put(sum_vartype(0.0), label->nonbonded->rNonbondedEnergy);
  newDW->put(matrix_sum(0.0), label->nonbonded->rNonbondedStress);

  SoleVariable<double> dependency;
  newDW->put(dependency, label->nonbonded->dNonbondedDependency);

}

void TwoBodyDeterministic::addSetupRequirements(Task* task,
                                                MDLabel* label) const {

  task->requires(Task::OldDW,
                 label->nonbonded->dNonbondedDependency,
                 Ghost::None, 0);

}

void TwoBodyDeterministic::addSetupComputes(Task* task,
                                            MDLabel* label) const {

  task->computes(label->nonbonded->dNonbondedDependency);
}

void TwoBodyDeterministic::setup(const ProcessorGroup*  pg,
                                 const PatchSubset*     patches,
                                 const MaterialSubset*  materials,
                                 DataWarehouse*         oldDW,
                                 DataWarehouse*         newDW,
                                 SimulationStateP&      simState,
                                 MDSystem*              systemInfo,
                                 const MDLabel*         label,
                                 coordinateSystem*      coordSys) {

}

void TwoBodyDeterministic::addCalculateRequirements(Task* task,
                                                    MDLabel* label) const {

  task->requires(Task::OldDW, label->nonbonded->dNonbondedDependency);

}

void TwoBodyDeterministic::addCalculateComputes(Task* task,
                                                MDLabel* label) const {
  // Provides force, energy, and stress tensor

  task->computes(label->nonbonded->pF_nonbonded_preReloc);
  task->computes(label->nonbonded->rNonbondedEnergy);
  task->computes(label->nonbonded->rNonbondedStress);

}

void TwoBodyDeterministic::calculate(const ProcessorGroup*  pg,
                                     const PatchSubset*     patches,
                                     const MaterialSubset*  materials,
                                     DataWarehouse*         oldDW,
                                     DataWarehouse*         newDW,
                                     SimulationStateP&      simState,
                                     MDSystem*              systemInfo,
                                     const MDLabel*         label,
                                     coordinateSystem*      coordSys)
{

  double cutoff2 = d_nonbondedRadius * d_nonbondedRadius;
  TwoBodyForcefield* forcefield = dynamic_cast<TwoBodyForcefield*>
                                       ( systemInfo->getForcefieldPointer() );

  // Initialize local accumulators
  double nbEnergy_patchLocal = 0;
  Matrix3 stressTensor_patchLocal( 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0);

  size_t numPatches = patches->size();
  size_t numMaterials = materials->size();
  SCIRun::Vector atomOffsetVector(0.0);

  // Loop over all patches
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch* currPatch = patches->get(patchIndex);
    // Internal patch material loop
    for (size_t localMaterialIndex = 0;
                localMaterialIndex < numMaterials;
              ++localMaterialIndex)
    {
      // Build particle set for on-patch atom/material set
      int               localMaterial = materials->get(localMaterialIndex);
      ParticleSubset*   localAtoms    = oldDW->getParticleSubset(localMaterial,
                                                                   currPatch);
      size_t localAtomCount = localAtoms->numParticles();

      // Get atom ID and positions
      constParticleVariable<long64> localParticleID;
      oldDW->get(localParticleID, label->global->pID, localAtoms);
      constParticleVariable<Point> localX;
      oldDW->get(localX, label->global->pX, localAtoms);

      // Get material map label for source atom type
      std::string localMaterialLabel =
          simState->getMDMaterial(localMaterial)->getMapLabel();

      // Initialize force variable
      ParticleVariable<Vector> pForce;
      newDW->allocateAndPut(pForce,
                            label->nonbonded->pF_nonbonded_preReloc,
                            localAtoms);
      for (size_t localAtomIndex = 0;
                  localAtomIndex < localAtomCount;
                ++localAtomIndex) // Initialize force array
      {
        pForce[localAtomIndex] = 0.0;
      }

      for (size_t neighborMaterialIndex = 0;
                  neighborMaterialIndex < numMaterials;
                ++neighborMaterialIndex)
      { // (Internal + ghost) patch material loop
        // Build particle set for on patch + nearby atoms
        int neighborMaterialID = materials->get(neighborMaterialIndex);
        ParticleSubset* neighborAtoms =
            oldDW->getParticleSubset(neighborMaterialID,
                                     currPatch,
                                     Ghost::AroundNodes,
                                     d_nonbondedGhostCells,
                                     label->global->pX);

        size_t neighborAtomCount = neighborAtoms->numParticles();

        // Get atom ID and positions
        constParticleVariable<long64> neighborParticleID;
        oldDW->get(neighborParticleID, label->global->pID, neighborAtoms);
        constParticleVariable<Point> neighborX;
        oldDW->get(neighborX, label->global->pX, neighborAtoms);
//        oldDW->get(neighborX, d_Label->pXLabel, neighborAtoms);

        // Get material map label for source atom type
        std::string neighborMaterialLabel = simState->getMDMaterial(neighborMaterialID)->getMapLabel();

        // All the local and neighbor related variables have been set up, get a potential and begin calculation
        NonbondedTwoBodyPotential* currentPotential = forcefield->getNonbondedPotential(localMaterialLabel, neighborMaterialLabel);

        for (size_t localAtomIndex = 0; localAtomIndex < localAtomCount; ++localAtomIndex) { // Loop over atoms in local patch
          for (size_t neighborAtomIndex = 0; neighborAtomIndex < neighborAtomCount; ++neighborAtomIndex) { // Loop over local plus nearby atoms
            if (localParticleID[localAtomIndex] != neighborParticleID[neighborAtomIndex]) { // Ensure we're not working with the same particle
              coordSys->minimumImageDistance(neighborX[neighborAtomIndex],
                                             localX[localAtomIndex],
                                             atomOffsetVector);
//              SCIRun::Vector atomOffsetVector = neighborPositions[neighborAtomIndex] - localPositions[localAtomIndex];
              // find minimum image of offset vector FIXME

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
  newDW->put(sum_vartype(0.5 * nbEnergy_patchLocal),
             label->nonbonded->rNonbondedEnergy);
  newDW->put(matrix_sum(0.5 * stressTensor_patchLocal),
             label->nonbonded->rNonbondedStress);
} // TwoBodyDeterministic::calculate

void TwoBodyDeterministic::addFinalizeRequirements(Task* task, MDLabel* d_label) const {
  // This space intentionally left blank
}

void TwoBodyDeterministic::addFinalizeComputes(Task* task, MDLabel* d_label) const {
  // What is the sound of one hand coding?
}

void TwoBodyDeterministic::finalize(const ProcessorGroup*   pg,
                                    const PatchSubset*      patches,
                                    const MaterialSubset*   materials,
                                    DataWarehouse*          oldDW,
                                    DataWarehouse*          newDW,
                                    SimulationStateP&       simState,
                                    MDSystem*               systemInfo,
                                    const MDLabel*          label,
                                    coordinateSystem*       coordSys) {
  // Nothing to put here now
}

void TwoBodyDeterministic::registerRequiredParticleStates(std::vector<const VarLabel*>& particleState,
                                                          std::vector<const VarLabel*>& particleState_preReloc,
                                                          MDLabel* d_label) const {
  //  We probably don't need these for relocation, but it may be easier to set them up that way than to do it any other way
  particleState.push_back(d_label->nonbonded->pF_nonbonded);
  particleState_preReloc.push_back(d_label->nonbonded->pF_nonbonded_preReloc);

}
