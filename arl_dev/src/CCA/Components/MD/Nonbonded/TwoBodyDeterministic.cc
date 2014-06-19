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
#include <CCA/Components/MD/CoordinateSystems/CoordinateSystem.h>

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
                                           d_nonbondedGhostCells(_nbGhost)
{
  // Empty
}

void TwoBodyDeterministic::addInitializeRequirements(Task* task,
                                                     MDLabel* label) const
{
  // Empty
}

void TwoBodyDeterministic::addInitializeComputes(Task* task,
                                                 MDLabel* label) const
{
  // Need to provide the particle variables we register
  task->computes(label->nonbonded->pF_nonbonded);

  // And reductions to initialize
  task->computes(label->nonbonded->rNonbondedEnergy);
  task->computes(label->nonbonded->rNonbondedStress);
}

void TwoBodyDeterministic::initialize(const ProcessorGroup*     pg,
                                      const PatchSubset*        patches,
                                      const MaterialSubset*     materials,
                                      DataWarehouse*          /*oldDW*/,
                                      DataWarehouse*            newDW,
                                      SimulationStateP&         simState,
                                      MDSystem*                 systemInfo,
                                      const MDLabel*            label,
                                      CoordinateSystem*         coordSys)
{
  // Do nothing
  std::cout << " TwoBodyDeterministic::Initialize" << std::endl;

  size_t numPatches     =   patches->size();
  size_t numAtomTypes   =   materials->size();

  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex)
  {
    const Patch*    currPatch   =   patches->get(patchIndex);
    for (size_t typeIndex = 0; typeIndex < numAtomTypes; ++typeIndex)
    {
      int   atomType    =   materials->get(typeIndex);
      ParticleSubset*   atomSubset  =   newDW->getParticleSubset(atomType,
                                                                 currPatch);
      ParticleVariable<Vector> pF_nb;
      newDW->allocateAndPut(pF_nb,
                            label->nonbonded->pF_nonbonded,
                            atomSubset);
      particleIndex numAtoms = atomSubset->numParticles();
      for (particleIndex atom = 0; atom < numAtoms; ++atom)
      {
        pF_nb[atom] = 0.0;
      }
    }
    newDW->put(matrix_sum(MDConstants::M3_0),
               label->nonbonded->rNonbondedStress);
    newDW->put(sum_vartype(0.0),
               label->nonbonded->rNonbondedEnergy);

  }

}

void TwoBodyDeterministic::addSetupRequirements(Task* task,
                                                MDLabel* label) const
{
  // None
}

void TwoBodyDeterministic::addSetupComputes(Task* task,
                                            MDLabel* label) const
{
  // None
}

void TwoBodyDeterministic::setup(const ProcessorGroup*  pg,
                                 const PatchSubset*     patches,
                                 const MaterialSubset*  materials,
                                 DataWarehouse*         oldDW,
                                 DataWarehouse*         newDW,
                                 SimulationStateP&      simState,
                                 MDSystem*              systemInfo,
                                 const MDLabel*         label,
                                 CoordinateSystem*      coordSys)
{
  // Do nothing

}

void TwoBodyDeterministic::addCalculateRequirements(Task* task,
                                                    MDLabel* label) const {

  int CUTOFF_CELLS = this->requiredGhostCells();
  task->requires(Task::OldDW,
                 label->global->pX,
                 Ghost::AroundNodes,
                 CUTOFF_CELLS);

  task->requires(Task::OldDW,
                 label->global->pID,
                 Ghost::AroundNodes,
                 CUTOFF_CELLS);

}

void TwoBodyDeterministic::addCalculateComputes(Task* task,
                                                MDLabel* label) const {
  // Provide per particle force
  task->computes(label->nonbonded->pF_nonbonded_preReloc);

  // Provide per patch contribution to energy and stress tensor
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
                                     CoordinateSystem*      coordSys)
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
    for (size_t sourceMatInd = 0; sourceMatInd < numMaterials; ++sourceMatInd)
    {
      // Build particle set for on-patch atom/material set
      int               sourceMaterial = materials->get(sourceMatInd);
      ParticleSubset*   sourceAtomSet  =
                            oldDW->getParticleSubset(sourceMaterial,
                                                     currPatch);
      size_t numSourceAtoms = sourceAtomSet->numParticles();

      // Get atom ID and positions
      constParticleVariable<long64> sourceID;
      oldDW->get(sourceID,
                 label->global->pID,
                 sourceAtomSet);

      constParticleVariable<Point> sourceX;
      oldDW->get(sourceX,
                 label->global->pX,
                 sourceAtomSet);

      // Get material map label for source atom type
      std::string sourceMaterialLabel =
          simState->getMDMaterial(sourceMaterial)->getMapLabel();

      // Initialize force variable
      ParticleVariable<Vector> pForce;
      newDW->allocateAndPut(pForce,
                            label->nonbonded->pF_nonbonded_preReloc,
                            sourceAtomSet);

      for (size_t sourceAtom = 0; sourceAtom < numSourceAtoms; ++sourceAtom)
      {
        // Initialize force array
        pForce[sourceAtom] = 0.0;
      }

      // (Internal + ghost) patch material loop
      for (size_t targetMatInd = 0; targetMatInd < numMaterials; ++targetMatInd)
      {
        // Build particle set for on patch + nearby atoms
        int targetMaterialID = materials->get(targetMatInd);
        ParticleSubset* targetAtoms =
                            oldDW->getParticleSubset(targetMaterialID,
                                                     currPatch,
                                                     Ghost::AroundNodes,
                                                     d_nonbondedGhostCells,
                                                     label->global->pX);

        size_t numTargetAtoms = targetAtoms->numParticles();

        // Get atom ID and positions
        constParticleVariable<long64> targetID;
        oldDW->get(targetID, label->global->pID, targetAtoms);
        constParticleVariable<Point> targetX;
        oldDW->get(targetX, label->global->pX, targetAtoms);

        // Get material map label for source atom type
        std::string targetMaterialLabel =
                        simState->getMDMaterial(targetMaterialID)->getMapLabel();

        // All the local and neighbor related variables have been set up,
        // get a potential and begin calculation
        NonbondedTwoBodyPotential* currentPotential =
            forcefield->getNonbondedPotential(sourceMaterialLabel,
                                              targetMaterialLabel);

        // Loop over atoms in local patch
        for (size_t sourceAtom = 0; sourceAtom < numSourceAtoms; ++sourceAtom)
        {
          // Loop over local plus nearby atoms
          for (size_t targetAtom = 0; targetAtom < numTargetAtoms; ++targetAtom)
          {
            // Ensure we're not working with the same particle
            if (sourceID[sourceAtom] != targetID[targetAtom])
            {
              coordSys->minimumImageDistance(targetX[targetAtom],
                                             sourceX[sourceAtom],
                                             atomOffsetVector);

              // Interaction is within range
              if (atomOffsetVector.length2() <= cutoff2)
              {
                SCIRun::Vector tempForce;
                double tempEnergy;
                currentPotential->fillEnergyAndForce(tempForce,
                                                     tempEnergy,
                                                     atomOffsetVector);

                nbEnergy_patchLocal     +=  tempEnergy;
                stressTensor_patchLocal +=  OuterProduct(atomOffsetVector,
                                                         tempForce);
                pForce[sourceAtom]      +=  tempForce;

              }
            }  // IDs not the same
          } // Loop over target Atoms
        }  // Loop over source Atoms

      }  // Loop over target materials

    }  // Loop over source materials

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
                                    CoordinateSystem*       coordSys) {
  // Nothing to put here now
}

void TwoBodyDeterministic::registerRequiredParticleStates(LabelArray& particleState,
                                                          LabelArray& particleState_preReloc,
                                                          MDLabel* d_label) const {
  //  We probably don't need these for relocation, but it may be easier to set
  //  them up that way than to do it any other way
  particleState.push_back(d_label->nonbonded->pF_nonbonded);
  particleState_preReloc.push_back(d_label->nonbonded->pF_nonbonded_preReloc);

}
