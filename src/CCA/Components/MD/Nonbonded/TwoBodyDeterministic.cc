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

#include <fstream>
#include <iomanip>


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
  task->computes(label->nonbonded->pNumPairsInCalc);

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
      ParticleVariable<long64> pNumPairsCalc;
      newDW->allocateAndPut(pNumPairsCalc,
                            label->nonbonded->pNumPairsInCalc,
                            atomSubset);

      particleIndex numAtoms = atomSubset->numParticles();
      for (particleIndex atom = 0; atom < numAtoms; ++atom)
      {
        pF_nb[atom] = 0.0;
        pNumPairsCalc[atom]=0;
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
  proc0cout << "Registering for " << CUTOFF_CELLS << " ghost cells required." << std::endl;
  task->requires(Task::OldDW,
                 label->global->pX,
                 Ghost::AroundCells,
                 CUTOFF_CELLS);

  task->requires(Task::OldDW,
                 label->global->pID,
                 Ghost::AroundCells,
                 CUTOFF_CELLS);

}

void TwoBodyDeterministic::addCalculateComputes(Task* task,
                                                MDLabel* label) const {
  // Provide per particle force
  task->computes(label->nonbonded->pF_nonbonded_preReloc);
  task->computes(label->nonbonded->pNumPairsInCalc_preReloc);

  // Provide per patch contribution to energy and stress tensor
  task->computes(label->nonbonded->rNonbondedEnergy);
  task->computes(label->nonbonded->rNonbondedStress);

}

void TwoBodyDeterministic::calculate(const ProcessorGroup*  pg,
                                     const PatchSubset*     patches,
                                     const MaterialSubset*  processorAtomTypes,
                                     DataWarehouse*         oldDW,
                                     DataWarehouse*         newDW,
                                     SimulationStateP&      simState,
                                     MDSystem*              systemInfo,
                                     const MDLabel*         label,
                                     CoordinateSystem*      coordSys)
{
  // TODO FIXME
  // Note:  It is theoretically possible that the material subset on a given
  // source patch doesn't contain a material type, but a nearby target patch
  // does.  If that's the case, we will fail to calculate the influence of that
  // target material type on all source material types in the patch, I believe.
  // This is important, and should be fixed!!!
  //
  // Maybe we can just grab a universal material subset for all materials and
  // use that to pull neighbor information?
  //
  // TODO FIXME (10/5/2014)
  double cutoff2 = d_nonbondedRadius * d_nonbondedRadius;
  TwoBodyForcefield* forcefield = dynamic_cast<TwoBodyForcefield*>
                                        ( systemInfo->getForcefieldPointer() );
  // Initialize local accumulators
  double nbEnergy_patchLocal = 0;
  Uintah::Matrix3 stressTensor_patchLocal = MDConstants::M3_0;

  size_t numPatches     = patches->size();
  size_t numAtomTypes   = processorAtomTypes->size();

//  int timestep = simState->getCurrentTopLevelTimeStep();
//  std::fstream debugOut;
//  std::string debugName = "source445.txt";
//  debugOut.open(debugName.c_str(),std::fstream::out | std::fstream::app);

  SCIRun::Vector atomOffsetVector(0.0);
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex)
  { // Loop over all patches
    const Patch* currPatch = patches->get(patchIndex);
    for (size_t sourceIndex = 0; sourceIndex < numAtomTypes; ++sourceIndex)
    { // Internal patch material loop
    // Build particle set for on-patch atom/material set
      int               sourceAtomType = processorAtomTypes->get(sourceIndex);
      ParticleSubset*   sourceAtomSet  = oldDW->getParticleSubset(sourceAtomType,
                                                                  currPatch);
      size_t            numSourceAtoms = sourceAtomSet->numParticles();

    // Get atom ID and positions
      constParticleVariable<long64> sourceID;
      oldDW->get(sourceID, label->global->pID, sourceAtomSet);
      constParticleVariable<Point> sourceX;
      oldDW->get(sourceX, label->global->pX, sourceAtomSet);
    // Get material map label for source atom type
      std::string sourceMaterialLabel = simState->getMDMaterial(sourceAtomType)
                                                  ->getMapLabel();
      // Initialize force variable
      ParticleVariable<Vector> pForce;
      newDW->allocateAndPut(pForce,
                            label->nonbonded->pF_nonbonded_preReloc,
                            sourceAtomSet);
      ParticleVariable<long64> numPairsInteracting;
      newDW->allocateAndPut(numPairsInteracting,
                            label->nonbonded->pNumPairsInCalc_preReloc,
                            sourceAtomSet);

//      for (size_t sourceAtom = 0; sourceAtom < numSourceAtoms; ++sourceAtom)
//      { // Initialize force array
//        pForce[sourceAtom] = 0.0;
//      }
      for (size_t sourceAtom = 0; sourceAtom < numSourceAtoms; ++sourceAtom)
      {
        pForce[sourceAtom] = 0.0;
        numPairsInteracting[sourceAtom] = 0;
      }

      // (Internal + ghost) patch material loop
      for (size_t targetIndex = 0; targetIndex < numAtomTypes; ++targetIndex)
      {
        // Build particle set for on patch + nearby atoms
        int             targetAtomType = processorAtomTypes->get(targetIndex);
        ParticleSubset* targetAtomSet;
        targetAtomSet = oldDW->getParticleSubset(targetAtomType,
                                                 currPatch
                                                 ,Ghost::AroundCells,
                                                 d_nonbondedGhostCells,
                                                 label->global->pX);
        size_t numTargetAtoms = targetAtomSet->numParticles();
        // Get atom ID and positions
        constParticleVariable<long64> targetID;
        oldDW->get(targetID, label->global->pID, targetAtomSet);
        constParticleVariable<Point> targetX;
        oldDW->get(targetX, label->global->pX, targetAtomSet);
        // Get material map label for source atom type
        std::string targetMaterialLabel =
                        simState->getMDMaterial(targetAtomType)->getMapLabel();

        // All the local and neighbor related variables have been set up,
        // get a potential and begin calculation
        NonbondedTwoBodyPotential* currPotential;
        currPotential = forcefield->getNonbondedPotential(sourceMaterialLabel,
                                                          targetMaterialLabel);

        for (size_t sourceAtom = 0; sourceAtom < numSourceAtoms; ++sourceAtom)
        { // Loop over atoms in local patch
          for (size_t targetAtom = 0; targetAtom < numTargetAtoms; ++targetAtom)
          { // Loop over local plus nearby atoms
            if (sourceID[sourceAtom] != targetID[targetAtom])
            { // Ensure we're not working with the same particle
              atomOffsetVector = targetX[targetAtom] - sourceX[sourceAtom];

//              if (sourceID[sourceAtom] == 445 && targetID[targetAtom] == 7947)
////              if ( targetID[targetAtom] == 7947)
////                if (sourceID[sourceAtom] == 445)
//              {
//                std::cerr << "Patch:  " << patchIndex
//                          << " dCell: " << currPatch->dCell() << " || "
//                          << " Source: " << sourceID[sourceAtom] << " - "
//                          << sourceX[sourceAtom]
//                          << " Range: "  << currPatch->getCellLowIndex() << " - "
//                          << currPatch->getCellHighIndex()
//                          << " O "
//                          << " Target: " << targetID[targetAtom] << " - "
//                          << targetX[targetAtom] << " : "
//                          << " Range: " << currPatch->getExtraCellLowIndex() << " - "
//                          << currPatch->getExtraCellHighIndex()
//                          << " Distance: "
//                          << atomOffsetVector.length2() << " < " << cutoff2
//                          << std::endl;
//              }

              if (atomOffsetVector.length2() <= cutoff2)
              { // Interaction is within range
                numPairsInteracting[sourceAtom]++;
//                if ((sourceID[sourceAtom] == 445) && (timestep <= 1))
//                {
//                  debugOut << " t: " << std::setw(2) << std::left
//                           << timestep
//                           << " S: " << std::setw(6) << std::right
//                           << sourceID[sourceAtom]
//                           << " T: " << std::setw(6) << std::right
//                           << targetID[targetAtom]
//                           << " TLoc: " << targetX[targetAtom]
//                           << " Dist: " << atomOffsetVector.length()
//                           << std::endl;
//                }
                SCIRun::Vector  tempForce;
                double          tempEnergy;
                currPotential->fillEnergyAndForce(tempForce,
                                                  tempEnergy,
                                                  atomOffsetVector);
                pForce[sourceAtom]      +=  tempForce;
                nbEnergy_patchLocal     +=  tempEnergy;
                stressTensor_patchLocal +=  OuterProduct(atomOffsetVector,
                                                         tempForce);

              }
            }  // IDs not the same
          } // Loop over target Atoms
        }  // Loop over source Atoms
      }  // Loop over target materials
    }  // Loop over source materials
  }  // Loop over patches
//  debugOut.close();
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
  particleState.push_back(d_label->nonbonded->pNumPairsInCalc);
  particleState_preReloc.push_back(d_label->nonbonded->pF_nonbonded_preReloc);
  particleState_preReloc.push_back(d_label->nonbonded->pNumPairsInCalc_preReloc);

}
