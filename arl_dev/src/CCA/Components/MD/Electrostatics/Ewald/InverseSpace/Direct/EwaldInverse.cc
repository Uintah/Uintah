/*
 *
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
 *
 * ----------------------------------------------------------
 * EwaldInverse.cc
 *
 *  Created on: Jul 28, 2015
 *      Author: jbhooper
 */

#include <CCA/Components/MD/Electrostatics/Ewald/InverseSpace/Direct/EwaldInverse.h>

using namespace Uintah;

void Ewald::addInitializeRequirements(Task* task, MDLabel* label) const
{

}

void Ewald::addInitializeComputes(Task* task, MDLabel* label) const
{
  // Reduction variables all electrostatic solvers will provide
  task->computes(label->electrostatic->rElectrostaticInverseEnergy);
  task->computes(label->electrostatic->rElectrostaticInverseStress);
  task->computes(label->electrostatic->rElectrostaticRealEnergy);
  task->computes(label->electrostatic->rElectrostaticRealStress);

  // Particle variables all electrostatic solvers will provide
  task->computes(label->electrostatic->pF_electroInverse);
  task->computes(label->electrostatic->pF_electroReal);

  if (f_polarizable)
  {
// Polarizable specific reduction variables
    task->computes(label->electrostatic->rElectrostaticInverseStressDipole);
// Polarizable specific particle variables
    task->computes(label->electrostatic->pMu);
    task->computes(label->electrostatic->pE_electroReal);
    task->computes(label->electrostatic->pE_electroInverse);
  }
}

void Ewald::addSetupRequirements(Task* task, MDLabel* label) const
{
  // Empty
}

void Ewald::addSetupComputes(Task* task, MDLabel* label) const
{
  // ??
}

void Ewald::addCalculateRequirements(Task* task, MDLabel* label) const
{
  task->requires(Task::OldDW, label->global->pX, Ghost::None, 0);
  task->requires(Task::OldDW, label->global->pID, Ghost::None, 0);

  if (f_polarizable) {
    task->requires(Task::OldDW, label->electrostatic->pMu, Ghost::None, 0);
  }
}

void Ewald::addCalculateComputes(Task* task, MDLabel* label) const
{
  // Reduction variables all electrostatic solvers will provide
  task->computes(label->electrostatic->rElectrostaticInverseEnergy);
  task->computes(label->electrostatic->rElectrostaticInverseStress);
  task->computes(label->electrostatic->rElectrostaticRealEnergy);
  task->computes(label->electrostatic->rElectrostaticRealStress);

  // Particle variables all electrostatic solvers will provide
  task->computes(label->electrostatic->pF_electroInverse);
  task->computes(label->electrostatic->pF_electroReal);

  if (f_polarizable)
  {
// Polarizable specific reduction variables
    task->computes(label->electrostatic->rElectrostaticInverseStressDipole);
// Polarizable specific particle variables
    task->computes(label->electrostatic->pMu);
    task->computes(label->electrostatic->pE_electroReal);
    task->computes(label->electrostatic->pE_electroInverse);
  }
}

void Ewald::addFinalizeRequirements(Task* task, MDLabel* label) const
{
  // Empty
};

void Ewald::addFinalizeComputes(Task* task, MDLabel* label) const
{
  // Empty
};

void Ewald::registerRequiredParticleStates(LabelArray& particleState,
                                           LabelArray& particleState_preReloc,
                                           MDLabel*    label) const
{
  if (f_polarizable) {
     particleState.push_back(label->electrostatic->pMu);
     particleState_preReloc.push_back(label->electrostatic->pMu_preReloc);
   }

   // We -probably- don't need relocatable Force information, however it may be
   // the easiest way to implement the required per-particle Force information.
   particleState.push_back(label->electrostatic->pF_electroInverse);
   particleState_preReloc.push_back(
                         label->electrostatic->pF_electroInverse_preReloc);
   particleState.push_back(label->electrostatic->pF_electroReal);
   particleState_preReloc.push_back(
                         label->electrostatic->pF_electroReal_preReloc);

   // Note:  Per particle charges may be required in some FF implementations
   //        (i.e. ReaxFF), however we will let the FF themselves register these
   //        variables if these are present and needed.

}

void Ewald::initializePrefactors()
{

  d_prefactor = LinearArray3<double>(d_kLimits.x(),
                                     d_kLimits.y(),
                                     d_kLimits.z(),
                                     0.0);

  d_stressPrefactor = LinearArray3<Uintah::Matrix3>(d_kLimits.x(),
                                                    d_kLimits.y(),
                                                    d_kLimits.z(),
                                                    MDConstants::M3_0)
  double invBeta2 = 1.0/(d_ewaldBeta*d_ewaldBeta);

  for (int x = 0; x < d_kLimits.x(); ++x)
  {
    for (int y = 0; y < d_kLimits.y(); ++y)
    {
      for (int z = 0; z < d_kLimits.z(); ++z)
      {
        if ((x !=0) || (y != 0) || (z!= 0))
        {
          SCIRun::Vector M = SCIRun::Vector(x,y,z);
          double M2 = (d_inverseUnitCell*M).length2();
          double Pi2M2OverBeta2 = MDConstants::PI2*M2*invBeta2;
          d_prefactor(x,y,z) = exp(-Pi2M2OverBeta2)/M2;
          d_stressPrefactor(x,y,z) = d_prefactor(x,y,z) *
             (MDConstants::M3_I -
              2.0 * ((1.0 + Pi2M2OverBeta2)/M2)*OuterProduct(M,M));
        }
      }
    }
  }
}

void Ewald::calculate(const ProcessorGroup*     pg,
                      const PatchSubset*        perProcPatches,
                      const MaterialSubset*     materials,
                            DataWarehouse*      parentOldDW,
                            DataWarehouse*      parentNewDW,
                      const SimulationStateP*   simState,
                            MDSystem*           systemInfo,
                      const MDLabel*            label,
                            CoordinateSystem*   coordSys,
                            SchedulerP&         subscheduler,
                      const LevelP&             level)
{
  const MaterialSet*    allMaterials        =   (*simState)->allMaterials();
  const MaterialSubset* allMaterialsUnion   =   allMaterials->getUnion();

  DataWarehouse::ScrubMode parentOldDW_scrubmode =
                           parentOldDW->setScrubbing(DataWarehouse::ScrubNone);

  DataWarehouse::ScrubMode parentNewDW_scrubmode =
                           parentNewDW->setScrubbing(DataWarehouse::ScrubNone);

  // Set up the subscheduler and initially populate the dipole
  GridP grid = level->getGrid();
  subscheduler->setParentDWs(parentOldDW, parentNewDW);
  subscheduler->advanceDataWarehouse(grid);
  subscheduler->setInitTimestep(true);

  DataWarehouse*    subOldDW    =   subscheduler->get_dw(2);
  DataWarehouse*    subNewDW    =   subscheduler->get_dw(3);

  if (f_polarizable)
  {
    subNewDW->transferFrom(parentOldDW,
                           label->electrostatic->pMu,
                           perProcPatches,
                           allMaterialsUnion);
  }
  subscheduler->setInitTimestep(false);

  bool              converged           =   false;
  int               numIterations       =   0;
  const PatchSet*   individualPatches   =   level->eachPatch();

  subscheduler->initialize(3,1);

  scheduleCalculateRealspace(pg, individualPatches, allMaterials,
                             subOldDW, subNewDW,
                             simState, label, coordSys,
                             subscheduler,
                             parentOldDW);

  scheduleCalculateFourierspace(pg, individualPatches, allMaterials,
                                subOldDW, subNewDW,
                                simState, label, coordSys,
                                subscheduler,
                                parentOldDW);

  if (f_polarizable)
  {
    scheduleUpdateFieldAndStress(pg, individualPatches, allMaterials,
                                 subOldDW, subNewDW,
                                 label, coordSys,
                                 subscheduler,
                                 parentOldDW);
    scheduleCalculateNewDipoles(pg, individualPatches, allMaterials,
                                subOldDW, subNewDW,
                                label,
                                subscheduler,
                                parentOldDW);
    scheduleCheckConvergence(pg, individualPatches, allMaterials,
                             subOldDW, subNewDW,
                             label,
                             subscheduler,
                             parentOldDW);
  }
  subscheduler->compile();

  while (!converged && (numIterations < d_maxPolarizableIterations))
  {
    converged = true;
    // Cycle data warehouses
    subscheduler->advanceDataWarehouse(grid);
    subscheduler->get_dw(3)->setScrubbing(DataWarehouse::ScrubNone);

    // Run our taskgraph
    subscheduler->execute();

    if (f_polarizable)
    {
    //Extract the reduced polarization deviation value from the subNewDW
      sum_vartype polarizationDeviation;
      subscheduler->get_dw(3)->get(polarizationDeviation,
                                   label->electrostatic->rPolarizationDeviation);
      double deviationValue = sqrt(polarizationDeviation);
      if (deviationValue > d_polarizationTolerance)
      {
        converged = false;
      }
    }
  }
  proc0cout << "Polarization loop completed with " << numIterations
            << " iterations." << std::endl;

  // Done with polarization, so associate the subNewDW variable name with the
  //correct DW one last time.
  subNewDW = subscheduler->get_dw(3);

  // We've converged, so can now calculate the self-correction terms for
  // electrostatics:
  double E_self = 0.0;
  double beta2  = d_ewaldBeta*d_ewaldBeta;
  double twoThirds = 2.0/3.0;

  // We cannot simply transfer dipoles from the subNewDW to the parentNewDW,
  //   because they have different variable names.  Therefore, we have to kludge this
  //   by looping through the patches.
  size_t numPatches = perProcPatches->size();
  size_t numAtomTypes = allMaterialsUnion->size();

  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex)
  {
    const Patch* currPatch = perProcPatches->get(patchIndex);
    for (size_t typeIndex = 0; typeIndex < numAtomTypes; ++typeIndex)
    {


//      int       atomType    =   materials->get(typeIndex);
      double    atomCharge  =   (*simState)->getMDMaterial(typeIndex)->getCharge();
      double    charge2     =   atomCharge*atomCharge;

      ParticleSubset* currPset = parentOldDW->getParticleSubset(typeIndex,
                                                                currPatch,
                                                                Ghost::None,
                                                                0,
                                                                label->global->pX);
      constParticleVariable<SCIRun::Vector> pMuSub, pFRealSub;
      subNewDW->get(pMuSub,
                    label->electrostatic->pMu,
                    currPset);
      subNewDW->get(pFRealSub,
                    label->electrostatic->pF_electroReal_preReloc,
                    currPset);

      //  E_self = - beta/sqrt(pi) sum(charge_i^2 + 2*beta^2*Dot(Mu_i,Mu_i)/3)
      int numAtoms = currPset->numParticles();
      for (int atom = 0; atom < numAtoms; ++atom)
      {
        E_self -= charge2 + twoThirds*beta2*(Dot(pMuSub[atom],pMuSub[atom]));
      }
      ParticleVariable<SCIRun::Vector> pMuParent, pFRealParent;
      parentNewDW->allocateAndPut(pMuParent,
                                  label->electrostatic->pMu_preReloc,
                                  currPset);
      parentNewDW->allocateAndPut(pFRealParent,
                                  label->electrostatic->pF_electroReal_preReloc,
                                  currPset);

    }
  }


  // Push energies to parent
  sum_vartype spmeEnergyTemp;
  subNewDW->get(spmeEnergyTemp,
                label->electrostatic->rElectrostaticRealEnergy);
  parentNewDW->put(spmeEnergyTemp,
                   label->electrostatic->rElectrostaticRealEnergy);


  subNewDW->get(spmeEnergyTemp,
                label->electrostatic->rElectrostaticInverseEnergy);

  double inverseEnergy = spmeEnergyTemp();
  inverseEnergy += d_ewaldBeta*E_self/MDConstants::rootPI;
  // Correct the inverse term for the spuriously included self interactions
  parentNewDW->put(sum_vartype(inverseEnergy),
                   label->electrostatic->rElectrostaticInverseEnergy);

  // Push stresses to parent
  matrix_sum spmeStressTemp;
  subNewDW->get(spmeStressTemp,
                label->electrostatic->rElectrostaticRealStress);
  parentNewDW->put(spmeStressTemp,
                   label->electrostatic->rElectrostaticRealStress);
  subNewDW->get(spmeStressTemp,
                label->electrostatic->rElectrostaticInverseStress);
  parentNewDW->put(spmeStressTemp,
                   label->electrostatic->rElectrostaticInverseStress);

  if (f_polarizable)
  {
    subNewDW->get(spmeStressTemp,
                  label->electrostatic->rElectrostaticInverseStressDipole);
    parentNewDW->put(spmeStressTemp,
                     label->electrostatic->rElectrostaticInverseStressDipole);
  }

  // Restore scrubbing state of the parent DW
  parentOldDW->setScrubbing(parentOldDW_scrubmode);
  parentNewDW->setScrubbing(parentNewDW_scrubmode);

  // Dipoles have converged, calculate forces
  if (f_polarizable) {
    calculateForceDipole(pg, perProcPatches, materials,
                         parentOldDW, parentNewDW,
                         simState, label, coordSys);
  }
  else
  {
    calculateForceCharge(pg, perProcPatches, materials,
                         parentOldDW, parentNewDW,
                         simState, label, coordSys);
  }



}
