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
 * Null.cc
 *
 *  Created on: Sep 26, 2014
 *      Author: jbhooper
 */

#include <CCA/Components/MD/Electrostatics/Null/Null.h>

using namespace Uintah;

ElectrostaticNull::ElectrostaticNull() {

}

ElectrostaticNull::~ElectrostaticNull() {

}

void ElectrostaticNull::initialize(const ProcessorGroup*   /*pg*/,
                                   const PatchSubset*        patches,
                                   const MaterialSubset*     atomTypes,
                                         DataWarehouse*    /*oldDW*/,
                                         DataWarehouse*      newDW,
                                   const SimulationStateP* /*simState*/,
                                         MDSystem*         /*systemInfo*/,
                                   const MDLabel*            label,
                                         CoordinateSystem* /*coordSys*/)
{
  size_t    numPatches      = patches->size();
  size_t    numAtomTypes    = atomTypes->size();

  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex)
  {
    const Patch*    patch   =   patches->get(patchIndex);
    newDW->put(sum_vartype(0.0),
               label->electrostatic->rElectrostaticInverseEnergy);
    newDW->put(sum_vartype(0.0),
               label->electrostatic->rElectrostaticRealEnergy);

    newDW->put(matrix_sum(MDConstants::M3_0),
               label->electrostatic->rElectrostaticInverseStress);
    newDW->put(matrix_sum(MDConstants::M3_0),
               label->electrostatic->rElectrostaticRealStress);

    for (size_t typeIndex = 0; typeIndex < numAtomTypes; ++typeIndex)
    {
      int atomType = atomTypes->get(typeIndex);
      ParticleSubset* atomSubset = newDW->getParticleSubset(atomType, patch);

      ParticleVariable<Vector> pF_real, pF_inverse;
      newDW->allocateAndPut(pF_real,
                            label->electrostatic->pF_electroReal,
                            atomSubset);
      newDW->allocateAndPut(pF_inverse,
                            label->electrostatic->pF_electroInverse,
                            atomSubset);
      particleIndex numAtoms = atomSubset->numParticles();
      for (particleIndex atom = 0; atom < numAtoms; ++atom)
      {
        pF_real[atom]       =   MDConstants::V_ZERO;
        pF_inverse[atom]    =   MDConstants::V_ZERO;
      }
    }
  }
}

void ElectrostaticNull::setup     (const ProcessorGroup*    /*pg*/,
                                   const PatchSubset*       /*patches*/,
                                   const MaterialSubset*   /*atomTypes*/,
                                         DataWarehouse*    /*oldDW*/,
                                         DataWarehouse*    /*newDW*/,
                                   const SimulationStateP* /*simState*/,
                                         MDSystem*         /*systemInfo*/,
                                   const MDLabel*          /*label*/,
                                         CoordinateSystem* /*coordSys*/)
{
  // Empty
}

void ElectrostaticNull::calculate (const ProcessorGroup*   /*pg*/,
                                   const PatchSubset*        patches,
                                   const MaterialSubset*     atomTypes,
                                         DataWarehouse*      oldDW,
                                         DataWarehouse*      newDW,
                                   const SimulationStateP* /*simState*/,
                                         MDSystem*         /*systemInfo*/,
                                   const MDLabel*            label,
                                         CoordinateSystem* /*coordSys*/,
                                         SchedulerP&       /*subscheduler*/,
                                   const LevelP&             level)
{
  size_t    numPatches      = patches->size();
  size_t    numAtomTypes    = atomTypes->size();

  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex)
  {
    const Patch*    patch   =   patches->get(patchIndex);
    newDW->put(sum_vartype(0.0),
               label->electrostatic->rElectrostaticInverseEnergy);
    newDW->put(sum_vartype(0.0),
               label->electrostatic->rElectrostaticRealEnergy);

    newDW->put(matrix_sum(MDConstants::M3_0),
               label->electrostatic->rElectrostaticInverseStress);
    newDW->put(matrix_sum(MDConstants::M3_0),
               label->electrostatic->rElectrostaticRealStress);

    for (size_t typeIndex = 0; typeIndex < numAtomTypes; ++typeIndex)
    {
      int atomType = atomTypes->get(typeIndex);
      ParticleSubset* atomSubset = oldDW->getParticleSubset(atomType, patch);

      ParticleVariable<Vector> pF_real, pF_inverse;
      newDW->allocateAndPut(pF_real,
                            label->electrostatic->pF_electroReal_preReloc,
                            atomSubset);
      newDW->allocateAndPut(pF_inverse,
                            label->electrostatic->pF_electroInverse_preReloc,
                            atomSubset);
      particleIndex numAtoms = atomSubset->numParticles();
      for (particleIndex atom = 0; atom < numAtoms; ++atom)
      {
        pF_real[atom]       =   MDConstants::V_ZERO;
        pF_inverse[atom]    =   MDConstants::V_ZERO;
      }
    }
  }
}

void ElectrostaticNull::finalize  (const ProcessorGroup*     pg,
                                   const PatchSubset*        patches,
                                   const MaterialSubset*     atomTypes,
                                         DataWarehouse*      oldDW,
                                         DataWarehouse*      newDW,
                                   const SimulationStateP* /*simState*/,
                                         MDSystem*         /*systemInfo*/,
                                   const MDLabel*            label,
                                         CoordinateSystem* /*coordSys*/)
{
  // Empty
}

void ElectrostaticNull::registerRequiredParticleStates(
                                    varLabelArray&         particleState,
                                    varLabelArray&         particleState_preReloc,
                                    MDLabel*            label) const
{
  particleState.push_back(label->electrostatic->pF_electroInverse);
  particleState_preReloc.push_back(
                          label->electrostatic->pF_electroInverse_preReloc);
  particleState.push_back(label->electrostatic->pF_electroReal);
  particleState_preReloc.push_back(
                          label->electrostatic->pF_electroReal_preReloc);
}

void ElectrostaticNull::addInitializeRequirements(
                                    Task*               task,
                                    MDLabel*            label) const
{
  // Empty
}

void ElectrostaticNull::addInitializeComputes(
                                    Task*               task,
                                    MDLabel*            label) const
{
  task->computes(label->electrostatic->rElectrostaticInverseEnergy);
  task->computes(label->electrostatic->rElectrostaticInverseStress);
  task->computes(label->electrostatic->rElectrostaticRealEnergy);
  task->computes(label->electrostatic->rElectrostaticRealStress);

  task->computes(label->electrostatic->pF_electroInverse);
  task->computes(label->electrostatic->pF_electroReal);
}

void ElectrostaticNull::addSetupRequirements(
                                    Task*               task,
                                    MDLabel*            label) const
{
  // Empty
}

void ElectrostaticNull::addSetupComputes(
                                    Task*               task,
                                    MDLabel*            label) const
{
  // Empty
}

void ElectrostaticNull::addCalculateRequirements(
                                    Task*               task,
                                    MDLabel*            label) const
{
  // Empty
}

void ElectrostaticNull::addCalculateComputes(
                                    Task*               task,
                                    MDLabel*            label) const
{
  task->computes(label->electrostatic->rElectrostaticInverseEnergy);
  task->computes(label->electrostatic->rElectrostaticInverseStress);
  task->computes(label->electrostatic->rElectrostaticRealEnergy);
  task->computes(label->electrostatic->rElectrostaticRealStress);

  task->computes(label->electrostatic->pF_electroInverse_preReloc);
  task->computes(label->electrostatic->pF_electroReal_preReloc);
}

void ElectrostaticNull::addFinalizeRequirements(
                                    Task*               task,
                                    MDLabel*            label) const
{
  // Empty
}

void ElectrostaticNull::addFinalizeComputes(
                                    Task*               task,
                                    MDLabel*            label) const
{
  // Empty
}
