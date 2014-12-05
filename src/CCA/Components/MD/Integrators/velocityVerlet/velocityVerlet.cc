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
 * velocityVerlet.cc
 *
 *  Created on: Oct 21, 2014
 *      Author: jbhooper
 */

#include <CCA/Components/MD/Integrators/velocityVerlet/velocityVerlet.h>

using namespace Uintah;

velocityVerlet::velocityVerlet(const VarLabel* _dt_label)
                             :dt_label(_dt_label),
                              d_integratorType("velocityVerlet")

{
  d_firstIntegration = true;
  d_previousKE   = 0.0;
  d_previousPE   = 0.0;
  d_previousTemp = 0.0;
  d_previousMass = 0.0;
  d_dt = 0.0;

  d_previousMomentum = MDConstants::V_ZERO;
  d_currentTimestep = 0;
}

velocityVerlet::~velocityVerlet()
{

}

void velocityVerlet::registerRequiredParticleStates(
                                           LabelArray& particleState,
                                           LabelArray& particleState_preReloc,
                                           MDLabel*    labels) const
{

  // Technically, not all integrators must necessarily track velocity
  // explicitly.  So the velocity label probably belongs to the integrator.
  particleState.push_back(labels->global->pV);
  particleState.push_back(labels->global->pID);

  particleState_preReloc.push_back(labels->global->pV_preReloc);
  particleState_preReloc.push_back(labels->global->pID_preReloc);

}

void velocityVerlet::addInitializeRequirements(Task*    task,
                                               MDLabel* labels
                                              ) const
{
  // Nothing to add
}

void velocityVerlet::addInitializeComputes(    Task*    task,
                                               MDLabel* labels
                                          ) const
{
  // Nothing to compute
}

void velocityVerlet::initialize(const ProcessorGroup*       pg,
                                const PatchSubset*          patches,
                                const MaterialSubset*       atomTypes,
                                      DataWarehouse*      /*oldDW*/,
                                      DataWarehouse*        newDW,
                                const SimulationStateP*     simState,
                                      MDSystem*             systemInfo,
                                const MDLabel*              label,
                                      CoordinateSystem*     coordSys)
{
  // No code
}

void velocityVerlet::addSetupRequirements(     Task*    task,
                                               MDLabel* labels
                                         ) const
{
  // Nothing to add - yet
}

void velocityVerlet::addSetupComputes(         Task*    task,
                                               MDLabel* labels
                                     ) const
{
  // Nothing to add - yet
}

void velocityVerlet::setup(     const ProcessorGroup*       pg,
                                const PatchSubset*          patches,
                                const MaterialSubset*       atomTypes,
                                      DataWarehouse*        oldDW,
                                      DataWarehouse*        newDW,
                                const SimulationStateP*     simState,
                                      MDSystem*             systemInfo,
                                const MDLabel*              label,
                                      CoordinateSystem*     coordSys)
{
  // No code
}

void velocityVerlet::addCalculateRequirements( Task*    task,
                                               MDLabel* labels
                                             ) const
{
  task->requires(Task::OldDW, labels->global->pX,  Ghost::None, 0);
  task->requires(Task::OldDW, labels->global->pV,  Ghost::None, 0);
  task->requires(Task::OldDW, labels->global->pID, Ghost::None, 0);

  task->requires(Task::OldDW, labels->global->rTotalMomentum);
  task->requires(Task::OldDW, labels->global->rTotalMass);
  task->requires(Task::OldDW, labels->global->rKineticEnergy);

  task->requires(Task::OldDW, labels->nonbonded->rNonbondedEnergy);
  task->requires(Task::OldDW, labels->electrostatic->rElectrostaticRealEnergy);
  task->requires(Task::OldDW, labels->electrostatic->rElectrostaticInverseEnergy);


  // Eventually the individual components should actually take care of
  // dropping their force contributions into a general "Force" array
  // so that I can just require the current forces here:
  // task->requires(Task::NewDW,
  //                labels->global->pF_preReloc,
  //                Ghost::None,
  //                0);
  task->requires(Task::NewDW,
                 labels->nonbonded->pF_nonbonded_preReloc,
                 Ghost::None,
                 0);
  task->requires(Task::NewDW,
                 labels->electrostatic->pF_electroInverse_preReloc,
                 Ghost::None,
                 0);
  task->requires(Task::NewDW,
                 labels->electrostatic->pF_electroReal_preReloc,
                 Ghost::None,
                 0);

  // Finally, we kinda need the timestep to integrate!
  task->requires(Task::OldDW, dt_label);
}

void velocityVerlet::addCalculateComputes(     Task*    task,
                                               MDLabel* labels
                                         ) const
{
  task->computes(labels->global->pX_preReloc);
  task->computes(labels->global->pV_preReloc);
  task->computes(labels->global->pID_preReloc);

  task->computes(labels->global->rKineticEnergy);
  task->computes(labels->global->rKineticStress);
  task->computes(labels->global->rTotalMomentum);
  task->computes(labels->global->rTotalMass);

}

void velocityVerlet::calculate( const ProcessorGroup*       pg,
                                const PatchSubset*          patches,
                                const MaterialSubset*       atomTypes,
                                      DataWarehouse*        oldDW,
                                      DataWarehouse*        newDW,
                                const SimulationStateP*     simState,
                                      MDSystem*             systemInfo,
                                const MDLabel*              label,
                                      CoordinateSystem*     coordSys)
{
  // Because of the way Uintah pushes reductions, we only ever have the
  // total energies from the previous timestep.
  sum_vartype       previousMass, previousKE, componentPE;
  sumvec_vartype    previousMomentum;
  oldDW->get(previousMass,     label->global->rTotalMass);
  oldDW->get(previousKE,       label->global->rKineticEnergy);
  oldDW->get(previousMomentum, label->global->rTotalMomentum);
  d_previousKE          = previousKE;
  d_previousMomentum    = previousMomentum;
  d_previousMass        = previousMass;

  oldDW->get(componentPE, label->nonbonded->rNonbondedEnergy);
  d_previousPE = componentPE;
  oldDW->get(componentPE, label->electrostatic->rElectrostaticRealEnergy);
  d_previousPE += componentPE;
  oldDW->get(componentPE, label->electrostatic->rElectrostaticInverseEnergy);
  d_previousPE += componentPE;

  delt_vartype delT;
  oldDW->get(delT, (*simState)->get_delt_label(), getLevel(patches));
  d_dt = delT;
  int d_currentTimestep = (*simState)->getCurrentTopLevelTimeStep() - 1;

  if (d_firstIntegration)
  {
    d_firstIntegration = false;
    firstIntegrate(patches, atomTypes, oldDW, newDW, simState, label);
  }
  else
  {
    integrate(patches, atomTypes, oldDW, newDW, simState, label);
  }
}

void velocityVerlet::addFinalizeRequirements(  Task*    task,
                                               MDLabel* labels
                                            ) const
{
  // No code
}

void velocityVerlet::addFinalizeComputes(      Task*    task,
                                               MDLabel* labels
                                        ) const
{
  // No code
}

void velocityVerlet::finalize(  const ProcessorGroup*       pg,
                                const PatchSubset*          patches,
                                const MaterialSubset*       atomTypes,
                                      DataWarehouse*        oldDW,
                                      DataWarehouse*        newDW,
                                const SimulationStateP*     simState,
                                      MDSystem*             systemInfo,
                                const MDLabel*              label,
                                      CoordinateSystem*     coordSys)
{
  // No Code
}

void velocityVerlet::firstIntegrate(const PatchSubset*          patches,
                                    const MaterialSubset*       atomTypes,
                                          DataWarehouse*        oldDW,
                                          DataWarehouse*        newDW,
                                    const SimulationStateP*     simState,
                                    const MDLabel*              label)
{
  SCIRun::Vector momentumFraction = d_previousMomentum/d_previousMass;
  // Accumulators for reduction variables
  double          totalMass     = 0.0;
  double          kineticEnergy = 0.0;
  Uintah::Matrix3 kineticStress = MDConstants::M3_0;
  SCIRun::Vector  totalMomentum = MDConstants::V_ZERO;

  // Normalization constants
  double forceNorm = 41.84;
  double velocNorm = 1.0e-5;
  double normKE    = 0.5*0.001*(1.0/4184.0);

  // Temporary vectors to avoid inner loop temp creation
  SCIRun::Vector F_nPlus1;
  int numPatches = patches->size();
  int numTypes   = atomTypes->size();

  for (int patchIndex = 0; patchIndex < numPatches; ++patchIndex)
  {
    const Patch* currPatch = patches->get(patchIndex);
    for (int typeIndex = 0; typeIndex < numTypes; ++typeIndex)
    {
      int       atomType    =   atomTypes->get(typeIndex);
      double    atomMass    =   (*simState)->getMDMaterial(atomType)->getMass();
      double    massInv     =   1.0/atomMass;

      ParticleSubset* atomSet = oldDW->getParticleSubset(atomType, currPatch);

      constParticleVariable<long64>         pID_n;
           ParticleVariable<long64>         pID_nPlus1;
      constParticleVariable<Point>          pX_n;
           ParticleVariable<Point>          pX_nPlus1;
      constParticleVariable<SCIRun::Vector> pV_n;
           ParticleVariable<SCIRun::Vector> pV_nPlus1;
      constParticleVariable<SCIRun::Vector> pF_nb_nPlus1, pF_eReal_nPlus1, pF_eInv_nPlus1;

      oldDW->get(pID_n,label->global->pID,atomSet);
      oldDW->get(pX_n, label->global->pX, atomSet);
      oldDW->get(pV_n, label->global->pV, atomSet);

      newDW->allocateAndPut(pID_nPlus1,label->global->pID_preReloc, atomSet);
      newDW->allocateAndPut(pX_nPlus1, label->global->pX_preReloc,  atomSet);
      newDW->allocateAndPut(pV_nPlus1, label->global->pV_preReloc,  atomSet);

      newDW->get(pF_nb_nPlus1,    label->nonbonded->pF_nonbonded_preReloc, atomSet);
      newDW->get(pF_eReal_nPlus1, label->electrostatic->pF_electroReal_preReloc, atomSet);
      newDW->get(pF_eInv_nPlus1,  label->electrostatic->pF_electroInverse_preReloc, atomSet);

      ParticleSubset::iterator atomBegin, atomEnd, atom;
      atomBegin = atomSet->begin();
      atomEnd   = atomSet->end();
      for (atom = atomBegin; atom != atomEnd; ++atom)
      {
        F_nPlus1 = pF_nb_nPlus1[*atom] + pF_eReal_nPlus1[*atom] + pF_eInv_nPlus1[*atom];
        pV_nPlus1[*atom] = (pV_n[*atom] - momentumFraction)
                         + 0.5 * F_nPlus1 * d_dt * forceNorm * massInv;

        kineticEnergy += atomMass * pV_nPlus1[*atom].length2();
        totalMomentum += atomMass * pV_nPlus1[*atom];
        totalMass     += atomMass;

        pX_nPlus1[*atom]   =  pX_n[*atom] + d_dt * velocNorm * pV_nPlus1[*atom];
        pID_nPlus1[*atom]  =  pID_n[*atom];
      } // Loop over atom subset
      ParticleSubset* delset = scinew ParticleSubset(0, atomType, currPatch);
      newDW->deleteParticles(delset);
    } // Loop over atom type
  } // Loop over patch
  kineticEnergy *= normKE;
  newDW->put( sum_vartype(kineticEnergy),    label->global->rKineticEnergy);
  newDW->put( sum_vartype(totalMass),        label->global->rTotalMass    );
  newDW->put( sumvec_vartype(totalMomentum), label->global->rTotalMomentum);
  newDW->put( matrix_sum(kineticStress),     label->global->rKineticStress);
} // velocityVerlet::firstIntegrate

void velocityVerlet::integrate(     const PatchSubset*          patches,
                                    const MaterialSubset*       atomTypes,
                                          DataWarehouse*        oldDW,
                                          DataWarehouse*        newDW,
                                    const SimulationStateP*     simState,
                                    const MDLabel*              label)
{
  SCIRun::Vector momentumFraction = d_previousMomentum/d_previousMass;
  // Accumulators for reduction variables
  double          totalMass     = 0.0;
  double          kineticEnergy = 0.0;
  Uintah::Matrix3 kineticStress = MDConstants::M3_0;
  SCIRun::Vector  totalMomentum = MDConstants::V_ZERO;

  // Normalization constants
  double forceNorm = 41.84;
  double velocNorm = 1.0e-5;
  double normKE    = 0.5*0.001*(1.0/4184.0);

  // Temporary vectors to avoid inner loop temp creation
  SCIRun::Vector F_nPlus1;
  int numPatches = patches->size();
  int numTypes   = atomTypes->size();

  for (int patchIndex = 0; patchIndex < numPatches; ++patchIndex)
  {
    const Patch* currPatch = patches->get(patchIndex);
    for (int typeIndex = 0; typeIndex < numTypes; ++typeIndex)
    {
      int       atomType    =   atomTypes->get(typeIndex);
      double    atomMass    =   (*simState)->getMDMaterial(atomType)->getMass();
      double    massInv     =   1.0/atomMass;

      ParticleSubset* atomSet = oldDW->getParticleSubset(atomType, currPatch);

      constParticleVariable<long64>         pID_n;
           ParticleVariable<long64>         pID_nPlus1;
      constParticleVariable<Point>          pX_n;
           ParticleVariable<Point>          pX_nPlus1;
      constParticleVariable<SCIRun::Vector> pV_n;
           ParticleVariable<SCIRun::Vector> pV_nPlus1;
      constParticleVariable<SCIRun::Vector> pF_nb_nPlus1, pF_eReal_nPlus1, pF_eInv_nPlus1;

      oldDW->get(pID_n,label->global->pID,atomSet);
      oldDW->get(pX_n, label->global->pX, atomSet);
      oldDW->get(pV_n, label->global->pV, atomSet);

      newDW->allocateAndPut(pID_nPlus1,label->global->pID_preReloc, atomSet);
      newDW->allocateAndPut(pX_nPlus1, label->global->pX_preReloc,  atomSet);
      newDW->allocateAndPut(pV_nPlus1, label->global->pV_preReloc,  atomSet);

      newDW->get(pF_nb_nPlus1,    label->nonbonded->pF_nonbonded_preReloc, atomSet);
      newDW->get(pF_eReal_nPlus1, label->electrostatic->pF_electroReal_preReloc, atomSet);
      newDW->get(pF_eInv_nPlus1,  label->electrostatic->pF_electroInverse_preReloc, atomSet);

      ParticleSubset::iterator atomBegin, atomEnd, atom;
      atomBegin = atomSet->begin();
      atomEnd   = atomSet->end();
      for (atom = atomBegin; atom != atomEnd; ++atom)
      {
        F_nPlus1 = pF_nb_nPlus1[*atom] + pF_eReal_nPlus1[*atom] + pF_eInv_nPlus1[*atom];
        pV_nPlus1[*atom] = (pV_n[*atom] - momentumFraction)
                         + 0.5 * F_nPlus1 * d_dt * forceNorm * massInv;

        kineticEnergy += atomMass * pV_nPlus1[*atom].length2();
        totalMomentum += atomMass * pV_nPlus1[*atom];
        totalMass     += atomMass;

        pV_nPlus1[*atom]  +=  0.5 * F_nPlus1 * d_dt * forceNorm * massInv;

        pX_nPlus1[*atom]   =  pX_n[*atom] + d_dt * velocNorm * pV_nPlus1[*atom];
        pID_nPlus1[*atom]  =  pID_n[*atom];
      } // Loop over atom subset
      ParticleSubset* delset = scinew ParticleSubset(0, atomType, currPatch);
      newDW->deleteParticles(delset);
    } // Loop over atom type
  } // Loop over patch
  kineticEnergy *= normKE;
  newDW->put( sum_vartype(kineticEnergy),    label->global->rKineticEnergy);
  newDW->put( sum_vartype(totalMass),        label->global->rTotalMass    );
  newDW->put( sumvec_vartype(totalMomentum), label->global->rTotalMomentum);
  newDW->put( matrix_sum(kineticStress),     label->global->rKineticStress);
} // velocityVerlet::integrate
