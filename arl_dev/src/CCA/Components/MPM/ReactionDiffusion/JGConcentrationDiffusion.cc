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

#include <CCA/Components/MPM/ReactionDiffusion/JGConcentrationDiffusion.h>
#include <CCA/Components/MPM/ReactionDiffusion/ReactionDiffusionLabel.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/MPMBoundCond.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Task.h>

#include <iostream>
using namespace std;
using namespace Uintah;


JGConcentrationDiffusion::JGConcentrationDiffusion(ProblemSpecP& ps, SimulationStateP& sS, MPMFlags* Mflag, string diff_type ):
  ScalarDiffusionModel(ps, sS, Mflag, diff_type) {
}

JGConcentrationDiffusion::~JGConcentrationDiffusion() {

}

void JGConcentrationDiffusion::scheduleComputeFlux(Task* task,
                                                   const MPMMaterial* matl,
                                                   const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType gnone = Ghost::None;

  task->requires(Task::OldDW, d_rdlb->pConcGradientLabel, matlset, gnone);
  task->computes(             d_rdlb->pFluxLabel,         matlset);
}

void JGConcentrationDiffusion::computeFlux(const Patch* patch, 
                                           const MPMMaterial* matl,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  ParticleInterpolator* interpolator = d_Mflag->d_interpolator->clone(patch);
  vector<IntVector> ni(interpolator->size());
  vector<Vector> d_S(interpolator->size());

  Vector dx = patch->dCell();
  double oodx[3];
  oodx[0] = 1.0/dx.x();
  oodx[1] = 1.0/dx.y();
  oodx[2] = 1.0/dx.z();

  int dwi = matl->getDWIndex();
  constParticleVariable<Vector>  pConcGradient;
  ParticleVariable<Vector>       pFlux;

  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

  old_dw->get(pConcGradient,     d_rdlb->pConcGradientLabel, pset);
  new_dw->allocateAndPut(pFlux,  d_rdlb->pFluxLabel,         pset);
  
  for (ParticleSubset::iterator iter  = pset->begin();iter!=pset->end();iter++){
    particleIndex idx = *iter;
    pFlux[idx] = diffusivity*pConcGradient[idx];
  } //End of Particle Loop

  delete interpolator;
}
