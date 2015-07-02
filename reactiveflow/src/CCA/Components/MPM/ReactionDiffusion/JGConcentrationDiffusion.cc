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

void JGConcentrationDiffusion::scheduleComputeFlux(Task* task, const MPMMaterial* matl,
                                                    const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType  gan = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;
  task->requires(Task::OldDW, d_lb->pXLabel,                  matlset, gan, NGP);
  task->requires(Task::OldDW, d_lb->pSizeLabel,               matlset, gan, NGP);
  task->requires(Task::OldDW, d_lb->pMassLabel,               matlset, gan, NGP);
  task->requires(Task::OldDW, d_lb->pVolumeLabel,             matlset, gan, NGP);
  task->requires(Task::OldDW, d_lb->pDeformationMeasureLabel, matlset, gan, NGP);
  task->requires(Task::NewDW, d_lb->gMassLabel,               matlset, gnone);
  task->requires(Task::NewDW, d_rdlb->gConcentrationLabel,    matlset, gan, 2*NGN);

  task->computes(d_rdlb->pFluxLabel,         matlset);
}

void JGConcentrationDiffusion::computeFlux(const Patch* patch, const MPMMaterial* matl,
                                           DataWarehouse* old_dw, DataWarehouse* new_dw)
{

  Ghost::GhostType  gac   = Ghost::AroundCells;
  Ghost::GhostType  gnone = Ghost::None;

  ParticleInterpolator* interpolator = d_Mflag->d_interpolator->clone(patch);
  vector<IntVector> ni(interpolator->size());
  vector<Vector> d_S(interpolator->size());

  Vector dx = patch->dCell();
  double oodx[3];
  oodx[0] = 1.0/dx.x();
  oodx[1] = 1.0/dx.y();
  oodx[2] = 1.0/dx.z();

  int dwi = matl->getDWIndex();
  constParticleVariable<Point>   px;
  constParticleVariable<double>  pvol,pMass;
  constParticleVariable<Matrix3> psize;
  constParticleVariable<Matrix3> deformationGradient;

  constNCVariable<double>       gConcentration,gMass;

  ParticleVariable<Vector>      pConcGradient;
  ParticleVariable<Vector>      pFlux;

  //ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch, gan, NGP, d_lb->pXLabel);
  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

  old_dw->get(px,           d_lb->pXLabel,                         pset);
  old_dw->get(pvol,         d_lb->pVolumeLabel,                    pset);
  old_dw->get(pMass,        d_lb->pMassLabel,                      pset);
  old_dw->get(psize,        d_lb->pSizeLabel,                      pset);
  old_dw->get(deformationGradient, d_lb->pDeformationMeasureLabel, pset);

  new_dw->get(gConcentration,      d_rdlb->gConcentrationLabel, dwi, patch, gac,2*NGN);
  new_dw->get(gMass,               d_lb->gMassLabel,            dwi, patch, gnone, 0);

  //new_dw->allocateAndPut(pConcGradient, d_rdlb->pConcGradientLabel, pset);
  new_dw->allocateTemporary(pConcGradient, pset);
  new_dw->allocateAndPut(pFlux,         d_rdlb->pFluxLabel,         pset);
  
  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
    particleIndex idx = *iter;

    // Get the node indices that surround the cell
    interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

    pConcGradient[idx] = Vector(0.0,0.0,0.0);
    pFlux[idx]         = Vector(0.0,0.0,0.0);
    for (int k = 0; k < d_Mflag->d_8or27; k++){
      for (int j = 0; j<3; j++) {
          pConcGradient[idx][j] += gConcentration[ni[k]] * d_S[k][j] * oodx[j];
      }
    }
    pFlux[idx] = diffusivity*pConcGradient[idx];
  } //End of Particle Loop

  delete interpolator;
}
