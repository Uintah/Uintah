/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <CCA/Components/MPM/Materials/Diffusion/DiffusionModels/JGConcentrationDiffusion.h>

#include <CCA/Components/MPM/Core/MPMDiffusionLabel.h>
#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>

using namespace std;
using namespace Uintah;


JGConcentrationDiffusion::JGConcentrationDiffusion(ProblemSpecP     & ps,
                                                   MaterialManagerP & sS,
                                                   MPMFlags         * Mflag,
                                                   string             diff_type
                                                  )
                                                   :ScalarDiffusionModel(ps,
                                                                         sS,
                                                                         Mflag,
                                                                         diff_type)
{
}

JGConcentrationDiffusion::~JGConcentrationDiffusion()
{

}
void JGConcentrationDiffusion::addInitialComputesAndRequires(      Task         * task,
                                                             const MPMMaterial  * matl,
                                                             const PatchSet     * patch
                                                            ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(d_lb->diffusion->pFlux,        matlset);
}

void JGConcentrationDiffusion::addParticleState(
                                                std::vector<const VarLabel*>& from,
                                                std::vector<const VarLabel*>& to
                                               ) const
{
  from.push_back(d_lb->diffusion->pFlux);

  to.push_back(d_lb->diffusion->pFlux_preReloc);
}

void JGConcentrationDiffusion::computeFlux(
                                           const Patch          * patch,
                                           const MPMMaterial    * matl,
                                                 DataWarehouse  * old_dw,
                                                 DataWarehouse  * new_dw
                                          )
{
  ParticleInterpolator* interpolator = d_Mflag->d_interpolator->clone(patch);
  vector<IntVector>     ni(interpolator->size());
  vector<Vector>        d_S(interpolator->size());

  Vector dx = patch->dCell();
  int   dwi = matl->getDWIndex();

  constParticleVariable<Vector>  pConcGradient;
  ParticleVariable<Vector>       pFlux;

  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

  old_dw->get(pConcGradient,     d_lb->diffusion->pGradConcentration,  pset);
  new_dw->allocateAndPut(pFlux,  d_lb->diffusion->pFlux_preReloc, pset);

  double timestep = 1.0e99;
  for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++)
  {
    particleIndex idx = *iter;

    pFlux[idx] = d_D0*pConcGradient[idx];
    timestep = min(timestep, computeStableTimeStep(d_D0, dx));
  } //End of Particle Loop

  new_dw->put(delt_vartype(timestep), d_lb->delTLabel, patch->getLevel());
  delete interpolator;
}

void JGConcentrationDiffusion::initializeSDMData(const Patch          * patch,
                                                 const MPMMaterial    * matl,
                                                       DataWarehouse  * new_dw
                                                )
{
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
  ParticleVariable<Vector>  pFlux;

  new_dw->allocateAndPut(pFlux, d_lb->diffusion->pFlux, pset);
  for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++)
  {
    pFlux[*iter] = Vector(0,0,0);
  }
}

void JGConcentrationDiffusion::scheduleComputeFlux(      Task         * task,
                                                   const MPMMaterial  * matl,
                                                   const PatchSet     * patch
                                                  ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType        gnone = Ghost::None;

  task->requires(Task::OldDW, d_lb->diffusion->pGradConcentration, matlset, gnone);

  task->computes(d_lb->diffusion->pFlux_preReloc, matlset);
  task->computes(d_lb->delTLabel,getLevel(patch));
}

void JGConcentrationDiffusion::addSplitParticlesComputesAndRequires(
                                                                          Task        * task,
                                                                    const MPMMaterial * matl,
                                                                    const PatchSet    * patches
                                                                   ) const
{
  // Do nothing for now.
}

void JGConcentrationDiffusion::splitSDMSpecificParticleData(
                                                            const Patch                 * patch,
                                                            const int                     dwi,
                                                            const int                     nDims,
                                                                  ParticleVariable<int> & prefOld,
                                                                  ParticleVariable<int> & pref,
                                                            const unsigned int            oldNumPart,
                                                            const int                     oldNumPartNeeded,
                                                                  DataWarehouse         * old_dw,
                                                                  DataWarehouse         * new_dw
                                                           )
{
  // Do nothing for now
}

void JGConcentrationDiffusion::outputProblemSpec(
                                                 ProblemSpecP & ps,
                                                 bool           output_rdm_tag
                                                ) const
{
  ProblemSpecP rdm_ps = ps;

  if (output_rdm_tag) {
    rdm_ps = ps->appendChild("diffusion_model");
    rdm_ps->setAttribute("type","jg");
  }
  ScalarDiffusionModel::baseOutputSDMProbSpec(rdm_ps);

  if(d_conductivity_equation){
    d_conductivity_equation->outputProblemSpec(rdm_ps);
  }
}
