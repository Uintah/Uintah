/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/MPM/Materials/Diffusion/DiffusionModels/RFConcDiffusion1MPM.h>

#include <CCA/Components/MPM/Core/MPMDiffusionLabel.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <Core/Grid/Task.h>

using namespace std;
using namespace Uintah;


RFConcDiffusion1MPM::RFConcDiffusion1MPM(
                                         ProblemSpecP     & ps,
                                         MaterialManagerP & sS,
                                         MPMFlags         * Mflag,
                                         string             diff_type
                                        ) :ScalarDiffusionModel(ps,
                                                                sS,
                                                                Mflag,
                                                                diff_type)
{

}

RFConcDiffusion1MPM::~RFConcDiffusion1MPM()
{

}

void RFConcDiffusion1MPM::addInitialComputesAndRequires(
                                                              Task         * task,
                                                        const MPMMaterial  * matl,
                                                        const PatchSet     * patches
                                                       ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  task->computes(d_lb->diffusion->pFlux,  matlset);
}

void RFConcDiffusion1MPM::addParticleState(
                                           std::vector<const VarLabel*> & from,
                                           std::vector<const VarLabel*> & to
                                          ) const
{
  from.push_back(d_lb->diffusion->pFlux);

  to.push_back(d_lb->diffusion->pFlux_preReloc);
}

void RFConcDiffusion1MPM::computeFlux(
                                      const Patch         * patch,
                                      const MPMMaterial   * matl,
                                            DataWarehouse * old_dw,
                                            DataWarehouse * new_dw
                                     )
{
  int dwi = matl->getDWIndex();

  constParticleVariable<Vector>  pConcGrad;
  ParticleVariable<Vector>       pFlux;

  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

  old_dw->get(pConcGrad,           d_lb->diffusion->pGradConcentration,       pset);
  new_dw->allocateAndPut(pFlux,    d_lb->diffusion->pFlux,             pset);

  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end();
                                                      iter++){
    particleIndex idx = *iter;

    pFlux[idx] = d_D0*pConcGrad[idx];
  } //End of Particle Loop
}

void RFConcDiffusion1MPM::initializeSDMData(
                                            const Patch         * patch,
                                            const MPMMaterial   * matl,
                                                  DataWarehouse * new_dw
                                           )
{
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<Vector>  pFlux;

  new_dw->allocateAndPut(pFlux,        d_lb->diffusion->pFlux,        pset);

  for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++)
  {
    pFlux[*iter] = Vector(0,0,0);
  }

}

void RFConcDiffusion1MPM::scheduleComputeFlux(
                                                    Task        * task,
                                              const MPMMaterial * matl,
                                              const PatchSet    * patch
                                             ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType  gnone = Ghost::None;
  task->requires(Task::OldDW, d_lb->diffusion->pGradConcentration, matlset, gnone);

  task->computes(d_lb->diffusion->pFlux,  matlset);
}

void RFConcDiffusion1MPM::addSplitParticlesComputesAndRequires(
                                                                     Task         * task,
                                                               const MPMMaterial  * matl,
                                                               const PatchSet     * patches
                                                              ) const
{
  // Do nothing for now
}

void RFConcDiffusion1MPM::splitSDMSpecificParticleData(
                                                       const Patch                  * Patch,
                                                       const int                      dwi,
                                                       const int                      nDims,
                                                             ParticleVariable<int>  & prefOld,
                                                             ParticleVariable<int>  & pref,
                                                       const unsigned int             oldNumPar,
                                                       const int                      numNewPartNeeded,
                                                             DataWarehouse          * old_dw,
                                                             DataWarehouse          * new_dw
                                                      )
{
  // Do nothing for now
}

void RFConcDiffusion1MPM::outputProblemSpec(
                                            ProblemSpecP  & ps,
                                            bool            output_rdm_tag
                                           ) const
{

  ProblemSpecP rdm_ps = ps;
  if (output_rdm_tag) {
    rdm_ps = ps->appendChild("diffusion_model");
    rdm_ps->setAttribute("type","rf1");
  }
  ScalarDiffusionModel::baseOutputSDMProbSpec(rdm_ps);
}
