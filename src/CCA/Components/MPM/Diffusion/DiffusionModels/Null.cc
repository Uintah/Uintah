/*
 * Null.cc
 *
 *  Created on: Feb 23, 2017
 *      Author: jbhooper
 *
 *
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

#include <CCA/Components/MPM/Diffusion/DiffusionModels/Null.h>

using namespace Uintah;

NullDiffusion::NullDiffusion(
                             ProblemSpecP      & probSpec  ,
                             SimulationStateP  & simState  ,
                             MPMFlags          * mFlags    ,
                             std::string         diff_type
                            ) : ScalarDiffusionModel(probSpec,
                                                     simState,
                                                     mFlags,
                                                     diff_type)
{

}

NullDiffusion::~NullDiffusion()
{

}

void NullDiffusion::addInitialComputesAndRequires(
                                                        Task        * task,
                                                  const MPMMaterial * matl,
                                                  const PatchSet    * patches
                                                 ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(d_lb->pFluxLabel, matlset);
}

void NullDiffusion::addParticleState(
                                     std::vector<const VarLabel*> & from,
                                     std::vector<const VarLabel*> & to
                                    ) const
{
  from.push_back(d_lb->pFluxLabel);
  to.push_back(d_lb->pFluxLabel_preReloc);

}

void NullDiffusion::computeFlux(
                                const Patch         * patch ,
                                const MPMMaterial   * matl  ,
                                      DataWarehouse * OldDW ,
                                      DataWarehouse * NewDW
                               )
{
  int dwi = matl->getDWIndex();
  ParticleVariable<Vector> pFlux;
  ParticleSubset* pset = OldDW->getParticleSubset(dwi, patch);
  NewDW->allocateAndPut(pFlux, d_lb->pFluxLabel_preReloc, pset);
  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); ++iter) {
    pFlux[*iter] = Vector(0.0, 0.0, 0.0);
  }
}

void NullDiffusion::initializeSDMData(
                                      const Patch         * patch ,
                                      const MPMMaterial   * matl  ,
                                            DataWarehouse * newDW
                                     )
{
  ParticleSubset* pset = newDW->getParticleSubset(matl->getDWIndex(), patch);
  ParticleVariable<Vector>  pFlux;

  newDW->allocateAndPut(pFlux, d_lb->pFluxLabel, pset);

  for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++)
  {
    pFlux[*iter] = Vector(0,0,0);
  }

}

void NullDiffusion::scheduleComputeFlux(
                                              Task        * task    ,
                                        const MPMMaterial * matl    ,
                                        const PatchSet    * patches
                       ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  task->computes(d_lb->pFluxLabel_preReloc, matlset);
}

void NullDiffusion::addSplitParticlesComputesAndRequires(
                                                               Task         * task    ,
                                                         const MPMMaterial  * matl    ,
                                                         const PatchSet     * patches
                                                        ) const
{

}

void NullDiffusion::splitSDMSpecificParticleData(
                                                 const Patch *                  patch   ,
                                                 const int                      dwi     ,
                                                 const int                      nDims   ,
                                                       ParticleVariable<int>  & prefOld ,
                                                       ParticleVariable<int>  & pref    ,
                                                 const unsigned int             oldNumPar ,
                                                 const int                      numNewPartNeeded ,
                                                       DataWarehouse          * oldDW   ,
                                                       DataWarehouse          * newDW
                                                )
{

}

void NullDiffusion::outputProblemSpec(
                                      ProblemSpecP  & ps,
                                      bool            output_rdm_tag
                                     ) const
{
  ProblemSpecP rdm_ps = ps;
  if (output_rdm_tag)
  {
    rdm_ps = ps->appendChild("diffusion_model");
    rdm_ps->setAttribute("type","null");
  }
  rdm_ps->appendElement("diffusivity", d_D0);
  rdm_ps->appendElement("max_concentration", d_MaxConcentration);
}

void NullDiffusion::scheduleComputeDivergence(
                                                    Task        * task    ,
                                              const MPMMaterial * matl    ,
                                              const PatchSet    * patches
                                             ) const
{
  task->computes(d_lb->gConcentrationRateLabel, matl->thisMaterial());
}

void NullDiffusion::computeDivergence(

                                      const Patch         * patch ,
                                      const MPMMaterial   * matl  ,
                                            DataWarehouse * oldDW ,
                                            DataWarehouse * newDW
                      )
{
  NCVariable<double> gConcRate;
  newDW->allocateAndPut(gConcRate, d_lb->gConcentrationRateLabel,
                        matl->getDWIndex(), patch);
  gConcRate.initialize(0.0);
}



