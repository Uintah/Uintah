/*
 * ActivatedDiffusion.cc
 *
 *  Created on: Feb 14, 2017
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

#include <CCA/Components/MPM/Diffusion/DiffusionModels/ActivatedDiffusion.h>

#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <vector>


namespace Uintah
{
  ActivatedDiffusion::ActivatedDiffusion(
                                         ProblemSpecP     & ps        ,
                                         SimulationStateP & simState  ,
                                         MPMFlags         * mFlag     ,
                                         std::string        diff_type
                                        )
                                        : ScalarDiffusionModel(
                                                               ps,
                                                               simState,
                                                               mFlag,
                                                               diff_type
                                                              )
  {
    ps->require("gasConstant", d_gasConstant);
    ps->require("activationEnergy", d_activationEnergy);

    d_perDegreeActivation = d_activationEnergy/d_gasConstant;

  }

  ActivatedDiffusion::~ActivatedDiffusion()
  {

  }

  void ActivatedDiffusion::addInitialComputesAndRequires(
                                                               Task         * task    ,
                                                         const MPMMaterial  * matl    ,
                                                         const PatchSet     * patches
                                                        ) const
  {
    const MaterialSubset  * matlset = matl->thisMaterial();
    task->computes(d_lb->pFluxLabel,  matlset);
  }

  void ActivatedDiffusion::addParticleState(
                                            std::vector<const VarLabel*>  & from,
                                            std::vector<const VarLabel*>  & to
                                           ) const
  {
    from.push_back(d_lb->pFluxLabel);
    to.push_back(d_lb->pFluxLabel_preReloc);
  }

  void ActivatedDiffusion::computeFlux(
                                    const Patch         * patch ,
                                    const MPMMaterial   * matl  ,
                                          DataWarehouse * OldDW ,
                                          DataWarehouse * NewDW
                                   )
  {

    Vector dx = patch->dCell();
    int    dwi = matl->getDWIndex();

    constParticleVariable<double> pTemperature;
    constParticleVariable<Vector> pGradConcentration;

    ParticleSubset* pset = OldDW->getParticleSubset(dwi, patch);

    OldDW->get(pGradConcentration, d_lb->pConcGradientLabel, pset);
    OldDW->get(pTemperature, d_lb->pTemperatureLabel, pset);

    ParticleVariable<Vector> pFluxNew;
    NewDW->allocateAndPut(pFluxNew, d_lb->pFluxLabel_preReloc, pset);

    double delT_local = 1.0e99;
    for (int pIdx = 0; pIdx < pset->numParticles(); ++pIdx)
    {
      double D = d_D0 * std::exp(-(d_perDegreeActivation/pTemperature[pIdx]));
      pFluxNew[pIdx] = D * pGradConcentration[pIdx];
      delT_local = std::min(delT_local, computeStableTimeStep(D,dx));
    }
    NewDW->put(delt_vartype(delT_local), d_lb->delTLabel, patch->getLevel());
  }

  void ActivatedDiffusion::initializeSDMData(
                                             const Patch          * patch   ,
                                             const MPMMaterial    * matl    ,
                                                   DataWarehouse  * NewDW
                                            )
  {
    ParticleVariable<Vector> pFlux;

    ParticleSubset* pset = NewDW->getParticleSubset(matl->getDWIndex(), patch);

    NewDW->allocateAndPut(pFlux, d_lb->pFluxLabel, pset);
    for(ParticleSubset::iterator pIdx = pset->begin(); pIdx < pset->end(); ++pIdx)
    {
      pFlux[*pIdx] = Vector(0.0);
    }
  }

  void ActivatedDiffusion::scheduleComputeFlux(
                                                     Task         * task  ,
                                               const MPMMaterial  * matl  ,
                                               const PatchSet     * patch
                                              ) const
  {
    const MaterialSubset  * matlset = matl->thisMaterial();
    Ghost::GhostType        gnone   = Ghost::None;

    task->requires(Task::OldDW, d_lb->pConcGradientLabel, matlset,  gnone);
    task->requires(Task::OldDW, d_lb->pTemperatureLabel,  matlset,  gnone);

    task->computes(d_lb->pFluxLabel_preReloc, matlset);
    task->computes(d_sharedState->get_delt_label(), getLevel(patch));

  }

  void ActivatedDiffusion::addSplitParticlesComputesAndRequires(
                                                                      Task        * task    ,
                                                                const MPMMaterial * matl    ,
                                                                const PatchSet    * patches
                                                               ) const
  {

  }

  void ActivatedDiffusion::splitSDMSpecificParticleData(
                                                        const Patch                  * patch           ,
                                                        const int                      dwi             ,
                                                        const int                      nDims           ,
                                                              ParticleVariable<int> & prefOld          ,
                                                              ParticleVariable<int> & pref             ,
                                                        const unsigned int            oldNumPar        ,
                                                        const int                     numNewPartNeeded ,
                                                              DataWarehouse         * OldDW            ,
                                                              DataWarehouse         * NewDW
                                                    )
  {

  }

  void ActivatedDiffusion::outputProblemSpec(
                                             ProblemSpecP & ps,
                                             bool           output_rdm_tag
                                            ) const
  {
    ProblemSpecP rdm_ps = ps;
    if (output_rdm_tag)
    {
      rdm_ps = ps->appendChild("diffusion_model");
      rdm_ps->setAttribute("type","activated");
    }
    rdm_ps->appendElement("diffusivity", d_D0);
    rdm_ps->appendElement("max_concentration", d_MaxConcentration);

    rdm_ps->appendElement("gasConstant",d_gasConstant);
    rdm_ps->appendElement("activationEnergy", d_activationEnergy);
  }

} // namespace Uintah



