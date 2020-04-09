/*
 * Activated2.cc
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

#include <CCA/Components/MPM/Diffusion/DiffusionModels/ActivatedDiffusion2.h>
#include <CCA/Components/MPM/Diffusion/DiffusionModels/AlNi_Diffusivity.h>

#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <vector>


namespace Uintah
{
  Activated2::Activated2(
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
    ps->getWithDefault("scaleMultiplier",d_multiplier,1.0);
    ps->getWithDefault("normalized",isConcNormalized, true);
  }

  Activated2::~Activated2()
  {

  }

  void Activated2::addInitialComputesAndRequires(
                                                               Task         * task    ,
                                                         const MPMMaterial  * matl    ,
                                                         const PatchSet     * patches
                                                        ) const
  {
    const MaterialSubset  * matlset = matl->thisMaterial();
    task->computes(d_lb->pFluxLabel,  matlset);
  }

  void Activated2::addParticleState(
                                            std::vector<const VarLabel*>  & from,
                                            std::vector<const VarLabel*>  & to
                                           ) const
  {
    from.push_back(d_lb->pFluxLabel);
    to.push_back(d_lb->pFluxLabel_preReloc);
  }

  void Activated2::computeFlux(
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

//    double R = 8.3144598; // Gas constant in J/(mol*K)
//    double Rinv = 1.0/R;
    double delT_local = 1.0e99;
    for (int pIdx = 0; pIdx < pset->numParticles(); ++pIdx)
    {
      double Temp = pTemperature[pIdx];
      double D = AlNi::Diffusivity(Temp)*d_multiplier;

      pFluxNew[pIdx] = D * pGradConcentration[pIdx];
      delT_local = std::min(delT_local, computeStableTimeStep(D,dx));
    }
    NewDW->put(delt_vartype(delT_local), d_lb->delTLabel, patch->getLevel());
  }

  void Activated2::initializeSDMData(
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

  void Activated2::scheduleComputeFlux(
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

  void Activated2::addSplitParticlesComputesAndRequires(
                                                                      Task        * task    ,
                                                                const MPMMaterial * matl    ,
                                                                const PatchSet    * patches
                                                               ) const
  {

  }

  void Activated2::splitSDMSpecificParticleData(
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

  void Activated2::outputProblemSpec(
                                             ProblemSpecP & ps,
                                             bool           output_rdm_tag
                                            ) const
  {
    ProblemSpecP rdm_ps = ps;
    if (output_rdm_tag)
    {
      rdm_ps = ps->appendChild("diffusion_model");
      rdm_ps->setAttribute("type","activated2");
    }
    rdm_ps->appendElement("diffusivity", d_D0);
    rdm_ps->appendElement("max_concentration", d_MaxConcentration);
    rdm_ps->appendElement("scalarMultiplier", d_multiplier);
    rdm_ps->appendElement("normalized", isConcNormalized);
  }

} // namespace Uintah



