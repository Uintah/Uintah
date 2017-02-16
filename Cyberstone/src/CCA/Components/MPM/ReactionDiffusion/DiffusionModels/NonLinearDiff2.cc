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

#include <CCA/Components/MPM/ReactionDiffusion/DiffusionModels/NonLinearDiff2.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/MPMBoundCond.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Task.h>

#include <iostream>
using namespace Uintah;

NonLinearDiff2::NonLinearDiff2(
                               ProblemSpecP     & ps,
                               SimulationStateP & sS,
                               MPMFlags         * Mflag,
                               std::string        diff_type
                              ) :ScalarDiffusionModel(ps,
                                                      sS,
                                                      Mflag,
                                                      diff_type)
{
  ps->require("tuning1", d_tuning1);
  ps->require("tuning2", d_tuning2);
}

NonLinearDiff2::~NonLinearDiff2()
{

}

void NonLinearDiff2::addInitialComputesAndRequires(
                                                         Task         * task,
                                                   const MPMMaterial  * matl,
                                                   const PatchSet     * patch
                                                  ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(d_lb->pDiffusivityLabel, matlset);
  task->computes(d_lb->pPressureLabel_t1, matlset);
  task->computes(d_lb->pConcInterpLabel,  matlset);
  task->computes(d_lb->pFluxLabel,        matlset);
}

void NonLinearDiff2::addParticleState(
                                      std::vector<const VarLabel*>& from,
                                      std::vector<const VarLabel*>& to
                                     ) const
{
  from.push_back(d_lb->pDiffusivityLabel);
  from.push_back(d_lb->pFluxLabel);

  to.push_back(d_lb->pDiffusivityLabel_preReloc);
  to.push_back(d_lb->pFluxLabel_preReloc);
}

void NonLinearDiff2::computeFlux(
                                 const Patch          * patch,
                                 const MPMMaterial    * matl,
                                       DataWarehouse  * old_dw,
                                       DataWarehouse  * new_dw
                                )
{

  ParticleInterpolator* interpolator = d_Mflag->d_interpolator->clone(patch);
  std::vector<IntVector> ni(interpolator->size());
  std::vector<double> S(interpolator->size());

  //double current_time1 = d_sharedState->getElapsedTime();

  int dwi = matl->getDWIndex();
  Vector dx = patch->dCell();

  constParticleVariable<Vector>  pConcGrad;
  constParticleVariable<Vector>  pESGradPotential;
  constParticleVariable<double>  pConcentration;
  constParticleVariable<double>  pESPotential;
  constParticleVariable<Matrix3> pStress;
  constParticleVariable<Point>   px;
  constParticleVariable<Matrix3> psize;
  constParticleVariable<Matrix3> pFOld;

  ParticleVariable<Vector>       pFlux;
  ParticleVariable<double>       pDiffusivity;

  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

  old_dw->get(px,             d_lb->pXLabel,                  pset);
  old_dw->get(pConcGrad,      d_lb->pConcGradientLabel,       pset);
  old_dw->get(pConcentration, d_lb->pConcentrationLabel,      pset);
  old_dw->get(pStress,        d_lb->pStressLabel,             pset);
  old_dw->get(pFOld,          d_lb->pDeformationMeasureLabel, pset);

  new_dw->get(psize,            d_lb->pSizeLabel_preReloc,    pset);
  new_dw->get(pESPotential,     d_lb->pESPotential,           pset);
  new_dw->get(pESGradPotential, d_lb->pESGradPotential,       pset);

  new_dw->allocateAndPut(pFlux,        d_lb->pFluxLabel_preReloc,        pset);
  new_dw->allocateAndPut(pDiffusivity, d_lb->pDiffusivityLabel_preReloc, pset);

  double D = diffusivity;
  double timestep = 1.0e99;
  double concentration = 0.0;

  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end();
                                                      iter++){
    particleIndex idx = *iter;

    interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],pFOld[idx]);

    concentration = pConcentration[idx];

    /*
    if(pConcentration[idx] < d_tuning1){
      pFlux[idx] = D * pConcGrad[idx];
    }else{
      double B = d_tuning2 * (1 - pConcentration[idx]);
      pFlux[idx] = D * pConcGrad[idx] - B * pESGradPotential[idx];
    }
    */
    pFlux[idx] = concentration * D * pESGradPotential[idx];

    pDiffusivity[idx] = D;
    timestep = std::min(timestep, computeStableTimeStep(D, dx));
  } //End of Particle Loop
  new_dw->put(delt_vartype(timestep), d_lb->delTLabel, patch->getLevel());
}

void NonLinearDiff2::initializeSDMData(
                                       const Patch          * patch,
                                       const MPMMaterial    * matl,
                                             DataWarehouse  * new_dw
                                      )
{
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<double>  pDiffusivity;
  ParticleVariable<double>  pPressure;
  ParticleVariable<double>  pConcInterp;
  ParticleVariable<Vector>  pFlux;

  new_dw->allocateAndPut(pDiffusivity, d_lb->pDiffusivityLabel, pset);
  new_dw->allocateAndPut(pFlux,        d_lb->pFluxLabel,        pset);

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++)
  {
    pDiffusivity[*iter] = diffusivity;
    pFlux[*iter]        = Vector(0,0,0);
  }
}

void NonLinearDiff2::scheduleComputeFlux(
                                               Task         * task,
                                         const MPMMaterial  * matl,
                                         const PatchSet     * patch
                                        ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType gnone = Ghost::None;
  //Ghost::GhostType gac   = Ghost::AroundCells;
  task->requires(Task::OldDW, d_lb->pXLabel,                  matlset, gnone);
  task->requires(Task::OldDW, d_lb->pConcGradientLabel,       matlset, gnone);
  task->requires(Task::OldDW, d_lb->pConcentrationLabel,      matlset, gnone);
  task->requires(Task::OldDW, d_lb->pStressLabel,             matlset, gnone);
  task->requires(Task::OldDW, d_lb->pDeformationMeasureLabel, matlset, gnone);

  task->requires(Task::NewDW, d_lb->pSizeLabel_preReloc,      matlset, gnone);
  task->requires(Task::NewDW, d_lb->pESPotential,             matlset, gnone);
  task->requires(Task::NewDW, d_lb->pESGradPotential,         matlset, gnone);

  task->computes(d_sharedState->get_delt_label(),getLevel(patch));

  task->computes(d_lb->pFluxLabel_preReloc,        matlset);
  task->computes(d_lb->pDiffusivityLabel_preReloc, matlset);
}

void NonLinearDiff2::addSplitParticlesComputesAndRequires(
                                                                Task        * task,
                                                          const MPMMaterial * matl,
                                                          const PatchSet    * patches
                                                         ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->modifies(d_lb->pDiffusivityLabel_preReloc, matlset);
  task->modifies(d_lb->pFluxLabel_preReloc,        matlset);
}

void NonLinearDiff2::splitSDMSpecificParticleData(
                                                  const Patch                 * patch,
                                                  const int                     dwi,
                                                  const int                     fourOrEight,
                                                        ParticleVariable<int> & prefOld,
                                                        ParticleVariable<int> & prefNew,
                                                  const unsigned int            oldNumPar,
                                                  const int                     numNewPartNeeded,
                                                        DataWarehouse         * old_dw,
                                                        DataWarehouse         * new_dw
                                                 )
{
  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

  ParticleVariable<double>  pDiffusivity;
  ParticleVariable<Vector>  pFlux;

  new_dw->getModifiable(pDiffusivity, d_lb->pDiffusivityLabel_preReloc,  pset);
  new_dw->getModifiable(pFlux,        d_lb->pFluxLabel_preReloc,         pset);

  ParticleVariable<double>  pDiffusivityTmp, pPressureTmp, pConcInterpTmp;
  ParticleVariable<Vector>  pFluxTmp;

  new_dw->allocateTemporary(pDiffusivityTmp, pset);
  new_dw->allocateTemporary(pFluxTmp,        pset);

  // copy data from old variables for particle IDs and the position vector
  for(unsigned int pp=0; pp<oldNumPar; ++pp )
  {
    pDiffusivityTmp[pp] = pDiffusivity[pp];
    pFluxTmp[pp]        = pFlux[pp];
  }

    int numRefPar=0;
    for(unsigned int idx=0; idx<oldNumPar; ++idx ){
      if(prefNew[idx]!=prefOld[idx]){  // do refinement!
        for(int i = 0;i<fourOrEight;i++){
          int new_index;
          if(i==0)
          {
            new_index=idx;
          }
          else
          {
            new_index=oldNumPar+(fourOrEight-1)*numRefPar+i;
          }
          pDiffusivityTmp[new_index] = pDiffusivity[idx];
          pFluxTmp[new_index]        = pFlux[idx];
        }
        numRefPar++;
      }
    }

    new_dw->put(pDiffusivityTmp, d_lb->pDiffusivityLabel_preReloc, true);
    new_dw->put(pFluxTmp,        d_lb->pFluxLabel_preReloc,        true);
}

void NonLinearDiff2::outputProblemSpec(
                                       ProblemSpecP & ps,
                                       bool           output_rdm_tag
                                      ) const
{

  ProblemSpecP rdm_ps = ps;
  if (output_rdm_tag) {
    rdm_ps = ps->appendChild("diffusion_model");
    rdm_ps->setAttribute("type","non_linear2");
  }

  rdm_ps->appendElement("diffusivity", diffusivity);
  rdm_ps->appendElement("max_concentration",max_concentration);
  rdm_ps->appendElement("tuning1", d_tuning1);
  rdm_ps->appendElement("tuning2", d_tuning2);

  if(d_conductivity_equation){
    d_conductivity_equation->outputProblemSpec(rdm_ps);
  }
}
