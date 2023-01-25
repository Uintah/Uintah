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

#include <CCA/Components/MPM/Materials/Diffusion/DiffusionModels/NonLinearDiff2.h>

#include <CCA/Components/MPM/Core/MPMDiffusionLabel.h>
#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Task.h>

#include <iostream>
using namespace Uintah;

NonLinearDiff2::NonLinearDiff2(
                               ProblemSpecP     & ps,
                               MaterialManagerP & sS,
                               MPMFlags         * Mflag,
                               std::string        diff_type
                              ) :ScalarDiffusionModel(ps,
                                                      sS,
                                                      Mflag,
                                                      diff_type)
{
  ps->require("boltzmann_const", d_boltz_const);
  ps->require("unit_charge", d_unit_charge);
  ps->require("operating_temp", d_operating_temp);
  d_alpha = 0;
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
  task->computes(d_lb->diffusion->pDiffusivity,   matlset);
  task->computes(d_lb->pPosChargeFluxLabel, matlset);
  task->computes(d_lb->pNegChargeFluxLabel, matlset);
}

void NonLinearDiff2::initializeSDMData(
                                       const Patch          * patch,
                                       const MPMMaterial    * matl,
                                             DataWarehouse  * new_dw
                                      )
{
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<double>  pDiffusivity;
  ParticleVariable<Vector>  pPosFlux;
  ParticleVariable<Vector>  pNegFlux;

  new_dw->allocateAndPut(pDiffusivity, d_lb->diffusion->pDiffusivity,   pset);
  new_dw->allocateAndPut(pPosFlux,     d_lb->pPosChargeFluxLabel, pset);
  new_dw->allocateAndPut(pNegFlux,     d_lb->pNegChargeFluxLabel, pset);

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++)
  {
    pDiffusivity[*iter] = d_D0;
    pPosFlux[*iter]     = Vector(0,0,0);
    pNegFlux[*iter]     = Vector(0,0,0);
  }
}

void NonLinearDiff2::addParticleState(
                                      std::vector<const VarLabel*>& from,
                                      std::vector<const VarLabel*>& to
                                     ) const
{
  from.push_back(d_lb->diffusion->pDiffusivity);
  from.push_back(d_lb->pPosChargeFluxLabel);
  from.push_back(d_lb->pNegChargeFluxLabel);

  to.push_back(d_lb->diffusion->pDiffusivity_preReloc);
  to.push_back(d_lb->pPosChargeFluxLabel_preReloc);
  to.push_back(d_lb->pNegChargeFluxLabel_preReloc);
}

void NonLinearDiff2::scheduleComputeFlux(
                                               Task         * task,
                                         const MPMMaterial  * matl,
                                         const PatchSet     * patch
                                        ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType gnone = Ghost::None;

  // task->requires(Task::OldDW, d_lb->simulationTimeLabel,);

  task->requires(Task::OldDW, d_lb->pPosChargeLabel, matlset, gnone);
  task->requires(Task::OldDW, d_lb->pNegChargeLabel, matlset, gnone);
  task->requires(Task::OldDW, d_lb->pPosChargeGradLabel, matlset, gnone);
  task->requires(Task::OldDW, d_lb->pNegChargeGradLabel, matlset, gnone);
  task->requires(Task::NewDW, d_lb->pESGradPotential, matlset, gnone);

  task->computes(d_lb->delTLabel,getLevel(patch));

  task->computes(d_lb->pPosChargeFluxLabel_preReloc, matlset);
  task->computes(d_lb->pNegChargeFluxLabel_preReloc, matlset);
  task->computes(d_lb->diffusion->pDiffusivity_preReloc,   matlset);
}

void NonLinearDiff2::computeFlux(
                                 const Patch          * patch,
                                 const MPMMaterial    * matl,
                                       DataWarehouse  * old_dw,
                                       DataWarehouse  * new_dw
                                )
{
  // Get the current simulation time
  // double simTime = d_materialManager->getElapsedSimTime();

  // simTime_vartype simTime;
  // old_dw->get(simTime, d_lb->simulationTimeLabel);

  int dwi = matl->getDWIndex();
  Vector dx = patch->dCell();

  constParticleVariable<double> pPosCharge;
  constParticleVariable<double> pNegCharge;
  constParticleVariable<Vector> pPosChargeGrad;
  constParticleVariable<Vector> pNegChargeGrad;
  constParticleVariable<Vector> pESGradPotential;

  ParticleVariable<Vector>       pPosFlux;
  ParticleVariable<Vector>       pNegFlux;
  ParticleVariable<double>       pDiffusivity;

  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

  old_dw->get(pPosCharge,       d_lb->pPosChargeLabel, pset);
  old_dw->get(pNegCharge,       d_lb->pNegChargeLabel, pset);
  old_dw->get(pPosChargeGrad,   d_lb->pPosChargeGradLabel, pset);
  old_dw->get(pNegChargeGrad,   d_lb->pNegChargeGradLabel, pset);
  new_dw->get(pESGradPotential, d_lb->pESGradPotential, pset);

  new_dw->allocateAndPut(pPosFlux,     d_lb->pPosChargeFluxLabel_preReloc, pset);
  new_dw->allocateAndPut(pNegFlux,     d_lb->pNegChargeFluxLabel_preReloc, pset);
  new_dw->allocateAndPut(pDiffusivity, d_lb->diffusion->pDiffusivity_preReloc,   pset);

  double D = d_D0;
  double timestep = 1.0e99;
  d_alpha = (D * d_unit_charge)/(d_boltz_const * d_operating_temp);

  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end();
                                                      iter++){
    particleIndex idx = *iter;

    pPosFlux[idx] =  pPosCharge[idx] * d_alpha * pESGradPotential[idx]
                  +  D * pPosChargeGrad[idx];
    pNegFlux[idx] = -pNegCharge[idx] * d_alpha * pESGradPotential[idx]
                  +  D * pNegChargeGrad[idx];

    pDiffusivity[idx] = D;
    timestep = std::min(timestep, computeStableTimeStep(D, dx));
  } //End of Particle Loop
  new_dw->put(delt_vartype(timestep), d_lb->delTLabel, patch->getLevel());
}

void NonLinearDiff2::scheduleComputeDivergence(       Task         * task,
                                                const MPMMaterial  * matl,
                                                const PatchSet     * patch
                                              ) const
{
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, d_lb->delTLabel);
  task->requires(Task::OldDW, d_lb->pXLabel,                   gan, NGP);
  task->requires(Task::NewDW, d_lb->pCurSizeLabel,             gan, NGP);
  task->requires(Task::OldDW, d_lb->pMassLabel,                gan, NGP);
  task->requires(Task::OldDW, d_lb->pVolumeLabel,              gan, NGP);

  task->requires(Task::NewDW, d_lb->pPosChargeFluxLabel_preReloc, gan, NGP);
  task->requires(Task::NewDW, d_lb->pNegChargeFluxLabel_preReloc, gan, NGP);

  task->computes(d_lb->gPosChargeRateLabel, matlset);
  task->computes(d_lb->gNegChargeRateLabel, matlset);

  task->computes(d_lb->diffusion->gConcentrationRate, matlset);
}

void NonLinearDiff2::computeDivergence(
                                       const Patch          * patch,
                                       const MPMMaterial    * matl,
                                             DataWarehouse  * old_dw,
                                             DataWarehouse  * new_dw
                                      )
{
  Ghost::GhostType  gan = Ghost::AroundNodes;
  int dwi = matl->getDWIndex();

  ParticleInterpolator*   interpolator = d_Mflag->d_interpolator->clone(patch);
  std::vector<IntVector>  ni(interpolator->size());
  std::vector<Vector>     d_S(interpolator->size());

  Vector dx = patch->dCell();
  double oodx[3];
  oodx[0] = 1.0/dx.x();
  oodx[1] = 1.0/dx.y();
  oodx[2] = 1.0/dx.z();

  constParticleVariable<Point>    px;
  constParticleVariable<double>   pvol;
  constParticleVariable<double>   pMass;
  constParticleVariable<Matrix3>  psize;
  constParticleVariable<Matrix3>  deformationGradient;
  constParticleVariable<Vector>   pPosChargeFlux;
  constParticleVariable<Vector>   pNegChargeFlux;

  NCVariable<double>              gPosChargeRate;
  NCVariable<double>              gNegChargeRate;
  NCVariable<double>              gConcRate;

  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch, gan, NGP,
                                                          d_lb->pXLabel);

  old_dw->get(px,                  d_lb->pXLabel,                  pset);
  old_dw->get(pvol,                d_lb->pVolumeLabel,             pset);
  old_dw->get(pMass,               d_lb->pMassLabel,               pset);
  new_dw->get(psize,               d_lb->pCurSizeLabel,            pset);

  new_dw->get(pPosChargeFlux,      d_lb->pPosChargeFluxLabel_preReloc, pset);
  new_dw->get(pNegChargeFlux,      d_lb->pNegChargeFluxLabel_preReloc, pset);

  new_dw->allocateAndPut(gPosChargeRate, d_lb->gPosChargeRateLabel,    dwi,patch);
  new_dw->allocateAndPut(gNegChargeRate, d_lb->gNegChargeRateLabel,    dwi,patch);
  new_dw->allocateAndPut(gConcRate,      d_lb->diffusion->gConcentrationRate,dwi,patch);

  gConcRate.initialize(0.0);
  gPosChargeRate.initialize(0.0);
  gNegChargeRate.initialize(0.0);

  // THIS IS COMPUTING A MASS WEIGHTED gConcRate.  THE DIVISION BY MASS, AND
  // SUBSEQUENT CALCULATIONS IS DONE IN computeAndIntegrateAcceleration.

  for(ParticleSubset::iterator iter = pset->begin();
                               iter != pset->end(); iter++){
    particleIndex idx = *iter;

    // Get the node indices that surround the cell
    int NN = interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,
                                                       psize[idx]);

    Vector PosJ = pPosChargeFlux[idx];
    Vector NegJ = pNegChargeFlux[idx];
    double PosCdot_cond = 0.0;
    double NegCdot_cond = 0.0;
    IntVector node(0,0,0);

    for (int k = 0; k < NN; k++){
      node = ni[k];
      if(patch->containsNode(node)){
        Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],d_S[k].z()*oodx[2]);
        PosCdot_cond = Dot(div, PosJ)*pMass[idx];
        NegCdot_cond = Dot(div, NegJ)*pMass[idx];

        gPosChargeRate[node] -= PosCdot_cond;
        gNegChargeRate[node] -= NegCdot_cond;
      }
    }
  } // End of Particle Loop
}

void NonLinearDiff2::addSplitParticlesComputesAndRequires(
                                                                Task        * task,
                                                          const MPMMaterial * matl,
                                                          const PatchSet    * patches
                                                         ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->modifies(d_lb->diffusion->pDiffusivity_preReloc, matlset);
  task->modifies(d_lb->diffusion->pFlux_preReloc,        matlset);
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

  new_dw->getModifiable(pDiffusivity, d_lb->diffusion->pDiffusivity_preReloc,  pset);
  new_dw->getModifiable(pFlux,        d_lb->diffusion->pFlux_preReloc,         pset);

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

    new_dw->put(pDiffusivityTmp, d_lb->diffusion->pDiffusivity_preReloc, true);
    new_dw->put(pFluxTmp,        d_lb->diffusion->pFlux_preReloc,        true);
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
  ScalarDiffusionModel::baseOutputSDMProbSpec(rdm_ps);
  rdm_ps->appendElement("boltzmann_const", d_boltz_const);
  rdm_ps->appendElement("unit_charge", d_unit_charge);
  rdm_ps->appendElement("operating_temp", d_operating_temp);

  if(d_conductivity_equation){
    d_conductivity_equation->outputProblemSpec(rdm_ps);
  }
}
