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

#include <CCA/Components/MPM/ReactionDiffusion/RFConcDiffusion1MPM.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/MPMBoundCond.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Task.h>

#include <iostream>
using namespace std;
using namespace Uintah;


RFConcDiffusion1MPM::RFConcDiffusion1MPM(ProblemSpecP& ps, SimulationStateP& sS, MPMFlags* Mflag, string diff_type):
  ScalarDiffusionModel(ps, sS, Mflag, diff_type) {

  //***********************************************************************
  // This is working code that uses pressure gradients to as part of
  // computing the flux of concentration.
  //
  //ps->require("partial_atomic_vol", partial_atomic_vol);
  //ps->require("operating_temp", operating_temp);
  //ps->require("boltzmann_const", boltzmann);
  //mech_val = (diffusivity * partial_atomic_vol)/(boltzmann * operating_temp);
  //
  //*************************************************************************

}

RFConcDiffusion1MPM::~RFConcDiffusion1MPM() {

}

void RFConcDiffusion1MPM::scheduleComputeFlux(Task* task, const MPMMaterial* matl, 
                                              const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType  gnone = Ghost::None;

  task->requires(Task::OldDW, d_lb->pConcGradientLabel,        matlset, gnone);
  task->computes(d_lb->pFluxLabel,  matlset);
  task->computes(d_sharedState->get_delt_label(),getLevel(patch));
}

void RFConcDiffusion1MPM::computeFlux(const Patch* patch,
                                      const MPMMaterial* matl,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();

  constParticleVariable<Vector>  pConcGrad;
  ParticleVariable<Vector>       pFlux;

  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

  old_dw->get(pConcGrad,           d_lb->pConcGradientLabel,       pset);
  new_dw->allocateAndPut(pFlux,    d_lb->pFluxLabel,             pset);

  //***********************************************************************
  // This is working code that uses pressure gradients to as part of
  // computing the flux of concentration.
  //
  //double oodx[3];
  //oodx[0] = 1.0/dx.x();
  //oodx[1] = 1.0/dx.y();
  //oodx[2] = 1.0/dx.z();
  //constParticleVariable<Point>   px;
  //constParticleVariable<double>  pMass;
  //constParticleVariable<Matrix3> psize;
  //constParticleVariable<Matrix3> deformationGradient;
  //constParticleVariable<double>  pConcentration;
  //constNCVariable<double>        gMass;
  //constNCVariable<double>        gHydrostaticStress;
  //ParticleVariable<Vector>       pHydroStressGradient;
  //old_dw->get(px,                  d_lb->pXLabel,                  pset);
  //old_dw->get(pMass,               d_lb->pMassLabel,               pset);
  //old_dw->get(psize,               d_lb->pSizeLabel,               pset);
  //old_dw->get(deformationGradient, d_lb->pDeformationMeasureLabel, pset);
  //old_dw->get(pConcentration,      d_lb->pConcentrationLabel,      pset);
  //old_dw->get(pConcGradient,       d_lb->pConcGradientLabel,       pset);
  //new_dw->get(gMass,              d_lb->gMassLabel,                dwi,
  //                                                            patch, gnone, 0);
  //new_dw->get(gHydrostaticStress, d_lb->gHydrostaticStressLabel, dwi,
  //new_dw->allocateTemporary(pHydroStressGradient, pset);
  //
  //************************************************************************

  double timestep = 1.0e99;
  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end();
                                                      iter++){
    particleIndex idx = *iter;

    pFlux[idx] = diffusivity*pConcGrad[idx];

    //***********************************************************************
    // This is working code that uses pressure gradients to as part of
    // computing the flux of concentration.
    //
    //interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],
    //                                          deformationGradient[idx]);
    //pHydroStressGradient[idx] = Vector(0.0,0.0,0.0);
    //for (int k = 0; k < d_Mflag->d_8or27; k++){
    //  for (int j = 0; j<3; j++) {
    //      pHydroStressGradient[idx][j]
    //                        += gHydrostaticStress[ni[k]] * d_S[k][j] * oodx[j];
    //  }
    //}
    //chem_potential = -diffusivity;
    //mech_potential = mech_val * (1 - pConcentration[idx]/max_concentration) * pConcentration[idx];
    //pFlux[idx] = chem_potential*pConcGradient[idx] + mech_potential*pHydroStressGradient[idx];
    //
    //**************************************************************************

    timestep = min(timestep, computeStableTimeStep(diffusivity, dx));
  } //End of Particle Loop

  new_dw->put(delt_vartype(timestep), d_lb->delTLabel, patch->getLevel());
}

void RFConcDiffusion1MPM::outputProblemSpec(ProblemSpecP& ps, bool output_rdm_tag)
{

  ProblemSpecP rdm_ps = ps;
  if (output_rdm_tag) {
    rdm_ps = ps->appendChild("diffusion_model");
    rdm_ps->setAttribute("type","rf1");
  }

  rdm_ps->appendElement("diffusivity",diffusivity);
  rdm_ps->appendElement("max_concentration",max_concentration);
}
