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

#include <CCA/Components/MPM/ReactionDiffusion/NonLinearDiff1.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/MPMBoundCond.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Task.h>

#include <iostream>
using namespace std;
using namespace Uintah;


NonLinearDiff1::NonLinearDiff1(ProblemSpecP& ps, SimulationStateP& sS, MPMFlags* Mflag, string diff_type):
  ScalarDiffusionModel(ps, sS, Mflag, diff_type) {

  ps->require("tuning1", tuning1);
  ps->require("tuning2", tuning2);

}

NonLinearDiff1::~NonLinearDiff1() {

}

void NonLinearDiff1::scheduleComputeFlux(Task* task, const MPMMaterial* matl, 
                                              const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType  gnone = Ghost::None;
  task->requires(Task::OldDW, d_lb->pConcGradientLabel,   matlset, gnone);
  task->requires(Task::OldDW, d_lb->pConcentrationLabel,  matlset, gnone);
  task->computes(d_sharedState->get_delt_label(),getLevel(patch));

  task->computes(d_lb->pFluxLabel,  matlset);
}

void NonLinearDiff1::computeFlux(const Patch* patch,
                                      const MPMMaterial* matl,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{

  int dwi = matl->getDWIndex();
  Vector dx = patch->dCell();

  constParticleVariable<Vector>  pConcGrad;
  constParticleVariable<double>  pConcentration;
  ParticleVariable<Vector>       pFlux;

  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

  old_dw->get(pConcGrad,        d_lb->pConcGradientLabel,  pset);
  old_dw->get(pConcentration,   d_lb->pConcentrationLabel, pset);
  new_dw->allocateAndPut(pFlux, d_lb->pFluxLabel,          pset);

  double non_lin_comp;
  double D;
  double timestep = 1.0e99;
  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end();
                                                      iter++){
    particleIndex idx = *iter;

    non_lin_comp = 1/(1-pConcentration[idx]) - 2 * tuning1 * pConcentration[idx];

		cout << "nlc: " << non_lin_comp << ", concentration: " << pConcentration[idx] << endl;

    if(non_lin_comp < tuning2){
      D = diffusivity * non_lin_comp;
    } else {
      D = diffusivity * tuning2;
    }

    pFlux[idx] = D*pConcGrad[idx];
    timestep = min(timestep, computeStableTimeStep(D, dx));
  } //End of Particle Loop
	cout << "timestep: " << timestep << endl;
  new_dw->put(delt_vartype(timestep), d_lb->delTLabel, patch->getLevel());
}

void NonLinearDiff1::outputProblemSpec(ProblemSpecP& ps, bool output_rdm_tag)
{

  ProblemSpecP rdm_ps = ps;
  if (output_rdm_tag) {
    rdm_ps = ps->appendChild("diffusion_model");
    rdm_ps->setAttribute("type","non_linear1");
  }

  rdm_ps->appendElement("diffusivity",diffusivity);
  rdm_ps->appendElement("max_concentration",max_concentration);
  rdm_ps->appendElement("tuning1",tuning1);
  rdm_ps->appendElement("tuning2",tuning2);
}
