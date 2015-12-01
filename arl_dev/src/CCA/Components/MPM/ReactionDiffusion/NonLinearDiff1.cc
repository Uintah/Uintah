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

#undef USE_PARTICLE_VALUES

NonLinearDiff1::NonLinearDiff1(ProblemSpecP& ps, SimulationStateP& sS, MPMFlags* Mflag, string diff_type):
  ScalarDiffusionModel(ps, sS, Mflag, diff_type) {

  use_pressure = false;
  ps->require("use_pressure", use_pressure);
  ps->require("tuning1", tuning1);
  ps->require("tuning2", tuning2);

  if(use_pressure){
    ps->require("tuning3", tuning3);
    ps->require("tuning4", tuning4);
    ps->require("tuning5", tuning5);
  }

}

NonLinearDiff1::~NonLinearDiff1() {

}

void NonLinearDiff1::scheduleComputeFlux(Task* task, const MPMMaterial* matl, 
                                              const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType gnone = Ghost::None;
  Ghost::GhostType gac   = Ghost::AroundCells;
  task->requires(Task::OldDW, d_lb->pXLabel,                  matlset, gnone);
  task->requires(Task::OldDW, d_lb->pConcGradientLabel,       matlset, gnone);
  task->requires(Task::OldDW, d_lb->pConcentrationLabel,      matlset, gnone);
  task->requires(Task::OldDW, d_lb->pStressLabel,             matlset, gnone);
  task->requires(Task::OldDW, d_lb->pDeformationMeasureLabel, matlset, gnone);

  task->requires(Task::NewDW, d_lb->pSizeLabel_preReloc,      matlset, gnone);
  task->requires(Task::NewDW, d_lb->gConcentrationLabel,      matlset, gac, NGN);
  task->requires(Task::NewDW, d_lb->gHydrostaticStressLabel,  matlset, gac, NGN);

  task->computes(d_sharedState->get_delt_label(),getLevel(patch));

  task->computes(d_lb->pFluxLabel,        matlset);
  task->computes(d_lb->pDiffusivityLabel, matlset);
  task->computes(d_lb->pPressureLabel_t1, matlset);
  //task->computes(d_lb->pPressureLabel_t2, matlset);
}

void NonLinearDiff1::computeFlux(const Patch* patch,
                                 const MPMMaterial* matl,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{

  Ghost::GhostType gac   = Ghost::AroundCells;
  ParticleInterpolator* interpolator = d_Mflag->d_interpolator->clone(patch);
  vector<IntVector> ni(interpolator->size());
  vector<double> S(interpolator->size());
  vector<Vector> d_S(interpolator->size());

  int dwi = matl->getDWIndex();
  Vector dx = patch->dCell();
  double oodx[3];
  oodx[0] = 1.0/dx.x();
  oodx[1] = 1.0/dx.y();
  oodx[2] = 1.0/dx.z();
  double comp_diffusivity;
  double neg_one_third = -1.0/3.0;

  constParticleVariable<Vector>  pConcGrad;
  constParticleVariable<double>  pConcentration;
  constParticleVariable<Matrix3> pStress;
  constParticleVariable<Point>   px;
  constParticleVariable<Matrix3> psize;
  constParticleVariable<Matrix3> pFOld;

  constNCVariable<double>  gConcentration;
  constNCVariable<double>  gHydroStress;

  ParticleVariable<Vector>       pFlux;
  ParticleVariable<double>       pDiffusivity;
  ParticleVariable<double>       pPressure1;
  //ParticleVariable<double>       pPressure2;

  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

  old_dw->get(px,             d_lb->pXLabel,                  pset);
  old_dw->get(pConcGrad,      d_lb->pConcGradientLabel,       pset);
  old_dw->get(pConcentration, d_lb->pConcentrationLabel,      pset);
  old_dw->get(pStress,        d_lb->pStressLabel,             pset);
  old_dw->get(pFOld,          d_lb->pDeformationMeasureLabel, pset);

  new_dw->get(psize,          d_lb->pSizeLabel_preReloc,      pset);

  new_dw->get(gConcentration, d_lb->gConcentrationLabel,     dwi, patch, gac, NGP);
  new_dw->get(gHydroStress,   d_lb->gHydrostaticStressLabel, dwi, patch, gac, NGP);

  new_dw->allocateAndPut(pFlux,        d_lb->pFluxLabel,        pset);
  new_dw->allocateAndPut(pDiffusivity, d_lb->pDiffusivityLabel, pset);
  new_dw->allocateAndPut(pPressure1,   d_lb->pPressureLabel_t1, pset);
  //new_dw->allocateAndPut(pPressure2,   d_lb->pPressureLabel_t2, pset);

  double non_lin_comp;
  double D;
  double timestep = 1.0e99;
  //double minD = 1.0e99;
  //double maxD = 0;
  double pressure;
  //double pressure1, pressure2;
  double concentration;
  Vector pressure_grad;
  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end();
                                                      iter++){
    particleIndex idx = *iter;

    //interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],pFOld[idx]);
    interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,psize[idx], pFOld[idx]);
    
#if defined USE_PARTICLE_VALUES
    //pressure1 = neg_one_third * pStress[idx].Trace(); 
    concentration = pConcentration[idx];
    pressure = neg_one_third * pStress[idx].Trace(); 
#else
    concentration = 0.0;
    pressure      = 0.0;
    pressure_grad = Vector(0.0,0.0,0.0);
    //pressure2      = 0.0;
    for(int k = 0; k < d_Mflag->d_8or27; k++) {
      IntVector node = ni[k];
      concentration += gConcentration[node] * S[k];
      pressure      -= gHydroStress[node]   * S[k];
      // testing to see how pressure gradients might affect concentration profile
      //for (int j = 0; j<3; j++) {
      //    pressure_grad[j] -= (gHydroStress[node]*tuning5) * d_S[k][j] * oodx[j];
      //}
    }
#endif

    //pressure = pressure1;
    //pressure = pressure2;
    comp_diffusivity = computeDiffusivityTerm(concentration, pressure);
    if(use_pressure){
      // normalize pressure to on order of 1
      pressure = pressure*tuning5;
      // set a floor for the minimum pressure
      // to be used in the calculation
      if(pressure < tuning3){
        pressure = tuning3;
      }
      // set a cap for the maximum pressure
      // to be used in the calculation
      if(pressure > tuning4){
        pressure = tuning4;
      }
      //D = comp_diffusivity*exp(tuning1*concentration);
      D = comp_diffusivity*exp(tuning1*concentration)*exp(-tuning2*pressure);
      //cout << "Pressure: " << pressure << ", Concentration: " << concentration << ", Diffusivity: " << D << endl;
      //cout << "Pressure1: " << pressure1 << ", Pressure2: " << pressure2;
      //cout << ", Conc1: " << conc1 << ", Conc2: " << conc2 << endl;
    }else{
      non_lin_comp = 1/(1-concentration) - 2 * tuning1 * concentration;
      //non_lin_comp = exp(tuning1 * pow(concentration,3));

      //cout << "nlc: " << non_lin_comp << ", concentration: " << pConcentration[idx] << endl;

      if(non_lin_comp < tuning2){
        D = comp_diffusivity * non_lin_comp;
      } else {
        D = comp_diffusivity * tuning2;
      }
    }

    pFlux[idx] = D * pConcGrad[idx];
    // testing to see how pressure gradients might affect concentration profile
    //pFlux[idx] = D * (pConcGrad[idx] + (1-concentration)*concentration*tuning2*pressure_grad);
    pDiffusivity[idx] = D;
    pPressure1[idx] = pressure;
    //pPressure2[idx] = pressure2;
    timestep = min(timestep, computeStableTimeStep(D, dx));
  } //End of Particle Loop
  //cout << "timestep: " << timestep << endl;
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
  rdm_ps->appendElement("use_pressure", use_pressure);
  rdm_ps->appendElement("tuning1",tuning1);
  rdm_ps->appendElement("tuning2",tuning2);

  if(use_pressure){
    rdm_ps->appendElement("tuning3", tuning3);
    rdm_ps->appendElement("tuning4", tuning4);
    rdm_ps->appendElement("tuning5", tuning5);
  }
}
