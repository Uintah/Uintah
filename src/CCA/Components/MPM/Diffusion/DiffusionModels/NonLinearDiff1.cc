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

#include <CCA/Components/MPM/Diffusion/DiffusionModels/NonLinearDiff1.h>

#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/MPMBoundCond.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Task.h>

#include <iostream>
using namespace Uintah;

#define USE_PARTICLE_VALUES

NonLinearDiff1::NonLinearDiff1(
                               ProblemSpecP     & ps,
                               SimulationStateP & sS,
                               MPMFlags         * Mflag,
                               std::string        diff_type
                              ) : ScalarDiffusionModel(ps,
                                                       sS,
                                                       Mflag,
                                                       diff_type)
{

  ProblemSpecP diff_curve = 0;
  ProblemSpecP time_point = 0;

  double time;
  std::string flux_direction;

  d_use_pressure = false;
  d_use_diff_curve = false;

  ps->require("use_pressure", d_use_pressure);
  ps->require("tuning1", d_tuning1);
  ps->require("tuning2", d_tuning2);

  if(d_use_pressure){
    ps->require("tuning3", d_tuning3);
    ps->require("tuning4", d_tuning4);
    ps->require("tuning5", d_tuning5);
  }

  diff_curve = ps->findBlock("diff_curve");
  if(diff_curve){
    d_use_diff_curve = true;
    std::cout << "!!!!!!!!!!!!!!!!!!!using diff curve!!!!!!!!!!!!!!!" << std::endl;
    std::cout << "This is experimental: diff_curve code still needs error checking" << std::endl;
    for (time_point = diff_curve->findBlock("time_point"); time_point != nullptr; time_point = time_point->findNextBlock("time_point")) {

      time_point->require("time", time);
      time_point->require("flux_direction", flux_direction);
      d_time_points.push_back(time);
      if(flux_direction == "in"){
        d_fd_directions.push_back(fd_in);
      }else if(flux_direction == "out"){
        d_fd_directions.push_back(fd_out);
      }else if(flux_direction == "transition"){
        d_fd_directions.push_back(fd_transition);
      }
    }
    for(unsigned int i =0; i < d_time_points.size(); i++){
      std::cout << "Time: " << d_time_points[i] << " Direction: " << d_fd_directions[i] << std::endl;
    }
    d_time_point1 = d_time_points[0];
    if(d_time_points.size() >= 1){
       d_time_point2 = d_time_points[1];
    }else{
       // needs to be an error for not having enough points
    }
    d_flux_direction = d_fd_directions[0];
    d_diff_curve_index = 0;
  }

}

NonLinearDiff1::~NonLinearDiff1()
{

}

void NonLinearDiff1::addInitialComputesAndRequires(
                                                         Task         * task,
                                                   const MPMMaterial  * matl,
                                                   const PatchSet     * patch
                                                  ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(d_lb->diffusion->pDiffusivity, matlset);
  task->computes(d_lb->pPressureLabel_t1, matlset);
  task->computes(d_lb->pConcInterpLabel,  matlset);
  task->computes(d_lb->diffusion->pFlux,        matlset);
}

void NonLinearDiff1::addParticleState(
                                      std::vector<const VarLabel*>& from,
                                      std::vector<const VarLabel*>& to
                                     ) const
{
  from.push_back(d_lb->diffusion->pDiffusivity);
  from.push_back(d_lb->pPressureLabel_t1);
  from.push_back(d_lb->pConcInterpLabel);
  from.push_back(d_lb->diffusion->pFlux);

  to.push_back(d_lb->diffusion->pDiffusivity_preReloc);
  to.push_back(d_lb->pPressureLabel_t1_preReloc);
  to.push_back(d_lb->pConcInterpLabel_preReloc);
  to.push_back(d_lb->diffusion->pFlux_preReloc);
}

void NonLinearDiff1::computeFlux(
                                 const Patch          * patch,
                                 const MPMMaterial    * matl,
                                       DataWarehouse  * old_dw,
                                       DataWarehouse  * new_dw
                                )
{
  Ghost::GhostType gac = Ghost::AroundCells;
  ParticleInterpolator* interpolator = d_Mflag->d_interpolator->clone(patch);
  std::vector<IntVector> ni(interpolator->size());
  std::vector<double> S(interpolator->size());

  // Get the current simulation time
  double current_time1 = d_sharedState->getElapsedSimTime();

  int dwi = matl->getDWIndex();
  Vector dx = patch->dCell();
  double comp_diffusivity;

  constParticleVariable<Vector>  pConcGrad;
  constParticleVariable<double>  pConcentration;
  constParticleVariable<Matrix3> pStress;
  constParticleVariable<Point>   px;
  constParticleVariable<Matrix3> psize;
  constParticleVariable<Matrix3> pFOld;
  constParticleVariable<double>  pDiffusivity_old;

  constNCVariable<double>  gConcentration;
  constNCVariable<double>  gHydroStress;

  ParticleVariable<Vector>       pFlux;
  ParticleVariable<double>       pDiffusivity;
  ParticleVariable<double>       pPressure1;
  ParticleVariable<double>       pConcInterp;

  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

  old_dw->get(px,             d_lb->pXLabel,                  pset);
  old_dw->get(pConcGrad,      d_lb->diffusion->pGradConcentration,       pset);
  old_dw->get(pConcentration, d_lb->diffusion->pConcentration,      pset);
  old_dw->get(pStress,        d_lb->pStressLabel,             pset);
  old_dw->get(pFOld,          d_lb->pDeformationMeasureLabel, pset);
  old_dw->get(pDiffusivity_old, d_lb->diffusion->pDiffusivity,      pset);

  new_dw->get(psize,          d_lb->pSizeLabel_preReloc,      pset);

  new_dw->get(gConcentration, d_lb->diffusion->gConcentration,     dwi, patch, gac, NGN);
  new_dw->get(gHydroStress,   d_lb->diffusion->gHydrostaticStress, dwi, patch, gac, NGN);

  new_dw->allocateAndPut(pFlux,        d_lb->diffusion->pFlux_preReloc,        pset);
  new_dw->allocateAndPut(pDiffusivity, d_lb->diffusion->pDiffusivity_preReloc, pset);

  // JBH -- FIXME -- TODO -- Add these to MPMDiffusion sublabel or to diffusion model specific sublabel?
  new_dw->allocateAndPut(pPressure1,   d_lb->pPressureLabel_t1_preReloc, pset);
  new_dw->allocateAndPut(pConcInterp,  d_lb->pConcInterpLabel_preReloc,  pset);

  double non_lin_comp = 0.0;
  double D = 0.0;
  double timestep = 1.0e99;
  double pressure = 0.0;
  double concentration = 0.0;

  if(d_use_diff_curve){
    if(current_time1 > d_time_point2){
      d_diff_curve_index++;
      d_time_point1 = d_time_points[d_diff_curve_index];
      d_time_point2 = d_time_points[d_diff_curve_index+1];
      d_flux_direction = d_fd_directions[d_diff_curve_index];
    }
    //cout << "Time: " << current_time1 << " t1: " << d_time_point1;
    //cout << ", t2: " << d_time_point2 << " fd: " << d_flux_direction << endl;
  }
  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end();
                                                      iter++){
    particleIndex idx = *iter;

    interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],pFOld[idx]);

#if defined USE_PARTICLE_VALUES
    double neg_one_third = -1.0/3.0;
    concentration = pConcentration[idx];
    pressure = neg_one_third * pStress[idx].Trace(); 
#else
    concentration = 0.0;
    pressure      = 0.0;
    for(int k = 0; k < NN; k++) {
      IntVector node = ni[k];
      concentration += gConcentration[node] * S[k];
      pressure      -= gHydroStress[node]   * S[k];
    }
#endif

    comp_diffusivity = computeDiffusivityTerm(concentration, pressure);
    if(d_use_diff_curve){
      if(d_flux_direction == fd_in || d_flux_direction == fd_transition){
        if(d_use_pressure){
          pressure = pressure * d_tuning5;
          if(pressure < d_tuning3){
            pressure = d_tuning3;
          }
          if(pressure > d_tuning4){
            pressure = d_tuning4;
          }
          D = comp_diffusivity*exp(d_tuning1 * concentration)*exp(-d_tuning2 * pressure);
        }else{
          non_lin_comp = exp(d_tuning1 * concentration);
          D = comp_diffusivity * non_lin_comp;
        }
      }else if(d_flux_direction == fd_out){
        D = pDiffusivity_old[idx];
      }
    }else{
      if(d_use_pressure){
        // normalize pressure to on order of 1
        // Why in heaven's name are you doing this instead of just rolling this
        // into tuning2?  You are essentially putting units into the code!  JG
        pressure = pressure * d_tuning5;
        // set a floor for the minimum pressure
        // to be used in the calculation
        if(pressure < d_tuning3){
          pressure = d_tuning3;
        }
        // set a cap for the maximum pressure
        // to be used in the calculation
        if(pressure > d_tuning4){
          pressure = d_tuning4;
        }
        D = comp_diffusivity*exp(d_tuning1 * concentration)*exp(-d_tuning2 * pressure);
        //cout << "Pressure: " << pressure << ", Concentration: " << concentration << ", Diffusivity: " << D << endl;
        //cout << "Pressure1: " << pressure1 << ", Pressure2: " << pressure2;
        //cout << ", Conc1: " << conc1 << ", Conc2: " << conc2 << endl;
      }else{
        non_lin_comp = exp(d_tuning1 * concentration);
  
        D = comp_diffusivity * non_lin_comp;
      }
    }

    pFlux[idx] = D * pConcGrad[idx];
    pDiffusivity[idx] = D;
    pPressure1[idx] = pressure;
    pConcInterp[idx] = concentration;
    timestep = std::min(timestep, computeStableTimeStep(D, dx));
  } //End of Particle Loop
  new_dw->put(delt_vartype(timestep), d_lb->delTLabel, patch->getLevel());
}

void NonLinearDiff1::initializeSDMData(const Patch* patch, const MPMMaterial* matl,
                                       DataWarehouse* new_dw)
{
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<double>  pDiffusivity;
  ParticleVariable<double>  pPressure;
  ParticleVariable<double>  pConcInterp;
  ParticleVariable<Vector>  pFlux;

  new_dw->allocateAndPut(pDiffusivity, d_lb->diffusion->pDiffusivity, pset);
  new_dw->allocateAndPut(pPressure,    d_lb->pPressureLabel_t1,    pset);
  new_dw->allocateAndPut(pConcInterp,  d_lb->pConcInterpLabel,  pset);
  new_dw->allocateAndPut(pFlux,        d_lb->diffusion->pFlux,        pset);

  for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++)
  {
    pDiffusivity[*iter] = d_D0;
    pPressure[*iter]    = 0.0;
    pConcInterp[*iter]  = 0.0;
    pFlux[*iter]        = Vector(0,0,0);
  }
}

void NonLinearDiff1::scheduleComputeFlux(Task* task, const MPMMaterial* matl,
                                              const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType gnone = Ghost::None;
  Ghost::GhostType gac   = Ghost::AroundCells;
  task->requires(Task::OldDW, d_lb->pXLabel,                        matlset, gnone);
  task->requires(Task::OldDW, d_lb->diffusion->pGradConcentration,  matlset, gnone);
  task->requires(Task::OldDW, d_lb->diffusion->pConcentration,      matlset, gnone);
  task->requires(Task::OldDW, d_lb->pStressLabel,                   matlset, gnone);
  task->requires(Task::OldDW, d_lb->pDeformationMeasureLabel,       matlset, gnone);
  task->requires(Task::OldDW, d_lb->diffusion->pDiffusivity,        matlset, gnone);

  task->requires(Task::NewDW, d_lb->pSizeLabel_preReloc,            matlset, gnone);
  task->requires(Task::NewDW, d_lb->diffusion->gConcentration,      matlset, gac, NGN);
  task->requires(Task::NewDW, d_lb->diffusion->gHydrostaticStress,  matlset, gac, NGN);

  task->computes(d_sharedState->get_delt_label(),getLevel(patch));

  task->computes(d_lb->diffusion->pFlux_preReloc,        matlset);
  task->computes(d_lb->diffusion->pDiffusivity_preReloc, matlset);
  // JBH -- FIXME -- TODO -- Add to MPMDiffusion sublabel or to diffusion model specific labels?
  task->computes(d_lb->pPressureLabel_t1_preReloc, matlset);
  task->computes(d_lb->pConcInterpLabel_preReloc,  matlset);
}

void NonLinearDiff1::addSplitParticlesComputesAndRequires(
                                                                Task        * task,
                                                          const MPMMaterial * matl,
                                                          const PatchSet    * patches
                                                         ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->modifies(d_lb->diffusion->pDiffusivity_preReloc, matlset);
  task->modifies(d_lb->pPressureLabel_t1_preReloc, matlset);
  task->modifies(d_lb->pConcInterpLabel_preReloc,  matlset);
  task->modifies(d_lb->diffusion->pFlux_preReloc,        matlset);
}

void NonLinearDiff1::splitSDMSpecificParticleData(
                                                  const Patch                 * patch,
                                                  const int dwi,
                                                  const int fourOrEight,
                                                        ParticleVariable<int> & prefOld,
                                                        ParticleVariable<int> & prefNew,
                                                  const unsigned int            oldNumPar,
                                                  const int                     numNewPartNeeded,
                                                        DataWarehouse*          old_dw,
                                                        DataWarehouse*          new_dw
                                                 )
{
	ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

	ParticleVariable<double>  pDiffusivity, pPressure, pConcInterp;
	ParticleVariable<Vector>  pFlux;

  new_dw->getModifiable(pDiffusivity, d_lb->diffusion->pDiffusivity_preReloc,  pset);
  new_dw->getModifiable(pPressure,    d_lb->pPressureLabel_t1_preReloc,  pset);
  new_dw->getModifiable(pConcInterp,  d_lb->pConcInterpLabel_preReloc,   pset);
  new_dw->getModifiable(pFlux,        d_lb->diffusion->pFlux_preReloc,         pset);

  ParticleVariable<double>  pDiffusivityTmp, pPressureTmp, pConcInterpTmp;
  ParticleVariable<Vector>  pFluxTmp;

  new_dw->allocateTemporary(pDiffusivityTmp, pset);
  new_dw->allocateTemporary(pPressureTmp,    pset);
  new_dw->allocateTemporary(pConcInterpTmp,  pset);
  new_dw->allocateTemporary(pFluxTmp,        pset);

	// copy data from old variables for particle IDs and the position vector
	for(unsigned int pp=0; pp < oldNumPar; ++pp )
	{
    pDiffusivityTmp[pp] = pDiffusivity[pp];
		pPressureTmp[pp]    = pPressure[pp];
		pConcInterpTmp[pp]  = pConcInterp[pp];
		pFluxTmp[pp]        = pFlux[pp];
  }

	int numRefPar=0;
	for(unsigned int idx=0; idx<oldNumPar; ++idx )
	{
	  if(prefNew[idx] != prefOld[idx])
	    {  // do refinement!
	      for(int i = 0; i<fourOrEight; i++)
	      {
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
	        pPressureTmp[new_index]    = pPressure[idx];
	        pConcInterpTmp[new_index]  = pConcInterp[idx];
	        pFluxTmp[new_index]        = pFlux[idx];
	      }
	      numRefPar++;
	    }
	  }

	  new_dw->put(pDiffusivityTmp, d_lb->diffusion->pDiffusivity_preReloc, true);
	  new_dw->put(pPressureTmp,    d_lb->pPressureLabel_t1_preReloc, true);
	  new_dw->put(pConcInterpTmp,  d_lb->pConcInterpLabel_preReloc,  true);
	  new_dw->put(pFluxTmp,        d_lb->diffusion->pFlux_preReloc,        true);
}

void NonLinearDiff1::outputProblemSpec(
                                       ProblemSpecP & ps,
                                       bool           output_rdm_tag
                                      ) const
{

  ProblemSpecP rdm_ps = ps;
  ProblemSpecP diff_curve_ps = 0;
  ProblemSpecP time_point_ps = 0;
  if (output_rdm_tag) {
    rdm_ps = ps->appendChild("diffusion_model");
    rdm_ps->setAttribute("type","non_linear1");
  }
  ScalarDiffusionModel::baseOutputSDMProbSpec(rdm_ps);

  rdm_ps->appendElement("use_pressure", d_use_pressure);
  rdm_ps->appendElement("tuning1", d_tuning1);
  rdm_ps->appendElement("tuning2", d_tuning2);

  if(d_conductivity_equation){
    d_conductivity_equation->outputProblemSpec(rdm_ps);
  }

  if(d_use_pressure){
    rdm_ps->appendElement("tuning3", d_tuning3);
    rdm_ps->appendElement("tuning4", d_tuning4);
    rdm_ps->appendElement("tuning5", d_tuning5);
  }

  //*********************************************************
  // Need to place code outputProblemSpec for diff_curve info
  //*********************************************************
  if(d_use_diff_curve){
    diff_curve_ps = rdm_ps->appendChild("diff_curve");
    for(unsigned int i =0; i < d_time_points.size(); i++){
      time_point_ps = diff_curve_ps->appendChild("time_point");
      time_point_ps->appendElement("time",d_time_points[i]);
      if(d_fd_directions[i] == fd_in){
        time_point_ps->appendElement("flux_direction", "in");
      }else if(d_fd_directions[i] == fd_out){
        time_point_ps->appendElement("flux_direction", "out");
      }else if(d_fd_directions[i] == fd_transition){
        time_point_ps->appendElement("flux_direction", "transition");
      }
    }
  }
}
