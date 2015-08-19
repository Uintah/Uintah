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
#include <CCA/Components/MPM/ReactionDiffusion/ReactionDiffusionLabel.h>
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

  ps->require("tuning1", omega);
  ps->require("ramp_time", ramp_time);
}

RFConcDiffusion1MPM::~RFConcDiffusion1MPM() {

}

void RFConcDiffusion1MPM::scheduleComputeFlux(Task* task, const MPMMaterial* matl, 
                                              const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType  gan = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;
  task->requires(Task::OldDW, d_lb->pXLabel,                   matlset, gan, NGP);
  task->requires(Task::OldDW, d_lb->pSizeLabel,                matlset, gan, NGP);
  task->requires(Task::OldDW, d_lb->pMassLabel,                matlset, gan, NGP);
  task->requires(Task::OldDW, d_lb->pVolumeLabel,              matlset, gan, NGP);
  task->requires(Task::OldDW, d_lb->pDeformationMeasureLabel,  matlset, gan, NGP);
  task->requires(Task::OldDW, d_lb->pXLabel,                   matlset, gan, NGP);
  //task->requires(Task::OldDW, d_lb->pLoadCurveIDLabel,         matlset, gan, NGP);
  task->requires(Task::OldDW, d_rdlb->pConcentrationLabel,     matlset, gan, NGP);

  task->requires(Task::NewDW, d_lb->gMassLabel,                matlset, gnone);
  task->requires(Task::NewDW, d_rdlb->gConcentrationLabel,     matlset, gan, 2*NGN);

  task->computes(d_rdlb->pFluxLabel,  matlset);
}

void RFConcDiffusion1MPM::computeFlux(const Patch* patch, const MPMMaterial* matl,
                                                DataWarehouse* old_dw, DataWarehouse* new_dw)
{

  Ghost::GhostType  gac   = Ghost::AroundCells;
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;


  ParticleInterpolator* interpolator = d_Mflag->d_interpolator->clone(patch);
  vector<IntVector> ni(interpolator->size());
  vector<Vector> d_S(interpolator->size());

  Vector dx = patch->dCell();
  double oodx[3];
  oodx[0] = 1.0/dx.x();
  oodx[1] = 1.0/dx.y();
  oodx[2] = 1.0/dx.z();
           
  int dwi = matl->getDWIndex();
  constParticleVariable<Point>   px;
  constParticleVariable<double>  pvol,pMass;
  constParticleVariable<Matrix3> psize;
  constParticleVariable<Matrix3> deformationGradient;
  constParticleVariable<double>  pConcentration;
  //constParticleVariable<int>     pLoadCurveID;

  constNCVariable<double>        gConcentration,gMass;

  ParticleVariable<Vector>       pConcGradient;
  ParticleVariable<Vector>       pFlux;

  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

  old_dw->get(px,                  d_lb->pXLabel,                  pset);
  old_dw->get(pvol,                d_lb->pVolumeLabel,             pset);
  old_dw->get(pMass,               d_lb->pMassLabel,               pset);
  old_dw->get(psize,               d_lb->pSizeLabel,               pset);
  old_dw->get(deformationGradient, d_lb->pDeformationMeasureLabel, pset);
  //old_dw->get(pLoadCurveID,        d_lb->pLoadCurveIDLabel,        pset);
  old_dw->get(pConcentration,      d_rdlb->pConcentrationLabel,    pset);

  new_dw->get(gConcentration,     d_rdlb->gConcentrationLabel,     dwi, patch, gac,2*NGN);
  new_dw->get(gMass,              d_lb->gMassLabel,                dwi, patch, gnone, 0);

  new_dw->allocateTemporary(pConcGradient, pset);
  new_dw->allocateAndPut(pFlux, d_rdlb->pFluxLabel,  pset);
  
  double Diff;
  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
    particleIndex idx = *iter;

    // Get the node indices that surround the cell
    interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

    pConcGradient[idx] = Vector(0.0,0.0,0.0);
    pFlux[idx] = Vector(0.0,0.0,0.0);
    for (int k = 0; k < d_Mflag->d_8or27; k++){
      for (int j = 0; j<3; j++) {
          pConcGradient[idx][j] += gConcentration[ni[k]] * d_S[k][j] * oodx[j];
      }
	  }

    //Diff = diffusivity*(1/(1-pConcentration[idx]) - 2*omega*pConcentration[idx]);
    Diff = diffusivity;

    pFlux[idx] = Diff*pConcGradient[idx];

    // this is a hack that uses LoadCurveID to identify boundary particles
    // works with nano_pillar3_2D_FBC
    /*
    if(pLoadCurveID[idx] == 1){
      pFlux[idx][0] = Diff;
      pFlux[idx][1] = 0.0;
      pFlux[idx][2] = 0.0;
    }
    if(pLoadCurveID[idx] == 2){
      pFlux[idx][0] = -Diff;
      pFlux[idx][1] = 0.0;
      pFlux[idx][2] = 0.0;
    }
    */
    //cout << "id: " << idx << " CG: " << pConcentrationGradient[idx] << ", PF: " << pPotentialFlux[idx] << endl;
  } //End of Particle Loop

	delete interpolator;
}

void RFConcDiffusion1MPM::scheduleInterpolateToParticlesAndUpdate(Task* task,
                                                           const MPMMaterial* matl,     
                                                           const PatchSet* patch) const
{
  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  const MaterialSubset* matlset = matl->thisMaterial();

  task->requires(Task::OldDW, d_sharedState->get_delt_label());
  task->requires(Task::OldDW, d_lb->pXLabel,                    gnone);
  task->requires(Task::OldDW, d_lb->pSizeLabel,                 gnone);
  task->requires(Task::OldDW, d_lb->pDeformationMeasureLabel,   gnone);
  task->requires(Task::OldDW, d_rdlb->pConcentrationLabel,      gnone);
  task->requires(Task::OldDW, d_lb->pLoadCurveIDLabel,          gnone);
  task->requires(Task::NewDW, d_rdlb->gConcentrationRateLabel,  gac,NGN);
  task->requires(Task::NewDW, d_rdlb->gConcentrationStarLabel,  gac,NGN);

  task->computes(d_rdlb->pConcentrationLabel_preReloc, matlset);
  task->computes(d_rdlb->pConcPreviousLabel_preReloc,  matlset);
  task->computes(d_rdlb->pConcGradientLabel_preReloc,  matlset);
}

void RFConcDiffusion1MPM::interpolateToParticlesAndUpdate(const Patch* patch,
                                                           const MPMMaterial* matl,
                                                           DataWarehouse* old_dw,
                                                           DataWarehouse* new_dw)
{
  Ghost::GhostType  gac   = Ghost::AroundCells;
  int dwi = matl->getDWIndex();

  double run_time = d_sharedState->getElapsedTime();
  double boundary_conc;

  if(run_time < ramp_time){
    boundary_conc = run_time/ramp_time;
  }else{
    boundary_conc = 1.0;
  }

  ParticleInterpolator* interpolator = d_Mflag->d_interpolator->clone(patch);
  vector<IntVector> ni(interpolator->size());
  vector<double> S(interpolator->size());
  vector<Vector> d_S(interpolator->size());
  Vector dx = patch->dCell();
  double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

  constParticleVariable<Point>   px;
  constParticleVariable<Matrix3> psize;
  constParticleVariable<Matrix3> pFOld;
  constParticleVariable<double>  pConcentration;
  constParticleVariable<int>     pLoadCurveID;
  constNCVariable<double>        gConcentrationRate;
  constNCVariable<double>        gConcentrationStar;
  
  ParticleVariable<double> pConcentrationNew;
  ParticleVariable<double> pConcPreviousNew;
  ParticleVariable<Vector> pConcGradNew;
  
  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
  
  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label(), patch->getLevel() );
  
  old_dw->get(px,             d_lb->pXLabel,                   pset);
  old_dw->get(psize,          d_lb->pSizeLabel,                pset);
  old_dw->get(pFOld,          d_lb->pDeformationMeasureLabel,  pset);
  old_dw->get(pLoadCurveID,   d_lb->pLoadCurveIDLabel,         pset);
  
  old_dw->get(pConcentration,     d_rdlb->pConcentrationLabel,     pset);
  new_dw->get(gConcentrationRate, d_rdlb->gConcentrationRateLabel, dwi, patch, gac, NGP);
  new_dw->get(gConcentrationStar, d_rdlb->gConcentrationStarLabel, dwi, patch, gac, NGP);

  new_dw->allocateAndPut(pConcentrationNew, d_rdlb->pConcentrationLabel_preReloc, pset);
  new_dw->allocateAndPut(pConcPreviousNew,  d_rdlb->pConcPreviousLabel_preReloc,  pset);
  new_dw->allocateAndPut(pConcGradNew,      d_rdlb->pConcGradientLabel_preReloc,  pset);

  for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){

    particleIndex idx = *iter;
    interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,psize[idx],pFOld[idx]);
    double concRate = 0.0;
		pConcGradNew[idx] = Vector(0.0, 0.0, 0.0);
    for (int k = 0; k < d_Mflag->d_8or27; k++) {
      IntVector node = ni[k];
      concRate += gConcentrationRate[node]   * S[k];
      for(int j = 0; j < 3; j++){
        pConcGradNew[idx][j] += gConcentrationStar[ni[k]] * d_S[k][j] * oodx[j];
      }
    }

    pConcentrationNew[idx] = pConcentration[idx] + concRate*delT;

    // this is a hack that uses LoadCurveID to identify boundary particles
    // works with nano_pillar3_2D_FBC
    if(pLoadCurveID[idx] == 1){
      pConcentrationNew[idx] = boundary_conc;
    }
    if(pLoadCurveID[idx] == 2){
      pConcentrationNew[idx] = boundary_conc;
    }
    if(pLoadCurveID[idx] == 3){
      pConcentrationNew[idx] = boundary_conc;
    }

    pConcPreviousNew[idx]  = pConcentration[idx];
  }
  delete interpolator;
}

