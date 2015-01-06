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

#include <CCA/Components/MPM/ReactionDiffusion/JGConcentrationDiffusion.h>
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


JGConcentrationDiffusion::JGConcentrationDiffusion(ProblemSpecP& ps, SimulationStateP& sS, MPMFlags* Mflag):
  ScalarDiffusionModel(ps, sS, Mflag) {
	
  ps->require("diffusivity",diffusivity);
}

JGConcentrationDiffusion::~JGConcentrationDiffusion() {

}

void JGConcentrationDiffusion::addInitialComputesAndRequires(Task* task,
                                                      const MPMMaterial* matl,
                                                      const PatchSet* patch) const{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(d_rdlb->pConcentrationLabel,  matlset);
  task->computes(d_rdlb->pConcPreviousLabel,   matlset);
  task->computes(d_rdlb->pdCdtLabel,           matlset);
}

void JGConcentrationDiffusion::initializeSDMData(const Patch* patch,
                                          const MPMMaterial* matl,
                                          DataWarehouse* new_dw)
{
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<double>  pConcentration;

  new_dw->allocateAndPut(pConcentration,   d_rdlb->pConcentrationLabel, pset);

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
    pConcentration[*iter] = 0.0;
  }
}

void JGConcentrationDiffusion::scheduleInterpolateParticlesToGrid(Task* task,
                                                           const MPMMaterial* matl,
                                                           const PatchSet* patch) const{

  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType  gan = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;

  task->requires(Task::OldDW, d_lb->pXLabel,    gan, NGP);
  task->requires(Task::OldDW, d_lb->pMassLabel, gan, NGP);
  task->requires(Task::OldDW, d_lb->pSizeLabel, gan, NGP);
  task->requires(Task::OldDW, d_lb->pDeformationMeasureLabel,gan, NGP);
  task->requires(Task::OldDW, d_rdlb->pConcentrationLabel, matlset, gan, NGP);
	task->requires(Task::NewDW, d_lb->gMassLabel, gnone);

  task->computes(d_rdlb->gConcentrationLabel,      matlset);
  task->computes(d_rdlb->gConcentrationNoBCLabel,  matlset);
  task->computes(d_rdlb->gConcentrationRateLabel,  matlset);

}

void JGConcentrationDiffusion::interpolateParticlesToGrid(const Patch* patch,
                                                   const MPMMaterial* matl,
                                                   DataWarehouse* old_dw,
																									 DataWarehouse* new_dw){

  Ghost::GhostType  gan = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;

  ParticleInterpolator* interpolator = d_Mflag->d_interpolator->clone(patch); 

  vector<IntVector> ni(interpolator->size());
  vector<double> S(interpolator->size());

  constParticleVariable<Point>  px;
  constParticleVariable<double> pmass;
  constParticleVariable<double> pConcentration;
  constParticleVariable<Matrix3> psize;
  constParticleVariable<Matrix3> pFOld;
	constNCVariable<double>       gmass;

  int dwi = matl->getDWIndex();
  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch, gan, NGP,
	                                                 d_lb->pXLabel);

  old_dw->get(px,             d_lb->pXLabel,                pset);
  old_dw->get(pmass,          d_lb->pMassLabel,             pset);
  old_dw->get(pConcentration, d_rdlb->pConcentrationLabel,  pset);
  old_dw->get(psize,          d_lb->pSizeLabel,             pset);
  old_dw->get(pFOld,          d_lb->pDeformationMeasureLabel,pset);
  new_dw->get(gmass,          d_lb->gMassLabel,        dwi, patch, gnone, 0);

  NCVariable<double> gconcentration;
  NCVariable<double> gconcentrationNoBC;
  NCVariable<double> gconcentrationRate;


  new_dw->allocateAndPut(gconcentration,      d_rdlb->gConcentrationLabel,
	                       dwi,  patch);
  new_dw->allocateAndPut(gconcentrationNoBC,  d_rdlb->gConcentrationNoBCLabel,
	                       dwi,  patch);
  new_dw->allocateAndPut(gconcentrationRate,  d_rdlb->gConcentrationRateLabel,
	                       dwi,  patch);


  gconcentration.initialize(0);
  gconcentrationNoBC.initialize(0);
  gconcentrationRate.initialize(0);
  
  int n8or27 = d_Mflag->d_8or27;
  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
    particleIndex idx = *iter;

    interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],pFOld[idx]);

    IntVector node;
    for(int k = 0; k < n8or27; k++) {
      node = ni[k];
      if(patch->containsNode(node)) {
        gconcentration[node] += pConcentration[idx] * pmass[idx] * S[k];
      }
    }
  }
  for(NodeIterator iter=patch->getExtraNodeIterator();
                   !iter.done();iter++){
    IntVector c = *iter; 
    gconcentration[c]   /= gmass[c];
    gconcentrationNoBC[c] = gconcentration[c];
  }

  MPMBoundCond bc;
  bc.setBoundaryCondition(patch,dwi,"SD-Type",gconcentration, d_Mflag->d_interpolator_type);

  delete interpolator;
}


void JGConcentrationDiffusion::scheduleComputeStep1(Task* task, const MPMMaterial* matl, 
		                                                    const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType  gan = Ghost::AroundNodes;
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gnone = Ghost::None;
  task->requires(Task::OldDW, d_lb->pXLabel,                         gan, NGP);
  task->requires(Task::OldDW, d_lb->pSizeLabel,                      gan, NGP);
  task->requires(Task::OldDW, d_lb->pMassLabel,                      gan, NGP);
  task->requires(Task::OldDW, d_lb->pVolumeLabel,                    gan, NGP);
  task->requires(Task::OldDW, d_lb->pDeformationMeasureLabel,        gan, NGP);
  task->requires(Task::NewDW, d_lb->gMassLabel,                      gnone);
  task->requires(Task::NewDW, d_rdlb->gConcentrationLabel,           gan, 2*NGN);
  task->computes(d_rdlb->gdCdtLabel,  matlset);

}

void JGConcentrationDiffusion::computeStep1(const Patch* patch, const MPMMaterial* matl,
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
  constParticleVariable<Point>  px;
  constParticleVariable<double> pvol,pMass;
  constParticleVariable<Matrix3> psize;
  constParticleVariable<Matrix3> deformationGradient;
  ParticleVariable<Vector>      pConcentrationGradient;
  constNCVariable<double>       gConcentration,gMass;
  NCVariable<double>            gdCdt;

  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch, gan, NGP, d_lb->pXLabel);

  old_dw->get(px,           d_lb->pXLabel,                         pset);
  old_dw->get(pvol,         d_lb->pVolumeLabel,                    pset);
  old_dw->get(pMass,        d_lb->pMassLabel,                      pset);
  old_dw->get(psize,        d_lb->pSizeLabel,                      pset);
  old_dw->get(deformationGradient, d_lb->pDeformationMeasureLabel, pset);
  new_dw->get(gConcentration, d_rdlb->gConcentrationLabel, dwi, patch, gac,2*NGN);
  new_dw->get(gMass,        d_lb->gMassLabel,        dwi, patch, gnone, 0);
  new_dw->allocateAndPut(gdCdt, d_rdlb->gdCdtLabel,    dwi, patch);
  new_dw->allocateTemporary(pConcentrationGradient, pset);
  
  gdCdt.initialize(0.);

  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
    particleIndex idx = *iter;

    // Get the node indices that surround the cell
    interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

    pConcentrationGradient[idx] = Vector(0.0,0.0,0.0);
    for (int k = 0; k < d_Mflag->d_8or27; k++){
      for (int j = 0; j<3; j++) {
          pConcentrationGradient[idx][j] += gConcentration[ni[k]] * d_S[k][j] * oodx[j];
      }
	  }
  } //End of Particle Loop

  for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
    particleIndex idx = *iter;
  
    // Get the node indices that surround the cell
    interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

    Vector dC_dx = pConcentrationGradient[idx];
    double Cdot_cond = 0.0;
    IntVector node(0,0,0);

    for (int k = 0; k < d_Mflag->d_8or27; k++){
      node = ni[k];
      if(patch->containsNode(node)){
        Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],d_S[k].z()*oodx[2]);
        Cdot_cond = Dot(div, dC_dx)*diffusivity;
        gdCdt[node] -= Cdot_cond;
      }
    }
  } // End of Particle Loop 

	delete interpolator;
}

void JGConcentrationDiffusion::scheduleComputeStep2(Task* task, const MPMMaterial* matl, 
		                                                const PatchSet* patch) const
{
  Ghost::GhostType  gnone = Ghost::None;
  Ghost::GhostType  gac   = Ghost::AroundCells;
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, d_sharedState->get_delt_label());

  task->requires(Task::NewDW, d_rdlb->gConcentrationLabel,     gnone);
  task->requires(Task::NewDW, d_rdlb->gConcentrationNoBCLabel, gnone);
  task->requires(Task::NewDW, d_rdlb->gdCdtLabel,              gnone);
  task->modifies(d_rdlb->gConcentrationRateLabel, matlset);
  task->computes(d_rdlb->gConcentrationStarLabel, matlset);

	// For testing purposes	
  task->requires(Task::OldDW, d_rdlb->pConcentrationLabel, matlset, gan, NGP);
  task->computes(d_rdlb->pConcentrationLabel,      matlset);
}

void JGConcentrationDiffusion::computeStep2(const Patch* patch, const MPMMaterial* matl,
                                            DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  Ghost::GhostType  gnone = Ghost::None;
  Ghost::GhostType  gac   = Ghost::AroundCells;
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  int dwi = matl->getDWIndex();

  constNCVariable<double> gConc_Old;
  constNCVariable<double> gConc_OldNoBC;
  constNCVariable<double> gdCdt;

  NCVariable<double> gConcRate;
  NCVariable<double> gConcStar;

  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label(), patch->getLevel() );
 
  new_dw->get(gConc_Old,     d_rdlb->gConcentrationLabel,     dwi, patch,gnone,0);
  new_dw->get(gConc_OldNoBC, d_rdlb->gConcentrationNoBCLabel, dwi, patch,gnone,0);
  new_dw->get(gdCdt,         d_rdlb->gdCdtLabel,              dwi, patch,gnone,0);

  new_dw->getModifiable(gConcRate,  d_rdlb->gConcentrationRateLabel, dwi,patch);
  new_dw->allocateAndPut(gConcStar, d_rdlb->gConcentrationStarLabel, dwi,patch);

  gConcStar.initialize(0.0);

  MPMBoundCond bc;

  for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
        gConcStar[c] = gConc_Old[c] + gdCdt[c] * delT;
  }

  bc.setBoundaryCondition(patch, dwi, "SD-Type", gConcStar, d_Mflag->d_interpolator_type);

  for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
    IntVector c = *iter;
    gConcRate[c] = (gConcStar[c] - gConc_OldNoBC[c]) / delT;
  }

	// For testing purposes
  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch, gan, NGP, d_lb->pXLabel);
  constParticleVariable<double> pConcentration;
  old_dw->get(pConcentration, d_rdlb->pConcentrationLabel,  pset);
  ParticleVariable<double>  pconcentration;
  new_dw->allocateAndPut(pconcentration,      d_rdlb->pConcentrationLabel, pset);
  for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
    particleIndex idx = *iter;
    pconcentration[idx] = pConcentration[idx]+1;
	}
}
