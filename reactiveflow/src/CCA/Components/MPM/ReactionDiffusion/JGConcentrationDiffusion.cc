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
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Grid/Task.h>

#include <iostream>
using namespace std;
using namespace Uintah;


JGConcentrationDiffusion::JGConcentrationDiffusion(ProblemSpecP& ps, MPMFlags* Mflag):
  ScalarDiffusionModel(ps,Mflag) {
  d_Mflag = Mflag;

  d_lb = scinew MPMLabel;
  d_rdlb = scinew ReactionDiffusionLabel();

  if(d_Mflag->d_8or27==8){
    NGP=1;
    NGN=1;
  } else {
    NGP=2;
    NGN=2;
  }

  if(d_Mflag->d_scalarDiffusion_type == "explicit"){
    do_explicit = true;
  }else{
    do_explicit = false;
  }
}

JGConcentrationDiffusion::~JGConcentrationDiffusion() {
  delete d_lb;
  delete d_rdlb;
}

void JGConcentrationDiffusion::addInitialComputesAndRequires(Task* task,
                                                      const MPMMaterial* matl,
                                                      const PatchSet* patch) const{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(d_rdlb->pConcentrationLabel,  matlset);
  //task->computes(d_rdlb->pConcPreviousLabel,   matlset);
  //task->computes(d_rdlb->pdCdtLabel,           matlset);
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

void JGConcentrationDiffusion::addInterpolateParticlesToGridCompAndReq(Task* task,
                                                                const MPMMaterial* matl,
                                                                const PatchSet* patch) const{

  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType  gan = Ghost::AroundNodes;

  //task->requires(Task::OldDW, d_lb->pXLabel,    gan, NGP);
  //task->requires(Task::OldDW, d_lb->pMassLabel, gan, NGP);
  task->requires(Task::OldDW, d_rdlb->pConcentrationLabel, matlset, gan, NGP);

  //task->computes(d_rdlb->gConcentrationLabel,      matlset);
  //task->computes(d_rdlb->gConcentrationNoBCLabel,  matlset);
  //task->computes(d_rdlb->gConcentrationRateLabel,  matlset);

  task->computes(d_rdlb->pConcentrationLabel,      matlset);
}

void JGConcentrationDiffusion::interpolateParticlesToGrid(const Patch* patch,
                                                   const MPMMaterial* matl,
                                                   DataWarehouse* old_dw,
																									 DataWarehouse* new_dw){

  Ghost::GhostType  gan = Ghost::AroundNodes;
  ParticleInterpolator* interpolator = d_Mflag->d_interpolator->clone(patch); 
  vector<IntVector> ni(interpolator->size());
  vector<double> S(interpolator->size());

  constParticleVariable<double> pConcentration;
  constParticleVariable<Point>  px;
  constParticleVariable<double> pmass;
  int dwi = matl->getDWIndex();
  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

  //old_dw->get(px,              d_lb->pXLabel,  pset);
  //old_dw->get(pmass,           d_lb->pMassLabel,  pset);
  old_dw->get(pConcentration,  d_rdlb->pConcentrationLabel,  pset);

  //NCVariable<double> gmass;
  //NCVariable<double> gconcentration;
  //NCVariable<double> gconcentrationNoBC;
  //NCVariable<double> gconcentrationRate;

  ParticleVariable<double>  pconcentration;

  /**
  new_dw->allocateAndPut(gconcentration,      d_rdlb->gConcentrationLabel,
	                       dwi,  patch);
  new_dw->allocateAndPut(gconcentrationNoBC,  d_rdlb->gConcentrationNoBCLabel,
	                       dwi,  patch);
  new_dw->allocateAndPut(gconcentrationRate,  d_rdlb->gConcentrationRateLabel,
	                       dwi,  patch);
  **/
  new_dw->allocateAndPut(pconcentration,      d_rdlb->pConcentrationLabel,
	                       pset);

  //gmass.initialize(0);
  //gconcentration.initialize(0);
  //gconcentrationNoBC.initialize(0);
  //gconcentrationRate.initialize(0);
  
  int n8or27 = d_Mflag->d_8or27;
  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
    particleIndex idx = *iter;
    //interpolator->findCellAndWeights(px[idx],ni,S);

    /**
    IntVector node;
    for(int k = 0; k < n8or27; k++) {
      node = ni[k];
      if(patch->containsNode(node)) {
        gmass[node]          += pmass[idx]                       * S[k];
        gconcentration[node] += pConcentration[idx] * pmass[idx] * S[k];
      }
    }
    **/
    pconcentration[idx] = pConcentration[idx];
  }
  /**
  for(NodeIterator iter=patch->getExtraNodeIterator();
                   !iter.done();iter++){
    IntVector c = *iter; 
    gconcentration[c]   /= gmass[c];
    gconcentrationNoBC[c] = gconcentration[c];
  }
  **/
  delete interpolator;
}
