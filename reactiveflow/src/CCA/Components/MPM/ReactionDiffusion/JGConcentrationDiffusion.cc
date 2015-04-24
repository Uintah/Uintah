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


JGConcentrationDiffusion::JGConcentrationDiffusion(ProblemSpecP& ps, SimulationStateP& sS, MPMFlags* Mflag, string diff_type ):
  ScalarDiffusionModel(ps, sS, Mflag, diff_type) {
	
  ps->require("diffusivity",diffusivity);
  ps->require("max_concentration",max_concentration);

  include_hydrostress = false;
}

JGConcentrationDiffusion::~JGConcentrationDiffusion() {

}

void JGConcentrationDiffusion::setIncludeHydroStress(bool value){
  include_hydrostress = value;
}
void JGConcentrationDiffusion::addInitialComputesAndRequires(Task* task,
                                                      const MPMMaterial* matl,
                                                      const PatchSet* patch) const{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(d_rdlb->pConcentrationLabel, matlset);
  task->computes(d_rdlb->pConcPreviousLabel,  matlset);
  task->computes(d_rdlb->maxHydroStressLabel);
  task->computes(d_rdlb->minHydroStressLabel);
}

void JGConcentrationDiffusion::initializeSDMData(const Patch* patch,
                                          const MPMMaterial* matl,
                                          DataWarehouse* new_dw)
{
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
  int dwi = matl->getDWIndex();

  ParticleVariable<double>  pConcentration;
  ParticleVariable<double>  pConcPrevious;

  new_dw->allocateAndPut(pConcentration,  d_rdlb->pConcentrationLabel, pset);
  new_dw->allocateAndPut(pConcPrevious,   d_rdlb->pConcPreviousLabel,  pset);

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
    pConcentration[*iter] = 0.0;
    pConcPrevious[*iter] = 0.0;
  }
  new_dw->put(max_vartype(0), d_rdlb->maxHydroStressLabel);
  new_dw->put(min_vartype(0), d_rdlb->minHydroStressLabel);
}

void JGConcentrationDiffusion::addParticleState(std::vector<const VarLabel*>& from,
                                                std::vector<const VarLabel*>& to)
{
  from.push_back(d_rdlb->pConcentrationLabel);
  from.push_back(d_rdlb->pConcPreviousLabel);

  to.push_back(d_rdlb->pConcentrationLabel_preReloc);
  to.push_back(d_rdlb->pConcPreviousLabel_preReloc);
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

  if(include_hydrostress){
    task->requires(Task::OldDW, d_lb->pStressLabel, matlset, gan, NGP);
    task->computes(d_rdlb->gHydrostaticStressLabel, matlset);
    task->computes(d_rdlb->pHydroStressLabel,       matlset);
    task->computes(d_rdlb->maxHydroStressLabel);
    task->computes(d_rdlb->minHydroStressLabel);
  }

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
  constParticleVariable<Matrix3> pStress;
	constNCVariable<double>       gmass;

  int dwi = matl->getDWIndex();
  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch, gan, NGP,
	                                                 d_lb->pXLabel);

  old_dw->get(px,             d_lb->pXLabel,                  pset);
  old_dw->get(pmass,          d_lb->pMassLabel,               pset);
  old_dw->get(pConcentration, d_rdlb->pConcentrationLabel,    pset);
  old_dw->get(psize,          d_lb->pSizeLabel,               pset);
  old_dw->get(pFOld,          d_lb->pDeformationMeasureLabel, pset);
  new_dw->get(gmass,          d_lb->gMassLabel,        dwi, patch, gnone, 0);

  ParticleVariable<double> phydrostress;
  NCVariable<double> gconcentration;
  NCVariable<double> gconcentrationNoBC;
  NCVariable<double> ghydrostaticstress;


  new_dw->allocateAndPut(gconcentration,      d_rdlb->gConcentrationLabel,
	                       dwi,  patch);
  new_dw->allocateAndPut(gconcentrationNoBC,  d_rdlb->gConcentrationNoBCLabel,
	                       dwi,  patch);

  gconcentration.initialize(0);
  gconcentrationNoBC.initialize(0);

  if(include_hydrostress){
    old_dw->get(pStress,        d_lb->pStressLabel,             pset);
    new_dw->allocateAndPut(phydrostress, d_rdlb->pHydroStressLabel, pset);
    new_dw->allocateAndPut(ghydrostaticstress,  d_rdlb->gHydrostaticStressLabel,
	                         dwi,  patch);
    ghydrostaticstress.initialize(0);
  }
  
  double maxhydrostress = 0;
  double minhydrostress = 0;

  int n8or27 = d_Mflag->d_8or27;
  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
    particleIndex idx = *iter;

    interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],pFOld[idx]);

    if(include_hydrostress){
      phydrostress[idx] = (pStress[idx].Trace())/3;

      if(phydrostress[idx] > maxhydrostress){
        maxhydrostress = phydrostress[idx];
      }
      if(phydrostress[idx] < minhydrostress){
        minhydrostress = phydrostress[idx];
      }
    }

    IntVector node;
    for(int k = 0; k < n8or27; k++) {
      node = ni[k];
      if(patch->containsNode(node)) {
        gconcentration[node] += pConcentration[idx] * pmass[idx] * S[k];
        if(include_hydrostress){
          ghydrostaticstress[node] += phydrostress[idx] * pmass[idx] * S[k];
        }
      }
    }
  }

  new_dw->put(max_vartype(maxhydrostress), d_rdlb->maxHydroStressLabel);
  new_dw->put(min_vartype(minhydrostress), d_rdlb->minHydroStressLabel);

  for(NodeIterator iter=patch->getExtraNodeIterator();
                   !iter.done();iter++){
    IntVector c = *iter; 
    gconcentration[c]   /= gmass[c];
    gconcentrationNoBC[c] = gconcentration[c];
    if(include_hydrostress){
      ghydrostaticstress[c] /= gmass[c];
    }
  }

  MPMBoundCond bc;
  bc.setBoundaryCondition(patch,dwi,"SD-Type",gconcentration, d_Mflag->d_interpolator_type);

  delete interpolator;
}


void JGConcentrationDiffusion::scheduleComputeFlux(Task* task, const MPMMaterial* matl, 
		                                                    const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType  gan = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;
  task->requires(Task::OldDW, d_lb->pXLabel,                  matlset, gan, NGP);
  task->requires(Task::OldDW, d_lb->pSizeLabel,               matlset, gan, NGP);
  task->requires(Task::OldDW, d_lb->pMassLabel,               matlset, gan, NGP);
  task->requires(Task::OldDW, d_lb->pVolumeLabel,             matlset, gan, NGP);
  task->requires(Task::OldDW, d_lb->pDeformationMeasureLabel, matlset, gan, NGP);
  task->requires(Task::NewDW, d_lb->gMassLabel,               matlset, gnone);
  task->requires(Task::NewDW, d_rdlb->gConcentrationLabel,    matlset, gan, 2*NGN);

  task->computes(d_rdlb->pConcGradientLabel, matlset);
  task->computes(d_rdlb->pFluxLabel,         matlset);

}

void JGConcentrationDiffusion::computeFlux(const Patch* patch, const MPMMaterial* matl,
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

  constNCVariable<double>       gConcentration,gMass;

  ParticleVariable<Vector>      pConcGradient;
  ParticleVariable<Vector>      pFlux;

  //ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch, gan, NGP, d_lb->pXLabel);
  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

  old_dw->get(px,           d_lb->pXLabel,                         pset);
  old_dw->get(pvol,         d_lb->pVolumeLabel,                    pset);
  old_dw->get(pMass,        d_lb->pMassLabel,                      pset);
  old_dw->get(psize,        d_lb->pSizeLabel,                      pset);
  old_dw->get(deformationGradient, d_lb->pDeformationMeasureLabel, pset);

  new_dw->get(gConcentration,      d_rdlb->gConcentrationLabel, dwi, patch, gac,2*NGN);
  new_dw->get(gMass,               d_lb->gMassLabel,            dwi, patch, gnone, 0);

  //new_dw->allocateAndPut(pConcGradient, d_rdlb->pConcGradientLabel, pset);
  new_dw->allocateTemporary(pConcGradient, pset);
  new_dw->allocateAndPut(pFlux,         d_rdlb->pFluxLabel,         pset);
  
  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
    particleIndex idx = *iter;

    // Get the node indices that surround the cell
    interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

    pConcGradient[idx] = Vector(0.0,0.0,0.0);
    pFlux[idx]         = Vector(0.0,0.0,0.0);
    for (int k = 0; k < d_Mflag->d_8or27; k++){
      for (int j = 0; j<3; j++) {
          pConcGradient[idx][j] += gConcentration[ni[k]] * d_S[k][j] * oodx[j];
      }
	  }
    pFlux[idx] = diffusivity*pConcGradient[idx];
  } //End of Particle Loop

	delete interpolator;
}

void JGConcentrationDiffusion::scheduleComputeDivergence(Task* task, const MPMMaterial* matl, 
		                                                const PatchSet* patch) const
{
  Ghost::GhostType  gnone = Ghost::None;
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, d_sharedState->get_delt_label());
  task->requires(Task::OldDW, d_lb->pXLabel,                         gan, NGP);
  task->requires(Task::OldDW, d_lb->pSizeLabel,                      gan, NGP);
  task->requires(Task::OldDW, d_lb->pMassLabel,                      gan, NGP);
  task->requires(Task::OldDW, d_lb->pVolumeLabel,                    gan, NGP);
  task->requires(Task::OldDW, d_lb->pDeformationMeasureLabel,        gan, NGP);
  task->requires(Task::NewDW, d_lb->gMassLabel,                      gnone);

  task->requires(Task::NewDW, d_rdlb->gConcentrationLabel,     gnone);
  task->requires(Task::NewDW, d_rdlb->gConcentrationNoBCLabel, gnone);
  task->requires(Task::NewDW, d_rdlb->pFluxLabel,              gan, NGP);

  task->computes(d_rdlb->gConcentrationRateLabel, matlset);
  task->computes(d_rdlb->gConcentrationStarLabel, matlset);
}

void JGConcentrationDiffusion::computeDivergence(const Patch* patch, const MPMMaterial* matl,
                                            DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  Ghost::GhostType  gnone = Ghost::None;
  Ghost::GhostType  gac   = Ghost::AroundCells;
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  int dwi = matl->getDWIndex();

  ParticleInterpolator* interpolator = d_Mflag->d_interpolator->clone(patch); 
  vector<IntVector> ni(interpolator->size());
  vector<Vector> d_S(interpolator->size());

  Vector dx = patch->dCell();
  double oodx[3];
  oodx[0] = 1.0/dx.x();
  oodx[1] = 1.0/dx.y();
  oodx[2] = 1.0/dx.z();

  constParticleVariable<Point>  px;
  constParticleVariable<double> pvol,pMass;
  constParticleVariable<Matrix3> psize;
  constParticleVariable<Matrix3> deformationGradient;
  constParticleVariable<Vector> pFlux;
  constNCVariable<double> gMass;
  constNCVariable<double> gConc_Old;
  constNCVariable<double> gConc_OldNoBC;

  NCVariable<double> gConcRate;
  NCVariable<double> gConcStar;
  NCVariable<double> gdCdt;

  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch, gan, NGP, d_lb->pXLabel);

  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label(), patch->getLevel() );

  old_dw->get(px,                  d_lb->pXLabel,                  pset);
  old_dw->get(pvol,                d_lb->pVolumeLabel,             pset);
  old_dw->get(pMass,               d_lb->pMassLabel,               pset);
  old_dw->get(psize,               d_lb->pSizeLabel,               pset);
  old_dw->get(deformationGradient, d_lb->pDeformationMeasureLabel, pset);

  new_dw->get(gMass,         d_lb->gMassLabel,                dwi, patch,gnone,0);
  new_dw->get(gConc_Old,     d_rdlb->gConcentrationLabel,     dwi, patch,gnone,0);
  new_dw->get(gConc_OldNoBC, d_rdlb->gConcentrationNoBCLabel, dwi, patch,gnone,0);
  new_dw->get(pFlux,         d_rdlb->pFluxLabel,      pset);

  new_dw->allocateAndPut(gConcRate, d_rdlb->gConcentrationRateLabel, dwi,patch);
  new_dw->allocateAndPut(gConcStar, d_rdlb->gConcentrationStarLabel, dwi,patch);

  new_dw->allocateTemporary(gdCdt,     patch);


  gdCdt.initialize(0.0);
  gConcStar.initialize(0.0);
  gConcRate.initialize(0.0);

  for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
    particleIndex idx = *iter;
  
    // Get the node indices that surround the cell
    interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

    Vector J = pFlux[idx];
    double Cdot_cond = 0.0;
    IntVector node(0,0,0);

    for (int k = 0; k < d_Mflag->d_8or27; k++){
      node = ni[k];
      if(patch->containsNode(node)){
        Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],d_S[k].z()*oodx[2]);
        Cdot_cond = Dot(div, J)*pMass[idx];
        gdCdt[node] -= Cdot_cond;
      }
    }
  } // End of Particle Loop 

  for(NodeIterator iter=patch->getExtraNodeIterator();
                   !iter.done();iter++){
    IntVector c = *iter; 
    gdCdt[c]   /= gMass[c];
  }

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
}

void JGConcentrationDiffusion::scheduleInterpolateToParticlesAndUpdate(Task* task,
                                                                       const MPMMaterial* matl, 
		                                                                   const PatchSet* patch) const
{
  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  const MaterialSubset* matlset = matl->thisMaterial();


  task->requires(Task::OldDW, d_sharedState->get_delt_label());
  task->requires(Task::OldDW, d_lb->pXLabel,                         gnone);
  task->requires(Task::OldDW, d_lb->pSizeLabel,                      gnone);
  task->requires(Task::OldDW, d_lb->pDeformationMeasureLabel,        gnone);
  task->requires(Task::OldDW, d_rdlb->pConcentrationLabel,           gnone);
  task->requires(Task::NewDW, d_rdlb->gConcentrationRateLabel,       gac,NGN);

  task->computes(d_rdlb->pConcentrationLabel_preReloc, matlset);
  task->computes(d_rdlb->pConcPreviousLabel_preReloc, matlset);
}

void JGConcentrationDiffusion::interpolateToParticlesAndUpdate(const Patch* patch,
                                                               const MPMMaterial* matl,
                                                               DataWarehouse* old_dw,
																															 DataWarehouse* new_dw)
{
  Ghost::GhostType  gnone = Ghost::None;
  Ghost::GhostType  gac   = Ghost::AroundCells;
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  int dwi = matl->getDWIndex();

  ParticleInterpolator* interpolator = d_Mflag->d_interpolator->clone(patch);
  vector<IntVector> ni(interpolator->size());
  vector<double> S(interpolator->size());
  vector<Vector> d_S(interpolator->size());

  constParticleVariable<Point>   px;
  constParticleVariable<Matrix3> psize;
  constParticleVariable<Matrix3> pFOld;
  constParticleVariable<double>  pConcentration;
  constNCVariable<double>        gConcentrationRate;

  ParticleVariable<double> pConcentrationNew; 
  ParticleVariable<double> pConcPreviousNew; 

  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label(), patch->getLevel() );

  old_dw->get(px,             d_lb->pXLabel,                   pset);
  old_dw->get(psize,          d_lb->pSizeLabel,                pset);
  old_dw->get(pFOld,          d_lb->pDeformationMeasureLabel,  pset);

  old_dw->get(pConcentration,     d_rdlb->pConcentrationLabel,     pset);
  new_dw->get(gConcentrationRate, d_rdlb->gConcentrationRateLabel, dwi, patch, gac, NGP);

  new_dw->allocateAndPut(pConcentrationNew,  d_rdlb->pConcentrationLabel_preReloc, pset);
  new_dw->allocateAndPut(pConcPreviousNew,   d_rdlb->pConcPreviousLabel_preReloc,  pset);

  for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){

    particleIndex idx = *iter;
    interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],pFOld[idx]);
    double concRate = 0.0;
    for (int k = 0; k < d_Mflag->d_8or27; k++) {
      IntVector node = ni[k];
      concRate += gConcentrationRate[node]   * S[k];
    }

    pConcentrationNew[idx] = pConcentration[idx] + concRate*delT;
		//if(pConcentrationNew[idx] > max_concentration){
		//  pConcentrationNew[idx] = max_concentration;
		//}
    pConcPreviousNew[idx] = pConcentration[idx];
	}
	delete interpolator;
}



void JGConcentrationDiffusion::scheduleFinalParticleUpdate(Task* task,
                                                           const MPMMaterial* matl, 
		                                                       const PatchSet* patch) const
{
  Ghost::GhostType gnone = Ghost::None;
  const MaterialSubset* matlset = matl->thisMaterial();


  task->requires(Task::OldDW, d_sharedState->get_delt_label());

  task->modifies(d_rdlb->pConcentrationLabel_preReloc, matlset);
}

void JGConcentrationDiffusion::finalParticleUpdate(const Patch* patch,
                                                   const MPMMaterial* matl,
                                                   DataWarehouse* old_dw,
																						  		 DataWarehouse* new_dw)
{
  constParticleVariable<double> pdCdt;
  ParticleVariable<double> pConcNew;
  int dwi = matl->getDWIndex();

  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

  new_dw->getModifiable(pConcNew, d_rdlb->pConcentrationLabel_preReloc,pset);

  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label(), patch->getLevel() );

  for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
    particleIndex idx = *iter;
  }
}
