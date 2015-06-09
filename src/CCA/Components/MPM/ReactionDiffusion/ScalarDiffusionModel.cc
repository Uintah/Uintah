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

#include <CCA/Components/MPM/ReactionDiffusion/ScalarDiffusionModel.h>
#include <CCA/Components/MPM/ReactionDiffusion/ReactionDiffusionLabel.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/MPMBoundCond.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/AMR.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
using namespace std;
using namespace Uintah;

static DebugStream cout_doing("AMRMPM", false);


ScalarDiffusionModel::ScalarDiffusionModel(ProblemSpecP& ps, SimulationStateP& sS, MPMFlags* Mflag, string diff_type)
{
  d_Mflag = Mflag;
  d_sharedState = sS;

  d_lb = scinew MPMLabel;
  d_rdlb = scinew ReactionDiffusionLabel();

  ps->require("diffusivity", diffusivity);
  ps->require("max_concentration", max_concentration);

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

  diffusion_type = diff_type;
  include_hydrostress = false;

  d_one_matl = scinew MaterialSubset();
  d_one_matl->add(0);
  d_one_matl->addReference();
}

ScalarDiffusionModel::~ScalarDiffusionModel() {
  delete d_lb;
  delete d_rdlb;

  if (d_one_matl->removeReference())
    delete d_one_matl;
}

string ScalarDiffusionModel::getDiffusionType(){
  return diffusion_type;
}

double ScalarDiffusionModel::getMaxConcentration(){
  return max_concentration;
}

void ScalarDiffusionModel::setIncludeHydroStress(bool value){
  include_hydrostress = value;
}

void ScalarDiffusionModel::addInitialComputesAndRequires(Task* task,
                                                         const MPMMaterial* matl,
                                                         const PatchSet* patch) const{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(d_rdlb->pConcentrationLabel, matlset);
  task->computes(d_rdlb->pConcPreviousLabel,  matlset);
}

void ScalarDiffusionModel::initializeSDMData(const Patch* patch,
                                             const MPMMaterial* matl,
                                             DataWarehouse* new_dw)
{
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<double>  pConcentration;
  ParticleVariable<double>  pConcPrevious;

  new_dw->allocateAndPut(pConcentration,  d_rdlb->pConcentrationLabel, pset);
  new_dw->allocateAndPut(pConcPrevious,   d_rdlb->pConcPreviousLabel,  pset);

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
    pConcentration[*iter] = 0.0;
    pConcPrevious[*iter]  = 0.0;
  }
}

void ScalarDiffusionModel::addParticleState(std::vector<const VarLabel*>& from,
                                            std::vector<const VarLabel*>& to)
{
  from.push_back(d_rdlb->pConcentrationLabel);
  from.push_back(d_rdlb->pConcPreviousLabel);

  to.push_back(d_rdlb->pConcentrationLabel_preReloc);
  to.push_back(d_rdlb->pConcPreviousLabel_preReloc);
}

void ScalarDiffusionModel::scheduleInterpolateParticlesToGrid(Task* task,
                                                   const MPMMaterial* matl,
                                                   const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType  gan = Ghost::AroundNodes;
//  Ghost::GhostType  gnone = Ghost::None;

  task->requires(Task::OldDW, d_lb->pXLabel,                 matlset, gan, NGP);
  task->requires(Task::OldDW, d_lb->pMassLabel,              matlset, gan, NGP);
  task->requires(Task::OldDW, d_lb->pSizeLabel,              matlset, gan, NGP);
  task->requires(Task::OldDW, d_lb->pDeformationMeasureLabel,matlset, gan, NGP);
  task->requires(Task::OldDW, d_lb->pStressLabel,            matlset, gan, NGP);
  task->requires(Task::OldDW, d_rdlb->pConcentrationLabel,   matlset, gan, NGP);
//task->requires(Task::NewDW, d_lb->gMassLabel,               matlset, gnone);

  task->computes(d_rdlb->gConcentrationLabel,      matlset);
  task->computes(d_rdlb->gHydrostaticStressLabel,  matlset);
  task->computes(d_rdlb->gConcentrationNoBCLabel,  matlset);
}

void ScalarDiffusionModel::interpolateParticlesToGrid(const Patch* patch,
                                                      const MPMMaterial* matl,
                                                      DataWarehouse* old_dw,
                                                      DataWarehouse* new_dw)
{
  Ghost::GhostType  gan = Ghost::AroundNodes;

  ParticleInterpolator* interpolator = d_Mflag->d_interpolator->clone(patch); 

  vector<IntVector> ni(interpolator->size());
  vector<double> S(interpolator->size());

  constParticleVariable<Point>  px;
  constParticleVariable<double> pmass;
  constParticleVariable<Matrix3> psize;
  constParticleVariable<Matrix3> pFOld;
  constParticleVariable<double> pConcentration;
  constParticleVariable<Matrix3> pStress;
//  constNCVariable<double>       gmass;

  int dwi = matl->getDWIndex();
  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch, gan, NGP,
	                                                 d_lb->pXLabel);

  old_dw->get(px,             d_lb->pXLabel,                  pset);
  old_dw->get(pmass,          d_lb->pMassLabel,               pset);
  old_dw->get(psize,          d_lb->pSizeLabel,               pset);
  old_dw->get(pFOld,          d_lb->pDeformationMeasureLabel, pset);
  old_dw->get(pConcentration, d_rdlb->pConcentrationLabel,    pset);
  old_dw->get(pStress,        d_lb->pStressLabel,             pset);

  NCVariable<double> gconcentration;
  NCVariable<double> gconcentrationNoBC;
  NCVariable<double> ghydrostaticstress;

  new_dw->allocateAndPut(gconcentration,      d_rdlb->gConcentrationLabel,
	                       dwi,  patch);
  new_dw->allocateAndPut(gconcentrationNoBC,  d_rdlb->gConcentrationNoBCLabel,
	                       dwi,  patch);
  new_dw->allocateAndPut(ghydrostaticstress,  d_rdlb->gHydrostaticStressLabel,
	                       dwi,  patch);

  gconcentration.initialize(0);
  gconcentrationNoBC.initialize(0);
  ghydrostaticstress.initialize(0);

  int n8or27 = d_Mflag->d_8or27;
  double one_third = 1./3.;
  for (ParticleSubset::iterator iter  = pset->begin(); 
                                iter != pset->end(); iter++){
    particleIndex idx = *iter;

    interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],pFOld[idx]);
    double phydrostress = one_third*pStress[idx].Trace();

    IntVector node;
    for(int k = 0; k < n8or27; k++) {
      node = ni[k];
      if(patch->containsNode(node)) {
        ghydrostaticstress[node] += phydrostress        * pmass[idx] * S[k];
        gconcentration[node]     += pConcentration[idx] * pmass[idx] * S[k];
      }
    }
  }

  // Mass Normalization takes place in 
  // ScalarDiffusion::MassNormalizeConcentration() task

  delete interpolator;
}

#if 0
//  You need particle data from the coarse levels at the CFI on the fine level
void ScalarDiffusionModel::scheduleInterpolateParticlesToGrid_CFI(Task* t,
                                                    const MPMMaterial* matl,
                                                    const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType  gac  = Ghost::AroundCells;
  Task::MaterialDomainSpec  ND  = Task::NormalDomain;

/*`==========TESTING==========*/
    // Linear 1 coarse Level cells:
    // Gimp:  2 coarse level cells:
//    int npc = d_nPaddingCells_Coarse;
    int npc = 1;  // For now...
/*===========TESTING==========`*/

  #define allPatches 0
  //__________________________________
  // Note: were using nPaddingCells to extract the region of coarse level
  // particles around every fine patch.   Technically, these are ghost
  // cells but somehow it works.
  t->requires(Task::NewDW, d_lb->gZOILabel,                d_one_matl,  Ghost::None, 0);
  t->requires(Task::OldDW, d_lb->pXLabel,                  allPatches, Task::CoarseLevel,matlset, ND, gac, npc);
  t->requires(Task::OldDW, d_lb->pMassLabel,               allPatches, Task::CoarseLevel,matlset, ND, gac, npc);
  t->requires(Task::OldDW, d_rdlb->pConcentrationLabel,    allPatches, Task::CoarseLevel,matlset, ND, gac, npc);
  t->requires(Task::OldDW, d_lb->pStressLabel,             allPatches, Task::CoarseLevel,matlset, ND, gac, npc);
//  t->requires(Task::OldDW, d_lb->pDeformationMeasureLabel, allPatches, Task::CoarseLevel,matlset, ND, gac, npc);
//  t->requires(Task::OldDW, d_lb->pSizeLabel,               allPatches, Task::CoarseLevel,matlset, ND, gac, npc);

  t->modifies(d_rdlb->gConcentrationLabel,     matlset);
  t->modifies(d_rdlb->gHydrostaticStressLabel, matlset);
}

void ScalarDiffusionModel::interpolateParticlesToGrid_CFI(
                                                 const PatchSubset* finePatches,
                                                 const MPMMaterial* mpm_matl,
                                                 DataWarehouse* old_dw,
                                                 DataWarehouse* new_dw)
{
  const Level* fineLevel = getLevel(finePatches);
  const Level* coarseLevel = fineLevel->getCoarserLevel().get_rep();
  IntVector refineRatio(fineLevel->getRefinementRatio());

  for(int fp=0; fp<finePatches->size(); fp++){
    const Patch* finePatch = finePatches->get(fp);
    printTask(finePatches,finePatch,cout_doing,
                       "Doing ScalarDiffusion::interpolateParticlesToGrid_CFI");

    ParticleInterpolator* interpolator = 
                                      d_Mflag->d_interpolator->clone(finePatch);

    constNCVariable<Stencil7> zoi_fine;
    new_dw->get(zoi_fine, d_lb->gZOILabel, 0, finePatch, Ghost::None, 0 );

    // Determine extents for coarser level particle data
    // Linear Interpolation:  1 layer of coarse level cells
    // Gimp Interpolation:    2 layers
/*==========TESTING==========*/
//    IntVector nLayers(d_nPadCellsCoarse,d_nPadCellsCoarse,d_nPadCellsCoarse);
    IntVector nLayers(1, 1, 1);  // For now... JG
    IntVector nPaddingCells = nLayers * (fineLevel->getRefinementRatio());
/*===========TESTING==========*/

    int nGhostCells = 0;
    bool returnExclusiveRange=false;
    IntVector cl_tmp, ch_tmp, fl, fh;

    getCoarseLevelRange(finePatch, coarseLevel, cl_tmp, ch_tmp, fl, fh,
                        nPaddingCells, nGhostCells,returnExclusiveRange);

    // expand cl_tmp when a neighor patch exists.
    // This patch owns the low nodes.
    // You need particles from the neighbor patch.
    cl_tmp -= finePatch->neighborsLow() * nLayers;

    // find the coarse patches under the fine patch.
    // You must add a single layer of padding cells.
    int padding = 1;
    Level::selectType coarsePatches;
    finePatch->getOtherLevelPatches(-1, coarsePatches, padding);

    int dwi = mpm_matl->getDWIndex();

    // get fine level nodal data
    NCVariable<double> gMass_fine;
    NCVariable<double> gConc_fine;
    NCVariable<double> gHStress_fine;

    new_dw->getModifiable(gMass_fine,d_lb->gMassLabel,           dwi,finePatch);
    new_dw->getModifiable(gConc_fine,d_rdlb->gConcentrationLabel,dwi,finePatch);
    new_dw->getModifiable(gHStress_fine,
                                 d_rdlb->gHydrostaticStressLabel,dwi,finePatch);

    // loop over the coarse patches under the fine patches.
    for(int cp=0; cp<coarsePatches.size(); cp++){
      const Patch* coarsePatch = coarsePatches[cp];

      // get coarse level particle data
      constParticleVariable<Point>  pX_coarse;
      constParticleVariable<double> pMass_coarse;
      constParticleVariable<double> pConc_coarse;
      constParticleVariable<Matrix3> pStress_coarse;
//    constParticleVariable<Matrix3> pDefMeasure_coarse;

      // coarseLow and coarseHigh cannot lie outside of the coarse patch
      IntVector cl = Max(cl_tmp, coarsePatch->getCellLowIndex());
      IntVector ch = Min(ch_tmp, coarsePatch->getCellHighIndex());

      ParticleSubset* pset=0;

      pset = old_dw->getParticleSubset(dwi, cl, ch,coarsePatch,d_lb->pXLabel);
      old_dw->get(pX_coarse,            d_lb->pXLabel,                  pset);
      old_dw->get(pMass_coarse,         d_lb->pMassLabel,               pset);
      old_dw->get(pConc_coarse,         d_rdlb->pConcentrationLabel,    pset);
      old_dw->get(pStress_coarse,       d_lb->pStressLabel,             pset);
//    old_dw->get(pDefMeasure_coarse,   d_lb->pDeformationMeasureLabel, pset);

      double one_third = 1./3.;
      for (ParticleSubset::iterator iter  = pset->begin();
                                    iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the fine patch cell
        vector<IntVector> ni;
        vector<double> S;

        double ConcMass         = pConc_coarse[idx]*pMass_coarse[idx];
        double phydrostressmass = one_third*pStress_coarse[idx].Trace()
                                           *pMass_coarse[idx];

        interpolator->findCellAndWeights_CFI(pX_coarse[idx],ni,S,zoi_fine);

        // Add each particle's contribution to the local mass & velocity 
        IntVector fineNode;
        for(int k = 0; k < (int) ni.size(); k++) {
          fineNode = ni[k];
          gConc_fine[fineNode]       += ConcMass         * S[k];
          gHStress_fine[fineNode]    += phydrostressmass * S[k];
        }
      }  // End of particle loop
    }  // loop over coarse patches
    delete interpolator;
  }  // End loop over fine patches
}
#endif

void ScalarDiffusionModel::scheduleComputeFlux(Task* task,
                                               const MPMMaterial* matl, 
		                               const PatchSet* patch) const
{
}

void ScalarDiffusionModel::computeFlux(const Patch* patch,
                                       const MPMMaterial* matl,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
}

void ScalarDiffusionModel::scheduleComputeDivergence(Task* task, 
                                                     const MPMMaterial* matl, 
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

void ScalarDiffusionModel::computeDivergence(const Patch* patch,
                                             const MPMMaterial* matl,
                                             DataWarehouse* old_dw, 
                                             DataWarehouse* new_dw)
{
  Ghost::GhostType  gnone = Ghost::None;
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
  new_dw->get(pFlux,         d_rdlb->pFluxLabel,              pset);

  new_dw->allocateAndPut(gConcRate,  d_rdlb->gConcentrationRateLabel, dwi,patch);
  new_dw->allocateAndPut(gConcStar,  d_rdlb->gConcentrationStarLabel, dwi,patch);

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

  bc.setBoundaryCondition(patch, dwi,"SD-Type", gConcStar, d_Mflag->d_interpolator_type);

  for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
    IntVector c = *iter;
    gConcRate[c] = (gConcStar[c] - gConc_OldNoBC[c]) / delT;
  }
}

#if 0
void ScalarDiffusionModel::scheduleInterpolateToParticlesAndUpdate(Task* task,
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
  task->computes(d_rdlb->pConcPreviousLabel_preReloc,  matlset);
}

void ScalarDiffusionModel::interpolateToParticlesAndUpdate(const Patch* patch,
                                                        const MPMMaterial* matl,
                                                        DataWarehouse* old_dw,
                                                        DataWarehouse* new_dw)
{
  Ghost::GhostType  gac   = Ghost::AroundCells;
  int dwi = matl->getDWIndex();

  ParticleInterpolator* interpolator = d_Mflag->d_interpolator->clone(patch);
  vector<IntVector> ni(interpolator->size());
  vector<double> S(interpolator->size());

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
    pConcentrationNew[idx] = min(pConcentrationNew[idx],max_concentration);
    pConcPreviousNew[idx]  = pConcentration[idx];
  }
  delete interpolator;
}
#endif
