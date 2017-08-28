/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include <CCA/Components/MPM/Diffusion/DiffusionModels/ScalarDiffusionModel.h>

#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/AMR.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Util/DebugStream.h>
#include <Core/Grid/Variables/NodeIterator.h>

#include <iostream>
#include <CCA/Components/MPM/Diffusion/ConductivityModels/BinaryEquation.h>
#include <CCA/Components/MPM/Diffusion/ConductivityModels/FixedEquation.h>

using namespace Uintah;

static DebugStream cout_doing("AMRMPM", false);

ScalarDiffusionModel::ScalarDiffusionModel(
                                           ProblemSpecP     & ps,
                                           SimulationStateP & sS,
                                           MPMFlags         * Mflag,
                                           std::string        diff_type
                                          ) 
{
  d_Mflag = Mflag;
  d_sharedState = sS;

  d_lb = scinew MPMLabel;

  ps->require("diffusivity", d_D0);
  ps->require("max_concentration", d_MaxConcentration);
  ps->getWithDefault("min_concentration", d_MinConcentration, 0.0);
  ps->getWithDefault("conc_tolerance", d_concTolerance, 1e-10);
  ps->getWithDefault("initial_concentration", d_InitialConcentration, 0.0);

  d_InverseMaxConcentration = 1.0/d_MaxConcentration;

  if(d_Mflag->d_8or27==8)
  {
    NGP=1;
    NGN=1;
  }
  else
  {
    NGP=2;
    NGN=2;
  }

  diffusion_type = diff_type;
  include_hydrostress = false;

  d_one_matl = scinew MaterialSubset();
  d_one_matl->add(0);
  d_one_matl->addReference();

  d_conductivity_equation = 0;
  ProblemSpecP cond_eq_ps = ps->findBlock("conductivity_equation");
  if(cond_eq_ps){
    std::string equation_type;
    cond_eq_ps->getAttribute("type", equation_type);
    if(equation_type == "fixed"){
      d_conductivity_equation = scinew FixedEquation(cond_eq_ps);
    }else if(equation_type == "binary"){
      d_conductivity_equation = scinew BinaryEquation(cond_eq_ps);
    }else{
      d_conductivity_equation = scinew ConductivityEquation(cond_eq_ps);
    }
  }
}

ScalarDiffusionModel::~ScalarDiffusionModel() {
  delete d_lb;

  if (d_one_matl->removeReference())
    delete d_one_matl;

  if(d_conductivity_equation){
    delete d_conductivity_equation;
    d_conductivity_equation = 0;
  }
}

// Common functions for all diffusion models
std::string ScalarDiffusionModel::getDiffusionType() const
{
  return diffusion_type;
}

double ScalarDiffusionModel::getMaxConcentration() const
{
  return d_MaxConcentration;
}

double ScalarDiffusionModel::getMinConcentration() const
{
  return d_MinConcentration;
}

double ScalarDiffusionModel::getConcentrationTolerance() const
{
  return d_concTolerance;
}

void ScalarDiffusionModel::setIncludeHydroStress(bool value){
  include_hydrostress = value;
}

void ScalarDiffusionModel::initializeTimeStep(
                                              const Patch         * patch,
                                              const MPMMaterial   * matl,
                                                    DataWarehouse * new_dw
                                             )
{
  Vector dx = patch->dCell();
  double timestep = 1.0e99;
  timestep = std::min(timestep, computeStableTimeStep(d_D0, dx));

  new_dw->put(delt_vartype(timestep), d_lb->delTLabel, patch->getLevel());
}

void ScalarDiffusionModel::scheduleComputeDivergence(      Task         * task,
                                                     const MPMMaterial  * matl,
                                                     const PatchSet     * patch
                                                    ) const
{
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, d_sharedState->get_delt_label());
  task->requires(Task::OldDW, d_lb->pXLabel,                   gan, NGP);
  task->requires(Task::OldDW, d_lb->pSizeLabel,                gan, NGP);
  task->requires(Task::OldDW, d_lb->pMassLabel,                gan, NGP);
  task->requires(Task::OldDW, d_lb->pVolumeLabel,              gan, NGP);
  task->requires(Task::OldDW, d_lb->pDeformationMeasureLabel,  gan, NGP);

  task->requires(Task::NewDW, d_lb->diffusion->pFlux_preReloc, gan, NGP);

  task->computes(d_lb->diffusion->gConcentrationRate, matlset);
}

void ScalarDiffusionModel::computeDivergence(
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
  Vector oodx(1.0/dx.x(),1.0/dx.y(),1.0/dx.z());

  constParticleVariable<Point>    px;
  constParticleVariable<double>   pvol;
  constParticleVariable<double>   pMass;
  constParticleVariable<Matrix3>  psize;
  constParticleVariable<Matrix3>  deformationGradient;
  constParticleVariable<Vector>   pFlux;
  constNCVariable<double>         gConc_OldNoBC;
  NCVariable<double>              gConcRate;

  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch, gan, NGP,
                                                          d_lb->pXLabel);

  old_dw->get(px,                  d_lb->pXLabel,                  pset);
  old_dw->get(pvol,                d_lb->pVolumeLabel,             pset);
  old_dw->get(pMass,               d_lb->pMassLabel,               pset);
  old_dw->get(psize,               d_lb->pSizeLabel,               pset);
  old_dw->get(deformationGradient, d_lb->pDeformationMeasureLabel, pset);
  new_dw->get(pFlux,               d_lb->diffusion->pFlux_preReloc,      pset);

  new_dw->allocateAndPut(gConcRate,  d_lb->diffusion->gConcentrationRate,dwi,patch);

  gConcRate.initialize(0.0);

  // THIS IS COMPUTING A MASS WEIGHTED gConcRate.  THE DIVISION BY MASS, AND
  // SUBSEQUENT CALCULATIONS IS DONE IN computeAndIntegrateAcceleration. 

  for(ParticleSubset::iterator iter = pset->begin();
                               iter != pset->end(); iter++){
    particleIndex idx = *iter;
  
    // Get the node indices that surround the cell
    int NN = interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,
                                           psize[idx],deformationGradient[idx]);

    Vector J = pFlux[idx];
    double Cdot_cond = 0.0;
    IntVector node(0,0,0);

    for (int k = 0; k < NN; k++){
      node = ni[k];
      if(patch->containsNode(node)) {
        Vector div(d_S[k]*oodx);
//        Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],d_S[k].z()*oodx[2]);
        Cdot_cond = Dot(div, J)*pMass[idx];
//        Cdot_cond = Dot(div, J)/* *pMass[idx]*/;
        gConcRate[node] -= Cdot_cond;
      }
    }
  } // End of Particle Loop
}

void ScalarDiffusionModel::scheduleComputeDivergence_CFI(      Task         * t,
                                                         const MPMMaterial  * matl,
                                                         const PatchSet     * patch
                                                        ) const
{
    Ghost::GhostType  gac  = Ghost::AroundCells;
    Task::MaterialDomainSpec  ND  = Task::NormalDomain;
    const MaterialSubset* matlset = matl->thisMaterial();

    /*`==========TESTING==========*/
      // Linear 1 coarse Level cells:
      // Gimp:  2 coarse level cells:
     int npc =  1; // d_nPaddingCells_Coarse;
    /*===========TESTING==========`*/

    #define allPatches 0
    #define allMatls 0
    //__________________________________
    // Note: were using nPaddingCells to extract the region of coarse level
    // particles around every fine patch.   Technically, these are ghost
    // cells but somehow it works.
    t->requires(Task::NewDW, d_lb->gZOILabel,     d_one_matl, Ghost::None,0);
    t->requires(Task::OldDW, d_lb->pXLabel,       allPatches, Task::CoarseLevel,allMatls, ND, gac, npc);
    t->requires(Task::OldDW, d_lb->pSizeLabel,    allPatches, Task::CoarseLevel,allMatls, ND, gac, npc);
    t->requires(Task::OldDW, d_lb->pMassLabel,    allPatches, Task::CoarseLevel,allMatls, ND, gac, npc);
    t->requires(Task::OldDW, d_lb->pDeformationMeasureLabel, allPatches, Task::CoarseLevel,allMatls, ND, gac, npc);
    t->requires(Task::NewDW, d_lb->diffusion->pFlux_preReloc,      allPatches, Task::CoarseLevel,allMatls, ND, gac, npc);

    t->modifies(d_lb->diffusion->gConcentrationRate, matlset);
}

void ScalarDiffusionModel::computeDivergence_CFI(const PatchSubset    * finePatches,
                                                 const MPMMaterial    * matl,
                                                       DataWarehouse  * old_dw,
                                                       DataWarehouse  * new_dw
                                                )
{
  int dwi = matl->getDWIndex();

  const Level* fineLevel    = getLevel(finePatches);
  const Level* coarseLevel  = fineLevel->getCoarserLevel().get_rep();

  IntVector refineRatio(fineLevel->getRefinementRatio());

  for(int p=0; p<finePatches->size(); p++)
  {
    const Patch* finePatch = finePatches->get(p);
    printTask(finePatches, finePatch,cout_doing,
                        "Doing ScalarDiffusionModel::computeInternalForce_CFI");

    ParticleInterpolator* interpolator =
                                      d_Mflag->d_interpolator->clone(finePatch);

    //__________________________________
    //          AT CFI
    if( fineLevel->hasCoarserLevel() && finePatch->hasCoarseFaces() )
    {
      // Determine extents for coarser level particle data
      // Linear Interpolation:  1 layer of coarse level cells
      // Gimp Interpolation:    2 layers
  /*`==========TESTING==========*/
//      IntVector nLayers(d_nPaddingCells_Coarse,
//                        d_nPaddingCells_Coarse,
//                        d_nPaddingCells_Coarse );
      IntVector nLayers(1, 1, 1 );
      IntVector nPaddingCells = nLayers * (fineLevel->getRefinementRatio());
      //cout << " nPaddingCells " << nPaddingCells << "nLayers " << nLayers << endl;
  /*===========TESTING==========`*/

      int nGhostCells = 0;
      bool returnExclusiveRange=false;
      IntVector cl_tmp, ch_tmp, fl, fh;

      getCoarseLevelRange(finePatch, coarseLevel, cl_tmp, ch_tmp, fl, fh,
                          nPaddingCells, nGhostCells,returnExclusiveRange);

      //  expand cl_tmp when a neighor patch exists.
      //  This patch owns the low nodes.  You need particles
      //  from the neighbor patch.
      cl_tmp -= finePatch->neighborsLow() * nLayers;

      // find the coarse patches under the fine patch.  
      // You must add a single layer of padding cells.
      int padding = 1;
      Level::selectType coarsePatches;
      finePatch->getOtherLevelPatches(-1, coarsePatches, padding);

      Matrix3 Id;
      Id.Identity();

      constNCVariable<Stencil7> zoi_fine;
      new_dw->get(zoi_fine, d_lb->gZOILabel, 0, finePatch, Ghost::None, 0 );

      NCVariable<double> gConcRate;
      new_dw->getModifiable(gConcRate, d_lb->diffusion->gConcentrationRate,
                                                     dwi, finePatch);

      // loop over the coarse patches under the fine patches.
      for(int cp=0; cp<coarsePatches.size(); cp++)
      {
        const Patch* coarsePatch = coarsePatches[cp];

        // get coarse level particle data 
        ParticleSubset* pset_coarse;
        constParticleVariable<Point>  px_coarse;
        constParticleVariable<Vector> pflux_coarse;
        constParticleVariable<double> pmass_coarse;

        // coarseLow and coarseHigh cannot lie outside of the coarse patch
        IntVector cl = Max(cl_tmp, coarsePatch->getCellLowIndex());
        IntVector ch = Min(ch_tmp, coarsePatch->getCellHighIndex());

        pset_coarse = old_dw->getParticleSubset(dwi, cl, ch, coarsePatch,
                                                              d_lb->pXLabel);

        // coarse level data
        old_dw->get(px_coarse,      d_lb->pXLabel,              pset_coarse);
        old_dw->get(pmass_coarse,   d_lb->pMassLabel,           pset_coarse);
        new_dw->get(pflux_coarse,   d_lb->diffusion->pFlux_preReloc,  pset_coarse);

        for (ParticleSubset::iterator iter = pset_coarse->begin();
                                      iter != pset_coarse->end();  iter++)
        {
          particleIndex idx = *iter;

          std::vector<IntVector>   ni;
          std::vector<double>       S;
          std::vector<Vector>     div;
          interpolator->findCellAndWeightsAndShapeDerivatives_CFI(
                                       px_coarse[idx], ni, S, div, zoi_fine);

          IntVector fineNode;
          for(int k = 0; k < (int)ni.size(); k++)
          {
            fineNode = ni[k];
            if( finePatch->containsNode( fineNode ) )
            {
               double Cdot_cond = Dot(div[k], pflux_coarse[idx])
                                            * pmass_coarse[idx];
//               double Cdot_cond = Dot(div[k], pflux_coarse[idx]);
//                                    /*      * pmass_coarse[idx]; */
               gConcRate[fineNode] -= Cdot_cond;
            }  // contains node
          }  // node loop
        }  // pset loop
      }  // coarse Patch loop
    }  // patch has CFI faces
    delete interpolator;
  }  // End fine patch loop
}

double ScalarDiffusionModel::computeStableTimeStep(
                                                   double Dif,
                                                   Vector dx
                                                  ) const
{
  // For a Forward Euler timestep the limiting factor is
  // dt < dx^2 / 2*D.
  Vector timeStep(dx.x()*dx.x(), dx.y()*dx.y(), dx.z()*dx.z());
  timeStep = timeStep/(Dif*2);
  return timeStep.minComponent();
}

bool ScalarDiffusionModel::usesChemicalPotential()
{
  // Override if model uses chemical potential.
  return false;
}

void ScalarDiffusionModel::addChemPotentialComputesAndRequires(
                                                                     Task           * task,
                                                               const MPMMaterial  * matl,
                                                               const PatchSet     * patches
                                                              ) const
{
 // Don't override if model doesn't use chemical potential.
//  Ghost::GhostType        gnone   = Ghost::None;
//  const MaterialSubset  * matlset = matl->thisMaterial();
//
//  task->requires(Task::OldDW, d_lb->pChemicalPotentialLabel, matlset, gnone);
//  task->computes(d_lb->pChemicalPotentialLabel_preReloc, matlset);
}

void ScalarDiffusionModel::calculateChemicalPotential(
                                                      const PatchSubset   * patches,
                                                      const MPMMaterial   * matl,
                                                            DataWarehouse * old_dw,
                                                            DataWarehouse * new_dw
                                                     )
{
  // Don't override if model doesn't use chemical potential.
//  for (int patchIndex = 0; patchIndex < patches->size(); ++patchIndex)
//  {
//    const Patch* patch = patches->get(patchIndex);
//    int dwi = matl->getDWIndex();
//    ParticleSubset  * pset = old_dw->getParticleSubset(dwi, patch);
//
//    constParticleVariable<double> pChemicalPotential;
//    old_dw->get(pChemicalPotential, d_lb->pChemicalPotentialLabel, pset);
//    ParticleVariable<double> pChemicalPotential_new;
//    new_dw->allocateAndPut(pChemicalPotential_new,
//                           d_lb->pChemicalPotentialLabel_preReloc, pset);
//
//    int numParticles = pset->numParticles();
//    for (int particleIndex = 0; particleIndex < numParticles; ++particleIndex)
//    {
//      pChemicalPotential_new[particleIndex] = pChemicalPotential[particleIndex];
//    }
//  }

}

double ScalarDiffusionModel::computeDiffusivityTerm(double concentration, double pressure)
{
  // This is just a function stub to be tied into the multiscale
  // component. JH, AH, CG

  return d_D0;
}

ConductivityEquation* ScalarDiffusionModel::getConductivityEquation(){
  return d_conductivity_equation;
}

void ScalarDiffusionModel::baseInitializeSDMData(
                                                 const Patch         * patch,
                                                 const MPMMaterial   * matl,
                                                       DataWarehouse * NewDW
                                                )
{
  ParticleVariable<double> pConcentration;


  ParticleSubset* pset = NewDW->getParticleSubset(matl->getDWIndex(), patch);
  NewDW->getModifiable(pConcentration, d_lb->diffusion->pConcentration, pset);

  for (int pIndex = 0; pIndex < pset->numParticles(); ++pIndex)
    {
      pConcentration[pIndex] = d_InitialConcentration;
    }

}


void ScalarDiffusionModel::baseOutputSDMProbSpec(
                                                 ProblemSpecP & probSpec  ,
                                                 bool           /* do_output */
                                                ) const
{
  probSpec->appendElement("diffusivity",d_D0);
  probSpec->appendElement("max_concentration",d_MaxConcentration);
  probSpec->appendElement("min_concentration",d_MinConcentration);
  probSpec->appendElement("conc_tolerance", d_concTolerance);
  probSpec->appendElement("initial_concentration", d_InitialConcentration);

  return;
}
