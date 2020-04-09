/*
 * BazantModel.cc
 *
 *  Created on: Jan 8, 2017
 *      Author: jbhooper
 *
 *
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/MPM/Diffusion/DiffusionModels/BazantModel.h>

#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <vector>
#include <random>

namespace Uintah
{
  BazantDiffusion::BazantDiffusion(
                                   ProblemSpecP     & ps,
                                   SimulationStateP & simState,
                                   MPMFlags         * mFlag,
                                   std::string        diff_type
                                  )
                                  : ScalarDiffusionModel(
                                                         ps,
                                                         simState,
                                                         mFlag,
                                                         diff_type
                                                        )
  {
    d_mismatchMagnitude = 0.0;
    d_includeMismatch = false;
    ps->require("BoltzmannConstant",d_unitBoltzmann);
    ps->require("regularSolutionParameter", d_regSolnParam);
    ps->require("penaltyCahnHilliard", d_CahnHilliardGradPenalty);
    ps->require("intercolantSiteDensity", d_intercSiteDensity);
    ps->require("molecularWeight", d_molWeight);

    ps->getWithDefault("externalChemicalPotential", d_muOther, 0.0);
    ProblemSpecP mismatchPS = ps->findBlock("MismatchStrain");
    if (mismatchPS)
    {
      // This is a clunky way to read a vector; I should fix the parser to take MATRIX. -- JBH
      d_includeMismatch = true;
      Vector misfitVector1, misfitVector2, misfitVector3;
      mismatchPS->require("v1", misfitVector1);
      mismatchPS->require("v2", misfitVector2);
      mismatchPS->require("v3", misfitVector3);
      d_latticeMisfit = Matrix3(misfitVector1.x(), misfitVector1.y(), misfitVector1.z(),
                                misfitVector2.x(), misfitVector2.y(), misfitVector2.z(),
                                misfitVector3.x(), misfitVector3.y(), misfitVector3.z());
    }
    concNormalized = false;


  }

  BazantDiffusion::~BazantDiffusion()
  {

  }

  void BazantDiffusion::addInitialComputesAndRequires(
                                                            Task    * task,
                                                      const MPMMaterial * matl,
                                                      const PatchSet    * patches
                                                     ) const
  {
    const MaterialSubset  * matlset = matl->thisMaterial();
    task->computes(d_lb->pFluxLabel,                        matlset);
  }

  void BazantDiffusion::addParticleState(
                                         std::vector<const VarLabel*> & from,
                                         std::vector<const VarLabel*> & to
                                        ) const
  {
    from.push_back(d_lb->pFluxLabel);
//    from.push_back(d_lb->pChemicalPotentialLabel);
 //   from.push_back(d_lb->pChemPotentialGradientLabel);

    to.push_back(d_lb->pFluxLabel_preReloc);
 //   to.push_back(d_lb->pChemicalPotentialLabel_preReloc);
 //   to.push_back(d_lb->pChemPotentialGradientLabel_preReloc);
  }

  void BazantDiffusion::computeFlux(
                                    const Patch         * patch,
                                    const MPMMaterial   * matl,
                                          DataWarehouse * OldDW,
                                          DataWarehouse * NewDW
                                   )
  {
    ParticleInterpolator  * interpolator = d_Mflag->d_interpolator->clone(patch);
    std::vector<IntVector>  ni(interpolator->size());
    std::vector<Vector>     d_S(interpolator->size());

    Vector  dx  = patch->dCell();
    int     dwi = matl->getDWIndex();

    constParticleVariable<double> pConcentration, pChemPotential, pTemperature;
    constParticleVariable<Vector> pGradConcentration, pGradChemPotential;

    ParticleSubset* pset = OldDW->getParticleSubset(dwi, patch);

    OldDW->get(pConcentration,      d_lb->pConcentrationLabel,              pset);
    OldDW->get(pGradConcentration,  d_lb->pConcGradientLabel,               pset);
    OldDW->get(pTemperature,        d_lb->pTemperatureLabel,                pset);

    NewDW->get(pGradChemPotential,  d_lb->pChemicalPotentialGradientLabel,  pset);

    ParticleVariable<Vector> pFluxNew;
    NewDW->allocateAndPut(pFluxNew, d_lb->pFluxLabel_preReloc,        pset);

    double maxFlux = 1e-25;
    for (int i = 0; i < pset->numParticles(); ++i )
    {
      // Flux calculation goes here.
      Vector concTerm = pGradConcentration[i];
      double thermalInv = 1.0/(pTemperature[i]*d_unitBoltzmann);
      Vector muTerm = thermalInv * pConcentration[i] * pGradChemPotential[i];
      pFluxNew[i] = d_D0 * (concTerm + muTerm);
      double fluxRate = pFluxNew[i].maxComponentMag();
      if (fluxRate > maxFlux) maxFlux = fluxRate;
    }

    double delT_local = computeStableTimeStep(maxFlux, dx);
    NewDW->put(delt_vartype(delT_local), d_lb->delTLabel, patch->getLevel());
  }

  void BazantDiffusion::initializeSDMData(
                                          const Patch          * patch,
                                          const MPMMaterial    * matl,
                                                DataWarehouse  * NewDW
                                         )
  {
    ParticleVariable<Vector> pFlux;
    ParticleVariable<double> pConcentration, pChemicalPotentialLabel;

    ParticleSubset* pset = NewDW->getParticleSubset(matl->getDWIndex(), patch);

    NewDW->allocateAndPut(pFlux,
                          d_lb->pFluxLabel                  , pset);

    NewDW->getModifiable(pConcentration, d_lb->pConcentrationLabel, pset);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normalDist(0.0, 0.005);
    std::uniform_real_distribution<> uniformDist(0.0, 1.0);

    for (int pIndex = 0; pIndex < pset->numParticles(); ++pIndex)
    {
      pFlux[pIndex] = Vector(0.0);
      double conc = pConcentration[pIndex];
//      if (uniformDist(gen) < 0.01)
//      {
        conc += normalDist(gen);
//      }
      if (conc < 0.0) conc = 0.0;
      if (conc > d_MaxConcentration) conc = d_MaxConcentration;
      pConcentration[pIndex] = 1.0;
      pConcentration[pIndex] = conc;
    }

  }

  void BazantDiffusion::scheduleComputeFlux(
                                                  Task        * task,
                                            const MPMMaterial * matl,
                                            const PatchSet    * patch
                                           ) const
  {
    const MaterialSubset *  matlset = matl->thisMaterial();
    Ghost::GhostType        gnone = Ghost::None;

    task->requires(Task::OldDW, d_lb->pConcentrationLabel, matlset, gnone);
    task->requires(Task::OldDW, d_lb->pConcGradientLabel, matlset, gnone);
//    task->requires(Task::NewDW, d_lb->pChemicalPotentialLabel, matlset, gnone);
    task->requires(Task::NewDW, d_lb->pChemicalPotentialGradientLabel, matlset, gnone);

    task->computes(d_lb->pFluxLabel_preReloc, matlset);
    task->computes(d_sharedState->get_delt_label(), getLevel(patch));
  }

  void BazantDiffusion::addSplitParticlesComputesAndRequires(
                                                                   Task         * task,
                                                             const MPMMaterial  * matl,
                                                             const PatchSet     * patches
                                                            ) const
  {

  }

  void BazantDiffusion::splitSDMSpecificParticleData(
                                                     const  Patch                 * patch,
                                                     const  int                     dwi,
                                                     const  int                     nDims,
                                                            ParticleVariable<int> & prefOld,
                                                            ParticleVariable<int> & pref,
                                                     const  unsigned int            oldNumPar,
                                                     const  int                     numNewPartNeeded,
                                                            DataWarehouse         * OldDW,
                                                            DataWarehouse         * NewDW
                                                    )
  {

  }

  void BazantDiffusion::outputProblemSpec(
                                          ProblemSpecP  & ps,
                                          bool            output_rdm_tag
                                         ) const
  {
    ProblemSpecP rdm_ps = ps;
    if (output_rdm_tag)
    {
      rdm_ps = ps->appendChild("diffusion_model");
      rdm_ps->setAttribute("type","bazant");
    }
    rdm_ps->appendElement("diffusivity",d_D0);
    rdm_ps->appendElement("max_concentration",d_MaxConcentration);

    rdm_ps->appendElement("BoltzmannConstant", d_unitBoltzmann);
    rdm_ps->appendElement("regularSolutionParameter", d_regSolnParam);
    rdm_ps->appendElement("penaltyCahnHilliard", d_CahnHilliardGradPenalty);
    rdm_ps->appendElement("intercolantSiteDensity", d_intercSiteDensity);
    rdm_ps->appendElement("externalChemicalPotential", d_muOther);
    rdm_ps->appendElement("molecularWeight", d_molWeight);
//    rdm_ps->appendElement("sitesPerVolume", d_diffusionSitesPerVolume);
//    rdm_ps->appendElement("volPerIntercSite", d_volPerSite);

    if (d_includeMismatch)
    {
      ProblemSpecP mm_ps = rdm_ps->appendChild("mismatchMatrix");
      mm_ps->appendElement("magnitude",d_mismatchMagnitude);
      Vector misfit(d_latticeMisfit(0,0),d_latticeMisfit(0,1),d_latticeMisfit(0,2));
      mm_ps->appendElement("v1",misfit);
      Vector misfit2(d_latticeMisfit(1,0),d_latticeMisfit(1,1),d_latticeMisfit(1,2));
      mm_ps->appendElement("v2",misfit2);
      Vector misfit3(d_latticeMisfit(2,0),d_latticeMisfit(2,1),d_latticeMisfit(2,2));
      mm_ps->appendElement("v3",misfit3);
    }
  }

//  void BazantDiffusion::scheduleComputeDivergence(
//                                                        Task        * task,
//                                                  const MPMMaterial * matl,
//                                                  const PatchSet    * patches
//                                                 ) const
//  {
//
//  }
//
//  void BazantDiffusion::computeDivergence(
//                                          const Patch         * patch,
//                                          const MPMMaterial   * matl,
//                                                DataWarehouse * OldDW,
//                                                DataWarehouse * NewDW
//                                         )
//  {
//
//  }

  bool BazantDiffusion::usesChemicalPotential() const
  {
    return true;
  }

  void BazantDiffusion::addChemPotentialComputesAndRequires(
                                                                  Task        * task,
                                                            const MPMMaterial * matl,
                                                            const PatchSet    * patches
                                                           ) const
  {
    Ghost::GhostType gac = Ghost::AroundCells;
    Ghost::GhostType gan = Ghost::AroundNodes;
    const MaterialSubset *  matlset = matl->thisMaterial();

    task->requires(Task::OldDW, d_lb->pConcentrationLabel,      matlset, gac, NGP);
    task->requires(Task::OldDW, d_lb->pConcGradientLabel,       matlset, gac, NGP);
    task->requires(Task::OldDW, d_lb->pStressLabel,             matlset, gac, NGP);
    task->requires(Task::OldDW, d_lb->pXLabel,                  matlset, gac, NGP);
    task->requires(Task::OldDW, d_lb->pSizeLabel,               matlset, gac, NGP);
    task->requires(Task::OldDW, d_lb->pDeformationMeasureLabel, matlset, gac, NGP);
    task->requires(Task::OldDW, d_lb->pTemperatureLabel,        matlset, gac, NGP);
    task->requires(Task::OldDW, d_lb->pMassLabel,               matlset, gac, NGP);

    task->requires(Task::NewDW, d_lb->gMassLabel,               matlset, gan, NGN);

    // Need strain eventually
     // task->requires(Task::OldDW, d_lb->pStrainLabel, matlset, gac, NGP);
    // Need potential at some point too...
     // task->requires(Task::OldDW, d_lb->pElectroPotential, matlset, gac, NGP);

//    task->computes(d_lb->gChemicalPotentialLabel, matlset);
    task->computes(d_lb->pChemicalPotentialLabel, matlset);
    task->computes(d_lb->pChemicalPotentialGradientLabel, matlset);
  }

  inline double BazantDiffusion::homogeneousMu(const double & conc ,
                                               const double & T    )
  {
    return (d_regSolnParam * (1.0 - 2.0 * conc) +
            d_unitBoltzmann * T * (std::log(conc) - std::log(1.0 - conc)));
  }

  double BazantDiffusion::isoCahnHilliardMu(const Vector                  & concGrad    ,
                                            const Vector                  & oodx        ,
                                            const int                       numNodes    ,
                                            const std::vector<IntVector>  & nodeIndices ,
                                            const std::vector<Vector>     & shapeDeriv  ,
                                            const Patch*                    patch       )
  {
    double mu = 0.0;
    IntVector node;
    for (int nIdx = 0; nIdx < numNodes; ++nIdx)
    {
      node = nodeIndices[nIdx];
      if(patch->containsNode(node)) {
        Vector div(shapeDeriv[nIdx]*oodx);
        mu += Dot(div,concGrad);
      }
    }
    return (mu);
  }

  void BazantDiffusion::calculateChemicalPotentialTake3(
                                                        const PatchSubset   * patches ,
                                                        const MPMMaterial   * matl    ,
                                                              DataWarehouse * oldDW   ,
                                                              DataWarehouse * newDW
                                                       )
  {
    Ghost::GhostType gan = Ghost::AroundNodes;
    Ghost::GhostType gap = Ghost::AroundCells;

    double prefactCahnHilliard = d_CahnHilliardGradPenalty/d_intercSiteDensity;

    for (int patchIdx = 0; patchIdx < patches->size(); ++patchIdx)
    {
      const Patch*          patch   = patches->get(patchIdx);
      ParticleInterpolator* pInterp = d_Mflag->d_interpolator->clone(patch);
      int interpPoints = pInterp->size();

      std::vector<IntVector> nodeList(interpPoints);
      std::vector<double>           S(interpPoints);
      std::vector<Vector>         d_S(interpPoints);

      Vector   dx = patch->dCell();
      Vector oodx = dx.inverse();

      int dwi = matl->getDWIndex();
      ParticleSubset* pSetGhost       = oldDW->getParticleSubset(dwi, patch, gap, NGP, d_lb->pXLabel);
      ParticleSubset* pSetNoGhost     = oldDW->getParticleSubset(dwi, patch, Ghost::None, 0, d_lb->pXLabel);
//      // These two -should- get the low/high index for the nodes to which the
//      // ghost particle set would interpolate onto for this patch, including the U/R
//      // edge which is normally excluded and which we need.
//      IntVector       ghostLowIndex   = pSetGhost->getLow();
//      IntVector       ghostHighIndex  = pSetGhost->getHigh();

      constParticleVariable<double>   pConcentration, pTemperature, pMass;
      constParticleVariable<Vector>   pConcGradient;
      constParticleVariable<Point>    pX;
      constParticleVariable<Matrix3>  pSize, pDefGrad;

      // Get outer particle set variables to interpolate to nodal values.
      oldDW->get(pConcentration, d_lb->pConcentrationLabel,       pSetGhost);
      oldDW->get(pConcGradient,  d_lb->pConcGradientLabel,        pSetGhost);
      oldDW->get(pX,             d_lb->pXLabel,                   pSetGhost);
      oldDW->get(pSize,          d_lb->pSizeLabel,                pSetGhost);
      oldDW->get(pDefGrad,       d_lb->pDeformationMeasureLabel,  pSetGhost);
      oldDW->get(pTemperature,   d_lb->pTemperatureLabel,         pSetGhost);
      oldDW->get(pMass,          d_lb->pMassLabel,                pSetGhost);

      // For phase-field strain
      constParticleVariable<Matrix3> pStress, pStrain;
      oldDW->get(pStress,        d_lb->pStressLabel,              pSetGhost);
      // oldDW->get(pStrain, d_lb->pStrainLabel, pSetGhost);

      // For coulombic potential
      constParticleVariable<double>  pCoulombPotential;







//      ParticleSubset* pSetNoGhost = oldDW->getParticleSubset(dwi, patch, gap, 0, d_lb->pXLabel);

    }
  }

  void BazantDiffusion::calculateChemicalPotential(
                                                         const PatchSubset    * patches,
                                                         const MPMMaterial    * matl,
                                                               DataWarehouse  * OldDW,
                                                               DataWarehouse  * NewDW
                                                        )
  {
    Ghost::GhostType gac = Ghost::AroundCells;
    Ghost::GhostType gan = Ghost::AroundNodes;

    double prefactCahnHilliard = d_CahnHilliardGradPenalty/d_intercSiteDensity;

    for (int patchIndex = 0; patchIndex < patches->size(); ++patchIndex)
    {
      const Patch* patch = patches->get(patchIndex);

      ParticleInterpolator* pInterp = d_Mflag->d_interpolator->clone(patch);
      int interpSize = pInterp->size();

      std::vector<IntVector> nodeIndices(interpSize);
      std::vector<double>    S(interpSize);
      std::vector<Vector>    d_S(interpSize);

      Vector dx = patch->dCell();
      Vector oodx = dx.inverse();

      int dwi = matl->getDWIndex();

      ParticleSubset* pset_ghost    = OldDW->getParticleSubset(dwi, patch, gac, NGP, d_lb->pXLabel);
      ParticleSubset* pset_noghost  = OldDW->getParticleSubset(dwi, patch, Ghost::None, 0, d_lb->pXLabel);
//      ParticleSubset* pset_noghost  = OldDW->getParticleSubset(dwi, patch, patch->getCellLowIndex(), patch->getCellHighIndex());

      constParticleVariable<double>   pConcentration, pTemperature, pMass;
      constParticleVariable<Vector>   pConcGradient;
      constParticleVariable<Point>    pX;
      constParticleVariable<Matrix3>  pSize, pDefGrad;

      OldDW->get(pConcentration,  d_lb->pConcentrationLabel,      pset_ghost);
      OldDW->get(pConcGradient,   d_lb->pConcGradientLabel,       pset_ghost);
      OldDW->get(pX,              d_lb->pXLabel,                  pset_ghost);
      OldDW->get(pSize,           d_lb->pSizeLabel,               pset_ghost);
      OldDW->get(pDefGrad,        d_lb->pDeformationMeasureLabel, pset_ghost);
      OldDW->get(pTemperature,    d_lb->pTemperatureLabel,        pset_ghost);
      OldDW->get(pMass,           d_lb->pMassLabel,               pset_ghost);

      // For phase-field strain
      constParticleVariable<Matrix3> pStress;
      OldDW->get(pStress,         d_lb->pStressLabel,             pset_ghost);
      constParticleVariable<Matrix3> pStrain;

      // For coulombic potential
      constParticleVariable<double>  pElectroPotential;

      // Grab the global mass grid with the proper extents for the calculations
      constNCVariable<double>  gMassGlobal;
      NewDW->get(gMassGlobal, d_lb->gMassLabel, dwi, patch, Ghost::AroundNodes, NGN);

      NCVariable<double> gridMassLocal;
      NCVariable<double> gridMu;
      NewDW->allocateTemporary(gridMassLocal,   patch, Ghost::AroundNodes, NGN);
      NewDW->allocateTemporary(gridMu,          patch, Ghost::AroundNodes, NGN);
      gridMassLocal.initialize(0.0);
      gridMu.initialize(0.0);


      ParticleVariable<double> pMu;
      NewDW->allocateAndPut(pMu, d_lb->pChemicalPotentialLabel, pset_noghost );

      Matrix3 mIdentity(1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0);

      // loop for non-ghost particles
      for (int pIdx = 0; pIdx < pset_noghost->numParticles(); ++pIdx)
      {
        pInterp->findCellAndWeightsAndShapeDerivatives(pX[pIdx], nodeIndices, S, d_S, pSize[pIdx], pDefGrad[pIdx]);

        double conc     = pConcentration[pIdx];
        Vector gradConc = pConcGradient[pIdx];

        if (!concNormalized) {
          conc      *= d_InverseMaxConcentration;
          gradConc  *= d_InverseMaxConcentration;
        }

        double mu_homogeneous = homogeneousMu(conc,pTemperature[pIdx]);
        Vector penaltyCH = mIdentity * gradConc;
        double mu_CahnHilliardIso = isoCahnHilliardMu(penaltyCH, oodx, interpSize, nodeIndices, d_S, patch);
        mu_CahnHilliardIso *= prefactCahnHilliard;
        double mu_total = mu_homogeneous - mu_CahnHilliardIso + d_muOther;
        pMu[pIdx] = mu_total;
        IntVector node;
        for (int pNode = 0; pNode < interpSize; ++pNode) {
          node = nodeIndices[pNode];
          if (patch->containsNode(node)) {
            double massWeightedShape = pMass[pIdx] * S[pNode];
            gridMu[node]        += (massWeightedShape * mu_total);
            gridMassLocal[node] += massWeightedShape;
          }
        }
      }

      // loop for ghost particles
      for (int pIdx = 0; pIdx < pset_ghost->numParticles(); ++pIdx)
      {
        if (!(patch->containsPoint(pX[pIdx])))
        {
          pInterp->findCellAndWeightsAndShapeDerivatives(pX[pIdx], nodeIndices, S, d_S, pSize[pIdx], pDefGrad[pIdx]);

          double conc     = pConcentration[pIdx];
          Vector gradConc = pConcGradient[pIdx];

          if (!concNormalized) {
            conc      *= d_InverseMaxConcentration;
            gradConc  *= d_InverseMaxConcentration;
          }

          double mu_homogeneous = homogeneousMu(conc,pTemperature[pIdx]);
          Vector penaltyCH = mIdentity * gradConc;
          double mu_CahnHilliardIso = isoCahnHilliardMu(penaltyCH, oodx, interpSize, nodeIndices, d_S, patch);
          mu_CahnHilliardIso *= prefactCahnHilliard;
          double mu_total = mu_homogeneous - mu_CahnHilliardIso + d_muOther;
          pMu[pIdx] = mu_total;
          IntVector node;
          for (int pNode = 0; pNode < interpSize; ++pNode) {
            node = nodeIndices[pNode];
            if (patch->containsNode(node)) {
              double massWeightedShape = pMass[pIdx] * S[pNode];
              gridMu[node]        += (massWeightedShape * mu_total);
              gridMassLocal[node] += massWeightedShape;
            }
          }

        }
      }

      NodeIterator nodeIt = patch->getExtraNodeIterator(NGN);
      for (nodeIt.begin(); !nodeIt.done(); ++nodeIt)
      {
        gridMu[*nodeIt] /= gridMassLocal[*nodeIt];
      }

//      // Loop for non-ghost particles.
//      for (int pIdx = 0; pIdx < pset_noghost->numParticles(); ++pIdx)
//      {
//        // get full set of Interpolator data
//        pInterp->findCellAndWeightsAndShapeDerivatives(pX[pIdx], nodeIndices,
//                                                       S, d_S,  pSize[pIdx],
//                                                       pDefGrad[pIdx]);
//
//        // Calculate chemical potential
//        double conc     = pConcentration[pIdx];
//        Vector gradConc = pConcGradient[pIdx];
//
////        double molIntercolant = pMass[pIdx]/d_molWeight;
////        double numIntercolant = molIntercolant * 6.0221409e+23;
//        double kT = pTemperature[pIdx] * d_unitBoltzmann;
//        if (!concNormalized)
//        {
//          conc      *= d_InverseMaxConcentration;
//          gradConc  *= d_InverseMaxConcentration;
//        }
//        double mu_homogeneous = homogeneousMu(conc,pTemperature[pIdx]);
////        double mu_homogeneous = d_regSolnParam*(1.0 - 2.0*conc)
////                                + kT * (std::log(conc) - std::log(1.0-conc));
//
//        // Calculate divergence of concentration gradient for Cahn-Hilliard term
//        Vector penaltyCH = (mIdentity * gradConc);
//        double mu_CahnHilliardIso = isoCahnHilliardMu(penaltyCH, oodx, interpSize, nodeIndices, d_S, patch);
////
////        IntVector node;
////        double mu_CahnHilliardIso = 0.0;
////        for (int nIdx = 0; nIdx < interpSize; ++nIdx) {
////          node = nodeIndices[nIdx];
////          if (patch->containsNode(node)) {
////            Vector div(d_S[nIdx]*oodx);
////            mu_CahnHilliardIso += Dot(div, penaltyCH);
////          }
////        }
//        mu_CahnHilliardIso *= prefactCahnHilliard;
////        double mu_CahnHilliardIso = -Dot(oodx, penaltyCH) * prefactCahnHilliard ;
//
//        // FIXME TODO JBH 2/2017
//        //  Should last term be Dot(oodx,d_energyGradientCoefficient*gradConc)?
//        double mu_total = mu_homogeneous - mu_CahnHilliardIso + d_muOther;
//        // Store on particle
//        pMu[pIdx] = mu_total;
//        // Volume weight for interpolation
//        mu_total *= pMass[pIdx];
//
//        IntVector node;
//        for (int pNode = 0; pNode < interpSize; ++pNode)
//        {
//          node = nodeIndices[pNode];
//          if (patch->containsNode(node))
//          {
//            gridMu[node]        += (S[pNode] * mu_total);
//            gridMassLocal[node] += (S[pNode] * pMass[pIdx]);
//          }
//        }
//      }
//

//      // Loop for ghost particles
//      for(int pIdx = 0; pIdx < pset_ghost->numParticles(); ++pIdx)
//      {
//        if (!(patch->containsPoint(pX[pIdx])))
//        {
//          // Point is in pset with ghost, but not in patch.
//          //  Therefore, particle is in ghost layer.
//          // Calculate chemical potential
//          // get full set of Interpolator data
//          pInterp->findCellAndWeightsAndShapeDerivatives(pX[pIdx], nodeIndices,
//                                                         S, d_S,  pSize[pIdx],
//                                                         pDefGrad[pIdx]);
//
//          // Calculate chemical potential
//          double conc     = pConcentration[pIdx];
//          Vector gradConc = pConcGradient[pIdx];
//
////          double molIntercolant = pMass[pIdx]/d_molWeight;
//          double kT = pTemperature[pIdx] * d_unitBoltzmann;
//          if (!concNormalized)
//          {
//            conc      *= d_InverseMaxConcentration;
//            gradConc  *= d_InverseMaxConcentration;
//          }
//          double mu_homogeneous = d_regSolnParam*(1.0 - 2.0*conc)
//                                  + kT * (std::log(conc) - std::log(1.0-conc));
//
//          // Calculate divergence of concentration gradient for Cahn-Hilliard term
//          Vector penaltyCH = (mIdentity * gradConc);
//          IntVector node;
//          double mu_CahnHilliardIso = 0.0;
//          for (int nIdx = 0; nIdx < interpSize; ++nIdx) {
//            node = nodeIndices[nIdx];
//            if (patch->containsNode(node)) {
//              Vector div(d_S[nIdx]*oodx);
//              mu_CahnHilliardIso += Dot(div, penaltyCH);
//            }
//          }
//          mu_CahnHilliardIso *= prefactCahnHilliard;
//  //        double mu_CahnHilliardIso = -Dot(oodx, penaltyCH) * prefactCahnHilliard ;
//
//          // FIXME TODO JBH 2/2017
//          //  Should last term be Dot(oodx,d_energyGradientCoefficient*gradConc)?
//          double mu_total = pMass[pIdx]*(mu_homogeneous-mu_CahnHilliardIso+d_muOther);
//          // Store on particle
//          //pMu[pIdx] = mu_total;
//
//          for (int pNode = 0; pNode < interpSize; ++pNode)
//          {
//            node = nodeIndices[pNode];
//            if (patch->containsNode(node))
//            {
//              gridMu[node]        += (S[pNode] * mu_total);
//              gridMassLocal[node] += (S[pNode] * pMass[pIdx]);
//            }
//          }
//
//        }
//      }
//

//      NodeIterator nodeIt = patch->getExtraNodeIterator(NGN);
//      for (nodeIt.begin(); !nodeIt.done(); ++nodeIt)
//      {
//        gridMu[*nodeIt] /= gMassGlobal[*nodeIt];
//      }
      // pMu and gridMu both calculated now.  Re-project gradient onto particles.
      // Loop through patch particles

      ParticleVariable<Vector> pGradMu;
      NewDW->allocateAndPut(pGradMu, d_lb->pChemicalPotentialGradientLabel, pset_noghost );

      for (ParticleSubset::iterator pIter = pset_noghost->begin(); pIter != pset_noghost->end(); ++pIter)
//      for (int pIdx = 0; pIdx < pset_noghost->numParticles(); ++pIdx)
      {
        particleIndex pIdx = *pIter;
        pInterp->findCellAndWeightsAndShapeDerivatives(pX[pIdx], nodeIndices,
                                                       S, d_S,  pSize[pIdx],
                                                       pDefGrad[pIdx]);
        pGradMu[pIdx] = Vector(0.0);
        for (int pNode = 0; pNode < interpSize; ++pNode)
        {
          IntVector node = nodeIndices[pNode];
          if (patch->containsNode(node))
          {
            pGradMu[pIdx] += d_S[pNode]*oodx*gridMu[node];
//            for (int axis = 0; axis < 3; ++axis)
//            {
//              pGradMu[pIdx][axis] += d_S[pNode][axis]*oodx[axis]*gridMu[node];
//            }
          }
        }
      }

    }
  }

//  void BazantDiffusion::calculateChemicalPotentialOld(
//                                                   const PatchSubset    * patches,
//                                                   const MPMMaterial    * matl,
//                                                         DataWarehouse  * old_dw,
//                                                         DataWarehouse  * new_dw
//                                                  )
//  {
//    // We don't -need- the chemical potential on the particles.  We only need
//    //   the gradient.
//    Ghost::GhostType gac = Ghost::AroundCells;
//    Ghost::GhostType gan = Ghost::AroundNodes;
//
//    // First iterate over patches and project to grid...
//    for (int patchIndex = 0; patchIndex < patches->size(); ++patchIndex)
//    {
//      const Patch* patch = patches->get(patchIndex);
//
//      ParticleInterpolator* pInterp = d_Mflag->d_interpolator->clone(patch);
//      int interpSize = pInterp->size();
//
//      std::vector<IntVector> nodeIndex(interpSize);
//      std::vector<double>    S(interpSize);
//
//      Vector dx = patch->dCell();
//      double oodx[3];
//      oodx[0] = 1.0/dx.x();
//      oodx[1] = 1.0/dx.y();
//      oodx[2] = 1.0/dx.z();
//
//      int dwi = matl->getDWIndex();
//
//      ParticleSubset* pset_ghost = old_dw->getParticleSubset(dwi, patch, gac, NGP, d_lb->pXLabel);
//
//      constParticleVariable<double>   pConcentration, pTemperature, pMass;
//      constParticleVariable<Vector>   pConcGradient;
//      constParticleVariable<Point>    pX;
//      constParticleVariable<Matrix3>  pSize, pDefGrad;
//
//      old_dw->get(pConcentration, d_lb->pConcentrationLabel,      pset_ghost);
//      old_dw->get(pConcGradient,  d_lb->pConcGradientLabel,       pset_ghost);
//      old_dw->get(pX,             d_lb->pXLabel,                  pset_ghost);
//      old_dw->get(pSize,          d_lb->pSizeLabel,               pset_ghost);
//      old_dw->get(pDefGrad,       d_lb->pDeformationMeasureLabel, pset_ghost);
//      old_dw->get(pTemperature,   d_lb->pTemperatureLabel,        pset_ghost);
//      old_dw->get(pMass,          d_lb->pMassLabel,               pset_ghost);
//
//      // For phase-field strain
//      constParticleVariable<Matrix3> pStress;
//      old_dw->get(pStress,        d_lb->pStressLabel,             pset_ghost);
//      constParticleVariable<Matrix3> pStrain;
////      new_dw->get(pStrain_new,        d_lb->pDiffusionStrainLabel,        pset_ghost);
//      // For coulombic potential
//      constParticleVariable<double> pElectroPotential;
//
//      // Grab mass at appropriate grid points.
////      constNCVariable<double> gMass;
////      old_dw->get(gMass, d_lb->gMassLabel, dwi, patch, Ghost::None, 0);
//
//      NCVariable<double> gChemicalPotential;
//      new_dw->allocateAndPut(gChemicalPotential, d_lb->gChemicalPotentialLabel, dwi, patch);
//      gChemicalPotential.initialize(0.0);
//
//      int numParticles = pset_ghost->numParticles();
//
//      for (int particleIndex = 0; particleIndex < numParticles; ++particleIndex)
//      {
//        int NN = pInterp->findCellAndWeights(pX[particleIndex], nodeIndex, S,
//                                             pSize[particleIndex],
//                                             pDefGrad[particleIndex]);
//        // Calculate the chemical potential contributions
//        double mu_total = 0.0;
//        double conc = pConcentration[particleIndex];
//        Vector gradConc = pConcGradient[particleIndex];
//        double kT = pTemperature[particleIndex] * d_unitBoltzmann;
//        // If pConc and pConcGrad are not normalized, do it here.
//        if (!concNormalized)
//        {
//          conc *= d_InverseMaxConcentration;
//          gradConc *= d_InverseMaxConcentration;
//        }
//
//        // Calculate individual component contributions to the chemical potential
//        double mu_homogeneous =
//          d_solutionParameter*d_diffusionSitesPerVolume*pMass[particleIndex]*(1.0-2.0*conc)
//            + 2.0 * kT * std::log(conc/(1.0-conc));
//        double mu_CahnHilliardIso = pMass[particleIndex]*d_volPerSite*0.5
//                            *d_energyGradientCoefficient*Dot(gradConc,gradConc);
//
//        mu_total = mu_homogeneous - mu_CahnHilliardIso - d_mu0;
//
//        // Now add it's contribution to our grid.
//        IntVector node;
//        for (int particleNode = 0; particleNode < NN; ++particleNode)
//        {
//          node = nodeIndex[particleNode];
//          if (patch->containsNode(node))
//          {
//            gChemicalPotential[node] += ( S[particleNode] * pMass[particleIndex]
//                                         * mu_total);
//          }
//        }
//      }
//
//    }
//
//    for (int patchIndex = 0; patchIndex < patches->size(); ++patchIndex)
//    {
//
//      Ghost::GhostType gnone = Ghost::None;
//      const Patch* patch = patches->get(patchIndex);
//
//      ParticleInterpolator* pInterp = d_Mflag->d_interpolator->clone(patch);
//      int interpSize = pInterp->size();
//
//      std::vector<IntVector> nodeIndex(interpSize);
//      std::vector<Vector>    d_S(interpSize);
//      std::vector<double>    S(interpSize);
//
//      Vector dx = patch->dCell();
//      double oodx[3];
//      oodx[0] = 1.0/dx.x();
//      oodx[1] = 1.0/dx.y();
//      oodx[2] = 1.0/dx.z();
//
//      int dwi = matl->getDWIndex();
//
//      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
//
//      constParticleVariable<Point>    pX;
//      constParticleVariable<Matrix3>  pSize, pDefGrad;
//      old_dw->get(pX,       d_lb->pXLabel,                  pset);
//      old_dw->get(pSize,    d_lb->pSizeLabel,               pset);
//      old_dw->get(pDefGrad, d_lb->pDeformationMeasureLabel, pset);
//
//      ParticleVariable<double> pChemicalPotential_new;
//      ParticleVariable<Vector> pChemicalPotentialGradient_new;
//      new_dw->allocateAndPut(pChemicalPotential_new,
//                             d_lb->pChemicalPotentialLabel,               pset);
//      new_dw->allocateAndPut(pChemicalPotentialGradient_new,
//                             d_lb->pChemicalPotentialGradientLabel,       pset);
//
//      constNCVariable<double> gMass;
//      new_dw->get(gMass, d_lb->gMassLabel, dwi, patch, Ghost::AroundNodes, NGN );
//
//      NCVariable<double> gChemicalPotential;
//      new_dw->getModifiable(gChemicalPotential,
//                            d_lb->gChemicalPotentialLabel, dwi, patch, Ghost::None, 0);
//
//      NodeIterator gIter = patch->getNodeIterator();
//      for (gIter.begin(); !gIter.done(); ++gIter)
//      {
//        IntVector node = *gIter;
//        if (patch->containsNode(node))
//        {
//          // Normalize chemical potential by total nodal mass.
//          gChemicalPotential[node] /= gMass[node];
//        }
//      }
//
//
//      ParticleSubset::iterator iter;
//      for (iter = pset->begin(); iter < pset->end(); ++iter)
//      {
//        int pIdx = *iter;
//        pChemicalPotential_new[pIdx] = 0.0;
//        pChemicalPotentialGradient_new[pIdx] = Vector(0.0);
//
//        int NN = pInterp->findCellAndWeightsAndShapeDerivatives(pX[pIdx],
//                                                                nodeIndex,
//                                                                S, d_S,
//                                                                pSize[pIdx],
//                                                                pDefGrad[pIdx]);
//        for (int nIdx = 0; nIdx < NN; ++nIdx)
//        {
//          IntVector node = nodeIndex[nIdx];
//          double nodalChemicalPotential = gChemicalPotential[node];
//          pChemicalPotential_new[pIdx] += S[nIdx] * nodalChemicalPotential;
//          for (int j = 0; j < 3; ++j)
//          {
//            double influence = d_S[nIdx][j] * oodx[j];
//            pChemicalPotentialGradient_new[pIdx][j] += nodalChemicalPotential * influence;
//          }
//        }
//      }
//    }
//  }

}


