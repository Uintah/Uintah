/*
 * CahnHilliard.cc
 *
 *  Created on: Apr 12, 2017
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

#include <CCA/Components/MPM/Diffusion/DiffusionModels/CahnHilliard.h>

#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>

#include <CCA/Components/MPM/Diffusion/DiffusionModels/Perlin.h>

#include <vector>
#include <random>

namespace Uintah
{
  CahnHilliardDiffusion::CahnHilliardDiffusion(
                                               ProblemSpecP       & probSpec  ,
                                               SimulationStateP   & simState  ,
                                               MPMFlags           * mFlag     ,
                                               std::string          diff_type
                                              )
                                              : ScalarDiffusionModel(
                                                                     probSpec ,
                                                                     simState ,
                                                                     mFlag    ,
                                                                     diff_type
                                                                    )
  {
    probSpec->require("gamma",d_gamma);
    probSpec->require("a", d_a);
    probSpec->require("b", d_b);
    probSpec->getWithDefault("random_scale",d_scalingFactor, 0.0);
    d_b2 = d_b*d_b;
    std::cout << "Cahn-Hilliard:\n\tPhase compositions - +/- " << std::sqrt(d_a)/d_b
              << "\n\tInnate timescale - " << d_gamma/(d_a*d_a*d_b2*d_b2*d_D0)
              << "\n\tInnate lengthscale - " << sqrt(d_gamma/d_a)/d_b
              << "\n\t a: " << d_a << " b: " << d_b << " gamma: " << d_gamma << " D: " << d_D0
              << std::endl;
  }

  CahnHilliardDiffusion::~CahnHilliardDiffusion()
  {

  }

  void CahnHilliardDiffusion::addInitialComputesAndRequires(
                                                                  Task        * task    ,
                                                            const MPMMaterial * matl    ,
                                                            const PatchSet    * patches
                                                           ) const
  {
    const MaterialSubset  * matlset = matl->thisMaterial();
    task->computes(d_lb->pFluxLabel,  matlset);
  }

  void CahnHilliardDiffusion::addParticleState(
                                               std::vector<const VarLabel*> & from  ,
                                               std::vector<const VarLabel*> & to
                                              ) const
  {
    from.push_back(d_lb->pFluxLabel);

    to.push_back(d_lb->pFluxLabel_preReloc);
  }

  void CahnHilliardDiffusion::computeFlux(
                                          const Patch         * patch ,
                                          const MPMMaterial   * matl  ,
                                                DataWarehouse * oldDW ,
                                                DataWarehouse * newDW
                                         )
  {
    ParticleInterpolator  * interpolator  = d_Mflag->d_interpolator->clone(patch);

    int interpPoints = interpolator->size();

    std::vector<IntVector>  nodeIndices(interpPoints);
    std::vector<double>     S(interpPoints);
    std::vector<Vector>     dS(interpPoints);

    Vector    dx  = patch->dCell();
    Vector  oodx  = dx.inverse();
    int       dwi = matl->getDWIndex();

    ParticleSubset* pset  = oldDW->getParticleSubset(dwi, patch);

    constParticleVariable<double> pTemperature;
    constParticleVariable<Vector> pGradChemPotential;

    oldDW->get(pTemperature, d_lb->pTemperatureLabel, pset);
    newDW->get(pGradChemPotential, d_lb->pChemicalPotentialGradientLabel, pset);

    ParticleVariable<Vector>  pFluxNew;
    newDW->allocateAndPut(pFluxNew, d_lb->pFluxLabel_preReloc, pset);

    double maxFlux = 1e-99;
    for (int pIdx = 0; pIdx < pset->numParticles(); ++pIdx)
    {
      // J = -D*grad(mu)
      pFluxNew[pIdx] =  d_D0 * pGradChemPotential[pIdx];
      maxFlux = std::max(maxFlux,pFluxNew[pIdx].maxComponentMag());
    }

    double delT_local = computeStableTimeStep(d_D0, dx);
    newDW->put(delt_vartype(delT_local), d_lb->delTLabel, patch->getLevel());
  }

  void CahnHilliardDiffusion::initializeSDMData(
                                                const Patch         * patch ,
                                                const MPMMaterial   * matl  ,
                                                      DataWarehouse * newDW
                                               )
  {
    ParticleSubset* pset  = newDW->getParticleSubset(matl->getDWIndex(), patch);

    PerlinNoise noiseSource;
    constParticleVariable<Point> pX;
    newDW->get(pX, d_lb->pXLabel, pset);
    int numMPMPoints = pset->numParticles();

    // Initialize a random offset for all particles
    ParticleVariable<double> concentrationOffset;
    newDW->allocateTemporary(concentrationOffset, pset);
    // Generate a random concentration value b/t [0..1) for every particle.
    ///  Using Perlin noise for some coherency in spatial distribution
    for (int pIdx = 0; pIdx < numMPMPoints; ++pIdx) {
      concentrationOffset[pIdx] = noiseSource.noise(pX[pIdx].x(), pX[pIdx].y(), pX[pIdx].z());
    }
    // Calculate the aggregate shift due to the random offset.  We will subtract
    //   totalOffset/numMPMPoints from each value so that the noise sums to
    //   zero net concentration differential.
    double totalOffset = 0.0;
    for (int pIdx = 0; pIdx < numMPMPoints; ++pIdx) {
      totalOffset += concentrationOffset[pIdx];
    }
    double uniformShift = totalOffset/static_cast<double> (numMPMPoints);
    // Now we scale the magnitude of the shifted function by our total overall
    // maximum desired amplitude.
    for (int pIdx = 0; pIdx < numMPMPoints; ++pIdx) {
      concentrationOffset[pIdx] -= uniformShift;
      concentrationOffset[pIdx] *= d_scalingFactor;
    }

    ParticleVariable<Vector> pFlux;
    ParticleVariable<double> pConcentration;
    newDW->allocateAndPut(pFlux,          d_lb->pFluxLabel,           pset);
    newDW->getModifiable(pConcentration,  d_lb->pConcentrationLabel,  pset);

    double minConc = getClampedMinConc();
    double maxConc = getClampedMaxConc();

    for (int pIndex = 0; pIndex < pset->numParticles(); ++pIndex) {
      pFlux[pIndex] = Vector(0.0);

      double conc = pConcentration[pIndex] + concentrationOffset[pIndex];
      pConcentration[pIndex] = std::max( minConc, std::min(maxConc, conc) );
    }

  }

  void CahnHilliardDiffusion::scheduleComputeFlux(
                                                        Task        * task  ,
                                                  const MPMMaterial * matl  ,
                                                  const PatchSet    * patch
                                                 ) const
  {
    const MaterialSubset  * matlset = matl->thisMaterial();
    Ghost::GhostType        gnone   = Ghost::None;

    task->requires(Task::NewDW, d_lb->pChemicalPotentialGradientLabel, matlset, gnone);

    task->computes(d_lb->pFluxLabel_preReloc, matlset);
    task->computes(d_sharedState->get_delt_label(), getLevel(patch));
  }

  void CahnHilliardDiffusion::addSplitParticlesComputesAndRequires(
                                                                         Task         * task  ,
                                                                   const MPMMaterial  * matl  ,
                                                                   const PatchSet     * patches
                                                                  ) const
  {

  }

  void CahnHilliardDiffusion::splitSDMSpecificParticleData(
                                                           const Patch                  * patch             ,
                                                           const int                      dwi               ,
                                                           const int                      nDims             ,
                                                                 ParticleVariable<int>  & prefOld           ,
                                                                 ParticleVariable<int>  & pref              ,
                                                           const unsigned int             oldNumPar         ,
                                                           const int                      numNewPartNeeded  ,
                                                                 DataWarehouse          * oldDW             ,
                                                                 DataWarehouse          * newDW
                                                          )
  {

  }

  void CahnHilliardDiffusion::outputProblemSpec(
                                                ProblemSpecP  & probSpec  ,
                                                bool            output_rdm_tag
                                               ) const
  {
    if (!output_rdm_tag) return;  // If for some reason we don't want to output, return
    ProblemSpecP rdm_ps = probSpec;
    rdm_ps = probSpec->appendChild("diffusion_model");
    rdm_ps->setAttribute("type","cahn-hilliard");
    // Output elements common to all diffusion models.
    baseOutputSDMProbSpec(probSpec, output_rdm_tag);
//    rdm_ps->appendElement("diffusivity",d_D0);
//    rdm_ps->appendElement("max_concentration",d_MaxConcentration);
//    rdm_ps->appendElement("min_concentration",d_MinConcentration);
//    rdm_ps->appendElement("conc_tolerance", d_concTolerance);
//    rdm_ps->appendElement("initial_concentration", d_InitialConcentration);

    // Output elements specific to just this diffusion model
    rdm_ps->appendElement("gamma",d_gamma);
    rdm_ps->appendElement("a", d_a);
    rdm_ps->appendElement("b", d_b);
    rdm_ps->appendElement("random_scale",d_scalingFactor);
  }

  bool CahnHilliardDiffusion::usesChemicalPotential() const
  {
    return true;
  }

  void CahnHilliardDiffusion::addChemPotentialComputesAndRequires(
                                                                        Task        * task    ,
                                                                  const MPMMaterial * matl    ,
                                                                  const PatchSet    * patches
                                                                 ) const
  {
    Ghost::GhostType gac = Ghost::AroundCells;
    Ghost::GhostType gan = Ghost::AroundNodes;
    const MaterialSubset  * matlset = matl->thisMaterial();

    // Particle variables we need for calculation of mu
    task->requires(Task::OldDW, d_lb->pConcentrationLabel,      matlset, gac, NGP);
    task->requires(Task::OldDW, d_lb->pConcGradientLabel,       matlset, gac, NGP);

    // Particle variables we need for calculation of divergence of concentration
    task->requires(Task::OldDW, d_lb->pXLabel,                  matlset, gac, NGP);
    task->requires(Task::OldDW, d_lb->pSizeLabel,               matlset, gac, NGP);
    task->requires(Task::OldDW, d_lb->pDeformationMeasureLabel, matlset, gac, NGP);
    task->requires(Task::OldDW, d_lb->pMassLabel,               matlset, gac, NGP);

    task->computes(d_lb->pChemicalPotentialLabel,         matlset);
    task->computes(d_lb->pChemicalPotentialGradientLabel, matlset);
  }


  void CahnHilliardDiffusion::calculateChemicalPotential(
                                                         const PatchSubset    * patches ,
                                                         const MPMMaterial    * matl    ,
                                                               DataWarehouse  * oldDW   ,
                                                               DataWarehouse  * newDW
                                                        )
  {
    Ghost::GhostType gac = Ghost::AroundCells;
    Ghost::GhostType gan = Ghost::AroundNodes;

    for (int patchIndex = 0; patchIndex < patches->size(); ++patchIndex)
    {
      const Patch* patch  = patches->get(patchIndex);

      ParticleInterpolator  * pInterp = d_Mflag->d_interpolator->clone(patch);
      int interpSize = pInterp->size();

      std::vector<IntVector>  nodeIndices(interpSize);
      std::vector<double>     S(interpSize);
      std::vector<Vector>     dS(interpSize);

      Vector oodx = (patch->dCell()).inverse();
      int dwi = matl->getDWIndex();

      ParticleSubset* pset_ghost    = oldDW->getParticleSubset(dwi, patch, gac, NGP, d_lb->pXLabel);
      ParticleSubset* pset_noghost  = oldDW->getParticleSubset(dwi, patch, Ghost::None, 0, d_lb->pXLabel);

      constParticleVariable<double>   pConcentration, pMass;
      constParticleVariable<Vector>   pConcGradient;
      constParticleVariable<Point>    pX;
      constParticleVariable<Matrix3>  pSize, pDefGrad;

      oldDW->get(pConcentration,  d_lb->pConcentrationLabel,      pset_ghost);
      oldDW->get(pConcGradient,   d_lb->pConcGradientLabel,       pset_ghost);
      oldDW->get(pX,              d_lb->pXLabel,                  pset_ghost);
      oldDW->get(pSize,           d_lb->pSizeLabel,               pset_ghost);
      oldDW->get(pDefGrad,        d_lb->pDeformationMeasureLabel, pset_ghost);
      oldDW->get(pMass,           d_lb->pMassLabel,               pset_ghost);

      NCVariable<double>gridMassLocal;
      NCVariable<double>gridMu;

      newDW->allocateTemporary(gridMassLocal, patch, Ghost::AroundNodes, NGN);
      newDW->allocateTemporary(gridMu,        patch, Ghost::AroundNodes, NGN);
      gridMassLocal.initialize(1e-200);
      gridMu.initialize(0.0);

      ParticleVariable<double> pMu;
      newDW->allocateAndPut(pMu, d_lb->pChemicalPotentialLabel, pset_noghost);
      Matrix3 mIdentity(1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0);

      // Loop for non-ghost particles
      for (int pIdx = 0; pIdx < pset_noghost->numParticles(); ++pIdx)
      {
        pInterp->findCellAndWeightsAndShapeDerivatives(pX[pIdx], nodeIndices, S, dS, pSize[pIdx], pDefGrad[pIdx]);

        double C      = pConcentration[pIdx];
        Vector gradC  = pConcGradient[pIdx];

        double mu_CHTerm = - d_gamma * (divConcentration(gradC, oodx, interpSize, nodeIndices, dS, patch));
        double mu_total = d_b2*C*(d_b2*C*C-d_a) + mu_CHTerm;
        pMu[pIdx] = mu_total;

        // Interpolate mu onto grid
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

      // Loop for ghost particles
      for (int pIdx = 0; pIdx < pset_ghost->numParticles(); ++pIdx)
      {
        if (!(patch->containsPoint(pX[pIdx])))
        {
          pInterp->findCellAndWeightsAndShapeDerivatives(pX[pIdx], nodeIndices, S, dS, pSize[pIdx], pDefGrad[pIdx]);

          double C      = pConcentration[pIdx];
          Vector gradC  = pConcGradient[pIdx];

          double mu_CHTerm = - d_gamma * (divConcentration(gradC, oodx, interpSize, nodeIndices, dS, patch));
          double mu_total = d_b2*C*(d_b2*C*C-d_a) + mu_CHTerm;

          // Interpolate mu onto grid
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

      // Normalize grid mu
      NodeIterator nodeIt = patch->getExtraNodeIterator(NGN);
      for (nodeIt.begin(); !nodeIt.done(); ++nodeIt)
      {
          double massNormedMu = gridMu[*nodeIt] / gridMassLocal[*nodeIt];
          gridMu[*nodeIt] = massNormedMu;
        //gridMu[*nodeIt] /= gridMassLocal[*nodeIt];
      }

      // Form gradMu
      ParticleVariable<Vector> pGradMu;
      newDW->allocateAndPut(pGradMu, d_lb->pChemicalPotentialGradientLabel, pset_noghost);

      int numNoGhost = pset_noghost->numParticles();
      for (int pIdx = 0; pIdx < numNoGhost; ++pIdx) {
        pInterp->findCellAndWeightsAndShapeDerivatives(pX[pIdx], nodeIndices, S, dS, pSize[pIdx], pDefGrad[pIdx]);

        Vector gradMu(0.0);
        for (int pNode = 0; pNode < interpSize; ++pNode)
        {
          IntVector node = nodeIndices[pNode];
          if (patch->containsNode(node))
          {
            gradMu += dS[pNode]*oodx*gridMu[node];
          }
        }
        pGradMu[pIdx] = gradMu;
      }
    }
  }

  double CahnHilliardDiffusion::divConcentration(
                                                 const Vector                 & concGrad  ,
                                                 const Vector                 & oodx      ,
                                                 const int                    & numNodes  ,
                                                 const std::vector<IntVector> & nodeIndices ,
                                                 const std::vector<Vector>    & shapeDeriv  ,
                                                 const Patch*                   patch
                                                ) const
  {
    double divC = 0.0;
    IntVector node;
    for (int nIdx = 0; nIdx < numNodes; ++nIdx)
    {
      node = nodeIndices[nIdx];
      if (patch->containsNode(node)) {
        Vector div(shapeDeriv[nIdx]*oodx);
        divC += Dot(div,concGrad);
      }
    }
    return (divC);
  }

}

