/*
 * BazantModel.h
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

#ifndef SRC_CCA_COMPONENTS_MPM_DIFFUSION_DIFFUSIONMODELS_BAZANTMODEL_H_
#define SRC_CCA_COMPONENTS_MPM_DIFFUSION_DIFFUSIONMODELS_BAZANTMODEL_H_

#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <CCA/Components/MPM/Diffusion/DiffusionModels/ScalarDiffusionModel.h>

#include <cmath>

namespace Uintah
{

  class BazantDiffusion : public ScalarDiffusionModel
  {
    public:
      BazantDiffusion(
                      ProblemSpecP      & ProbSpec,
                      SimulationStateP  & SimState,
                      MPMFlags          * MFlag,
                      std::string         diff_type
                     );
     ~BazantDiffusion();

      // Required implementations
      virtual void addInitialComputesAndRequires(
                                                       Task         * task,
                                                 const MPMMaterial  * matl,
                                                 const PatchSet     * patches
                                                ) const ;
      virtual void addParticleState(
                                    std::vector<const VarLabel*>  & from,
                                    std::vector<const VarLabel*>  & to
                                   ) const;

      virtual void computeFlux(
                               const Patch          * patch,
                               const MPMMaterial    * matl,
                                     DataWarehouse  * OldDW,
                                     DataWarehouse  * NewDW
                              );
      virtual void initializeSDMData(
                                     const Patch          * patch,
                                     const MPMMaterial    * matl,
                                           DataWarehouse  * NewDW
                                    );
      virtual void scheduleComputeFlux(
                                             Task * task,
                                       const MPMMaterial  * matl,
                                       const PatchSet     * patch
                                      ) const;

      virtual void addSplitParticlesComputesAndRequires(
                                                              Task  * task,
                                                        const MPMMaterial * matl,
                                                        const PatchSet    * patches
                                                       ) const;

      virtual void splitSDMSpecificParticleData(
                                                const Patch                 * patch,
                                                const int                     dwi,
                                                const int                     nDims,
                                                      ParticleVariable<int> & prefOld,
                                                      ParticleVariable<int> & pref,
                                                const unsigned int            oldNumPar,
                                                const int                     numNewPartNeeded,
                                                      DataWarehouse         * OldDW,
                                                      DataWarehouse         * NewDW
                                               );

      virtual void  outputProblemSpec(
                                      ProblemSpecP  & ps,
                                      bool            output_rdm_tag = true
                                     ) const;

//      virtual void scheduleComputeDivergence(
//                                                   Task         * task,
//                                             const MPMMaterial  * matl,
//                                             const PatchSet     * patches
//                                            ) const;
//
//      virtual void computeDivergence(
//                                     const Patch          * patch,
//                                     const MPMMaterial    * matl,
//                                           DataWarehouse  * OldDW,
//                                           DataWarehouse  * NewDW
//                                    );

      virtual bool usesChemicalPotential() const;

      virtual void addChemPotentialComputesAndRequires(
                                                             Task         * task,
                                                       const MPMMaterial  * matl,
                                                       const PatchSet     * patches
                                                      ) const;
      virtual void calculateChemicalPotential(
                                              const PatchSubset   * patches,
                                              const MPMMaterial   * matl,
                                                    DataWarehouse * old_dw,
                                                    DataWarehouse * new_dw
                                             );
      virtual void calculateChemicalPotentialTake3( const PatchSubset *, const MPMMaterial *, DataWarehouse*, DataWarehouse*);

    private:
      inline double homogeneousMu(const double & conc  ,
                                  const double & T     );

      double isoCahnHilliardMu(const Vector                  & concGrad    ,
                               const Vector                  & oodx        ,
                               const int                       numNodes    ,
                               const std::vector<IntVector>  & nodeIndices ,
                               const std::vector<Vector>     & shapeDeriv  ,
                               const Patch                   * patch       );

      // Required components for this model
      double d_regSolnParam;            // Enthalpic contribution for regular solution theory
      double d_unitBoltzmann;           // Base value of k_B in consistent units
      double d_CahnHilliardGradPenalty; // Cahn-Hilliard term that penalizes interface
      double d_intercSiteDensity;       // Density of intercalant species at 100% loading
      double d_molWeight;               // Molecular weight of full concentration compound

      // Optional components for this model
      double d_muOther;                 // External chemical potential from source not in this model.
      double d_mismatchMagnitude;
      bool   d_includeMismatch;

      bool concNormalized;
      Matrix3 d_latticeMisfit;
  };
}



#endif /* SRC_CCA_COMPONENTS_MPM_DIFFUSION_DIFFUSIONMODELS_BAZANTMODEL_H_ */
