/*
 * Activated2.h
 *
 *  Created on: Feb 14, 2017
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

#ifndef SRC_CCA_COMPONENTS_MPM_DIFFUSION_DIFFUSIONMODELS_Activated2_H_
#define SRC_CCA_COMPONENTS_MPM_DIFFUSION_DIFFUSIONMODELS_Activated2_H_

#include <CCA/Components/MPM/Diffusion/DiffusionModels/ScalarDiffusionModel.h>

namespace Uintah
{
  class Activated2: public ScalarDiffusionModel
  {
    public:
      Activated2(
                         ProblemSpecP      & ps,
                         SimulationStateP  & simState,
                         MPMFlags          * MFlag,
                         std::string         diff_type
                        );

     ~Activated2();

     // Required interfaces for ScalarDiffusionModel
     virtual void addInitialComputesAndRequires(
                                                      Task        * task    ,
                                                const MPMMaterial * matl    ,
                                                const PatchSet    * patches
                                               ) const;

     virtual void addParticleState(
                                   std::vector<const VarLabel*> & from  ,
                                   std::vector<const VarLabel*> & to
                                  ) const;

     virtual void computeFlux(
                              const Patch         * patch ,
                              const MPMMaterial   * matl  ,
                                    DataWarehouse * OldDW ,
                                    DataWarehouse * NewDW
                             );

     virtual void initializeSDMData(
                                    const Patch         * patch ,
                                    const MPMMaterial   * matl  ,
                                          DataWarehouse * NewDW
                                   );

     virtual void scheduleComputeFlux(
                                            Task        * task  ,
                                      const MPMMaterial * matl  ,
                                      const PatchSet    * patch
                                     ) const;

     virtual void addSplitParticlesComputesAndRequires(
                                                             Task         * task    ,
                                                       const MPMMaterial  * matl    ,
                                                       const PatchSet     * patches
                                                      ) const;

     virtual void splitSDMSpecificParticleData(
                                               const Patch                  * patch,
                                               const int                      dwi,
                                               const int                      nDims,
                                                     ParticleVariable<int>  & prefOld,
                                                     ParticleVariable<int>  & pref,
                                               const unsigned int             oldNumPar,
                                               const int                      numNewPartNeeded,
                                                     DataWarehouse          * OldDW,
                                                     DataWarehouse          * NewDW
                                              );

     virtual void outputProblemSpec(
                                    ProblemSpecP  & ps,
                                    bool            output_rdm_tag = true
                                   ) const;


    private:

     bool isConcNormalized;

     double d_multiplier;

     Matrix3 d_latticeMisfit;
  };

}


#endif /* SRC_CCA_COMPONENTS_MPM_DIFFUSION_DIFFUSIONMODELS_Activated2_H_ */
