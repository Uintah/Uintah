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

#ifndef UINTAH_RD_NONLINEARDIFF1_H
#define UINTAH_RD_NONLINEARDIFF1_H

/*
 * This Non-Linear Diffusivity model contains two different models for
 * computing the diffusion coefficient. The first uses only concentration
 * as the input variable and is based on the following papers:
 *
 * Two-Phase Electrochemical Lithiation in Amorphous Silicon
 * Jiang Wei Wang, Yu He, Feifei Fan, Xiao Hua Liu, Shuman Xia, Yang Liu,
 * C. Thomas Harris, Hong Li, Jian Yu Huang, Scott X. Mao, and Ting Zhu
 * Nano Letters 2013 13 (2), 709-715
 *
 * Size-Dependent Fracture of Silicon Nanoparticles During Lithiation
 * Xiao Hua Liu, Li Zhong, Shan Huang, Scott X. Mao, Ting Zhu, and Jian Yu Huang
 * ACS Nano 2012 6 (2), 1522-1531
 *
 * Also note the supplementary information associated with each paper
 * for more details.
 *
 * The second model uses both concentration and pressure as inputs for the 
 * calculation of the diffusion coefficient.
 */

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <vector>
#include <string>
#include <CCA/Components/MPM/Materials/Diffusion/DiffusionModels/ScalarDiffusionModel.h>

namespace Uintah {

  class Task;
  class MPMFlags;
  class MPMLabel;
  class MPMMaterial;
  class DataWarehouse;
  class ProcessorGroup;

  class NonLinearDiff1 : public ScalarDiffusionModel {
  public:
    
     NonLinearDiff1(
                    ProblemSpecP      & ps,
                    MaterialManagerP  & sS,
                    MPMFlags          * Mflag,
                    std::string         diff_type
                   );
    ~NonLinearDiff1();

    // Interface requirements
    virtual void addInitialComputesAndRequires(      Task         * task,
                                               const MPMMaterial  * matl,
                                               const PatchSet     * patches
                                              ) const ;

    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to
                                 ) const;


    virtual void computeFlux(const Patch          * patch,
                             const MPMMaterial    * matl,
                                   DataWarehouse  * old_dw,
                                   DataWarehouse  * new_dw
                            );

    virtual void initializeSDMData(const Patch          * patch,
                                   const MPMMaterial    * matl,
                                         DataWarehouse  * new_dw
                                  );

    virtual void scheduleComputeFlux(      Task         * task,
                                     const MPMMaterial  * matl,
                                     const PatchSet     * patch
                                    ) const;

    virtual void addSplitParticlesComputesAndRequires(      Task        * task,
                                                      const MPMMaterial * matl,
                                                      const PatchSet    * patches
                                                     ) const;

    virtual void splitSDMSpecificParticleData(const Patch                 * patch,
                                              const int dwi,
                                              const int nDims,
                                                    ParticleVariable<int> & prefOld,
                                                    ParticleVariable<int> & pref,
                                              const unsigned int            oldNumPar,
                                              const int                     numNewPartNeeded,
                                                    DataWarehouse         * old_dw,
                                                    DataWarehouse         * new_dw
                                             );
    virtual void outputProblemSpec(
                                   ProblemSpecP & ps,
                                   bool           output_rdm_tag = true
                                  ) const ;

    // Overridden functions

  private:
    double d_tuning1;
    double d_tuning2;
    double d_tuning3;
    double d_tuning4;
    double d_tuning5;
    bool d_use_pressure;
    bool d_use_diff_curve;
    FluxDirection d_flux_direction;
    double d_time_point1;
    double d_time_point2;
    int d_diff_curve_index;

    std::vector<double> d_time_points;
    std::vector<FluxDirection> d_fd_directions;

    NonLinearDiff1(const NonLinearDiff1&);
    NonLinearDiff1& operator=(const NonLinearDiff1&);
  };
} // end namespace Uintah
#endif
