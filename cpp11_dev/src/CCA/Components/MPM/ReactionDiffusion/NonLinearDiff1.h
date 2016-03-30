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

#include <CCA/Components/MPM/ReactionDiffusion/ScalarDiffusionModel.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <vector>
#include <string>
using namespace std;

namespace Uintah {

  class Task;
  class MPMFlags;
  class MPMLabel;
  class MPMMaterial;
  class DataWarehouse;
  class ProcessorGroup;


  enum FluxDirection{
    fd_in,
    fd_out,
    fd_transition
  };
  
  class NonLinearDiff1 : public ScalarDiffusionModel {
  public:
    
    NonLinearDiff1(ProblemSpecP& ps, SimulationStateP& sS, MPMFlags* Mflag,
                        std::string diff_type);
    ~NonLinearDiff1();

    virtual void scheduleComputeFlux(Task* task, const MPMMaterial* matl, 
		                                      const PatchSet* patch) const;

    virtual void computeFlux(const Patch* patch, const MPMMaterial* matl,
                             DataWarehouse* old_dw, DataWarehouse* new_dw);

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_rdm_tag = true);

  private:
    double tuning1;
    double tuning2;
    double tuning3;
    double tuning4;
    double tuning5;
    bool use_pressure;
    bool use_diff_curve;

    vector<double> time_points;
    vector<FluxDirection> fd_directions;

    NonLinearDiff1(const NonLinearDiff1&);
    NonLinearDiff1& operator=(const NonLinearDiff1&);
  };
} // end namespace Uintah
#endif
