/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#ifndef __COMMONIFCONCDIFF_H__
#define __COMMONIFCONCDIFF_H__

#include <CCA/Components/MPM/ReactionDiffusion/SDInterfaceModel.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  class Task;
  class MPMFlags;
  class MPMLabel;
  class MPMMaterial;
  class DataWarehouse;
  class ProcessorGroup;

  
  class CommonIFConcDiff : public SDInterfaceModel {
  public:
    
    CommonIFConcDiff(ProblemSpecP& ps, SimulationStateP& sS, MPMFlags* Mflag);
    ~CommonIFConcDiff();

//    virtual void initializeSDMData(const Patch* patch, DataWarehouse* new_dw);

    virtual void computeDivergence(const Patch* patch, DataWarehouse* old_dw,
		                               DataWarehouse* new_dw);

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_sdim_tag = true);

  protected:
    MPMLabel* d_lb;
    MPMFlags* d_Mflag;
    SimulationStateP d_sharedState;

    int NGP, NGN;
    std::string diffusion_type;
    int numMPMmatls;
    bool include_hydrostress;

    CommonIFConcDiff(const CommonIFConcDiff&);
    CommonIFConcDiff& operator=(const CommonIFConcDiff&);
    
  };
  
} // end namespace Uintah
#endif
