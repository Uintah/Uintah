/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#ifndef __NULLIFCONCDIFF_H__
#define __NULLIFCONCDIFF_H__

#include <CCA/Components/MPM/ReactionDiffusion/SDInterfaceModel.h>

namespace Uintah {

  class NullIFConcDiff : public SDInterfaceModel {
  public:
    
    NullIFConcDiff(ProblemSpecP& ps, SimulationStateP& sS, MPMFlags* Mflag);
    virtual ~NullIFConcDiff();

    virtual void addComputesAndRequiresInterpolated(SchedulerP & sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls);

    virtual void sdInterfaceInterpolated(const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw);

    virtual void addComputesAndRequiresDivergence(SchedulerP & sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls);

    virtual void sdInterfaceDivergence(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  protected:
    MPMLabel* d_lb;
    MPMFlags* d_Mflag;
    SimulationStateP d_sharedState;
    ReactionDiffusionLabel* d_rdlb;

    int NGP, NGN;
    int numMPMmatls;
    bool do_explicit;

    NullIFConcDiff(const NullIFConcDiff&);
    NullIFConcDiff& operator=(const NullIFConcDiff&);
    
  };
  
} // end namespace Uintah
#endif
