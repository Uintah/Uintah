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

#ifndef UINTAH_CONCENTRATION_DIFFUSION_H
#define UINTAH_CONCENTRATION_DIFFUSION_H

#include <CCA/Ports/SchedulerP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/SimulationStateP.h>

namespace Uintah {

  class MPMLabel;
  class MPMFlags;
  class DataWarehouse;
  class ProcessorGroup;
  
  class ConcentrationDiffusion {
  public:
    
    ConcentrationDiffusion(SimulationStateP& ss,MPMLabel* lb, MPMFlags* mflags);
    ~ConcentrationDiffusion();

    void scheduleComputeInternalDiffusionRate(SchedulerP&, const PatchSet*,
                                         const MaterialSet*);
                                         
    void scheduleComputeNodalConcentrationFlux(SchedulerP&, const PatchSet*,
                                      const MaterialSet*);
    
    void scheduleSolveDiffusionEquations(SchedulerP&, const PatchSet*,
                                    const MaterialSet*);
    
    void scheduleIntegrateDiffusionRate(SchedulerP&, const PatchSet*,
                                          const MaterialSet*);
    
    void computeInternalDiffusionRate(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw);
                                 
    void computeNodalConcentrationFlux(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* ,
                              DataWarehouse* /*old_dw*/,
                              DataWarehouse* new_dw);
                                                  
    void solveDiffusionEquations(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* /*old_dw*/,
                            DataWarehouse* new_dw);
    
    void integrateDiffusionRate(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw);

  private:
    MPMLabel* d_lb;
    MPMFlags* d_flag;
    SimulationStateP d_sharedState;
    int NGP, NGN;

    ConcentrationDiffusion(const ConcentrationDiffusion&);
    ConcentrationDiffusion& operator=(const ConcentrationDiffusion&);
    
  };
  
} // end namespace Uintah
#endif
