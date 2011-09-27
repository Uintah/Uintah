/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef UINTAH_HEAT_CONDUCTION_H
#define UINTAH_HEAT_CONDUCTION_H

#include <CCA/Ports/SchedulerP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/SimulationStateP.h>

namespace Uintah {

  class MPMLabel;
  class MPMFlags;
  class DataWarehouse;
  class ProcessorGroup;
  
  class HeatConduction {
  public:
    
    HeatConduction(SimulationStateP& ss,MPMLabel* lb, MPMFlags* mflags);
    ~HeatConduction();

    void scheduleComputeInternalHeatRate(SchedulerP&, const PatchSet*,
                                         const MaterialSet*);
                                         
    void scheduleComputeNodalHeatFlux(SchedulerP&, const PatchSet*,
                                      const MaterialSet*);
    
    void scheduleSolveHeatEquations(SchedulerP&, const PatchSet*,
                                    const MaterialSet*);
    
    void scheduleIntegrateTemperatureRate(SchedulerP&, const PatchSet*,
                                          const MaterialSet*);
    
    void computeInternalHeatRate(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw);
                                 
    void computeNodalHeatFlux(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* ,
                              DataWarehouse* /*old_dw*/,
                              DataWarehouse* new_dw);
                                                  
    void solveHeatEquations(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* /*old_dw*/,
                            DataWarehouse* new_dw);
    
    void integrateTemperatureRate(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw);

  private:
    MPMLabel* d_lb;
    MPMFlags* d_flag;
    SimulationStateP d_sharedState;
    int NGP, NGN;

    HeatConduction(const HeatConduction&);
    HeatConduction& operator=(const HeatConduction&);
    
  };
  
} // end namespace Uintah
#endif
