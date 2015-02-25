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

#ifndef Packages_Uintah_CCA_Ports_AnalysisModule_h
#define Packages_Uintah_CCA_Ports_AnalysisModule_h

#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/ComputeSet.h>

#include <Core/Geometry/Vector.h>

namespace Uintah {

  class DataWarehouse;
  class ICELabel;
  class Material;
  class Patch;
  

  class AnalysisModule {

  public:
    
    AnalysisModule();
    AnalysisModule(ProblemSpecP& prob_spec, SimulationStateP& sharedState, Output* dataArchiver);
    virtual ~AnalysisModule();

    virtual void problemSetup(const ProblemSpecP& params,
                              const ProblemSpecP& restart_prob_spec,
                              GridP& grid,
                              SimulationStateP& state) = 0;
                              
                              
    virtual void scheduleInitialize(SchedulerP& sched,
                                    const LevelP& level) =0;
    
    virtual void restartInitialize() = 0;
    
    virtual void scheduleDoAnalysis(SchedulerP& sched,
                                    const LevelP& level) =0;
    
    virtual void scheduleDoAnalysis_preReloc(SchedulerP& sched,
                                    const LevelP& level) =0;
  };
}

#endif
