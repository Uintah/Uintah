#ifndef Packages_Uintah_CCA_Ports_AnalysisModule_h
#define Packages_Uintah_CCA_Ports_AnalysisModule_h

#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>

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
                              GridP& grid,
                              SimulationStateP& state) = 0;
                              
    virtual void scheduleInitialize(SchedulerP& sched,
                                    const LevelP& level) =0;
    
    virtual void restartInitialize() = 0;
    
    virtual void scheduleDoAnalysis(SchedulerP& sched,
                                    const LevelP& level) =0;
    
  };
}

#endif
