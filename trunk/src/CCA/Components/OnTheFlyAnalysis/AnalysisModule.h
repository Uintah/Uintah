#ifndef Packages_Uintah_CCA_Ports_AnalysisModule_h
#define Packages_Uintah_CCA_Ports_AnalysisModule_h

#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/ComputeSet.h>

#include <SCIRun/Core/Geometry/Vector.h>

#include <CCA/Components/OnTheFlyAnalysis/uintahshare.h>
namespace Uintah {

  class DataWarehouse;
  class ICELabel;
  class Material;
  class Patch;
  

  class UINTAHSHARE AnalysisModule {

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
