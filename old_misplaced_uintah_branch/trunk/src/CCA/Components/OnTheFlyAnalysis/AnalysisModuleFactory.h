#ifndef Packages_Uintah_CCA_Components_OnTheFlyAnalysis_Factory_h
#define Packages_Uintah_CCA_Components_OnTheFlyAnalysis_Factory_h

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Ports/Output.h>

#include <CCA/Components/OnTheFlyAnalysis/uintahshare.h>

namespace Uintah {
  class AnalysisModule;
  
  class UINTAHSHARE AnalysisModuleFactory{
    public:
      AnalysisModuleFactory();
      ~AnalysisModuleFactory();
      
      static AnalysisModule* create(const ProblemSpecP& prob_spec,
                                    SimulationStateP& sharedState,
                                    Output* dataArchiever);
    };
}

#endif 
