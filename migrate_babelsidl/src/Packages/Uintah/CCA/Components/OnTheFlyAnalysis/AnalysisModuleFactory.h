#ifndef Packages_Uintah_CCA_Components_OnTheFlyAnalysis_Factory_h
#define Packages_Uintah_CCA_Components_OnTheFlyAnalysis_Factory_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/CCA/Ports/Output.h>

#include <Packages/Uintah/CCA/Components/OnTheFlyAnalysis/share.h>

namespace Uintah {
  class AnalysisModule;
  
  class SCISHARE AnalysisModuleFactory{
    public:
      AnalysisModuleFactory();
      ~AnalysisModuleFactory();
      
      static AnalysisModule* create(const ProblemSpecP& prob_spec,
                                    SimulationStateP& sharedState,
                                    Output* dataArchiever);
    };
}

#endif 
