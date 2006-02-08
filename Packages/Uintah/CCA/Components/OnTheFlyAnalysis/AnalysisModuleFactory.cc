#include <Packages/Uintah/CCA/Components/OnTheFlyAnalysis/AnalysisModuleFactory.h>
#include <Packages/Uintah/CCA/Components/OnTheFlyAnalysis/lineExtract.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>

using namespace std;
using namespace Uintah;

AnalysisModuleFactory::AnalysisModuleFactory()
{
}

AnalysisModuleFactory::~AnalysisModuleFactory()
{
}

AnalysisModule* AnalysisModuleFactory::create(const ProblemSpecP& prob_spec,
                                              SimulationStateP&   sharedState,
                                              Output* dataArchiver)
{
  string module("");
  ProblemSpecP da_ps = prob_spec->findBlock("DataAnalysis");

  if (da_ps) {
    ProblemSpecP module_ps = da_ps->findBlock("Module");
    if(!module_ps){
      throw ProblemSetupException("\nERROR:<DataAnalysis>, could not find find <Module> tag \n",__FILE__, __LINE__);
    }
    map<string,string> attributes;
    module_ps->getAttributes(attributes);
    module = attributes["name"];
    
    if (module == "lineExtract") {
      return (scinew lineExtract(module_ps, sharedState, dataArchiver));
    } else {
      throw ProblemSetupException("\nERROR:<DataAnalysis> Unknown analysis module.  "+module,__FILE__, __LINE__);
    }
    
  } else {
    return 0;
  }


}
