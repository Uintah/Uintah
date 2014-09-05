#include <Packages/Uintah/CCA/Components/ICE/Advection/AdvectionFactory.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/FirstOrderAdvector.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/SecondOrderAdvector.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <map>
#include <string>
#include <stdlib.h>


using namespace std;
using namespace Uintah;

Advector* AdvectionFactory::create(ProblemSpecP& ps,
                                   bool& d_useCompatibleFluxes)
{
  ProblemSpecP advect_ps = ps->findBlock("advection");
  if(!advect_ps){
    throw ProblemSetupException("Cannot find advection tag", __FILE__, __LINE__);
  }
  
  map<string,string> advect_options;
  advect_ps->getAttributes(advect_options);
  
  if(advect_options.find("type") == advect_options.end()){
    throw ProblemSetupException("No type for advection", __FILE__, __LINE__);
  }  

  
  //__________________________________
  //  check for compatible fluxes tag
  bool found = false;
  if(advect_options.count("useCompatibleFluxes") || 
     advect_options.count("compatibleFluxes")  ){
     found = true;
     
     if(advect_options["useCompatibleFluxes"] == "false" ||
        advect_options["compatibleFluxes"]    == "false"){
        d_useCompatibleFluxes = false;
        cout << "\n--------ICE::Warning:  You've turned off compatible fluxes.\n"<< endl;
     }
  } 
  
  
  // bulletproofing
  if (advect_options.size() > 1 && found == false){
        string warn="\n\n ERROR: Advection operator flags: "
                    " Did you misspell compatibleFluxes = 'true/false' in the input file?\n";
    throw ProblemSetupException(warn, __FILE__, __LINE__);
  }
  
  //__________________________________
  // Find the advection operator type
  if (advect_options["type"] == "FirstOrder"){
    return(scinew FirstOrderAdvector());
  } 
  else if (advect_options["type"] == "SecondOrder"){
    return(scinew SecondOrderAdvector());
  }
  else {
    throw ProblemSetupException("Unknown advection Type R ("+advect_options["type"]+")",
                                __FILE__, __LINE__); 
  }
}
