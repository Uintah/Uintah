#include <Packages/Uintah/CCA/Components/ICE/Advection/AdvectionFactory.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/FirstOrderAdvector.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/FirstOrderCEAdvector.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/SecondOrderAdvector.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/SecondOrderCEAdvector.h>
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
    throw ProblemSetupException("Cannot find advection tag");
  }
  
  map<string,string> advect_options;
  advect_ps->getAttributes(advect_options);
  
  if(advect_options.find("type") == advect_options.end()){
    throw ProblemSetupException("No type for advection");
  }  


  //__________________________________
  //  check for compatible fluxes tag
  d_useCompatibleFluxes = false;
  if(advect_options.count("useCompatibleFluxes") || 
     advect_options.count("compatibleFluxes")    ||
     advect_options.count("compatible") ){
     d_useCompatibleFluxes = true; 
  } // bulletproofing
  if (advect_options.size() > 1 && d_useCompatibleFluxes == false){
        string warn="\n\n ERROR: Advection operator flags: "
                    " Did you misspell compatibleFluxes = 'true' in the input file\n";
    throw ProblemSetupException(warn);
  }
  
  //__________________________________
  // Find the advection operator type
  if (advect_options["type"] == "FirstOrder"){
    return(scinew FirstOrderAdvector());
  }
  else if (advect_options["type"] == "FirstOrderCE"){ 
    return(scinew FirstOrderCEAdvector());
  }
  else if (advect_options["type"] == "SecondOrder"){
    return(scinew SecondOrderAdvector());
  }
  else if (advect_options["type"] == "SecondOrderCE") {
    string warn="\n\n ERROR:SecondOrderCE has a bug in it.  "
                "\nTodd use ICE/performanceTest.ups to find it\n \n";
   throw ProblemSetupException(warn);
    //return(scinew SecondOrderCEAdvector());
  }else {
    throw ProblemSetupException("Unknown advection Type R ("+advect_options["type"]+")"); 
  }
}
