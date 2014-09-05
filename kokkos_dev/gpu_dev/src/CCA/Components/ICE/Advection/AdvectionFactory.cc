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


#include <CCA/Components/ICE/Advection/AdvectionFactory.h>
#include <CCA/Components/ICE/Advection/FirstOrderAdvector.h>

#include <sci_defs/cuda_defs.h>
#ifdef HAVE_CUDA
#include <CCA/Components/ICE/Advection/FirstOrderAdvectorGPU.h>
#endif
#include <CCA/Components/ICE/Advection/SecondOrderAdvector.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <map>
#include <string>
#include <cstdlib>


using namespace std;
using namespace Uintah;

Advector* AdvectionFactory::create(ProblemSpecP& ps,
                                   bool& d_useCompatibleFluxes,
                                   int& d_OrderOfAdvection)
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
    d_OrderOfAdvection = 1;
    return(scinew FirstOrderAdvector());
  } 
#ifdef HAVE_CUDA
  if (advect_options["type"] == "FirstOrderGPU"){
    d_OrderOfAdvection = 1;
    return(scinew FirstOrderAdvectorGPU());
  } 
#endif
  else if (advect_options["type"] == "SecondOrder"){
    d_OrderOfAdvection = 2;
    return(scinew SecondOrderAdvector());
  }
  else {
    throw ProblemSetupException("Unknown advection Type R ("+advect_options["type"]+")",
                                __FILE__, __LINE__); 
  }
}
