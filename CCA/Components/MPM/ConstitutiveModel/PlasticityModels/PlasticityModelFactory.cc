/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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


#include "PlasticityModelFactory.h"                                             
#include "IsoHardeningPlastic.h"
#include "JohnsonCookPlastic.h"
#include "ZAPlastic.h"
#include "MTSPlastic.h"
#include "SCGPlastic.h"
#include "PTWPlastic.h"
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>
#include <string>
using std::cerr;
using std::ifstream;
using std::ofstream;

using namespace Uintah;

PlasticityModel* PlasticityModelFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP child = ps->findBlock("plasticity_model");
   if(!child)
      throw ProblemSetupException("Cannot find plasticity_model tag", __FILE__, __LINE__);
   string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("No type for plasticity_model", __FILE__, __LINE__);
   if (mat_type == "isotropic_hardening")
      return(scinew IsoHardeningPlastic(child));
   else if (mat_type == "johnson_cook")
      return(scinew JohnsonCookPlastic(child));
   else if (mat_type == "zerilli_armstrong")
      return(scinew ZAPlastic(child));
   else if (mat_type == "mts_model")
      return(scinew MTSPlastic(child));
   else if (mat_type == "steinberg_cochran_guinan")
      return(scinew SCGPlastic(child));
   else if (mat_type == "preston_tonks_wallace")
      return(scinew PTWPlastic(child));
   else {
      //cerr << "**WARNING** Creating default isotropic hardening plasticity model" << endl;
      //return(scinew IsoHardeningPlastic(child));
      throw ProblemSetupException("Unknown Plasticity Model ("+mat_type+")", __FILE__, __LINE__);
   }
}

PlasticityModel* 
PlasticityModelFactory::createCopy(const PlasticityModel* pm)
{
   if (dynamic_cast<const IsoHardeningPlastic*>(pm))
      return(scinew IsoHardeningPlastic(dynamic_cast<const 
                                        IsoHardeningPlastic*>(pm)));

   else if (dynamic_cast<const JohnsonCookPlastic*>(pm))
      return(scinew JohnsonCookPlastic(dynamic_cast<const 
                                       JohnsonCookPlastic*>(pm)));

   else if (dynamic_cast<const ZAPlastic*>(pm))
      return(scinew ZAPlastic(dynamic_cast<const ZAPlastic*>(pm)));

   else if (dynamic_cast<const MTSPlastic*>(pm))
      return(scinew MTSPlastic(dynamic_cast<const MTSPlastic*>(pm)));
   
   else if (dynamic_cast<const SCGPlastic*>(pm))
      return(scinew SCGPlastic(dynamic_cast<const SCGPlastic*>(pm)));

   else if (dynamic_cast<const PTWPlastic*>(pm))
      return(scinew PTWPlastic(dynamic_cast<const PTWPlastic*>(pm)));
   
   else {
      //cerr << "**WARNING** Creating copy of default isotropic hardening plasticity model" << endl;
      //return(scinew IsoHardeningPlastic(dynamic_cast<const 
      //                                  IsoHardeningPlastic*>(pm)));
      throw ProblemSetupException("Cannot create copy of unknown plasticity model", __FILE__, __LINE__);
   }
}

