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



#include "PressureModelFactory.h"
#include "Pressure_Hypoelastic.h"
#include "Pressure_Hyperelastic.h"
#include "Pressure_MieGruneisen.h"
#include "Pressure_Borja.h"
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>
#include <string>
using std::cerr;
using std::ifstream;
using std::ofstream;

using namespace UintahBB;
using namespace Uintah;

PressureModel* PressureModelFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP child = ps->findBlock("equation_of_state");
   if(!child) {

      cerr << "**WARNING** Creating default hyperelastic equation of state" << endl;
      return(scinew Pressure_Hyperelastic(child));

      //cerr << "**WARNING** Creating default linear equation of state" << endl;
      //return(scinew Pressure_Hypoelastic(child));
      //throw ProblemSetupException("Cannot find equation_of_state tag", __FILE__, __LINE__);
   }
   string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("No type for equation_of_state", __FILE__, __LINE__);
   
   if (mat_type == "mie_gruneisen")
      return(scinew Pressure_MieGruneisen(child));
   else if (mat_type == "default_hypo")
      return(scinew Pressure_Hypoelastic(child));
   else if (mat_type == "default_hyper")
      return(scinew Pressure_Hyperelastic(child));
   else if (mat_type == "borja")
      return(scinew Pressure_Borja(child));
   else {
      cerr << "**WARNING** Creating default hyperelastic equation of state" << endl;
      return(scinew Pressure_Hyperelastic(child));
      //throw ProblemSetupException("Unknown MPMEquation of State Model ("+mat_type+")", __FILE__, __LINE__);
   }
 

   //return 0;
}

PressureModel* 
PressureModelFactory::createCopy(const PressureModel* eos)
{
   if (dynamic_cast<const Pressure_MieGruneisen*>(eos))
      return(scinew Pressure_MieGruneisen(dynamic_cast<const Pressure_MieGruneisen*>(eos)));

   if (dynamic_cast<const Pressure_Borja*>(eos))
      return(scinew Pressure_Borja(dynamic_cast<const Pressure_Borja*>(eos)));

   else if (dynamic_cast<const Pressure_Hypoelastic*>(eos))
      return(scinew Pressure_Hypoelastic(dynamic_cast<const Pressure_Hypoelastic*>(eos)));

   else {
      cerr << "**WARNING** Creating a copy of the default hyperelastic equation of state" << endl;
      return(scinew Pressure_Hyperelastic(dynamic_cast<const Pressure_Hyperelastic*>(eos)));
      //throw ProblemSetupException("Cannot create copy of unknown MPM EOS", __FILE__, __LINE__);
   }

   //return 0;
}
