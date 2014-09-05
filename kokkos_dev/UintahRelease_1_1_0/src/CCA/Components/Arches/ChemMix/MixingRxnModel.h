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


//----- MixingRxnModel.h --------------------------------------------------

#ifndef Uintah_Component_Arches_MixingRxnModel_h
#define Uintah_Component_Arches_MixingRxnModel_h

// Uintah includes
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CCVariable.h>

// C++ includes
#include     <vector>
#include     <map>
#include     <string>
#include     <stdexcept>


/** 
* @class  MixingRxnModel
* @author Charles Reid
* @date   Nov, 22 2008
* 
* @brief Base class for mixing/reaction tables interfaces 
*    
*
    This MixingRxnModel class provides a representation of the mixing 
    and reaction model for the Arches code (interfaced through Properties.cc).  
    The MixingRxnModel class is a base class that allows for child classes 
    that each provide a specific representation of specific mixing and 
    reaction table formats.  

    Tables are pre-processed by using any number of programs (DARS, Cantera, TabProps, 
    etc.).  
* 
*/ 


namespace Uintah {
class MixingRxnModel{

public:

  typedef map<unsigned int, CCVariable<double>* > VarMap;

  MixingRxnModel();

  virtual ~MixingRxnModel();

  /** @brief Interface the the input file.  Get table name, then read table data into object */
  virtual void problemSetup( const ProblemSpecP& params ) = 0;

  /** @brief Returns a vector of the state space for a given set of independent parameters */
  virtual void getState( VarMap ivVars, VarMap dvVars, const Patch* patch ) = 0;

  /** @brief Checks for consistency between the requested independent variables and those actually in the table along with the 
   *    dependent variables and those in the table */
  virtual void const verifyTable( bool diagnosticMode,
                            bool strictMode )  = 0;

  /** @brief Returns a list of dependent variables */
  virtual const std::vector<std::string> & getDepVars() = 0;

  /** @brief Returns a list of independent variables */ 
  virtual const std::vector<std::string> & getIndepVars() = 0;


protected :


private:

}; // end class MixingRxnModel
  
} // end namespace Uintah

#endif
