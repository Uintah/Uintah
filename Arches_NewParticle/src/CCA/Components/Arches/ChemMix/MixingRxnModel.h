/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Parallel/Parallel.h>

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
 
class ArchesLabel; 
class TimeIntegratorLabel; 
class MixingRxnModel{

public:

  // Useful typedefs
  typedef std::map<string, const VarLabel* >           VarMap;
  typedef std::map<string, CCVariable<double>* >       CCMap; 

  MixingRxnModel( const ArchesLabel* labels, const MPMArchesLabel* MAlabels );

  virtual ~MixingRxnModel();

  /** @brief Interface the the input file.  Get table name, then read table data into object */
  virtual void problemSetup( const ProblemSpecP& params ) = 0;

  /** @brief Returns a vector of the state space for a given set of independent parameters */
  virtual void sched_getState( const LevelP& level, 
                                SchedulerP& sched, 
                                const TimeIntegratorLabel* time_labels, 
                                const bool initialize,
                                const bool with_energy_exch,
                                const bool modify_ref_den ) = 0;

  /** @brief Returns a list of dependent variables */
  virtual const std::vector<std::string> & getAllDepVars() = 0;

  /** @brief Returns a list of independent variables */ 
  virtual const std::vector<std::string> & getAllIndepVars() = 0;

  /** @brief Computes the heat loss value */ 
  virtual void sched_computeHeatLoss( const LevelP& level, 
                                      SchedulerP& sched,
                                      const bool initialize, const bool calcEnthalpy ) = 0;

  /** @brief Initializes the enthalpy for the first time step */ 
  virtual void sched_computeFirstEnthalpy( const LevelP& level, 
                                          SchedulerP& sched ) = 0; 

  /** @brief Provides access for models, algorithms, etc. to add additional table lookup variables. */
  virtual void addAdditionalDV( std::vector<string>& vars );

protected :

  VarMap d_dvVarMap; 
  VarMap d_ivVarMap; 

  /** @brief Sets the mixing table's dependent variable list. */
  void setMixDVMap( const ProblemSpecP& root_params ); 

  const ArchesLabel* d_lab; 
  const MPMArchesLabel* d_MAlab;


private:

  /** @brief  Insert the name of a dependent variable into the dependent variable map (dvVarMap), which maps strings to VarLabels */
  inline void insertIntoMap( const string var_name ){

    VarMap::iterator i = d_dvVarMap.find( var_name ); 

    if ( i == d_dvVarMap.end() ) {

      const VarLabel* the_label = VarLabel::create( var_name, CCVariable<double>::getTypeDescription() ); 

      i = d_dvVarMap.insert( make_pair( var_name, the_label ) ).first; 

      proc0cout << "  Adding variables for table lookup: " << endl; 
      proc0cout << "    ---> " << var_name << endl; 

    } 
    return; 
  };


}; // end class MixingRxnModel
  
} // end namespace Uintah

#endif
