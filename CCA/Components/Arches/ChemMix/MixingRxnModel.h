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

#include <CCA/Components/Arches/Mixing/InletStream.h>
#include <CCA/Components/Arches/Mixing/Stream.h>

// Uintah includes
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/DebugStream.h>

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
 
// setenv SCI_DEBUG TABLE_DEBUG:+ 
static DebugStream cout_tabledbg("TABLE_DEBUG",false);

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

  /** @brief Computes the heat loss value */ 
  virtual void sched_computeHeatLoss( const LevelP& level, 
                                      SchedulerP& sched,
                                      const bool initialize, const bool calcEnthalpy ) = 0;

  /** @brief Initializes the enthalpy for the first time step */ 
  virtual void sched_computeFirstEnthalpy( const LevelP& level, 
                                          SchedulerP& sched ) = 0; 

  /** @brief Provides access for models, algorithms, etc. to add additional table lookup variables. */
  void addAdditionalDV( std::vector<string>& vars );

  /** @brief Needed for the old properties method until it goes away */ 
  virtual void oldTableHack( const InletStream&, Stream&, bool, const std::string ) = 0; 

  /** @brief Needed for dumb MPMArches */ 
  virtual void sched_dummyInit( const LevelP& level, 
                                    SchedulerP& sched ) = 0;

	/** @brief Returns the value of a single variable given the iv vector 
	 * This will be useful for other classes to have access to */
	virtual double getTableValue( std::vector<double>, std::string ) = 0; 

	/** @brief Return a reference to the independent variables */
	inline const VarMap getIVVars(){ return d_ivVarMap; }; 

	/** @brief Return a reference to the dependent variables */ 
	inline const VarMap getDVVars(){ return d_dvVarMap; }; 

  /** @brief Return a string list of all independent variable names in order */ 
  inline std::vector<string>& getAllIndepVars(){ return d_allIndepVarNames; }; 

  /** @brief Return a string list of dependent variables names in the order they were read */ 
  inline std::vector<string>& getAllDepVars(){ return d_allDepVarNames; };

protected :

  VarMap d_dvVarMap;   ///< Dependent variable map
  VarMap d_ivVarMap;   ///< Independent variable map

  /** @brief Sets the mixing table's dependent variable list. */
  void setMixDVMap( const ProblemSpecP& root_params ); 

  const ArchesLabel* d_lab;               ///< Arches labels
  const MPMArchesLabel* d_MAlab;          ///< MPMArches labels

  bool d_coldflow;                        ///< Will not compute heat loss and will not initialized ethalpy
  bool d_adiabatic;                       ///< Will not compute heat loss
  bool d_coal_table;                      ///< Flagged as a coal table or not
  bool d_use_mixing_model;                ///< Turn on/off mixing model

  std::string d_fp_label;                 ///< Primary mixture fraction name for a coal table
  std::string d_eta_label;                ///< Eta mixture fraction name for a coal table
  std::vector<string> d_allIndepVarNames; ///< Vector storing all independent variable names from table file
  std::vector<string> d_allDepVarNames;   ///< Vector storing all dependent variable names from the table file

  /** @brief  Insert the name of a dependent variable into the dependent variable map (dvVarMap), which maps strings to VarLabels */
  inline void insertIntoMap( const string var_name ){

    VarMap::iterator i = d_dvVarMap.find( var_name ); 

    if ( i == d_dvVarMap.end() ) {

      const VarLabel* the_label = VarLabel::create( var_name, CCVariable<double>::getTypeDescription() ); 

      i = d_dvVarMap.insert( make_pair( var_name, the_label ) ).first; 

      proc0cout << "    ---> " << var_name << endl; 

    } 
    return; 
  };

private:


}; // end class MixingRxnModel
  
} // end namespace Uintah

#endif
