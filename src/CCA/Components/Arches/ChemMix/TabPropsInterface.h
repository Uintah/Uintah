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


//----- TabPropsInterface.h --------------------------------------------------

#ifndef Uintah_Component_Arches_TabPropsInterface_h
#define Uintah_Component_Arches_TabPropsInterface_h

// includes for Arches
#include <CCA/Components/Arches/ChemMix/TabProps/StateTable.h>
#include <CCA/Components/Arches/Mixing/InletStream.h>
#include <CCA/Components/Arches/Mixing/Stream.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>


/**
 * @class  TabPropsInterface
 * @author Charles Reid
 * @date   Nov 11, 2008
 *
 * @brief Table interface for those created with TabProps.
 *
    Dependent variables are B-Splined, and spline coefficients are put into an
    HDF5 formated file.  This class creates a TabProps StateTable object,
    reads datafrom a table into the StateTable object, and can query the
    StateTable object for the value of a dependent variable given values
    for independent variables, as well as return names for independent
    and dependent variables, and verify tables by checking the names of
    the dependent variables requested by the user in the input file to
    dependent variables tabulated in the table. Functionality will also be
    added to utilize the StateTable functions to convert the table data to
    a matlab file to easily investigate the results of the table creation.

 */

namespace Uintah {
class ArchesLabel; 
class TimeIntegratorLabel; 
class TabPropsInterface : public MixingRxnModel {

public:

  TabPropsInterface( const ArchesLabel* labels );

  ~TabPropsInterface();


  //see MixingRxnModel.h
  void problemSetup( const ProblemSpecP& params );
  
  /** @brief Compare dependent variables found in input file to dependent variables found in table file */
  void const verifyDV( bool diagnosticMode, bool strictMode );

  /** @brief Compare independent variables found in input file to independent variables found in table file */
  void const verifyIV( bool diagnosticMode, bool strictMode );

  //see MixingRxnModel.h
  void const verifyTable( bool diagnosticMode, bool strictMode );

  /** @brief Gets the thermochemical state for a patch */
  void sched_getState( const LevelP& level, 
                       SchedulerP& sched, 
                       const TimeIntegratorLabel* time_labels, 
                       const bool initialize,          // tells the method to allocateAndPut
                       const bool with_energy_exch,    // tells the method that energy exchange is on
                       const bool modify_ref_den );    // tells the method to modify the reference density 

  /** @brief See sched_get_state */ 
  void getState( const ProcessorGroup* pc, 
                 const PatchSubset* patches, 
                 const MaterialSubset* matls, 
                 DataWarehouse* old_dw, 
                 DataWarehouse* new_dw, 
                 const TimeIntegratorLabel* time_labels, 
                 const bool initialize, 
                 const bool with_energy_exch, 
                 const bool modify_ref_den );

  /** @brief Gets the thermochemical state for a patch */
  void sched_computeHeatLoss( const LevelP& level, 
                              SchedulerP& sched, 
                              const bool intialize_me ); 

  /** @brief See sched_computeHeatLoss */ 
  void computeHeatLoss( const ProcessorGroup* pc, 
                        const PatchSubset* patches, 
                        const MaterialSubset* matls, 
                        DataWarehouse* old_dw, 
                        DataWarehouse* new_dw, 
                        const bool intialize_me ); 


  /** @brief A temporary solution to deal with boundary conditions on properties until Properties.cc is eliminated */ 
  void oldTableHack( const InletStream& inStream, Stream& outStream, bool calcEnthalpy, const string bc_type );

  /** @brief This will initialize the enthalpy to a table value for the first timestep */ 
  void sched_computeFirstEnthalpy( const LevelP& level, SchedulerP& sched ); 
  /** @brief See sched_computeFirstEnthalpy */ 
  void computeFirstEnthalpy( const ProcessorGroup* pc, 
                             const PatchSubset* patches, 
                             const MaterialSubset* matls, 
                             DataWarehouse* old_dw, 
                             DataWarehouse* new_dw ); 

  // Load list of dependent variables from the table
  // Return vector<string>& (reference to allDepVarNames())
  const vector<string> & getAllDepVars();

  // Load list of independent variables from the table
  // Return vector<string>& (reference to allIndepVarNames())
  const vector<string> & getAllIndepVars();

  inline double getSingleState( string dv, vector<double> iv ) {
    double result = 0.0; 
    return result = d_statetbl.query(  dv, &iv[0] ); 
  };


protected :

private:

  // boolean to tell you if table has been loaded
  bool d_table_isloaded;
  
  // booleans for verification methods
  bool d_diagnostic_mode;
  bool d_strict_mode;


  // heat loss values for non-adiabatic conditions 
  double d_hl_outlet; 
  double d_hl_pressure; 
  double d_hl_scalar_init; 

  IntVector d_ijk_den_ref; 

  // vectors to store independent, dependent variable names from table file
  vector<string> d_allIndepVarNames;
  vector<string> d_allDepVarNames;

  // vectors to store independent, dependent variable names from input file
  vector<string> d_allUserDepVarNames;
  vector<string> d_allUserIndepVarNames;
    
  // vector to store independent variable values for call to StateTable::query
  vector<double> d_indepVarValues;

  // StateTable object to represent the table data
  StateTable d_statetbl;

  // string to hold filename
  string d_tableFileName;

  // A dependent variable container
  struct ADepVar {

    string name; 
    CCVariable<double> data; 

  };

}; // end class TabPropsInterface
  
} // end namespace Uintah

#endif
