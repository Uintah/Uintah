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
#include <tabprops/StateTable.h>
#include <CCA/Components/Arches/Mixing/InletStream.h>
#include <CCA/Components/Arches/Mixing/Stream.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>


/**
 * @class  TabPropsInterface
 * @author Jeremy Thornock
 * @date   May 2009 
 *
 * @brief Table interface for those created with TabProps.
 *
 *
The UPS interface for TabProps is: 
\code
    <Properties>
        <TabProps>
            <inputfile>REQUIRED STRING</inputfile>
            <strict_mode>OPTIONAL BOOL</strict_mode>
            <diagnostic_mode>OPTIONAL BOOL</diagnostic_mode>
        </TabProps>
    </Properties>

    <DataArchiver>
        <save name=STRING table_lookup="true"> <!-- note that STRING must match the name in the table -->
    </DataArchiver>
 \endcode
 * Any variable that is saved to the UDA in the dataarchiver block is automatically given a VarLabel.  
 *
 * To-do's: 
 *  - Need to add support for multiple scalar variance
 * 
 *
 *
 */

namespace Uintah {
class ArchesLabel; 
class MPMArchesLabel; 
class TimeIntegratorLabel; 
class TabPropsInterface : public MixingRxnModel {

public:

  TabPropsInterface( const ArchesLabel* labels, const MPMArchesLabel* MAlabels );

  ~TabPropsInterface();

  /** @brief See MixingRxnModel.h */ 
  void problemSetup( const ProblemSpecP& params );
  
  /** @brief Compare dependent variables found in input file to dependent variables found in table file */
  void const verifyDV( bool diagnosticMode, bool strictMode );

  /** @brief Compare independent variables found in input file to independent variables found in table file */
  void const verifyIV( bool diagnosticMode, bool strictMode );

  /// @brief See MixingRxnModel.h 
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
                              const bool intialize_me, const bool calcEnthalpy ); 

  /** @brief See sched_computeHeatLoss */ 
  void computeHeatLoss( const ProcessorGroup* pc, 
                        const PatchSubset* patches, 
                        const MaterialSubset* matls, 
                        DataWarehouse* old_dw, 
                        DataWarehouse* new_dw, 
                        const bool intialize_me, const bool calcEnthalpy ); 


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

  /** @brief Returns a list of all tabulated dependent variables */ 
  const vector<string> & getAllDepVars();

  /** @brief Returns a list of all independent variables */ 
  const vector<string> & getAllIndepVars();

  inline double getSingleState( string dv, vector<double> iv ) {
    double result = 0.0; 
    return result = d_statetbl.query(  dv, &iv[0] ); 
  };

  /** @brief Dummy initialization as required by MPMArches */
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );
  /** @brief See sched_dummyInit */ 
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );


protected :

private:

  /// Indicates if the table has been loaded. 
  bool d_table_isloaded; 
 
  /// Checks saved variables from UPS against those in the table
  bool d_diagnostic_mode;  
  /// Causes the code to exit if saved variables indicated in UPS do not match with those in the table
  bool d_strict_mode;


  /// Heat loss value at outlet boundary types
  double d_hl_outlet; 
  /// Heat loss value at pressure boundary types
  double d_hl_pressure; 
  /// Initial heat loss value in the domain 
  double d_hl_scalar_init; 

  /// Reference density location 
  IntVector d_ijk_den_ref; 

  /// All independent variable names
  vector<string> d_allIndepVarNames;
  /// All dependent variable names
  vector<string> d_allDepVarNames;
  
  /// All user requested dependent variable names
  vector<string> d_allUserDepVarNames;
    
  /// StateTable object representing the tabular information 
  StateTable d_statetbl;

  /// A dependent variable container
  struct ADepVar {

    string name; 
    CCVariable<double> data; 

  };

}; // end class TabPropsInterface
  
} // end namespace Uintah

#endif
