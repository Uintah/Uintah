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


//----- ClassicTableInterface.h --------------------------------------------------

#ifndef Uintah_Component_Arches_ClassicTableInterface_h
#define Uintah_Component_Arches_ClassicTableInterface_h

#include <tabprops/StateTable.h>
#include <CCA/Components/Arches/Mixing/InletStream.h>
#include <CCA/Components/Arches/Mixing/Stream.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <Core/Util/DebugStream.h>

#include   <string>

/**
 * @class  ClassicTableInterface
 * @author Jeremy Thornock
 * @date   Jan 2011
 *
 * @brief Table interface for those created with the Classic Arches Format 
 *
 * @todo
 *
 * @details
 * This class provides and interface to classic Arches formatted tables.  
 
This code checks for the following tags/attributes in the input file:
The UPS interface is: 

\code
    <Properties>
      <ClassicTable>
        <inputfile>REQUIRED STRING</inputfile>
      </ClassicTable>
    </Properties>

    <DataArchiver>
        <save name=STRING table_lookup="true"> <!-- note that STRING must match the name in the table -->
    </DataArchiver>
\endcode

 * Any variable that is saved to the UDA in the dataarchiver block is automatically given a VarLabel.  
 *
 * If you have trouble reading your table, you can "setenv SCI_DEBUG TABLE_DEBUG:+" to get a 
 * report of what is going on in the table reader.
 *
 *
*/


namespace Uintah {


class ArchesLabel; 
class MPMArchesLabel; 
class TimeIntegratorLabel; 
class ClassicTableInterface : public MixingRxnModel {

public:

  ClassicTableInterface( const ArchesLabel* labels, const MPMArchesLabel* MAlabels );

  ~ClassicTableInterface();

  void problemSetup( const ProblemSpecP& params );
  
  /** @brief Gets the thermochemical state for a patch 
      @param initialize         Tells the method to allocateAndPut 
      @param with_energy_exch   Tells the method that energy exchange is on
      @param modify_ref_den     Tells the method to modify the reference density */
  void sched_getState( const LevelP& level, 
                       SchedulerP& sched, 
                       const TimeIntegratorLabel* time_labels, 
                       const bool initialize,
                       const bool with_energy_exch,
                       const bool modify_ref_den ); 

  /** @brief Gets the thermochemical state for a patch 
      @param initialize         Tells the method to allocateAndPut 
      @param with_energy_exch   Tells the method that energy exchange is on
      @param modify_ref_den     Tells the method to modify the reference density */
  void getState( const ProcessorGroup* pc, 
                 const PatchSubset* patches, 
                 const MaterialSubset* matls, 
                 DataWarehouse* old_dw, 
                 DataWarehouse* new_dw, 
                 const TimeIntegratorLabel* time_labels, 
                 const bool initialize, 
                 const bool with_energy_exch, 
                 const bool modify_ref_den );

  /** @brief Schedule computeHeatLoss */
  void sched_computeHeatLoss( const LevelP& level, 
                              SchedulerP& sched, 
                              const bool intialize_me, const bool calcEnthalpy ); 

  /** @brief  Computes the heat loss from the table */
  void computeHeatLoss( const ProcessorGroup* pc, 
                        const PatchSubset* patches, 
                        const MaterialSubset* matls, 
                        DataWarehouse* old_dw, 
                        DataWarehouse* new_dw, 
                        const bool intialize_me, 
                        const bool calcEnthalpy ); 


  /** @brief A temporary solution to deal with boundary conditions on properties until Properties.cc is eliminated */ 
  void oldTableHack( const InletStream& inStream, Stream& outStream, bool calcEnthalpy, const string bc_type );

  /** @brief  schedules computeFirstEnthalpy */
  void sched_computeFirstEnthalpy( const LevelP& level, SchedulerP& sched ); 

  /** @brief This will initialize the enthalpy to a table value for the first timestep */
  void computeFirstEnthalpy( const ProcessorGroup* pc, 
                             const PatchSubset* patches, 
                             const MaterialSubset* matls, 
                             DataWarehouse* old_dw, 
                             DataWarehouse* new_dw ); 

  /** @brief    Load list of dependent variables from the table 
      @returns  A vector<string>& that is a reference to the list of all dependent variables */
  const vector<string> & getAllDepVars();

  /** @brief    Load list of independent variables from the table
      @returns  A vector<string>& that is a reference to the list of all independent variables */ 
  const vector<string> & getAllIndepVars();

  /** @brief Dummy initialization as required by MPMArches */
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );

  /** @brief Dummy initialization as required by MPMArches */
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

  /** @brief Load table into memory */ 
  void loadMixingTable( const string & inputfile );

  /** @brief Interpolate values from the table using the species index */ 
  double tableLookUp( std::vector<double> iv, int var_index);

  enum BoundaryType { DIRICHLET, NEUMANN };

  struct DepVarCont {

    CCVariable<double>* var; 
    int index; 

  }; 

  typedef std::map<string, DepVarCont >       DepVarMap;
  typedef std::map<string, int >               IndexMap; 

protected :

private:

  bool d_table_isloaded;    ///< Boolean: has the table been loaded?
  
  double d_hl_scalar_init;  ///< Heat loss value for non-adiabatic conditions
  // Specifically for the classic table: 
  double d_f_stoich;        ///< Stoichiometric mixture fraction 
  double d_H_fuel;          ///< Fuel Enthalpy
  double d_H_air;           ///< Oxidizer Enthalpy
  
  int d_indepvarscount;     ///< Number of independent variables
  int d_varscount;          ///< Total dependent variables

  IntVector d_ijk_den_ref;                ///< Reference density location

  IndexMap d_depVarIndexMap;              ///< Reference to the integer location of the variable
  IndexMap d_enthalpyVarIndexMap;         ///< Referece to the integer location of variables for heat loss calculation

  std::vector<string> d_allIndepVarNames;      ///< Vector storing all independent variable names from table file
  std::vector<int>    d_allIndepVarNum;        ///< Vector storing the grid size for the Independant variables
  std::vector<string> d_allDepVarNames;        ///< Vector storing all dependent variable names from the table file
  std::vector<string> d_allDepVarUnits;        ///< Units for the dependent variables 

  vector<string> d_allUserDepVarNames;    ///< Vector storing all independent varaible names requested in input file

  //previous Arches specific variables: 
  std::vector<std::vector<double> > i1; 
  std::vector<double> i2; 
  std::vector<double> i3; 
  std::vector <std::vector <double> > table;


  /// A dependent variable wrapper
  struct ADepVar {
    string name; 
    CCVariable<double> data; 
  };

  /// @brief Method to find the index for any dependent variable.  
  int inline findIndex( std::string name ){ 

    int index = -1; 

    for ( int i = 0; i < d_varscount; i++ ) { 

      if ( name.compare( d_allDepVarNames[i] ) == 0 ) {
        index = i; 
        break; 
      }
    }

    if ( index == -1 ) {
      ostringstream exception;
      exception << "Error: The variable " << name << " was not found in the table." << "\n" << 
        "Please check your input file and try again. " << endl;
      throw InternalError(exception.str(),__FILE__,__LINE__);
    }

    return index; 
  }

  void getIndexInfo(); 
  void getEnthalpyIndexInfo(); 

}; // end class ClassicTableInterface
  
} // end namespace Uintah

#endif
