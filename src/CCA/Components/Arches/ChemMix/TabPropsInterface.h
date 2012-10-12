/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//----- TabPropsInterface.h --------------------------------------------------

#ifndef Uintah_Component_Arches_TabPropsInterface_h
#define Uintah_Component_Arches_TabPropsInterface_h

#include <tabprops/StateTable.h>

#include <tabprops/Archive.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>


/**
 * @class  TabPropsInterface
 * @author Jeremy Thornock, Charles Reid
 * @date   May 2009
 *
 * @brief Table interface for those created with TabProps.
 *
 * @todo
 * Add support for multiple scalar variance
 *
 * @details
 * This class provides and interface to TabProps formatted tables.  See the
 * TabProps project here:
 * https://software.crsim.utah.edu/trac/wiki/TabProps
 * to get more information regarding TabProps and the its tabluar format.

This code checks for the following tags/attributes in the input file:
The UPS interface for TabProps is:

\code
<TabProps                       spec="OPTIONAL NO_DATA" >
  <inputfile                    spec="REQUIRED STRING" /> <!-- table to be opened -->
  <cold_flow                    spec="OPTIONAL BOOLEAN"/> <!-- used for simple stream mixing (no rxn) -->
  <hl_scalar_init               spec="OPTIONAL DOUBLE" /> <!-- initial heat loss value in the domain -->
  <noisy_hl_warning             spec="OPTIONAL NO_DATA"/> <!-- warn when heat loss is clipped to bounds -->
  <lower_hl_bound               spec="OPTIONAL DOUBLE"/> <!-- In the property table, the lower bound for heat loss.  default = -1 -->
  <upper_hl_bound               spec="OPTIONAL DOUBLE"/> <!-- In the property table, the upper bound for heat loss.  default = +1 -->
  <coal                         spec="OPTIONAL NO_DATA"
                                attribute1="fp_label REQUIRED STRING"
                                attribute2="eta_label REQUIRED STRING"/>
                                <!-- Attributes must match the transported IVs specified in the TransportEqn node -->
</TabProps>

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
class TabPropsInterface : public MixingRxnModel {

public:

  TabPropsInterface( ArchesLabel* labels, const MPMArchesLabel* MAlabels );

  ~TabPropsInterface();

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

  /** @brief      Returns a single dependent variable, given a vector of independent variable values
      @param dv   The name of the dependent variable to look up in the table
      @param iv   The vector of indepenent variable values */
  inline double getSingleState( string dv, vector<double> iv ) {
    double result = 0.0;
    //cout_tabledbg << "From your table, looking up: " << dv << endl;
    return result = d_statetbl.query(  dv, &iv[0] );
  };

  /** @brief          Returns a single dependent variable, given a vector of independent variable values
      @param spline   The spline information for the dep. var.
      @param iv       The vector of indepenent variable values */
  inline double getSingleState( const InterpT* spline, std::string dv, vector<double> iv ) {
    double result = 0.0;
    //cout_tabledbg << "From your table, looking up a variable using spline information: " << dv << endl;
    return result = d_statetbl.query(  spline, &iv[0] );
  };

  /** @brief Dummy initialization as required by MPMArches */
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );

  /** @brief Dummy initialization as required by MPMArches */
  void dummyInit( const ProcessorGroup* pc,
                  const PatchSubset* patches,
                  const MaterialSubset* matls,
                  DataWarehouse* old_dw,
                  DataWarehouse* new_dw );

  /** @brief Gets the Spline information for TabProps.  Spline info is used because it is more efficient that passing strings */
  void getSplineInfo();
  /** @brief Gets the Spline information for TabProps.  This is specific to the enthalpy vars */
  void getEnthalpySplineInfo();

  typedef std::map<std::string, const InterpT*>   SplineMap;

  enum BoundaryType { DIRICHLET, NEUMANN };

  struct DepVarCont {

    CCVariable<double>* var;
    const InterpT* spline;

  };

  typedef std::map<string, DepVarCont >       DepVarMap;

  double getTableValue( std::vector<double>, std::string );

	void tableMatching(){};

protected :

private:

  bool d_table_isloaded;    ///< Boolean: has the table been loaded?
  bool d_noisy_hl_warning;  ///< Provide information about heat loss clipping

  double d_hl_scalar_init;  ///< Heat loss value for non-adiabatic conditions
  double d_hl_lower_bound;  ///< Heat loss lower bound
  double d_hl_upper_bound;  ///< Heat loss upper bound

  IntVector d_ijk_den_ref;                ///< Reference density location


  vector<string> d_allUserDepVarNames;    ///< Vector storing all independent varaible names requested in input file

  StateTable d_statetbl;                  ///< StateTable object to represent the table data
  SplineMap  d_depVarSpline;              ///< Map of spline information for each dependent var
  SplineMap  d_enthalpyVarSpline;         ///< Holds the sensible and adiabatic enthalpy spline information
                                          // note that this ^^^ is a bit of a quick fix. Should find a better way later.

  /// A dependent variable wrapper
  struct ADepVar {
    string name;
    CCVariable<double> data;
  };

  /** @brief  Helper for filling the spline map */
  inline void insertIntoSplineMap( const string var_name, const InterpT* spline ){

    SplineMap::iterator i = d_depVarSpline.find( var_name );

    if ( i == d_depVarSpline.end() ) {

      cout_tabledbg << "Inserting " << var_name << " spline information into storage." << endl;

      i = d_depVarSpline.insert( make_pair( var_name, spline ) ).first;

    }
    return;
  };

}; // end class TabPropsInterface

} // end namespace Uintah

#endif
