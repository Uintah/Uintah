/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//----- ClassicTableInterface.h --------------------------------------------------

#ifndef Uintah_Component_Arches_ClassicTableInterface_h
#define Uintah_Component_Arches_ClassicTableInterface_h

#include <sci_defs/kokkos_defs.h>
#include <CCA/Components/Arches/ChemMixV2/ClassicTable.h>

#define MAX_NUM_DEP_VARS 15

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
class BoundaryCondition_new;
class MixingRxnModel;

class ClassicTableInterface : public MixingRxnModel {

public:

  ClassicTableInterface( MaterialManagerP& materialManager );

  ~ClassicTableInterface();

  void problemSetup( const ProblemSpecP& params );

  /** @brief Gets the thermochemical state for a patch
      @param initialize         Tells the method to allocateAndPut
      @param modify_ref_den     Tells the method to modify the reference density */
  void sched_getState( const LevelP& level,
                       SchedulerP& sched,
                       const int time_substep,
                       const bool initialize,
                       const bool modify_ref_den );

  /** @brief Gets the thermochemical state for a patch
      @param initialize         Tells the method to allocateAndPut
      @param modify_ref_den     Tells the method to modify the reference density */
  void getState( const ProcessorGroup* pc,
                 const PatchSubset* patches,
                 const MaterialSubset* matls,
                 DataWarehouse* old_dw,
                 DataWarehouse* new_dw,
                 const int time_substep,
                 const bool initialize,
                 const bool modify_ref_den );


  /** @brief returns the heat loss bounds from the table **/
  inline std::vector<double> get_hl_bounds(){
    std::vector<double> bounds;
    bounds.push_back(d_hl_lower_bound);
    bounds.push_back(d_hl_upper_bound);
    return bounds; };

  /*********interp derived classes*****************************************/

  /** @brief A base class for Interpolation */

  //*******************************************end interp classes//

  typedef std::map<std::string, DepVarCont >       DepVarMap;
  typedef std::map<std::string, int >               IndexMap;

  /** @brief Return a table lookup for a variable given the independent variable space. **/
  double getTableValue( std::vector<double>, std::string );

  /** @brief Match the requested dependent variable with their table index. **/
  void tableMatching();

  /** @brief Return a table lookup for a variable given the independent variables and set of inerts (may be an empty set) - Grid wise **/
  double getTableValue( std::vector<double> iv, std::string depend_varname, StringToCCVar inert_mixture_fractions, IntVector c);

  /** @brief Return a table lookup for a variable given the independent variables and set of inerts (may be an empty set) - single point wise**/
  double getTableValue( std::vector<double> iv, std::string depend_varname, doubleMap inert_mixture_fractions, bool do_inverse = false );

  /** @brief Method to find the index for any dependent variable.  **/
  int inline findIndex( std::string name ){

    int index = -1;

    for ( int i = 0; i < d_varscount; i++ ) {

      if ( name.compare( d_allDepVarNames[i] ) == 0 ) {
        index = i;
        break;
      }
    }

    if ( index == -1 ) {
      std::ostringstream exception;
      exception << "Error: The variable " << name << " was not found in the table." << "\n" <<
        "Please check your input file and try again. Total variables in table = " << d_allDepVarNames.size()<< std::endl;
      throw InternalError(exception.str(),__FILE__,__LINE__);
    }

    return index;
  }

private:

  Interp_class<MAX_NUM_DEP_VARS> * ND_interp; ///< classic table object

  bool d_table_isloaded;    ///< Boolean: has the table been loaded?

  // Specifically for the classic table:
  double d_hl_lower_bound;  ///< Heat loss lower bound
  double d_hl_upper_bound;  ///< Heat loss upper bound


  int d_indepvarscount;       ///< Number of independent variables
  int d_varscount;            ///< Total dependent variables
  int d_nDepVars;             ///< number of dependent variables requested by arches

  std::string d_enthalpy_name;

  IndexMap d_depVarIndexMap;                      ///< Reference to the integer location of the variable
  IndexMap d_enthalpyVarIndexMap;                 ///< Reference to the integer location of variables for heat loss calculation


  /// A dependent variable wrapper
  void getIndexInfo();
  void getEnthalpyIndexInfo();

}; // end class ClassicTableInterface
} // end namespace Uintah

#endif
