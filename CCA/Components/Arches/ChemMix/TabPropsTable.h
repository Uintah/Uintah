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


//----- TabPropsTable.h --------------------------------------------------

#ifndef Uintah_Component_Arches_TabPropsTable_h
#define Uintah_Component_Arches_TabPropsTable_h

// includes for Arches
#include <CCA/Components/Arches/ChemMix/TabProps/StateTable.h>


/**
 * @class  TabPropsTable
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
class TabPropsTable : public MixingRxnModel {

public:

  TabPropsTable();

  ~TabPropsTable();

  typedef map<unsigned int, const VarLabel* > LabelMap;
  typedef map<unsigned int, bool> BoolMap;

  //see MixingRxnModel.h
  void problemSetup( const ProblemSpecP& params );
  
  /** @brief Compare dependent variables found in input file to dependent variables found in table file */
  void const verifyDV( bool diagnosticMode, bool strictMode );

  /** @brief Compare independent variables found in input file to independent variables found in table file */
  void const verifyIV( bool diagnosticMode, bool strictMode );

  //see MixingRxnModel.h
  void const verifyTable( bool diagnosticMode, bool strictMode );

  //see MixingRxnModel.h
  void getState(VarMap ivVar, VarMap dvVar, const Patch* patch);

  // Load list of dependent variables from the table
  // Return vector<string>& (reference to allDepVarNames())
  const vector<string> & getDepVars();

  // Load list of independent variables from the table
  // Return vector<string>& (reference to allIndepVarNames())
  const vector<string> & getIndepVars();

protected :

private:

  // boolean to tell you if table has been loaded
  bool b_table_isloaded;
  
  // booleans for verification methods
  bool b_diagnostic_mode;
  bool b_strict_mode;

  // vectors to store independent, dependent variable names from table file
  vector<string> allIndepVarNames;
  vector<string> allDepVarNames;

  // vectors to store independent, dependent variable names from input file
  vector<string> allUserDepVarNames;
  vector<string> allUserIndepVarNames;
    
  // vector to store independent variable values for call to StateTable::query
  // HOW TO INITIALIZE TO BE CORREC SIZE?
  vector<double> indepVarValues;

  // StateTable object to represent the table data
  StateTable statetbl;

  // string to hold filename
  string tableFileName;

}; // end class TabPropsTable
  
} // end namespace Uintah

#endif
