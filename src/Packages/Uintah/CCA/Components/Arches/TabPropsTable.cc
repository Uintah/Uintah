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


//----- TabPropsTable.cc --------------------------------------------------

// includes for Arches
#include <Packages/Uintah/CCA/Components/Arches/TabProps/StateTable.h>
#include <Packages/Uintah/CCA/Components/Arches/TabPropsTable.h>
#include <Packages/Uintah/CCA/Components/Arches/MixingRxnTable.h>
#include <Packages/Uintah/CCA/Components/Arches/Properties.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>

// includes for Uintah
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <dirent.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;



/**************************************************************************
 TabPropsTable.cc

INPUT FILE TAGS
This code checks for the following tags/attributes in the input file:
    <Properties>
        <TabPropsTable>
            <table_file_name>THIS FIELD IS REQUIRED</table_file_name>
            <strict_mode>true/false (not required)</strict_mode>
            <diagnostic_mode>true/false (not required)</diagnostic_mode>
        </TabPropsTable>
    </Properties>

    <DataArchiver>
        ...
        <save name="BlahBlahBlah" table_lookup="true/false">
        ...
    </DataArchiver>

This is used to construct the vector of user-requested dependent variables.

WARNINGS
The code will throw exceptions for the following reasons:
- no <table_file_name> specified in the intput file (ProblemSpecP require() method throws the error)
- the getState method is run without the problemSetup method being run (b/c the problemSetup sets 
  the boolean b_table_isloaded to true when a table is loaded, and you can't run getState without 
  first loading a table)
- if bool strictMode is true, and a dependent variable specified in the input file does not
  match the names of any of the dependent variables in the table
- the getDepVars or getIndepVars methods are run on a TabPropsTable object which hasn't loaded
  a table yet (i.e. hasn't run problemSetup method yet) 

***************************************************************************/



//****************************************************************************
// Default constructor for TabPropsTable
//****************************************************************************
TabPropsTable::TabPropsTable()
{
}

//****************************************************************************
// Destructor
//****************************************************************************
TabPropsTable::~TabPropsTable()
{
}

//****************************************************************************
// TabPropsTable problemSetup
//
// Obtains parameters from the input file (via an instance of the ProblemSpecP 
// class). Once the relevant info is obtained from the input file, a list of 
// user-requested dependent variables, and a list of independent and dependent 
// variables contained in the table, are all populated. A verification routine
// is then run to compare the user-requested dependent variables to the 
// dependent variables contained in the table.
// 
//****************************************************************************
void
TabPropsTable::problemSetup( const ProblemSpecP& propertiesParameters )
{
  /*

  problemSetup notes:
  --------------------
  input file: 
  <Properties>
  ...
    <TabPropsTable>
      <table_file_name>...</table_file_name>
    </TabPropsTable>
  ...
  </Properties>

  Properties.cc:
  --------------
  1. Create ProblemSpecP object "db" for <Properties> tags
  2. Create new TabPropsTable object using "scinew TabPropsTable"
  (it is up to the object to know what it needs to extract from the <Properties> tags)
  3. Run TabPropsTable.problemSetup(db)

  with regard to the list of independent and dependent variables:
  properties.cc needs the list in CCVariable format (to interface w/ data warehouse)
  tabprops.cc needs the list in vector<string> format (to interface w/ TabProps)
  Properties.cc extracts the list from the input file and adds it to a "table lookup list"
  two different maps, to connect strings (the "regular" name) to VarLabel and CCVariable
    VarMap d_varMap
    LabelMap d_labelMap
  in Properties::computeProps, iterate through the varMap and call getstate for each one

  TabPropsTable::problemSetup
  ---------------------------

  */

  // Step 1 - create sub-ProblemSpecP object
  string tableFileName;
  ProblemSpecP db_tabpropstable = propertiesParameters->findBlock("TabPropsTable");
  
  // Step 2 - obtain object parameters
  db_tabpropstable->require( "table_file_name", tableFileName );
  db_tabpropstable->getWithDefault( "strict_mode", b_strict_mode, false );
  db_tabpropstable->getWithDefault( "diagnostic_mode", b_diagnostic_mode, false );

  // Step 3 - Check for and deal with filename extension
  // - if table file name has .h5 extension, remove it
  // - otherwise, assume it is an .h5 file but no extension was given
  string extension (tableFileName.end()-3,tableFileName.end());
  if( extension == ".h5" || extension == ".H5" ) {
    tableFileName = tableFileName.substr( 0, tableFileName.size() - 3 );
  }

  // Step 4 - Check if tableFileName exists
  DIR *check = opendir(tableFileName.c_str());
  if ( check == NULL ) {
    ostringstream warn;
    warn << "ERROR:Arches:TabPropsTable  The table " << tableFileName << " does not exist. ";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  
  // Step 5 - Load data from HDF5 file into StateTable
  statetbl.read_hdf5(tableFileName);

  // Now TabPropsTable must obtain three lists:
  // - user-defined dependent variables
  // - all dependent variables in table
  // - all independent variables in table
  // 
  // This is done so that TabPropsTable can check to see if user-defined dependent variables
  // match table dependent variables (in verifyTable method)

  // Step 6 - get user-defined/user-requested dependent variables
  ProblemSpecP db_rootnode = propertiesParameters; // need a non-const instance of propertiesParameters
  ProblemSpecP db_vars;

  db_rootnode = db_rootnode->getRootNode(); // this throws a 'discards qualifiers' error if db_rootnode is const
  db_vars = db_rootnode->findBlock("DataArchiver");
  if (db_vars) {
    for( ProblemSpecP db_var = db_vars->findBlock("save");db_var !=0; 
       db_var = db_var->findNextBlock("save") ) {
      string tableLookup;
      string name;
      db_var->getAttribute("table_lookup", tableLookup);
      db_var->getAttribute("label", name);

      if( tableLookup=="true" ){
        allUserDepVarNames.push_back( name ); 
      } 
    } 
  } 

  // Step 7 - read all of the independent and dependent variables in the table into memory
  allIndepVarNames = statetbl.get_indepvar_names();
  allDepVarNames   = statetbl.get_depvar_names();

  // Step 8 - set "is the table loaded?" boolean to true
  b_table_isloaded = true;

  // Step 9 - verify table
  // diagnostic mode - checks variables requested saved in the input file to variables found in the table
  // strict mode - if a dependent variable is requested saved in the input file, but NOT found in the table, throw error
  verifyTable( b_diagnostic_mode, b_strict_mode );

}

//****************************************************************************
// TabPropsTable getState 
//
// This method calls the StateTable::query method. It passes independent 
// variable values and user-requested dependent variables to StateTable::query
// (which then performs a table lookup), and the values of the requested
// dependent variables are returned in a map<string,double>.
//
// This method is called from Properties::computeProps
//
//****************************************************************************
map<string,double>
TabPropsTable::getState( const double * indepVarValues, vector<string> userDepVarNames )
{
  if( b_table_isloaded == false ) {
    throw InternalError("You requested a thermodynamic state, but you did not specify which table you wanted to use!!!",__FILE__,__LINE__);
  }

  map<string,double> returnValuesMap;
  for( unsigned int i=0 ; i < userDepVarNames.size() ; i++) {
    // obtain numerical value from table and put in myQueryResults
    //myQueryResults[i] = statetbl.query( userDepVarNames[i], &indepVarValues[i] );
    // put the string and the double into the map
    //returnValuesMap.insert(make_pair(userDepVarNames[i],myQueryResults[i]));

    // Put the string (passed to the function when it is called) and the double (obtained from the StateTable::query function)
    //  into the depVarValuesMap
    returnValuesMap.insert(make_pair(userDepVarNames[i],statetbl.query( userDepVarNames[i], &indepVarValues[i] ) ) );
  }
  return returnValuesMap;
}

//****************************************************************************
// TabPropsTable verifyTable 
//
// Compares list of user-requested dependent variables (obtained from input file)
// to list of dependent variables contained in the table.
//
// If strict mode is on, an exception will be thrown if any user-requested dependent
// variables are not found in the table.
// 
// If diagnostic mode is on, each user-requested dependent variable will be listed
// and a message will say whether that dependent variable was found in the table
// or not.
//
//****************************************************************************
void
TabPropsTable::verifyTable(  bool diagnosticMode,
                             bool strictMode ) const
{
  // for j in list of user-requested dependent variables,
  //  for i in list of all dependent variables,
  //   if userdefineddepvars[j] == alldepvars[i]
  //    some_boolean = true;
  // 
  int numNegativeResults = 0;
  bool toggle;
  std::vector<bool> myVerifyResults;

  for( unsigned int i=0; i < allUserDepVarNames.size(); i++) {
    toggle = false; // toggle = "is this user-requested dependent variable in the table?"
    
    for( unsigned int j=0; j < allDepVarNames.size(); j++) {
      if( allUserDepVarNames[i] == allDepVarNames[j] ) {
        toggle = true;
      } 
    }
     
    if( toggle == false ) { 
      numNegativeResults += 1; 
    }
    myVerifyResults.push_back(toggle);
  } // myVerifyResults = size of allUserDepVarNames (contains bool toggle)

  proc0cout << "Hello, this is the TabProps table reader module of the Arches code." << endl;
  proc0cout << "I am checking the dependent variables requested in the input files against the dependent variables in the TabProps table." << endl;

  if(diagnosticMode == true){
    //print results
    for( unsigned int i=0; i < allUserDepVarNames.size(); i++) {
      if( myVerifyResults[i] == true ) {
        proc0cout << "----- The dependent variable " << allUserDepVarNames[i] << " was found in the table." << endl;
      } 
      else if(myVerifyResults[i] == false) {
        proc0cout << "XXXXX The dependent variable " << allUserDepVarNames[i] << " was NOT found in the table!!!" << endl;
      }
    }

    proc0cout << endl;
    proc0cout << "==================================================================" << endl;
    proc0cout << endl;
  }

  if(numNegativeResults > 0){ // some of the dependent variables the user requested are NOT in the table

    if(strictMode == true) {
      throw InternalError( "You specified dependent variables in your input file that you wanted to save, "
                             "but some of them were not found in the table!!!",__FILE__,__LINE__);
    } else if(strictMode == false) {
       
      // if not strict mode, just print a warning
      proc0cout << "WARNING!!!" << endl;
      proc0cout << "The table verification routine found " << numNegativeResults << " dependent variable(s) requested in your input file that were NOT found in the table." << endl;
      proc0cout << "Did you want one of the following dependent variables in the table?" << endl;
      for( unsigned int j=0; j < allDepVarNames.size(); j++) {
        proc0cout << allDepVarNames[j] << endl;
      }
  
    }
  }  
  else if(numNegativeResults == 0) {
    //you're OK, either way (strict mode or not doesn't matter)
    proc0cout << "Success!" << endl;
    proc0cout << "All " << allUserDepVarNames.size() << " of the dependent variables you requested in your input file were found in the table." << endl;
  } 
}

//****************************************************************************
// TabPropsTable getDepVars
//
// This method will first check to see if the table is loaded; if it is, it
// will return a reference to allDepVarNames, which is a private vector<string>
// of the TabPropsTable class
//
// (If it is a reference to a private class variable, can the reciever of
//  the reference vector<string> still access it?)
//
//****************************************************************************
const vector<string> &
TabPropsTable::getDepVars()
{
  if( b_table_isloaded == true ) {
    vector<string>& allDepVarNames_ref(allDepVarNames);
    return allDepVarNames_ref;
  } else {
    ostringstream exception;
    exception << "Error: You requested a list of dependent variables " <<
                 "before specifying the table that you were using. " << endl;
    throw InternalError(exception.str(),__FILE__,__LINE__);
  }
}

//****************************************************************************
// TabPropsTable getIndepVars
//
// This method will first check to see if the table is loaded; if it is, it
// will return a reference to allIndepVarNames, which is a private
// vector<string> of the TabPropsTable class
// 
// (If it is a reference to a private class variable, can the reciever of
//  the reference vector<string> still access it?)
// 
//****************************************************************************
const vector<string> &
TabPropsTable::getIndepVars()
{
  if( b_table_isloaded == true ) {
    vector<string>& allIndepVarNames_ref(allIndepVarNames);
    return allIndepVarNames_ref;
  } 
  else {
    ostringstream exception;
    exception << "Error: You requested a list of independent variables " <<
                 "before specifying the table that you were using. " << endl;
    throw InternalError(exception.str(),__FILE__,__LINE__);
  }
}

