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


//----- TabPropsTable.cc --------------------------------------------------

// includes for Arches
#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <CCA/Components/Arches/ChemMix/TabProps/StateTable.h>
#include <CCA/Components/Arches/ChemMix/TabPropsTable.h>
#include <CCA/Components/Arches/Properties.h>

// includes for Uintah
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>
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
        <save name="BlahBlahBlah" table_lookup="true/false">
    </DataArchiver>

This is used to construct the vector of user-requested dependent variables.

WARNINGS
The code will throw exceptions for the following reasons:
- no <table_file_name> specified in the intput file
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
// Obtain parameters from the input file
// Construct lists of independent and dependent variables (from user and from table)
// Verify that these variables match
// 
//****************************************************************************
void
TabPropsTable::problemSetup( const ProblemSpecP& propertiesParameters )
{
  // Create sub-ProblemSpecP object
  string tableFileName;
  ProblemSpecP db_tabpropstable = propertiesParameters->findBlock("TabPropsTable");
  
  // Obtain object parameters
  db_tabpropstable->require( "table_file_name", tableFileName );
  db_tabpropstable->getWithDefault( "strict_mode", b_strict_mode, false );
  db_tabpropstable->getWithDefault( "diagnostic_mode", b_diagnostic_mode, false );

  // Check for and deal with filename extension
  // - if table file name has .h5 extension, remove it
  // - otherwise, assume it is an .h5 file but no extension was given
  string extension (tableFileName.end()-3,tableFileName.end());
  if( extension == ".h5" || extension == ".H5" ) {
    tableFileName = tableFileName.substr( 0, tableFileName.size() - 3 );
  }

  // Check if tableFileName exists
  DIR *check = opendir(tableFileName.c_str());
  if ( check == NULL ) {
    ostringstream warn;
    warn << "ERROR:Arches:TabPropsTable - The table " << tableFileName << " does not exist. ";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  
  // Load data from HDF5 file into StateTable
  statetbl.read_hdf5(tableFileName);

  // Extract independent and dependent variables from input file
  ProblemSpecP db_rootnode = propertiesParameters;
  ProblemSpecP db_vars;

  db_rootnode = db_rootnode->getRootNode();
  db_vars = db_rootnode->findBlock("DataArchiver");
  if (db_vars) {
    // Extract DVs from input file
    for(ProblemSpecP db_dv = db_vars->findBlock("save"); db_dv !=0; 
        db_dv = db_dv->findNextBlock("save") ) {
      string tableLookup;
      string dvName;
      db_dv->getAttribute("table_lookup", tableLookup);
      db_dv->getAttribute("label", dvName);

      if( tableLookup.compare("true")==0 ){
        allUserDepVarNames.push_back( dvName ); 
      } 
    } 
    // Extract IVs from input file
    for(ProblemSpecP db_iv = db_vars->findBlock("independent_variable"); db_iv != 0; db_iv = db_iv->findNextBlock("independent_variable")) {
        string ivName;
      db_iv->getAttribute("name",ivName);
      allUserIndepVarNames.push_back( ivName );
    }
  }

  // Extract independent and dependent variables from the table
  allIndepVarNames = statetbl.get_indepvar_names();
  allDepVarNames   = statetbl.get_depvar_names();

  // Confirm that table has been loaded into memory
  b_table_isloaded = true;

  // Verify table
  // - diagnostic mode - checks variables requested saved in the input file to variables found in the table
  // - strict mode - if a dependent variable is requested saved in the input file, but NOT found in the table, throw error
  verifyTable( b_diagnostic_mode, b_strict_mode );
}



//****************************************************************************
// TabPropsTable verifyTable 
//
//****************************************************************************
void const 
TabPropsTable::verifyTable(  bool diagnosticMode,
                             bool strictMode )
{
  verifyIV(diagnosticMode, strictMode);
  verifyDV(diagnosticMode, strictMode);
}


//****************************************************************************
// TabPropsTable verifyDV
//
//****************************************************************************
void const
TabPropsTable::verifyDV( bool diagnosticMode, bool strictMode )
{
  int numNegativeResults = 0;
  bool toggle;
  std::vector<bool> myVerifyResults;

  // assume toggle is false, but if match is found set toggle to true
  for(unsigned int i=0; i < allUserDepVarNames.size(); i++) {
    toggle = false;
    for(unsigned int j=0; j < allDepVarNames.size(); i++) {
      if(allUserDepVarNames[i] == allDepVarNames[j]) {
        toggle = true;
      }
    }
    if(toggle==false) {
      numNegativeResults += 1;
    }
    myVerifyResults.push_back(toggle);
  }

  proc0cout << "Arches:TabPropsTable will now check dependent variables requested in input file and compare to dependent variables in table." << endl;

  if(diagnosticMode == true) {
    // print full results
    for(unsigned int i=0; i < allUserDepVarNames.size(); i++) {
      if(myVerifyResults[i]==true) {
        cout << "The dependent variable " << allUserDepVarNames[i] << " was found in the table." << endl;
      } else if(myVerifyResults[i]==false) {
        cout << "WARNING: The dependent variable " << allUserDepVarNames[i] << " was NOT found in the table." << endl;
      }
    }
    proc0cout << "The following is a list of dependent variables found in the table:" << endl;
    for(unsigned int j=0; j < allDepVarNames.size(); j++) {
      proc0cout << allDepVarNames[j] << endl;
    }
  }

  if(numNegativeResults > 0) {
    if(strictMode == true) {
      throw InternalError("You requested in your input file that dependent variables be saved, but some were not found in the table.",__FILE__,__LINE__);
    } else if(strictMode == false) {
      proc0cout << "WARNING: Table verification routine found " << numNegativeResults <<
                   " dependent variables requested in your input file that were not found in the table." << endl;

      proc0cout << "The following is a list of dependent variables found in the table:" << endl;
      for(unsigned int j=0; j < allDepVarNames.size(); j++) {
        proc0cout << allDepVarNames[j] << endl;
      }
    }
  }
  else if(numNegativeResults == 0) {
    proc0cout << "Success!" << endl;
    proc0cout << "All " << allUserDepVarNames.size() << " of the dependent variables requested were found in the table." << endl;
  }

}

//****************************************************************************
// TabPropsTable verifyIV
//
//****************************************************************************
void const 
TabPropsTable::verifyIV( bool diagnosticMode, bool strictMode ) 
{
  int numNegativeResults = 0;
  bool toggle;
  std::vector<bool> myVerifyResults;

  // assume toggle is false, but if match is found set toggle to true
  for(unsigned int i=0; i < allUserIndepVarNames.size(); i++) {
    toggle = false;
    for(unsigned int j=0; j < allIndepVarNames.size(); i++) {
      if(allUserIndepVarNames[i] == allIndepVarNames[j]) {
        toggle = true;
      }
    }
    if(toggle==false) {
      numNegativeResults += 1;
    }
    myVerifyResults.push_back(toggle);
  }

  proc0cout << "Arches:TabPropsTable will now check independent variables requested in input file and compare to independent variables in table." << endl;

  if(diagnosticMode == true) {
    // print full results
    for(unsigned int i=0; i < allUserIndepVarNames.size(); i++) {
      if(myVerifyResults[i]==true) {
        cout << "The independent variable " << allUserIndepVarNames[i] << " was found in the table." << endl;
      } else if(myVerifyResults[i]==false) {
        cout << "WARNING: The independent variable " << allUserIndepVarNames[i] << " was NOT found in the table." << endl;
      }
    }
    proc0cout << "The following is a list of independent variables found in the table:" << endl;
    for(unsigned int j=0; j < allIndepVarNames.size(); j++) {
      proc0cout << allIndepVarNames[j] << endl;
    }
  }

  if(numNegativeResults > 0) {
    if(strictMode == true) {
      throw InternalError("You requested in your input file that independent variables be saved, but some were not found in the table.",__FILE__,__LINE__);
    } else if(strictMode == false) {
      proc0cout << "WARNING: Table verification routine found " << numNegativeResults <<
                   " independent variables requested in your input file that were not found in the table." << endl;

      proc0cout << "The following is a list of independent variables found in the table:" << endl;
      for(unsigned int j=0; j < allIndepVarNames.size(); j++) {
        proc0cout << allIndepVarNames[j] << endl;
      }
    }
  }
  else if(numNegativeResults == 0) {
    proc0cout << "Success!" << endl;
    proc0cout << "All " << allUserIndepVarNames.size() << " of the independent variables requested were found in the table." << endl;
  }
}



//****************************************************************************
// TabPropsTable getState 
//
// Call the StateTable::query method for each cell on a given patch
// Called from Properties::computeProps
//
//****************************************************************************
void
TabPropsTable::getState( VarMap ivVar, VarMap dvVar, const Patch* patch )
{
  if( b_table_isloaded == false ) {
    throw InternalError("ERROR:Arches:TabPropsTable - You requested a thermodynamic state, but no "
                        "table has been loaded. You must specify a table filename in your input file.",__FILE__,__LINE__);
  }
  
  for (CellIterator iCell = patch->getCellIterator(); !iCell.done(); ++iCell) {
    IntVector currCell = *iCell;
    
    // loop over all independent variables to extract IV values at currCell
    for (unsigned int i=0; i<ivVar.size(); ++i) { 
      VarMap::iterator iVar = ivVar.find(i);
      indepVarValues[i] = (*iVar->second)[currCell];
    }
 
    // loop over all dependent variables to query table and record values
    for (unsigned int i=0; i<allDepVarNames.size(); ++i) {
      VarMap::iterator iVar = dvVar.find(i);
      (*iVar->second)[currCell] = statetbl.query(allDepVarNames[i], &indepVarValues[0]);
    }

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

