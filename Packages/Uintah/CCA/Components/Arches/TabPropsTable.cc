//----- TabPropsTable.cc --------------------------------------------------
//
// See TabPropsTable.h for more information.
//

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
            <diagnostic_mode>true/false (not req)</diagnostic_mode>
        </TabPropsTable>
    </Properties>

    <DataArchiver>
        ...
        <save name="BlahBlahBlah" table_lookup="true/false">
        ...
    </DataArchiver>

This is used to construct the vector of user-requested dependent variables...

WARNINGS
The code will throw exceptions for the following reasons:
- no <table_file_name> specified in the intput file (ProblemSpecP require() method throws the error)
- the getState method is run without the problemSetup method being run
         (the problemSetup sets the boolean b_table_isloaded to true when data is loaded from a table, and you can't getState without data)
- if bool strictMode is true, and a dependent variable specified in the input file does not match the names of any of the dependent variables in the table
- the getDepVars or getIndepVars methods are run on a TabPropsTable object which hasn't loaded data from a table yet (i.e. hasn't run problemSetup method yet) 

*/



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
// Obtain parameters from input file using Properties.cc and ProblemSpecP object
// The problemSetup() method first retrieves relevant information from the input file
// it then populates the list of independent and dependent variables contained inthe table,
// and the list of user-requested dependent variables from the input file.
// It then runs a verification routine to compare the dependent variables requested
// by the user to the dependent variables contained in the table and ensure the former
// is a subset of the latter.
//****************************************************************************
void
TabPropsTable::problemSetup( const ProblemSpecP& propertiesParameters )
{
  /*

  problemSetup notes:
  --------------------
  input file: include <TabPropsTable> tags
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

  why not have TabPropsTable extract relevant info itself?
  does properties.cc even need the list of user-defined dependent variables?
  properties.cc needs the list in CCVariable format (to interface w/ data warehouse)
  tabprops.cc needs the list in vector<string> format (to interface w/ TabProps)

  TabPropsTable::problemSetup
  ---------------------------

  */

  // Step 1 - create sub-ProblemSpecP object
  // and Step 2 - obtain object parameters
  string tableFileName;
  ProblemSpecP db_tabpropstable = propertiesParameters->findBlock("TabPropsTable");
  
  db_tabpropstable->require( "table_file_name", tableFileName );
  db_tabpropstable->getWithDefault( "strict_mode", b_strict_mode, false );
  db_tabpropstable->getWithDefault( "diagnostic_mode", b_diagnostic_mode, false );

  // Step 3 - obtain user-defined input parameters

  // Step 4 - Check for and deal with filename extension
  // - if table file name has .h5 extension, remove it
  // - otherwise, assume it is an .h5 file but no extension was given
  string extension (tableFileName.end()-3,tableFileName.end());
  if( extension == ".h5" || extension == ".H5" ) {
    tableFileName = tableFileName.substr( 0, tableFileName.size() - 3 );
  }
//__________________________________
//
// Shouldn't you check if the tableFileName exists?  Something like
//      DIR *check = opendir(tableFileName.c_str());
//      if ( check == NULL){
//        ostringstream warn;
//        warn << "ERROR:Arches:TabPropsTable  The table << tableFileName<< does not exist. ";
//        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
//      }  -- Todd
//__________________________________


  // Step 5 - Load data into StateTable
  statetbl.read_hdf5(tableFileName);

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

  // note: it's up to MixingRxn/TabPropsTable class to get list of user-requested dependent variables
  // TabProps needs the list of variables in a string array format
  // but Properties.cc needs the list of varibles in a CCVariable format
  // so dealing with the user-requested vars as a string array should be dealt with in tabprops, not in properties


  // Step 7 - read independent and dependent variables into memory
  allIndepVarNames = statetbl.get_indepvar_names();
  allDepVarNames   = statetbl.get_depvar_names();

  // Step 8 - set is table loaded boolean to true
  b_table_isloaded = true;

  // Step 9 - verify table
  verifyTable( b_diagnostic_mode, b_strict_mode );

}

//****************************************************************************
// TabPropsTable getState 
//
//      allUserDepVarNames[] is a class variable
//      problemSetup() method populates the allUserDepVarNames[] vector
//
//****************************************************************************
const vector<double>
TabPropsTable::getState( const double * indepVarValues )
{
  if( b_table_isloaded == false ) {
    throw InternalError("You requested a thermodynamic state, but you did not specify which table you wanted to use!!!",__FILE__,__LINE__);
  }

  // for each dependent variable, query table
  for( unsigned int i=0 ; i < allUserDepVarNames.size() ; i++) {
    myQueryResults[i] = statetbl.query( allUserDepVarNames[i], &indepVarValues[i] );
  }
  return myQueryResults;
}

//****************************************************************************
// TabPropsTable verifyTable 
//****************************************************************************
void
TabPropsTable::verifyTable(  bool diagnosticMode,
                             bool strictMode ) const
{
  // 4. for loop: for j in list of user-requested dependent variables,
  //        for loop: for i in list of all dependent variables,
  //            if userdefineddepvars[j] == alldepvars[i]
  //                some_boolean = true;
  // 
  int numNegativeResults = 0;
  bool toggle;
  std::vector<bool> myVerifyResults;

  for( unsigned int i=0; i < allUserDepVarNames.size(); i++) {
    toggle = false; // toggle = "is this user-requested dep var in the table?"
    
    for( unsigned int j=0; j < allDepVarNames.size(); j++) {
      if( allUserDepVarNames[i] == allDepVarNames[j] ) {
        toggle = true;
      } 
    }
     
    if( toggle == false ) { 
      numNegativeResults += 1; 
    }
    myVerifyResults.push_back(toggle);
  } // myVerifyResults - size of allUserDepVarNames (contains bool toggle)

//__________________________________
// Should be using proc0cout instead of cout  --Todd
//__________________________________
  
  cout << "Hello, this is the TabProps table reader module of the Arches code." << endl;
  cout << "I am checking the dependent variables requested in the input files against the dependent variables in the TabProps table." << endl;

  if(diagnosticMode == true){
    //print results
    for( unsigned int i=0; i < allUserDepVarNames.size(); i++) {
      if( myVerifyResults[i] == true ) {
        cout << "----- The dependent variable " << allUserDepVarNames[i] << " was found in the table." << endl;
      } 
      else if(myVerifyResults[i] == false) {
        cout << "XXXXX The dependent variable " << allUserDepVarNames[i] << " was NOT found in the table!!!" << endl;
      }
    }

    cout << endl;
    cout << "==================================================================" << endl;
    cout << endl;
  }

  if(numNegativeResults > 0){ // some of the dependent variables the user requested are NOT in the table

    if(strictMode == true) {
      throw InternalError( "You specified dependent variables in your input file that you wanted to save, "
                             "but some of them were not found in the table!!!",__FILE__,__LINE__);
    }else if(strictMode == false) {
       
      // if not strict mode, just print a warning
      cout << "WARNING!!!" << endl;
      cout << "The table verification routine found " << numNegativeResults << " dependent variable(s) requested in your input file that were NOT found in the table." << endl;
      cout << "Did you want one of the following dependent variables in the table?" << endl;
      for( unsigned int j=0; j < allDepVarNames.size(); j++) {
        cout << allDepVarNames[j] << endl;
      }
  
    }
  }  
  else if(numNegativeResults == 0) {
    //you're OK, either way (strict mode or not doesn't matter)
    cout << "SUCCESS!" << endl;
    cout << "All " << allUserDepVarNames.size() << " of the dependent variables you requested in your input file were found in the table." << endl;
  } 
}

//****************************************************************************
// TabPropsTable getDepVars
//
// only do once
// subsequently see if bool iscached
// if is not, srt it using statetable methods
// if yes just grab a private mutable
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
// does the reference being passed still WORK for who it's being passed to?
// (do they have the permission to see what's there?  to change it?)
// (SHOULD THEY have permission to see it? change it?)
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

