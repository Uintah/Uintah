//----- TabPropsTable.h --------------------------------------------------

#ifndef Uintah_Component_Arches_TabPropsTable_h
#define Uintah_Component_Arches_TabPropsTable_h

// constructor
// destructor
// problemSetup
// getState
// verifyTable
// getIndepVars
// getDepVars

// includes for Arches
#include <Packages/Uintah/CCA/Components/Arches/MixingRxnTable.h>
#include <Packages/Uintah/CCA/Components/Arches/TabProps/StateTable.h>
#include <Packages/Uintah/CCA/Components/Arches/Properties.h>

// includes for Uintah
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>

// includes for C++
#include <sgi_stl_warnings_off.h>
#include     <vector>
#include     <map>
#include     <string>
#include     <stdexcept>
#include <sgi_stl_warnings_on.h>

/***************************************************************************
CLASS
    TabPropsTable 
       
GENERAL INFORMATION
    TabPropsTable.h - representation of reaction and mixing table created using Dr. James Sutherland's TabProps program.  Dependent variables are B-Splined, and spline coefficients are put into an HDF5 formated file.  This class creates a TabProps StateTable object, reads data from a table into the StateTable object, and can query the StateTable object for the value of a dependent variable given values for independent variables, as well as return names for independent and dependent variables, and verify tables by checking the names of the dependent variables requested by the user in the input file to dependent variables tabulated in the table.  Functionality will also be added to utilize the StateTable functions to convert the table data to a matlab file to easily investigate the results of the table creation.

    Author: Charles Reid (charles.reid@utah.edu)
    
    Creation Date : 11-13-2008

    C-SAFE
    
    Copyright U of U 2008

KEYWORDS
    Mixing Table 

DESCRIPTION
    TabPropsTable is a child class of MixingRxnTable.  Its methods are specific to tables created using Dr. James Sutherland's TabProps program.  While there are many programs available with which to create tables (DARS, Cantera, Chemkin, etc.), each of these specific table formats can be read into TabProps by writing a unique reader class into TabProps (for example, TabProps/src/prepro/rxnmdl/JCSFlamelets.C, which interfaces with the custom format of a flamelet code that Dr. Sutherland also wrote), the data splined, and the spline coefficients pushed into the HDF5 file.

PATTERNS
    None

WARNINGS

NOTES

POSSIBLE REVISIONS

***************************************************************************/

namespace Uintah {
class Properties;
class TabPropsTable : public MixingRxnTable {

public:

  // GROUP: Constructors:
  // Constructs an instance of MixingRxnTable
  TabPropsTable();

  
  // GROUP: Destructors :
  // Destructor
  // putting virtual here messes things up for compile time
  ~TabPropsTable();


  // GROUP: Problem Setup 
  // Set up the problem specs (from the input file)
  // Get table name, read table data into object
  void problemSetup( const ProblemSpecP& params );


  // GROUP: Actual Action Methods :
  // Actually obtain properties
  const std::vector<double> getState( const double * indepVarValues);


  // GROUP: Verify Methods :
  // Methods used in verifying the table
  void verifyTable( bool diagnosticMode,
                    bool strictMode );


  // GROUP: Get Methods :
  // Get non-state space information from the table
  
  // Load list of dependent variables from the table
  // returns a reference to private var allDepVarNames
  // this may need to be changed to return reference to public variable instead`
  const std::vector<std::string> & getDepVars();

  // Load list of independent variables from the table
  // returns a reference to private var allIndepVarNames
  // this may need to be changed to return reference to public variable instead
  const std::vector<std::string> & getIndepVars();

protected :

private:

    // create booleans to tell you if table has been loaded
    bool b_table_isloaded;
    // note: when you load the table, you also load the dep/indep vars list
    bool b_diagnostic_mode;
    bool b_strict_mode;

    // create vectors to store independent, dependent, and user-requested dependent variables
    std::vector<std::string> allIndepVarNames;
    std::vector<std::string> allDepVarNames;
    std::vector<std::string> allUserDepVarNames;
    
    // create vector to store table query results
    std::vector<double> myQueryResults;
    std::vector<bool> myVerifyResults;

    // create a StateTable object to represent the table data
    StateTable statetbl;

    // create string to hold filename (accessed by problemSetup, verifyTable)
    std::string tableFileName;

}; // end class TabPropsTable
  
} // end namespace Uintah

#endif
