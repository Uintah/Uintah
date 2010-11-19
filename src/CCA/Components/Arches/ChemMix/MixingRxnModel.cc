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


//----- MixingRxnModel.cc --------------------------------------------------
//
// See MixingRxnModel.h for additional information.
//

// includes for Arches
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <CCA/Components/Arches/Properties.h>
#include <CCA/Components/Arches/Arches.h>

// includes for Uintah
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>

// includes for C++
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <fcntl.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <unistd.h>
//======================================================

using namespace std;
using namespace Uintah;

//---------------------------------------------------------------------------
MixingRxnModel::MixingRxnModel( const ArchesLabel* labels, const MPMArchesLabel* MAlab ):
  d_lab(labels), d_MAlab(MAlab)
{
}

//---------------------------------------------------------------------------
MixingRxnModel::~MixingRxnModel()
{

  for ( VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){
    VarLabel::destroy( i->second ); 
  }
  for ( VarMap::iterator i = d_ivVarMap.begin(); i != d_ivVarMap.end(); ++i ){
    VarLabel::destroy( i->second ); 
  }
}

//---------------------------------------------------------------------------
// Set Mix Dependent Variable Map
//---------------------------------------------------------------------------
void 
MixingRxnModel::setMixDVMap( const ProblemSpecP& root_params )
{

  const ProblemSpecP db = root_params; 
  ProblemSpecP db_vars  = db->findBlock("DataArchiver");

  string table_lookup; 
  string var_name; 

  if (db_vars) {

    proc0cout << "  The following table variables are requested by the user: " << endl; 

    for (ProblemSpecP db_dv = db_vars->findBlock("save"); 
          db_dv !=0; db_dv = db_dv->findNextBlock("save")){

      table_lookup = "false";
      var_name     = "false";
      
      db_dv->getAttribute( "table_lookup", table_lookup );
      db_dv->getAttribute( "label", var_name );

      if ( table_lookup == "true" )
        insertIntoMap( var_name ); 

    }
  }

  // Add a few extra variables to the dependent variable map that are required by the algorithm 
  // NOTE: These are required FOR NOW by the algorithm while NewStaticMixingTable still lives. 
  //       They will be removed once the conversion to TabProps is complete. 
  proc0cout << "    (below required by the CFD algorithm)" << endl; 
  var_name = "density"; 
  insertIntoMap( var_name ); 
  if ( !d_coldflow ){ 
    var_name = "temperature"; 
    insertIntoMap( var_name ); 
    var_name = "heat_capacity"; 
    //var_name = "specificheat"; 
    insertIntoMap( var_name ); 
    var_name = "CO2"; 
    insertIntoMap( var_name ); 
    var_name = "H2O"; 
    insertIntoMap( var_name ); 
  }

  proc0cout << endl;
}


//---------------------------------------------------------------------------
// Add Additional Table Lookup Variables
//---------------------------------------------------------------------------
void 
MixingRxnModel::addAdditionalDV( std::vector<string>& vars )
{
  proc0cout << "  Adding these additional variables for table lookup: " << endl; 
  for ( std::vector<string>::iterator ivar = vars.begin(); ivar != vars.end(); ivar++ ) { 
    insertIntoMap( *ivar ); 
  }
}
