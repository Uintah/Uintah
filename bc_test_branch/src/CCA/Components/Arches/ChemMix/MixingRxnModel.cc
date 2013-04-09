/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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
MixingRxnModel::MixingRxnModel( ArchesLabel* labels, const MPMArchesLabel* MAlab ):
  d_lab(labels), d_MAlab(MAlab)
{
  d_does_post_mixing = false; 
}

//---------------------------------------------------------------------------
MixingRxnModel::~MixingRxnModel()
{
  for ( VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){
    VarLabel::destroy( i->second ); 
  }
  delete _iv_transform; 
}

//---------------------------------------------------------------------------
void 
MixingRxnModel::problemSetupCommon( const ProblemSpecP& params )
{

  ProblemSpecP db = params; 

  db->getWithDefault( "temperature_label_name", _temperature_label_name, string("temperature") ); 

  // create a transform object
  d_has_transform = true; 
  if ( db->findBlock("coal") ) {

    double constant = 0.0; 
    _iv_transform = scinew CoalTransform( constant ); 

  } else if ( db->findBlock("acidbase") ) {

    doubleMap::iterator iter = d_constants.find( "transform_constant" ); 

    if ( iter == d_constants.end() ) {
      throw ProblemSetupException( "Could not find transform_constant for the acid/base table.",__FILE__, __LINE__ ); 
    } else { 
      double constant = iter->second; 
      _iv_transform = scinew CoalTransform( constant ); 
    }

  } else if ( db->findBlock("slowfastchem") ) { 

    _iv_transform = scinew SlowFastTransform(); 

  } else if ( db->findBlock("inert_mixing") ) {

    _iv_transform = scinew InertMixing(); 

  } else { 

    _iv_transform = scinew NoTransform();
    d_has_transform = false; 

  }

  bool check_transform = _iv_transform->problemSetup( db, d_allIndepVarNames ); 

  if ( !check_transform ){ 
    throw ProblemSetupException( "Could not properly setup independent variable transform based on input.",__FILE__,__LINE__); 
  }

  doubleMap::iterator iter = d_constants.find("H_ox");
  if ( iter == d_constants.end() ){ 
    proc0cout << " WARNING: Cannot find #KEY H_ox=*value* in the table.\n"; 
    proc0cout << "          Make sure that your case doesn't require it. \n"; 
    //throw ProblemSetupException( "H_ox (pure oxidizer enthalpy) required as an input.",__FILE__,__LINE__);
  } else { 
    _H_ox = iter->second; 
  }

  iter = d_constants.find("H_fuel");
  if ( !(iter == d_constants.end()) ){ 
    _H_fuel = iter->second; 
  }

  // For inert stream mixing // 
  d_inertMap.clear(); 
  if ( db->findBlock( "post_mix" ) ){ 

    ProblemSpecP db_inert = db->findBlock( "post_mix" ); 

    for ( ProblemSpecP db_st = db_inert->findBlock("stream"); db_st != 0; db_st = db_st->findNextBlock("stream") ){ 

      std::string phi; 
      db_st->getAttribute("transport_label", phi); 

      doubleMap var_values; 
      var_values.clear(); 
      bool found_vars = false; 

      proc0cout << "For inert: " << phi << ", adding species: " << std::endl;
      for ( ProblemSpecP db_var = db_st->findBlock("var"); db_var != 0; db_var = db_var->findNextBlock("var") ){ 

        std::string label; 
        double value; 
      
        db_var->getAttribute("label",label); 
        db_var->getAttribute("value",value);

        doubleMap::iterator iter = var_values.find( label ); 
        if ( iter == var_values.end() ){ 
          var_values.insert( std::make_pair( label, value ) );
          proc0cout << " ---> " << label << ", " << value << std::endl;
          found_vars = true; 
        }

      } 
      proc0cout << "\n";

      if ( found_vars ){ 

        d_does_post_mixing = true; 
        InertMasterMap::iterator iter = d_inertMap.find( phi ); 
        if ( iter == d_inertMap.end() ){ 
          d_inertMap.insert( std::make_pair( phi, var_values ) );   
        } 

      } else { 

        proc0cout << "Warning: Intert stream " << phi << " was not added because no species were found in UPS file!" << std::endl;

      } 
    } 
  } 

  // need the reference denisty point: (also in PhysicalPropteries object but this was easier than passing it around)
  const ProblemSpecP db_root = db->getRootNode(); 
  db_root->findBlock("PhysicalConstants")->require("reference_point", d_ijk_den_ref);  
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
    //var_name = "heat_capacity"; 
    var_name = "specificheat"; 
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
MixingRxnModel::TransformBase::TransformBase(){}
MixingRxnModel::TransformBase::~TransformBase(){}
MixingRxnModel::NoTransform::NoTransform(){}
MixingRxnModel::NoTransform::~NoTransform(){}
MixingRxnModel::CoalTransform::CoalTransform( double constant ) : d_constant(constant){}
MixingRxnModel::CoalTransform::~CoalTransform(){}
MixingRxnModel::SlowFastTransform::SlowFastTransform(){}
MixingRxnModel::SlowFastTransform::~SlowFastTransform(){}
MixingRxnModel::InertMixing::InertMixing(){}
MixingRxnModel::InertMixing::~InertMixing(){}

