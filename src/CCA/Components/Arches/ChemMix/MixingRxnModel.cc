/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/UPSHelper.h>
#include <CCA/Components/Arches/GridTools.h>

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
#include <unistd.h>
//======================================================

using namespace std;
using namespace Uintah;

//---------------------------------------------------------------------------
MixingRxnModel::MixingRxnModel( MaterialManagerP& materialManager ) :
m_materialManager(materialManager)
{

  d_does_post_mixing = false;

  m_matl_index = 0;

  // Time Step
  m_timeStepLabel = VarLabel::create(timeStep_name, timeStep_vartype::getTypeDescription());

}

//---------------------------------------------------------------------------
MixingRxnModel::~MixingRxnModel()
{
  for ( VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){
    VarLabel::destroy( i->second );
  }
  for ( VarMap::iterator i = d_oldDvVarMap.begin(); i != d_oldDvVarMap.end(); ++i ){
    VarLabel::destroy( i->second );
  }
  delete _iv_transform;
  VarLabel::destroy(m_denRefArrayLabel);

  VarLabel::destroy(m_timeStepLabel);
}

//---------------------------------------------------------------------------
void
MixingRxnModel::problemSetupCommon( const ProblemSpecP& params, MixingRxnModel* const model )
{

  ProblemSpecP db = params;

  using namespace ArchesCore;

  m_denRefArrayLabel = VarLabel::create( "denRefArray", CCVariable<double>::getTypeDescription() );
  //resolve some common labels:

  std::string density_name = parse_ups_for_role( DENSITY, db, "densityCP");
  m_densityLabel = VarLabel::find(density_name);
  if ( m_densityLabel == NULL ){
    throw InvalidValue("Error: Cannot resolve density label.",__FILE__,__LINE__);
  }

  m_volFractionLabel = VarLabel::find("volFraction");
  if ( m_volFractionLabel == NULL ){
    GridVarMap<CCVariable<double> > varmap;
    m_volFractionLabel = VarLabel::find(varmap.vol_frac_name);
    if ( m_volFractionLabel == NULL ){
      throw InvalidValue("Error: Cannot resolve volume fraction label.",__FILE__, __LINE__);
    }
  }

  if ( m_denRefArrayLabel == NULL ){
    throw InvalidValue("Error: ref density label not resolved by the table.", __FILE__, __LINE__ );
  }
  if ( m_densityLabel == NULL ){
    throw InvalidValue("Error: density label not resolved by the table.", __FILE__, __LINE__ );
  }
  if ( m_volFractionLabel == NULL ){
    throw InvalidValue("Error: vol fraction label not resolved by the table.", __FILE__, __LINE__ );
  }

  db->getWithDefault("temperature_label_name", _temperature_label_name, "temperature");

  // create a transform object
  d_has_transform = true;
  if ( db->findBlock("coal") ) {

    _iv_transform = scinew CoalTransform( d_constants, model );

  } else if ( db->findBlock("rcce") ){

    _iv_transform = scinew CoalTransform( d_constants, model );

  } else if ( db->findBlock("rcce_fp") ){

    _iv_transform = scinew RCCETransform( d_constants, model );

  } else if ( db->findBlock("rcce_eta") ){

    _iv_transform = scinew RCCETransform( d_constants, model );

  } else if ( db->findBlock("acidbase") ) {

    _iv_transform = scinew AcidBase( d_constants, model );

  } else if ( db->findBlock("inert_mixing") ) {

    _iv_transform = scinew InertMixing( d_constants, model );

  } else if ( db->findBlock("standard_flamelet" ) ) {

    _iv_transform = scinew SingleMF( d_constants, model );

  } else if ( db->findBlock("standard_equilibrium" ) ) {

    _iv_transform = scinew SingleMF( d_constants, model );

  } else if ( db->findBlock("single_iv") ) {

    _iv_transform = scinew SingleIV( d_constants, model );

  } else if ( db->findBlock("mixfrac_with_heatloss") ) {

    _iv_transform = scinew MFHLTransform( d_constants, model );

  } else {

    _iv_transform = scinew NoTransform();
    d_has_transform = false;

  }

  bool check_transform = _iv_transform->problemSetup( db, d_allIndepVarNames );

  if ( !check_transform ){
    throw ProblemSetupException( "Could not properly setup independent variable transform based on input.",__FILE__,__LINE__);
  }

  bool ignoreDensityCheck = false;
  if ( db->findBlock("ignore_iv_density_check"))
    ignoreDensityCheck = true;

  // For inert stream mixing //
  d_inertMap.clear();
  if ( db->findBlock( "post_mix" ) ){

    ProblemSpecP db_inert = db->findBlock( "post_mix" );

    for ( ProblemSpecP db_st = db_inert->findBlock("stream"); db_st != nullptr; db_st = db_st->findNextBlock("stream") ){

      std::string phi;
      db_st->getAttribute("transport_label", phi);

      doubleMap var_values;
      var_values.clear();
      bool found_vars = false;

      proc0cout << "For inert: " << phi << ", adding species:\n";
      for ( ProblemSpecP db_var = db_st->findBlock("var"); db_var != nullptr; db_var = db_var->findNextBlock("var") ){

        std::string label;
        double value;

        db_var->getAttribute("label",label);
        db_var->getAttribute("value",value);

        doubleMap::iterator iter = var_values.find( label );
        if ( iter == var_values.end() ){
          var_values.insert( std::make_pair( label, value ) );
          proc0cout << " Adding ---> " << label << ", " << value << "\n";
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

        proc0cout << "Warning: Intert stream " << phi << " was not added because no species were found in UPS file!\n";

      }

      // inert mixture fraction must be transported
      EqnFactory& eqn_factory = EqnFactory::self();

      if ( eqn_factory.find_scalar_eqn( phi ) ){
        EqnBase& eqn = eqn_factory.retrieve_scalar_eqn( phi );

        //check if it uses a density guess (which it should)
        //if it isn't set properly, then do it automagically for the user
        if (!eqn.getDensityGuessBool() && !ignoreDensityCheck ){
          proc0cout << " Warning: For equation named " << phi << "\n"
            << "     Density guess must be used for this equation because it determines properties.\n"
            << "     Automatically setting density guess = true.\n";
          eqn.setDensityGuessBool( true );
        }
      } else {
        string err_msg;
        err_msg = "Error: For inert mixture fraction named "+phi+".  Cannot find an associated <TransportEqn>.\n";
        throw ProblemSetupException( err_msg,__FILE__,__LINE__);
      }
    }
  }

  // need the reference denisty point: (also in PhysicalPropteries object but this was easier than passing it around)
  const ProblemSpecP db_root = db->getRootNode();
  db_root->findBlock("PhysicalConstants")->require("reference_point", d_ijk_den_ref);
  d_user_ref_density = false;
  d_reference_density = 1.0;
  if ( db->findBlock("reference_density") ){
    db->findBlock("reference_density")->getAttribute("value",d_reference_density);
    d_user_ref_density = true;
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

    proc0cout << "  The following table variables are requested by the user:\n";

    for( ProblemSpecP db_dv = db_vars->findBlock("save"); db_dv != nullptr; db_dv = db_dv->findNextBlock("save") ) {

      table_lookup = "false";
      var_name     = "false";

      db_dv->getAttribute( "table_lookup", table_lookup );
      db_dv->getAttribute( "label", var_name );

      if ( table_lookup == "true" ) {
        bool test = insertIntoMap( var_name );
        if ( !test ) {
          throw InvalidValue( "Error: Could not insert the following into the table lookup: " + var_name, __FILE__, __LINE__ );
        }
      }
    }
  }

  // Add a few extra variables to the dependent variable map that are required by the algorithm
  // NOTE: These are required FOR NOW by the algorithm while NewStaticMixingTable still lives.
  //       They will be removed once the conversion to TabProps is complete.
  proc0cout << "    (below required by the CFD algorithm)\n";
  var_name = "density";
  bool test = insertIntoMap( var_name );
  if ( !test ){
    throw InvalidValue("Error: Could not insert the following into the table lookup: "+var_name,
                       __FILE__,__LINE__);
  }
  if ( !d_coldflow ){
    var_name = "temperature";
    test = insertIntoMap( var_name );
    if ( !test ){
      throw InvalidValue("Error: Could not insert the following into the table lookup: "+var_name,
                         __FILE__,__LINE__);
    }
    //var_name = "heat_capacity";
    var_name = "specificheat";
    test = insertIntoMap( var_name );
    if ( !test ){
      throw InvalidValue("Error: Could not insert the following into the table lookup: "+var_name,
                         __FILE__,__LINE__);
    }
    var_name = "CO2";
    test = insertIntoMap( var_name );
    if ( !test ){
     throw InvalidValue("Error: Could not insert the following into the table lookup: "+var_name,
                        __FILE__,__LINE__);
    }
    var_name = "H2O";
    test = insertIntoMap( var_name );
    if ( !test ){
     throw InvalidValue("Error: Could not insert the following into the table lookup: "+var_name,
                        __FILE__,__LINE__);
    }
  }

  proc0cout << "\n";
}
//---------------------------------------------------------------------------
// Add Additional Table Lookup Variables
//---------------------------------------------------------------------------
void
MixingRxnModel::addAdditionalDV( std::vector<string>& vars )
{
  proc0cout << "  Adding these additional variables for table lookup:\n";
  for ( std::vector<string>::iterator ivar = vars.begin(); ivar != vars.end(); ivar++ ) {

    bool test = insertIntoMap( *ivar );
    if ( !test ){
     throw InvalidValue("Error: Could not insert the following into the table lookup: "+*ivar,
                        __FILE__,__LINE__);
    }

  }
}

void
MixingRxnModel::sched_checkTableBCs( const LevelP& level, SchedulerP& sched )
{
  string taskname = "MixingRxnModel::checkTableBCs";
  Task* tsk = scinew Task(taskname, this, &MixingRxnModel::checkTableBCs);
  sched->addTask( tsk, level->eachPatch(), m_materialManager->allMaterials( "Arches" )  );
}
void
MixingRxnModel::checkTableBCs( const ProcessorGroup* pc,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw )
{
  for (int p=0; p < patches->size(); p++){

    //Ghost::GhostType gn = Ghost::None;
    const Patch* patch = patches->get(p);

    vector<Patch::FaceType> bf;
    vector<Patch::FaceType>::const_iterator bf_iter;
    patch->getBoundaryFaces(bf);

    //int totalIVs = d_allIndepVarNames.size();
    // Loop over all boundary faces on this patch
    for (bf_iter = bf.begin(); bf_iter != bf.end(); bf_iter++){

      Patch::FaceType face = *bf_iter;
      IntVector insideCellDir = patch->faceDirection(face);

      int numChildren = patch->getBCDataArray(face)->getNumberChildren(m_matl_index);
      for (int child = 0; child < numChildren; child++){

        // look to make sure every variable has a BC set:
        // stuff the bc values into a container for use later
        for ( int i = 0; i < (int) d_allIndepVarNames.size(); i++ ){

          std::string variable_name = d_allIndepVarNames[i];
          string bc_kind="NotSet";
          double bc_value = 0.0;
          std::string bc_s_value = "NA";
          bool foundIterator = false;
          std::string face_name;
          Iterator bound_ptr;

          getBCKind( patch, face, child, variable_name, m_matl_index, bc_kind, face_name );

          if ( bc_kind == "FromFile" ){
            foundIterator =
              getIteratorBCValue<std::string>( patch, face, child, variable_name, m_matl_index, bc_s_value, bound_ptr );
          } else {
            foundIterator =
              getIteratorBCValue<double>( patch, face, child, variable_name, m_matl_index, bc_value, bound_ptr );
          }
          if ( !foundIterator ){
            throw InvalidValue( "Error: Table Independent variable missing a boundary condition spec: "+variable_name+" on face: "+face_name, __FILE__, __LINE__);
          }
        }
      }
    }
  }
}

MixingRxnModel::TransformBase::TransformBase(){}
MixingRxnModel::TransformBase::~TransformBase(){}
MixingRxnModel::NoTransform::NoTransform(){}
MixingRxnModel::NoTransform::~NoTransform(){}
MixingRxnModel::CoalTransform::CoalTransform( std::map<string,double>& keys, MixingRxnModel* const model ) : _keys(keys), _model(model){}
MixingRxnModel::CoalTransform::~CoalTransform(){}
MixingRxnModel::AcidBase::AcidBase( std::map<string,double>& keys, MixingRxnModel* const model ) : _keys(keys), _model(model){}
MixingRxnModel::AcidBase::~AcidBase(){}
MixingRxnModel::RCCETransform::RCCETransform( std::map<string, double>& keys, MixingRxnModel* const model ) : _keys(keys), _model(model) {}
MixingRxnModel::RCCETransform::~RCCETransform(){}
MixingRxnModel::InertMixing::InertMixing( std::map<string,double>& keys, MixingRxnModel* const model ) : _keys(keys), _model(model) {}
MixingRxnModel::InertMixing::~InertMixing(){}
MixingRxnModel::SingleMF::SingleMF( std::map<string,double>& keys, MixingRxnModel* const model) : _keys(keys), _model(model) {};
MixingRxnModel::SingleMF::~SingleMF(){};
MixingRxnModel::SingleIV::SingleIV( std::map<string,double>& keys, MixingRxnModel* const model) : _keys(keys), _model(model) {};
MixingRxnModel::SingleIV::~SingleIV(){};
MixingRxnModel::MFHLTransform::MFHLTransform( std::map<string,double>& keys, MixingRxnModel* const model) : _keys(keys), _model(model) {};
MixingRxnModel::MFHLTransform::~MFHLTransform(){};
