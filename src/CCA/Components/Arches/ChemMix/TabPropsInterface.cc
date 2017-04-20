/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

//----- TabPropsInterface.cc --------------------------------------------------

// includes for Arches
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <tabprops/StateTable.h>
#include <CCA/Components/Arches/ChemMix/TabPropsInterface.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelFactory.h>

#include <CCA/Ports/Scheduler.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>

// includes for Uintah
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>
#include <dirent.h>
#include <fstream>

using namespace std;
using namespace Uintah;

//---------------------------------------------------------------------------
// Default Constructor
//---------------------------------------------------------------------------
TabPropsInterface::TabPropsInterface( SimulationStateP& sharedState ) :
MixingRxnModel( sharedState )
{}

//---------------------------------------------------------------------------
// Default Destructor
//---------------------------------------------------------------------------
TabPropsInterface::~TabPropsInterface()
{
}

//---------------------------------------------------------------------------
// Problem Setup
//---------------------------------------------------------------------------
void
TabPropsInterface::problemSetup( const ProblemSpecP& propertiesParameters )
{
  // Create sub-ProblemSpecP object
  string tableFileName;
  ProblemSpecP db_tabprops = propertiesParameters->findBlock("TabProps");
  ProblemSpecP db_properties_root = propertiesParameters;

  // Obtain object parameters
  db_tabprops->require( "inputfile", tableFileName );
  db_tabprops->getWithDefault( "cold_flow", d_coldflow, false);

  // need the reference denisty point: (also in PhysicalPropteries object but this was easier than passing it around)
  const ProblemSpecP db_root = db_tabprops->getRootNode();
  db_root->findBlock("PhysicalConstants")->require("reference_point", d_ijk_den_ref);

  // READ TABLE:
  std::ifstream inFile( tableFileName.c_str(), std::ios_base::in );
  try {
    InputArchive ia( inFile );
    ia >> BOOST_SERIALIZATION_NVP( d_statetbl );
  } catch (...) {
    throw ProblemSetupException( "Could not open table file "+tableFileName+": Boost error opening file.",__FILE__,__LINE__);
  }

  // Extract independent and dependent variables from input file
  ProblemSpecP db_rootnode = propertiesParameters;
  db_rootnode = db_rootnode->getRootNode();

  proc0cout << endl;
  proc0cout << "--- TabProps information --- " << endl;
  proc0cout << endl;

  setMixDVMap( db_rootnode );

  // Get Spline information for a more efficient TabProps
  getSplineInfo();
  getEnthalpySplineInfo(); // maybe want a more elegant way of handling this?

  // Extract independent and dependent variables from the table
  d_allIndepVarNames = d_statetbl.get_indepvar_names();
  d_allDepVarNames   = d_statetbl.get_depvar_names();

  proc0cout << "  Now matching user-defined IV's with table IV's" << endl;
  proc0cout << "     Note: If sus crashes here, check to make sure your" << endl;
  proc0cout << "           <TransportEqns><eqn> names match those in the table. " << endl;

  cout_tabledbg << " Creating the independent variable map " << endl;
  for ( unsigned int i = 0; i < d_allIndepVarNames.size(); ++i ){

    //put the right labels in the label map
    string var_name = d_allIndepVarNames[i];

    const VarLabel* the_label = 0;
    the_label = VarLabel::find( var_name );

    if ( the_label == 0 ) {

      throw InvalidValue( "Error: Could not locate the label for a table parameter: "+var_name, __FILE__, __LINE__);

    } else {

      d_ivVarMap.insert( make_pair(var_name, the_label) );

      // if this parameter is a transported, then density guess must be true.
      EqnFactory& eqn_factory = EqnFactory::self();

      if ( eqn_factory.find_scalar_eqn( var_name ) ){
        EqnBase& eqn = eqn_factory.retrieve_scalar_eqn( var_name );

        //check if it uses a density guess (which it should)
        //if it isn't set properly, then do it automagically for the user
        if (!eqn.getDensityGuessBool()){
          proc0cout << " Warning: For equation named " << var_name << endl
            << "     Density guess must be used for this equation because it determines properties." << endl
            << "     Automatically setting density guess = true. " << endl;
          eqn.setDensityGuessBool( true );
        }
      }
    }
  }

  proc0cout << "  Matching sucessful!" << endl;
  proc0cout << endl;

  problemSetupCommon( db_tabprops, this );

  // Confirm that table has been loaded into memory
  d_table_isloaded = true;

  proc0cout << "--- End TabProps information --- " << endl;
  proc0cout << endl;
}

//---------------------------------------------------------------------------
// schedule get State
//---------------------------------------------------------------------------
void
TabPropsInterface::sched_getState( const LevelP& level,
                                   SchedulerP& sched,
                                   const int time_substep,
                                   const bool initialize_me,
                                   const bool modify_ref_den )

{
  string taskname = "TabPropsInterface::getState";
  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &TabPropsInterface::getState, time_substep, initialize_me, modify_ref_den );

  // independent variables :: these must have been computed previously
  for ( MixingRxnModel::VarMap::iterator i = d_ivVarMap.begin(); i != d_ivVarMap.end(); ++i ) {

    tsk->requires( Task::NewDW, i->second, gn, 0 );

  }

  // dependent variables
  if ( initialize_me ) {

    for ( MixingRxnModel::VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ) {
      tsk->computes( i->second );
    }

  } else {

    for ( MixingRxnModel::VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ) {
      tsk->modifies( i->second );
    }

  }

  // other variables
  tsk->modifies( m_densityLabel );  // lame .... fix me
  tsk->requires( Task::NewDW, m_volFractionLabel, gn, 0 );

  if ( modify_ref_den ){
    if ( time_substep == 0 ){
      tsk->computes( m_denRefArrayLabel );
    }
  } else {
    if ( time_substep == 0 ){
      tsk->computes( m_denRefArrayLabel );
      tsk->requires( Task::OldDW, m_denRefArrayLabel, Ghost::None, 0);
    }
  }

  sched->addTask( tsk, level->eachPatch(), m_sharedState->allArchesMaterials() );
}

//---------------------------------------------------------------------------
// get State
//---------------------------------------------------------------------------
void
TabPropsInterface::getState( const ProcessorGroup* pc,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             const int time_substep,
                             const bool initialize_me,
                             const bool modify_ref_den )
{
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType gn = Ghost::None;
    const Patch* patch = patches->get(p);

    // volume fraction:
    constCCVariable<double> eps_vol;
    new_dw->get( eps_vol, m_volFractionLabel, m_matl_index, patch, gn, 0 );

    //independent variables:
    std::vector<constCCVariable<double> > indep_storage;

    for ( int i = 0; i < (int) d_allIndepVarNames.size(); i++ ){

      VarMap::iterator ivar = d_ivVarMap.find( d_allIndepVarNames[i] );

      constCCVariable<double> the_var;
      new_dw->get( the_var, ivar->second, m_matl_index, patch, gn, 0 );
      indep_storage.push_back( the_var );

    }

    // dependent variables:
    CCVariable<double> mpmarches_denmicro;

    DepVarMap depend_storage;
    if ( initialize_me ) {

      for ( VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){

        DepVarCont storage;

        storage.var = new CCVariable<double>;
        new_dw->allocateAndPut( *storage.var, i->second, m_matl_index, patch );
        (*storage.var).initialize(0.0);

        SplineMap::iterator i_spline = d_depVarSpline.find( i->first );
        storage.spline = i_spline->second;

        depend_storage.insert( make_pair( i->first, storage ));

      }

    } else {

      for ( VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){

        DepVarCont storage;

        storage.var = new CCVariable<double>;
        new_dw->getModifiable( *storage.var, i->second, m_matl_index, patch );

        SplineMap::iterator i_spline = d_depVarSpline.find( i->first );
        storage.spline = i_spline->second;

        depend_storage.insert( make_pair( i->first, storage ));

      }

    }

    CCVariable<double> arches_density;
    new_dw->getModifiable( arches_density, m_densityLabel, m_matl_index, patch );

    // Go through the patch and populate the requested state variables
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;

      // fill independent variables
      std::vector<double> iv;
      for ( std::vector<constCCVariable<double> >::iterator i = indep_storage.begin(); i != indep_storage.end(); ++i ) {
        iv.push_back( (*i)[c] );
      }

      _iv_transform->transform( iv, 0.0 );

      // retrieve all depenedent variables from table
      for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){

        double table_value = getSingleState( i->second.spline, i->first, iv );
          table_value *= eps_vol[c];
        (*i->second.var)[c] = table_value;

        if (i->first == "density") {
          arches_density[c] = table_value;
        }

      }
    }

    // set boundary property values:
    vector<Patch::FaceType> bf;
    vector<Patch::FaceType>::const_iterator bf_iter;
    patch->getBoundaryFaces(bf);

    // Loop over all boundary faces on this patch
    for (bf_iter = bf.begin(); bf_iter != bf.end(); bf_iter++){

      Patch::FaceType face = *bf_iter;
      IntVector insideCellDir = patch->faceDirection(face);

      int numChildren = patch->getBCDataArray(face)->getNumberChildren(m_matl_index);
      for (int child = 0; child < numChildren; child++){

        std::vector<double> iv;
        Iterator nu;
        Iterator bound_ptr;

        std::vector<TabPropsInterface::BoundaryType> which_bc;
        std::vector<double> bc_values;

        // look to make sure every variable has a BC set:
        for ( int i = 0; i < (int) d_allIndepVarNames.size(); i++ ){
          std::string variable_name = d_allIndepVarNames[i];

          const BoundCondBase* bc = patch->getArrayBCValues( face, m_matl_index,
                                                             variable_name, bound_ptr,
                                                             nu, child );

          const BoundCond<double> *new_bcs =  dynamic_cast<const BoundCond<double> *>(bc);
          if ( new_bcs == 0 ) {
            cout << "Error: For variable named " << variable_name << endl;
            throw InvalidValue( "Error: When trying to compute properties at a boundary, found boundary specification missing in the <Grid> section of the input file.", __FILE__, __LINE__);
          }

          double bc_value     = new_bcs->getValue();
          std::string bc_kind = new_bcs->getBCType();

          if ( bc_kind == "Dirichlet" ) {
            which_bc.push_back(TabPropsInterface::DIRICHLET);
          } else if (bc_kind == "Neumann" ) {
            which_bc.push_back(TabPropsInterface::NEUMANN);
          } else
            throw InvalidValue( "Error: BC type not supported for property calculation", __FILE__, __LINE__ );

          // currently assuming a constant value across the mesh.
          bc_values.push_back( bc_value );

          bc_values.push_back(0.0);
          which_bc.push_back(TabPropsInterface::DIRICHLET);

          delete bc;

        }

        // now use the last bound_ptr to loop over all boundary cells:
        for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++){

          IntVector c   =   *bound_ptr;
          IntVector cp1 = ( *bound_ptr - insideCellDir );

          // again loop over iv's and fill iv vector
          for ( int i = 0; i < (int) d_allIndepVarNames.size(); i++ ){

            switch (which_bc[i]) {
              case TabPropsInterface::DIRICHLET:
                iv.push_back( bc_values[i] );
                break;
              case TabPropsInterface::NEUMANN:
                iv.push_back(0.5*(indep_storage[i][c] + indep_storage[i][cp1]));
                break;
              default:
                throw InvalidValue( "Error: BC type not supported for property calculation", __FILE__, __LINE__ );
            }
          }

          _iv_transform->transform( iv, 0.0 );

          // now get state for boundary cell:
          for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){

            double table_value = getSingleState( i->second.spline, i->first, iv );
            table_value *= eps_vol[c];
            (*i->second.var)[c] = table_value;

            if (i->first == "density") {
              //double ghost_value = 2.0*table_value - arches_density[cp1];
              arches_density[c] = table_value;
              //arches_density[c] = ghost_value;
            }
          }
          iv.clear();
        }
      }
    }

    for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){
      delete i->second.var;
    }

    // reference density modification
    if ( modify_ref_den ) {

      throw InvalidValue( "Error: Reference denisty not implement yet in TabProps. Code fix needed, yo.", __FILE__, __LINE__ );

      //actually modify the reference density value:
      //double den_ref = get_reference_density(arches_density, cell_type);
      //if ( time_substep == 0 ){
      //  CCVariable<double> den_ref_array;
      //  new_dw->allocateAndPut(den_ref_array, m_denRefArrayLabel, m_matl_index, patch );

      //  for (CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++ ){
      //    den_ref_array[c] = den_ref;
      //  }

      //}

    } else {

      //just carry forward:
      if ( time_substep == 0 ){
        CCVariable<double> den_ref_array;
        constCCVariable<double> old_den_ref_array;
        new_dw->allocateAndPut(den_ref_array, m_denRefArrayLabel, m_matl_index, patch );
        old_dw->get(old_den_ref_array, m_denRefArrayLabel, m_matl_index, patch, Ghost::None, 0 );
        den_ref_array.copyData( old_den_ref_array );
      }
    }
  }
}

//---------------------------------------------------------------------------
// Get Spline information
//---------------------------------------------------------------------------
void
TabPropsInterface::getSplineInfo()
{

  for ( MixingRxnModel::VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ) {

    const InterpT* spline = d_statetbl.find_entry( i->first );

    if ( spline == nullptr ) {
      ostringstream exception;
      exception << "Error: could not find spline information for variable " << i->first << " \n" <<
        "Please check your dependent variable list and match it to your requested variables. " << endl;
      throw InternalError(exception.str(), __FILE__, __LINE__);
    }

    insertIntoSplineMap( i->first, spline );

  }
}

//---------------------------------------------------------------------------
// Get Enthalpy Spline information
//---------------------------------------------------------------------------
void
TabPropsInterface::getEnthalpySplineInfo()
{

  cout_tabledbg << "TabPropsInterface::getEnthalpySplineInfo(): Looking for sensibleenthalpy" << endl;
  const InterpT* spline = d_statetbl.find_entry( "sensibleenthalpy" );

  d_enthalpyVarSpline.insert( make_pair( "sensibleenthalpy", spline ));

  cout_tabledbg << "TabPropsInterface::getEnthalpySplineInfo(): Looking for adiabaticenthalpy" << endl;
  spline = d_statetbl.find_entry( "adiabaticenthalpy" );

  d_enthalpyVarSpline.insert( make_pair( "adiabaticenthalpy", spline ));

}

//-----------------------------------------------------------------------------------
//
double TabPropsInterface::getTableValue( std::vector<double> iv, std::string variable )
{
  double value = getSingleState( variable, iv );
  return value;
}
