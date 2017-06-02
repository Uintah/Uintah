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

//----- ColdFlow.cc --------------------------------------------------

// includes for Arches
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <CCA/Components/Arches/ChemMix/ColdFlow.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelFactory.h>

// includes for Uintah
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>

using namespace std;
using namespace Uintah;

//---------------------------------------------------------------------------
// Default Constructor
//---------------------------------------------------------------------------
ColdFlow::ColdFlow( SimulationStateP& sharedState ) :
  MixingRxnModel( sharedState )
{
  d_coldflow = true;
}

//---------------------------------------------------------------------------
// Default Destructor
//---------------------------------------------------------------------------
ColdFlow::~ColdFlow()
{}

//---------------------------------------------------------------------------
// Problem Setup
//---------------------------------------------------------------------------
  void
ColdFlow::problemSetup( const ProblemSpecP& db )
{
  // Create sub-ProblemSpecP object
  ProblemSpecP db_coldflow = db;

  // Hard code these since they are not read in from an external source
  d_allDepVarNames.push_back("density");
  d_allDepVarNames.push_back("temperature");

  // Need the reference denisty point: (also in PhysicalPropteries object but this was easier than passing it around)
  const ProblemSpecP db_root = db_coldflow->getRootNode();
  db_root->findBlock("PhysicalConstants")->require("reference_point", d_ijk_den_ref);

  // d_stream[ kind index (density, temperature) ][ stream index ( 0,1) ]
  ProblemSpecP db_str1 = db_coldflow->findBlock("stream_0"); //corresponds to f = 1 stream
  ProblemSpecP db_str2 = db_coldflow->findBlock("stream_1"); //corresponds to f = 0 stream

  double den_1;
  double den_2;

  if ( db_str1 ){
    db_str1->getAttribute("density",den_1);
    d_stream[0][0] = 1.0/den_1;
    db_str1->getAttribute("temperature",d_stream[1][0]);
  } else {
    throw InvalidValue("Error: steam_0 not specified in cold flow model.",__FILE__,__LINE__);
  }

  if ( db_str2 ){
    db_str2->getAttribute("density",den_2);
    d_stream[0][1] = 1.0/den_2;
    db_str2->getAttribute("temperature",d_stream[1][1]);
  } else {
    throw InvalidValue("Error: steam_1 not specified in cold flow model.",__FILE__,__LINE__);
  }

  // allow speciation
  for ( ProblemSpecP db_sp = db_str1->findBlock("species"); db_sp != nullptr; db_sp = db_sp->findNextBlock("species") ){

    double value;
    string label;

    db_sp->getAttribute( "label", label );
    db_sp->getAttribute( "value", value );

    species_s1.insert(make_pair(label,value));

    bool test = insertIntoMap( label );

    if ( !test ){
      throw InvalidValue("Error: Could not insert the following into the table lookup: "+label,
                         __FILE__,__LINE__);
    }
  }

  for ( ProblemSpecP db_sp = db_str2->findBlock("species"); db_sp != nullptr; db_sp = db_sp->findNextBlock("species") ){

    double value;
    string label;

    db_sp->getAttribute( "label", label );
    db_sp->getAttribute( "value", value );

    species_s2.insert(make_pair(label,value));

    bool test = insertIntoMap( label );
    if ( !test ){
      throw InvalidValue("Error: Could not insert the following into the table lookup: "+label,
                         __FILE__,__LINE__);
    }

  }

  // bullet proofing
  for ( map<string,double>::iterator iter_1 = species_s1.begin(); iter_1 != species_s1.end(); iter_1++ ){

    string label = iter_1->first;

    // currently don't allow for a unique species to be defined in both streams.
    // This just simplifies life.
    for ( map<string,double>::iterator iter_2 = species_s2.begin(); iter_2 != species_s2.end(); iter_2++ ){
      string label_check = iter_2->first;
      if ( label == label_check ){
        std::cout << " For: " << label_check << " and " << label << endl;
        throw ProblemSetupException("Error: Cold Flow model does not currently allow for the same species to be defined in both streams",
            __FILE__, __LINE__ );
      }
    }
  }

  db_coldflow->findBlock( "mixture_fraction")->getAttribute("label",d_cold_flow_mixfrac);

  proc0cout << endl;
  proc0cout << "--- Cold Flow information --- " << endl;
  proc0cout << endl;

  // This sets the table lookup variables and saves them in a map
  // Map<string name, Label>
  bool test = insertIntoMap( "density" );
  if ( !test ){
    throw InvalidValue("Error: Could not insert the following into the table lookup: density",
                       __FILE__,__LINE__);
  }
  test = insertIntoMap( "temperature" );
  if ( !test ){
    throw InvalidValue("Error: Could not insert the following into the table lookup: temperature",
                       __FILE__,__LINE__);
  }

  proc0cout << "  Now matching user-defined IV's with table IV's" << endl;
  proc0cout << "     Note: If sus crashes here, check to make sure your" << endl;
  proc0cout << "           <TransportEqns><eqn> names match those in the table. " << endl;

  cout_tabledbg << " Creating the independent variable map " << endl;

  d_allIndepVarNames.push_back( d_cold_flow_mixfrac );

  //put the right labels in the label map
  string varName = d_allIndepVarNames[0];

  cout_tabledbg << " Variable: " << varName << " being inserted into the indep. var map"<< endl;

  EqnFactory& eqn_factory = EqnFactory::self();
  EqnBase& eqn = eqn_factory.retrieve_scalar_eqn( varName );
  d_ivVarMap.insert(make_pair(varName, eqn.getTransportEqnLabel()));

  proc0cout << "  Matching sucessful!" << endl;
  proc0cout << endl;

  problemSetupCommon( db_coldflow, this );

  //Automatically adding density_old to the table lookup because this
  //is needed for scalars that aren't solved on stage 1:
  ChemHelper& helper = ChemHelper::self();
  helper.add_lookup_species( "density", ChemHelper::OLD );

  proc0cout << "--- End Cold Flow information --- " << endl;
  proc0cout << endl;

}

//---------------------------------------------------------------------------
// schedule get State
//---------------------------------------------------------------------------
  void
ColdFlow::sched_getState( const LevelP& level,
    SchedulerP& sched,
    const int time_substep,
    const bool initialize_me,
    const bool modify_ref_den )
{
  string taskname = "ColdFlow::getState";
  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &ColdFlow::getState, time_substep, initialize_me, modify_ref_den );

  // independent variables :: these must have been computed previously
  for ( MixingRxnModel::VarMap::iterator i = d_ivVarMap.begin(); i != d_ivVarMap.end(); ++i ) {

    tsk->requires( Task::NewDW, i->second, gn, 0 );

  }

  if ( initialize_me ) {

    for ( MixingRxnModel::VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ) {
      tsk->computes( i->second );
      MixingRxnModel::VarMap::iterator check_iter = d_oldDvVarMap.find( i->first + "_old");
      if ( check_iter != d_oldDvVarMap.end() ){
        if ( m_sharedState->getCurrentTopLevelTimeStep() != 0 ){
          tsk->requires( Task::OldDW, i->second, Ghost::None, 0 );
        }
      }
    }

    for ( MixingRxnModel::VarMap::iterator i = d_oldDvVarMap.begin(); i != d_oldDvVarMap.end(); ++i ) {
      tsk->computes( i->second );
    }

  } else {

    for ( MixingRxnModel::VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ) {
      tsk->modifies( i->second );
    }
    for ( MixingRxnModel::VarMap::iterator i = d_oldDvVarMap.begin(); i != d_oldDvVarMap.end(); ++i ) {
      tsk->modifies( i->second );
    }

  }

  // other variables
  tsk->modifies( m_densityLabel );  // lame .... fix me

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

  tsk->requires( Task::NewDW, m_volFractionLabel, gn, 0 );

  // for inert mixing
  for ( InertMasterMap::iterator iter = d_inertMap.begin(); iter != d_inertMap.end(); iter++ ){
    const VarLabel* label = VarLabel::find( iter->first );
    tsk->requires( Task::NewDW, label, gn, 0 );
  }

  sched->addTask( tsk, level->eachPatch(), m_sharedState->allArchesMaterials() );
}

//---------------------------------------------------------------------------
// get State
//---------------------------------------------------------------------------
  void
ColdFlow::getState( const ProcessorGroup* pc,
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

    // dependent variables
    CCVariable<double> mpmarches_denmicro;

    DepVarMap depend_storage;
    if ( initialize_me ) {

      for ( VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){

        DepVarCont storage;

        storage.var = new CCVariable<double>;
        new_dw->allocateAndPut( *storage.var, i->second, m_matl_index, patch );
        (*storage.var).initialize(0.0);

        depend_storage.insert( make_pair( i->first, storage ));

        std::string name = i->first+"_old";
        VarMap::iterator i_old = d_oldDvVarMap.find(name);

        if ( i_old != d_oldDvVarMap.end() ){
          if ( old_dw != 0 ){

            //copy from old DW
            constCCVariable<double> old_t_value;
            CCVariable<double> old_tpdt_value;
            old_dw->get( old_t_value, i->second, m_matl_index, patch, gn, 0 );
            new_dw->allocateAndPut( old_tpdt_value, i_old->second, m_matl_index, patch );

            old_tpdt_value.copy( old_t_value );

          } else {

            //just allocated it because this is the Arches::Initialize
            CCVariable<double> old_tpdt_value;
            new_dw->allocateAndPut( old_tpdt_value, i_old->second, m_matl_index, patch );
            old_tpdt_value.initialize(0.0);

          }
        }
      }

    } else {

      for ( VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){

        DepVarCont storage;

        storage.var = new CCVariable<double>;
        new_dw->getModifiable( *storage.var, i->second, m_matl_index, patch );

        depend_storage.insert( make_pair( i->first, storage ));

        std::string name = i->first+"_old";
        VarMap::iterator i_old = d_oldDvVarMap.find(name);

        if ( i_old != d_oldDvVarMap.end() ){
          //copy current value into old
          CCVariable<double> old_value;
          new_dw->getModifiable( old_value, i_old->second, m_matl_index, patch );
          old_value.copy( *storage.var );
        }

      }

    }

    std::map< std::string, int> iter_to_index;
    for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){

      // this just maps the iterator to an index so that density and temperature can be
      // easily identified:
      if ( i->first == "density" )
        iter_to_index.insert( make_pair( i->first, 0 ));
      else if ( i->first == "temperature" )
        iter_to_index.insert( make_pair( i->first, 1 ));

    }

    // for inert mixing
    StringToCCVar inert_mixture_fractions;
    inert_mixture_fractions.clear();
    for ( InertMasterMap::iterator iter = d_inertMap.begin(); iter != d_inertMap.end(); iter++ ){
      const VarLabel* label = VarLabel::find( iter->first );
      constCCVariable<double> variable;
      new_dw->get( variable, label, m_matl_index, patch, gn, 0 );
      ConstVarContainer container;
      container.var = variable;

      inert_mixture_fractions.insert( std::make_pair( iter->first, container) );

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

      // retrieve all depenedent variables from table
      for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){

        if ( i->first == "density" || i->first == "temperature" ) {

          std::map< std::string, int>::iterator i_to_i = iter_to_index.find( i->first );
          double table_value = coldFlowMixing( iv, i_to_i->second );

          // for post look-up mixing
          for (StringToCCVar::iterator inert_iter = inert_mixture_fractions.begin();
              inert_iter != inert_mixture_fractions.end(); inert_iter++ ){

            double inert_f = inert_iter->second.var[c];
            doubleMap inert_species_map_list = d_inertMap.find( inert_iter->first )->second;

            post_mixing( table_value, inert_f, i->first, inert_species_map_list );
          }

          table_value *= eps_vol[c];

          if (i->first == "density") {
            if ( table_value > 0.0 ){
              (*i->second.var)[c] = 1.0/table_value;
              arches_density[c] = 1.0/table_value;
            } else {
              (*i->second.var)[c] = 0.0;
              arches_density[c] = 0.0;
            }
          } else if ( i->first == "temperature") {
            (*i->second.var)[c] = table_value;
          }
        } else {

          //speciation
          string species_name = i->first;

          //look in stream 1:
          doubleMap::iterator sp_iter = species_s1.find( species_name );
          if ( sp_iter != species_s1.end() ){

            (*i->second.var)[c] = sp_iter->second * iv[0];

          }

          //look in stream 2:
          sp_iter = species_s2.find( species_name );
          if ( sp_iter != species_s2.end() ){

            (*i->second.var)[c] = sp_iter->second * ( 1.0 - iv[0]);

          }

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
        string bc_kind = "NotSet";
        string face_name;

        std::vector<ColdFlow::BoundaryType> which_bc;
        double bc_value = 0.0;
        std::string bc_s_value = "NA";

        int totalIVs = d_allIndepVarNames.size();
        int counter = 0;

        // look to make sure every variable has a BC set:
        for ( int i = 0; i < (int) d_allIndepVarNames.size(); i++ ){

          std::string variable_name = d_allIndepVarNames[i];

          getBCKind( patch, face, child, variable_name, m_matl_index, bc_kind, face_name );

          bool foundIterator = false;
          if ( bc_kind == "FromFile" ){
            foundIterator =
              getIteratorBCValue<std::string>( patch, face, child, variable_name, m_matl_index, bc_s_value, bound_ptr );
            counter++;
          } else {
            foundIterator =
              getIteratorBCValue<double>( patch, face, child, variable_name, m_matl_index, bc_value, bound_ptr );
            counter++;
          }

          if ( !foundIterator ){
            throw InvalidValue( "Error: Missing boundary condition for table variable: "+variable_name, __FILE__, __LINE__ );
          }

          if ( bc_kind == "Dirichlet" ) {
            which_bc.push_back(ColdFlow::DIRICHLET);
          } else if (bc_kind == "Neumann" ) {
            which_bc.push_back(ColdFlow::NEUMANN);
          } else if (bc_kind == "FromFile") {
            which_bc.push_back(ColdFlow::FROMFILE);
          } else {
            cout << " For face: " << face << endl;
            throw InvalidValue( "Error: BC type not supported for property calculation on face.", __FILE__, __LINE__ );
          }
        }

        if ( counter != totalIVs ){
          stringstream msg;
          msg << "Error: For face " << face << " there are missing IVs in the boundary specification." << endl;
          throw InvalidValue( msg.str(), __FILE__, __LINE__);
        }

        // now use the last bound_ptr to loop over all boundary cells:
        for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++){

          IntVector c   =   *bound_ptr;
          IntVector cp1 = ( *bound_ptr - insideCellDir );

          // again loop over iv's and fill iv vector
          for ( int i = 0; i < (int)d_allIndepVarNames.size(); i++ ){

            iv.push_back( 0.5 * ( indep_storage[i][c] + indep_storage[i][cp1]));

          }

          // now get state for boundary cell:
          for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){

            if ( i->first == "density" || i->first == "temperature" ) {
              std::map< std::string, int>::iterator i_to_i = iter_to_index.find( i->first );
              double table_value = coldFlowMixing( iv, i_to_i->second );

              // for post look-up mixing
              for (StringToCCVar::iterator inert_iter = inert_mixture_fractions.begin();
                  inert_iter != inert_mixture_fractions.end(); inert_iter++ ){

                double inert_f = inert_iter->second.var[c];
                doubleMap inert_species_map_list = d_inertMap.find( inert_iter->first )->second;

                post_mixing( table_value, inert_f, i->first, inert_species_map_list );
              }

              table_value *= eps_vol[c];

              if (i->first == "density") {

                if ( table_value > 0.0 ){
                  double ghost_value = 2.0/table_value - arches_density[cp1];
                  (*i->second.var)[c] = ghost_value;
                  arches_density[c] = ghost_value;
                } else {
                  (*i->second.var)[c] = 0.0;
                  arches_density[c] = 0.0;
                }

              } else if ( i->first == "temperature") {
                (*i->second.var)[c] = table_value;
              }

            } else {

              //speciation
              string species_name = i->first;

              //look in stream 1:
              map<string,double>::iterator sp_iter = species_s1.find( species_name );
              if ( sp_iter != species_s1.end() ){

                (*i->second.var)[c] = sp_iter->second * iv[0];

              }

              //look in stream 2:
              sp_iter = species_s2.find( species_name );
              if ( sp_iter != species_s2.end() ){

                (*i->second.var)[c] = sp_iter->second * ( 1.0 - iv[0]);

              }
            }
          }
          iv.resize(0);
        }
      }
    }

    for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){
      delete i->second.var;
    }

    // reference density modification
    if ( modify_ref_den ) {

      //actually modify the reference density value:
      double den_ref = get_reference_density(arches_density, eps_vol);
      if ( time_substep == 0 ){
        CCVariable<double> den_ref_array;
        new_dw->allocateAndPut(den_ref_array, m_denRefArrayLabel, m_matl_index, patch );

        for (CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++ ){
          IntVector c = *iter;
          den_ref_array[c] = den_ref;
        }

      }

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

double
ColdFlow::coldFlowMixing( std::vector<double>& iv, int pos )
{

  double value = iv[0] * d_stream[pos][0] + ( 1 - iv[0] ) * d_stream[pos][1];
  return value;

}

double ColdFlow::getTableValue( std::vector<double> iv, std::string variable )
{

  if ( variable == "density" ){

    int pos = 0;
    double value = coldFlowMixing( iv, pos );
    value = 1.0 / value;
    return value;

  } else if ( variable == "temperature" ) {

    int pos = 1;
    double value = coldFlowMixing( iv, pos );
    return value;

  } else {

    // a bit dangerous?
    //
    double value = 0;
    return value;

  }
}
