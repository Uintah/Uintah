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

//----- ClassicTableInterface.cc --------------------------------------------------

// includes for Arches
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <CCA/Components/Arches/ChemMix/ClassicTableInterface.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelFactory.h>
#include <CCA/Components/Arches/PropertyModels/HeatLoss.h>

// includes for Uintah
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/InvalidState.h>
#include <Core/Parallel/MasterLock.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>

#include <zlib.h>

#include <cstdio>
#include <sstream>

#define OLD_TABLE 1
#undef OLD_TABLE

using namespace std;
using namespace Uintah;

namespace {

using Mutex = Uintah::MasterLock;
Mutex dependency_map_mutex{};
Mutex enthalpy_map_mutex{};

}

//---------------------------------------------------------------------------
// Default Constructor
//---------------------------------------------------------------------------
ClassicTableInterface::ClassicTableInterface( SimulationStateP& sharedState )
  : MixingRxnModel( sharedState )
{
  m_matl_index = 0;
  _boundary_condition = scinew BoundaryCondition_new( m_matl_index );
}

//---------------------------------------------------------------------------
// Default Destructor
//---------------------------------------------------------------------------
ClassicTableInterface::~ClassicTableInterface()
{
  delete _boundary_condition;
  delete ND_interp;
}

//---------------------------------------------------------------------------
// Problem Setup
//---------------------------------------------------------------------------
  void
ClassicTableInterface::problemSetup( const ProblemSpecP& db )
{
  // Create sub-ProblemSpecP object
  string tableFileName;
  ProblemSpecP db_classic = db;

  // Obtain object parameters
  db_classic->require( "inputfile", tableFileName );
  db_classic->getWithDefault( "cold_flow", d_coldflow, false);

  // READ TABLE:
  proc0cout << "----------Mixing Table Information---------------  " << endl;

  int table_size = 0;
  char* table_contents=nullptr;
  std::string uncomp_table_contents;

  int mpi_rank = Parallel::getMPIRank();

#ifndef OLD_TABLE

  if (mpi_rank == 0) {
    try {
      table_size = gzipInflate( tableFileName, uncomp_table_contents );
    }
    catch( Exception & e ) {
      throw ProblemSetupException( string("Call to gzipInflate() failed: ") + e.message(), __FILE__, __LINE__ );
    }

    table_contents = (char*) uncomp_table_contents.c_str();
    proc0cout << tableFileName << " is " << table_size << " bytes" << endl;
  }

  Uintah::MPI::Bcast(&table_size,1,MPI_INT,0,
      Parallel::getRootProcessorGroup()->getComm());

  if (mpi_rank != 0) {
    table_contents = scinew char[table_size];
  }

  Uintah::MPI::Bcast(table_contents, table_size, MPI_CHAR, 0,
      Parallel::getRootProcessorGroup()->getComm());

  std::stringstream table_contents_stream;
  table_contents_stream << table_contents;
#endif

#ifdef OLD_TABLE
  gzFile gzFp = gzopen(tableFileName.c_str(),"r");
  if( gzFp == nullptr ) {
    // If errno is 0, then not enough memory to uncompress file.
    proc0cout << "Error with gz in opening file: " << tableFileName << ". Errno: " << errno << "\n";
    throw ProblemSetupException("Unable to open the given input file: " + tableFileName, __FILE__, __LINE__);
  }

  loadMixingTable(gzFp, tableFileName );
  gzrewind(gzFp);
  checkForConstants(gzFp, tableFileName );
  gzclose(gzFp);
#else
  loadMixingTable(table_contents_stream, tableFileName );
  table_contents_stream.seekg(0);
  checkForConstants(table_contents_stream, tableFileName );
  if (mpi_rank != 0)
    delete [] table_contents;
#endif

  proc0cout << "-------------------------------------------------  " << endl;

  // Extract independent and dependent variables from input file
  ProblemSpecP db_rootnode = db;
  db_rootnode = db_rootnode->getRootNode();

  proc0cout << endl;
  proc0cout << "--- Classic Arches table information --- " << endl;
  proc0cout << endl;

  // This sets the table lookup variables and saves them in a map
  // Map<string name, Label>
  setMixDVMap( db_rootnode );

  proc0cout << "  Now matching user-defined IV's with table IV's" << endl;
  proc0cout << "     Note: If sus crashes here, check to make sure your" << endl;
  proc0cout << "           <TransportEqns><eqn> names match those in the table. " << endl;

  cout_tabledbg << " Creating the independent variable map " << endl;

  bool ignoreDensityCheck = false;
  if ( db_classic->findBlock("ignore_iv_density_check"))
    ignoreDensityCheck = true;

  size_t numIvVarNames = d_allIndepVarNames.size();
  for ( unsigned int i = 0; i < numIvVarNames; ++i ){

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
        if ( !eqn.getDensityGuessBool() && !ignoreDensityCheck ){
          proc0cout << "\n WARNING: For equation named " << var_name << endl
            << "          Density guess must be used for this equation because it determines properties." << endl
            << "          Automatically setting density guess = true. \n";
          eqn.setDensityGuessBool( true );
        }

      } else {

        PropertyModelFactory& propFactory = PropertyModelFactory::self();
        PropertyModelFactory::PropMap& all_prop_models = propFactory.retrieve_all_property_models();
        bool matched_heatloss = false;

        for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
              iprop != all_prop_models.end(); iprop++){

          PropertyModelBase* prop_model = iprop->second;
          const std::string prop_name = prop_model->retrieve_property_name();

          if ( var_name == prop_name ){

            std::string prop_type = prop_model->getPropType();
            if ( prop_type == "heat_loss" ){

              matched_heatloss = true;

              HeatLoss* hl_model = dynamic_cast<HeatLoss*>(prop_model);
              std::string enthalpy_name = hl_model->get_enthalpy_name();
              EqnBase& enthalpy_eqn = eqn_factory.retrieve_scalar_eqn( enthalpy_name );
              enthalpy_eqn.setDensityGuessBool(true);

            }
          }
        }

        if  ( matched_heatloss ){
          proc0cout << "\n WARNING: For table variable: " << var_name <<  endl;
          proc0cout << "          Density guess must be used for the enthalpy equation because it determines properties." << endl;
          proc0cout << "          Automatically setting/ensuring density guess = true (same as stage=0). \n" << endl;
        } else {
          proc0cout << "\n WARNING: An independent variable, " << var_name << ", wasn\'t found  " << std::endl;
          proc0cout << "          as a transported variable or as a heat_loss model. Arches " << std::endl;
          proc0cout << "          will assume that this is intentional and not force a  " << std::endl;
          proc0cout << "          specific algorithmic ordering for this variable. Sometimes the " <<  std::endl;
          proc0cout << "          model itself will force the correct ordering, which may or " <<  std::endl;
          proc0cout << "          may not be the case here.  \n" <<  std::endl;
        }

      }
    }
  }

  proc0cout << "  Matching sucessful!" << endl;
  proc0cout << endl;

  // Confirm that table has been loaded into memory
  d_table_isloaded = true;

  // Match the requested dependent variables with their table index:
  getIndexInfo();
  if (!d_coldflow)
    getEnthalpyIndexInfo();

  problemSetupCommon( db_classic, this );

  d_hl_upper_bound = 1;
  d_hl_lower_bound = -1;

  if ( _iv_transform->has_heat_loss() ){

    const vector<double> hl_bounds = _iv_transform->get_hl_bounds( indep_headers, d_allIndepVarNum);

    d_hl_lower_bound = hl_bounds[0];
    d_hl_upper_bound = hl_bounds[1];

    proc0cout << "\n Lower bounds on heat loss = " << d_hl_lower_bound << endl;
    proc0cout << " Upper bounds on heat loss = " << d_hl_upper_bound << endl;

    PropertyModelFactory& propFactory = PropertyModelFactory::self();
    PropertyModelFactory::PropMap& all_prop_models = propFactory.retrieve_all_property_models();
    for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
        iprop != all_prop_models.end(); iprop++){

      PropertyModelBase* prop_model = iprop->second;
      if ( prop_model->getPropType() == "heat_loss" ){

        HeatLoss* hl_model = dynamic_cast<HeatLoss*>(prop_model);

        std::string h_name = hl_model->get_hs_label_name();
        bool test = insertIntoMap( h_name );

        if ( !test ){
        throw InvalidValue("Error: could not insert the following into the lookup map: "+h_name,
                           __FILE__,__LINE__);
        }

        h_name = hl_model->get_ha_label_name();
        test = insertIntoMap( h_name );

        if ( !test ){
        throw InvalidValue("Error: could not insert the following into the lookup map: "+h_name,
                           __FILE__,__LINE__);
        }

      }
    }

  }

  //Automatically adding density_old to the table lookup because this
  //is needed for scalars that aren't solved on stage 1:
  ChemHelper& helper = ChemHelper::self();
  helper.add_lookup_species( "density", ChemHelper::OLD );
  helper.set_table_constants( &d_constants );

  proc0cout << "\n --- End Classic Arches table information --- " << endl;
  proc0cout << endl;

}

void ClassicTableInterface::tableMatching(){
  // Match the requested dependent variables with their table index:
  // Must do this again in case a source or model added more species --
  getIndexInfo();
}

//---------------------------------------------------------------------------
// schedule get State
//---------------------------------------------------------------------------
  void
ClassicTableInterface::sched_getState( const LevelP& level,
                                       SchedulerP& sched,
                                       const int time_substep,
                                       const bool initialize_me,
                                       const bool modify_ref_den )

{
  string taskname = "ClassicTableInterface::getState";
  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &ClassicTableInterface::getState, time_substep, initialize_me, modify_ref_den );

  // independent variables :: these must have been computed previously
  for ( MixingRxnModel::VarMap::iterator i = d_ivVarMap.begin(); i != d_ivVarMap.end(); ++i ) {

    tsk->requires( Task::NewDW, i->second, gn, 0 );

  }

  // ensure that dependent variables are matched to their index.
  getIndexInfo();

  // dependent variables
  if ( initialize_me ) {

      d_nDepVars =0;
    for ( MixingRxnModel::VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ) {
      d_nDepVars++;
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
ClassicTableInterface::getState( const ProcessorGroup* pc,
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

    int nIndVars = (int)d_allIndepVarNames.size(); // number of independent variables
    for ( int i = 0; i < nIndVars; i++ ){

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

        storage.var = scinew CCVariable<double>;
        new_dw->allocateAndPut( *storage.var, i->second, m_matl_index, patch );
        (*storage.var).initialize(0.0);

        IndexMap::iterator i_index = d_depVarIndexMap.find( i->first );
        storage.index = i_index->second;

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

        storage.var = scinew CCVariable<double>;
        new_dw->getModifiable( *storage.var, i->second, m_matl_index, patch );

        IndexMap::iterator i_index = d_depVarIndexMap.find( i->first );
        storage.index = i_index->second;

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

    std::vector<int> depVarIndexes(d_nDepVars);
    int ix=0;
    for (DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i){
      depVarIndexes[ix]=i->second.index;
       ix++;
    }

    // Go through the patch and populate the requested state variables
    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
    Uintah::parallel_for(range,  [&]( int i,  int j, int k){

      std::vector<double> iv1(nIndVars);

      // fill independent variables
      int ix=0;
      for ( std::vector<constCCVariable<double> >::iterator iter = indep_storage.begin(); iter != indep_storage.end(); ++iter ) {
        iv1[ix]=(*iter)(i,j,k);
        ix++;
      }

      // do a variable transform (if specified)
      double total_inert_f = 0.0;
      for (StringToCCVar::iterator inert_iter = inert_mixture_fractions.begin();
          inert_iter != inert_mixture_fractions.end(); inert_iter++ ){

        double inert_f = inert_iter->second.var(i,j,k);
        total_inert_f += inert_f;
      }

      _iv_transform->transform( iv1, total_inert_f );

      std::vector<double> depVarValues;
      //get all the needed varaible values from table with only one search
      depVarValues = ND_interp->find_val(iv1, depVarIndexes );// would it be faster to pass by reference?  ~12 dependent variables for coal in 1-2017

      int depVarCount = 0;
      //now deal with the mixing and density checks same as before
      for ( DepVarMap::iterator iter = depend_storage.begin(); iter != depend_storage.end(); ++iter ){

        // for post look-up mixing
        for (StringToCCVar::iterator inert_iter = inert_mixture_fractions.begin();
            inert_iter != inert_mixture_fractions.end(); inert_iter++ ){

          double inert_f = inert_iter->second.var(i,j,k);
          doubleMap inert_species_map_list = d_inertMap.find( inert_iter->first )->second;

          double temp_table_value = depVarValues[depVarCount];
          if ( iter->first == "density" ){
            temp_table_value = 1.0/depVarValues[depVarCount];
          }

          post_mixing( temp_table_value, inert_f, iter->first, inert_species_map_list );

          if ( iter->first == "density" ){
            depVarValues[depVarCount] = 1.0 / temp_table_value;
          } else {
            depVarValues[depVarCount] = temp_table_value;
          }
        }

        depVarValues[depVarCount] *= eps_vol(i,j,k);
        (*iter->second.var)(i,j,k) = depVarValues[depVarCount];

        if (iter->first == "density") {

          arches_density(i,j,k) = depVarValues[depVarCount];

        }
        depVarCount++;
      }

    });

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

        //int totalIVs = d_allIndepVarNames.size();
        int counter = 0;

        // use the first IV to get the iterator:
        std::string variable_name = d_allIndepVarNames[0];
        string bc_kind="NotSet";
        double bc_value = 0.0;
        std::string bc_s_value = "NA";
        std::string face_name;

        getBCKind( patch, face, child, variable_name, m_matl_index, bc_kind, face_name );

        if ( bc_kind == "FromFile" ){
            getIteratorBCValue<std::string>( patch, face, child, variable_name, m_matl_index, bc_s_value, bound_ptr );
          counter++;
        } else {
            getIteratorBCValue<double>( patch, face, child, variable_name, m_matl_index, bc_value, bound_ptr );
          counter++;
        }

        for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++){

          IntVector c   =   *bound_ptr;
          IntVector cp1 = ( *bound_ptr - insideCellDir );

          // again loop over iv's and fill iv vector
          for ( int i = 0; i < (int) d_allIndepVarNames.size(); i++ ){

            iv.push_back( 0.5 * ( indep_storage[i][c] + indep_storage[i][cp1]) );

          }

          double total_inert_f = 0.0;
          for (StringToCCVar::iterator inert_iter = inert_mixture_fractions.begin();
              inert_iter != inert_mixture_fractions.end(); inert_iter++ ){

            total_inert_f += 0.5 * ( inert_iter->second.var[c] + inert_iter->second.var[cp1] );

          }

          _iv_transform->transform( iv, total_inert_f );

          //Get all the dependant variables with one look up
          std::vector<int> indepVarIndexes;
          std::vector<double> depVarValues;
          for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){
            indepVarIndexes.push_back( i->second.index );
          }
          depVarValues = ND_interp->find_val(iv, indepVarIndexes );

          //take care of the mixing and density the same
          int depVarCount = 0;
          // now get state for boundary cell:
          for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){

            // for post look-up mixing
            for (StringToCCVar::iterator inert_iter = inert_mixture_fractions.begin();
                inert_iter != inert_mixture_fractions.end(); inert_iter++ ){

              doubleMap inert_species_map_list = d_inertMap.find( inert_iter->first )->second;

              double temp_table_value = depVarValues[depVarCount];
              if ( i->first == "density" ){
                temp_table_value = 1.0/depVarValues[depVarCount];
              }

              post_mixing( temp_table_value, total_inert_f, i->first, inert_species_map_list );

              if ( i->first == "density" ){
                depVarValues[depVarCount] = 1.0 / temp_table_value;
              } else {
                depVarValues[depVarCount] = temp_table_value;
              }
            }

            depVarValues[depVarCount] *= eps_vol[c];
            double ghost_value = 2.0*depVarValues[depVarCount] - (*i->second.var)[cp1];
            (*i->second.var)[c] = ghost_value;
            //(*i->second.var)[c] = table_value;

            if (i->first == "density")
              arches_density[c] = ghost_value;
            depVarCount++;
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
      DepVarMap::iterator i = depend_storage.find("density");
      std::vector<double> iv = _iv_transform->get_reference_iv();

      std::vector<int> varIndex (1, i->second.index );
      std::vector<double> denValue(1, 0.0);
      denValue = ND_interp->find_val( iv, varIndex );
      double den_ref = denValue[0];

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

//-------------------------------------
  void
ClassicTableInterface::getIndexInfo()
{
  for ( MixingRxnModel::VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){

    std::string name = i->first;
    int index = findIndex( name );

    IndexMap::iterator iter;

    dependency_map_mutex.lock();
    {
       iter = d_depVarIndexMap.find(name);
    }
    dependency_map_mutex.unlock();

    // Only insert variable if it isn't already there.
    if ( iter == d_depVarIndexMap.end() ) {
      cout_tabledbg << " Inserting " << name << " index information into storage." << endl;

      dependency_map_mutex.lock();
      {
        iter = d_depVarIndexMap.insert(make_pair(name, index)).first;
      }
      dependency_map_mutex.unlock();
    }
  }
}

//-------------------------------------
  void
ClassicTableInterface::getEnthalpyIndexInfo()
{
  if ( !d_coldflow){

    enthalpy_map_mutex.lock();
    {
      cout_tabledbg << "ClassicTableInterface::getEnthalpyIndexInfo(): Looking up sensible enthalpy" << endl;
      int index = findIndex("sensibleenthalpy");

      d_enthalpyVarIndexMap.insert(make_pair("sensibleenthalpy", index));

      cout_tabledbg << "ClassicTableInterface::getEnthalpyIndexInfo(): Looking up adiabatic enthalpy" << endl;
      index = findIndex("adiabaticenthalpy");
      d_enthalpyVarIndexMap.insert(make_pair("adiabaticenthalpy", index));

      index = findIndex("density");
      d_enthalpyVarIndexMap.insert(make_pair("density", index));
    }
    enthalpy_map_mutex.unlock();
  }
}

//-------------------------------------
void
ClassicTableInterface::loadMixingTable(gzFile &fp, const string & inputfile )
{

  proc0cout << " Preparing to read the table inputfile:   " << inputfile << "\n";
  d_indepvarscount = getInt( fp );

  proc0cout << " Total number of independent variables: " << d_indepvarscount << endl;

  d_allIndepVarNames = vector<std::string>(d_indepvarscount);

  d_allIndepVarNum = vector<int>(d_indepvarscount);

  for (int ii = 0; ii < d_indepvarscount; ii++){
    std::string varname = getString( fp );
    d_allIndepVarNames[ii] = varname;
  }
  for (int ii = 0; ii < d_indepvarscount; ii++){
    int grid_size = getInt( fp );
    d_allIndepVarNum[ii] = grid_size;
  }

  d_varscount = getInt( fp );
  proc0cout << " Total dependent variables in table: " << d_varscount << endl;

  d_allDepVarNames = vector<std::string>(d_varscount);
  for (int ii = 0; ii < d_varscount; ii++) {

    std::string variable;
    variable = getString( fp );
    d_allDepVarNames[ii] = variable ;

  }

  // Units
  d_allDepVarUnits = vector<std::string>(d_varscount);
  for (int ii = 0; ii < d_varscount; ii++) {
    std::string units = getString( fp );
    d_allDepVarUnits[ii] =  units ;
  }

  //indep vars grids
  indep_headers = vector<vector<double> >(d_indepvarscount);  //vector contains 2 -> N dimensions
  for (int i = 0; i < d_indepvarscount - 1; i++) {
    indep_headers[i] = vector<double>(d_allIndepVarNum[i+1]);
  }
  i1 = vector<vector<double> >(d_allIndepVarNum[d_indepvarscount-1]);
  for (int i = 0; i < d_allIndepVarNum[d_indepvarscount-1]; i++) {
    i1[i] = vector<double>(d_allIndepVarNum[0]);
  }
  //assign values (backwards)
  for (int i = d_indepvarscount-2; i>=0; i--) {
    for (int j = 0; j < d_allIndepVarNum[i+1] ; j++) {
      double v = getDouble( fp );
      indep_headers[i][j] = v;
    }
  }

  int size=1;
  //ND size
  for (int i = 0; i < d_indepvarscount; i++) {
    size = size*d_allIndepVarNum[i];
  }


#ifdef UINTAH_ENABLE_KOKKOS
  tempTableContainer table("ClassicMixingTable",d_varscount,size);
#else
  table = tempTableContainer (d_varscount , std::vector<double> (size,0.0));
#endif

  int size2 = size/d_allIndepVarNum[d_indepvarscount-1];
  proc0cout << "Table size " << size << endl;

  proc0cout << "Reading in the dependent variables: " << endl;
  bool read_assign = true;
  if (d_indepvarscount > 1) {
    for (int kk = 0; kk < d_varscount; kk++) {
      proc0cout << " loading ---> " << d_allDepVarNames[kk] << endl;

      for (int mm = 0; mm < d_allIndepVarNum[d_indepvarscount-1]; mm++) {
        if (read_assign) {
          for (int i = 0; i < d_allIndepVarNum[0]; i++) {
            double v = getDouble(fp);
            i1[mm][i] = v;
          }
        } else {
          //read but don't assign inbetween vals
          for (int i = 0; i < d_allIndepVarNum[0]; i++) {
            getDouble(fp);
          }
        }
        for (int j=0; j<size2; j++) {
          double v = getDouble(fp);
#ifdef UINTAH_ENABLE_KOKKOS
          table(kk,j + mm*size2) = v;
#else
          table[kk][j + mm*size2] = v;
#endif
        }
      }
      if ( read_assign ) { read_assign = false; }
    }
  } else {
    for (int kk = 0; kk < d_varscount; kk++) {
      proc0cout << "loading --->" << d_allDepVarNames[kk] << endl;
      if (read_assign) {
        for (int i=0; i<d_allIndepVarNum[0]; i++) {
          double v = getDouble(fp);
          i1[0][i] = v;
        }
      } else {
        for (int i=0; i<d_allIndepVarNum[0]; i++) {
          getDouble(fp);
        }
      }
      for (int j=0; j<size; j++) {
        double v = getDouble(fp);
#ifdef UINTAH_ENABLE_KOKKOS
        table(kk, j) = v;
#else
        table[kk][j] = v;
#endif
      }
      if (read_assign){read_assign = false;}
    }
  }


  if (d_indepvarscount == 1) {
    ND_interp = new Interp1(d_allIndepVarNum, table, i1);
  } else if (d_indepvarscount == 2) {
    ND_interp = new Interp2(d_allIndepVarNum, table, indep_headers, i1);
  } else if (d_indepvarscount == 3) {
    ND_interp = new Interp3(d_allIndepVarNum, table, indep_headers, i1);
  } else if (d_indepvarscount == 4) {
    ND_interp = new Interp4(d_allIndepVarNum, table, indep_headers, i1);
  } else {  //IV > 4
    ND_interp = new InterpN(d_allIndepVarNum, table, indep_headers, i1, d_indepvarscount);
  }

  proc0cout << "Table successfully loaded into memory!" << endl;

}

void
ClassicTableInterface::loadMixingTable(stringstream& table_stream,
    const string & inputfile )
{

  proc0cout << " Preparing to read the table inputfile:   " << inputfile << "\n";
  d_indepvarscount = getInt( table_stream );

  proc0cout << " Total number of independent variables: " << d_indepvarscount << endl;

  d_allIndepVarNames = vector<std::string>(d_indepvarscount);

  d_allIndepVarNum = vector<int>(d_indepvarscount);

  for (int ii = 0; ii < d_indepvarscount; ii++){
    std::string varname = getString( table_stream );
    d_allIndepVarNames[ii] = varname;
  }
  for (int ii = 0; ii < d_indepvarscount; ii++){
    int grid_size = getInt( table_stream );
    d_allIndepVarNum[ii] = grid_size;
  }

  d_varscount = getInt( table_stream );
  proc0cout << " Total dependent variables in table: " << d_varscount << endl;

  d_allDepVarNames = vector<std::string>(d_varscount);
  for (int ii = 0; ii < d_varscount; ii++) {

    std::string variable;
    variable = getString( table_stream );
    d_allDepVarNames[ii] = variable ;

  }

  // Units
  d_allDepVarUnits = vector<std::string>(d_varscount);
  for (int ii = 0; ii < d_varscount; ii++) {
    std::string units = getString( table_stream );
    d_allDepVarUnits[ii] =  units ;
  }

  //indep vars grids
  indep_headers = vector<vector<double> >(d_indepvarscount);  //vector contains 2 -> N dimensions
  for (int i = 0; i < d_indepvarscount - 1; i++) {
    indep_headers[i] = vector<double>(d_allIndepVarNum[i+1]);
  }
  i1 = vector<vector<double> >(d_allIndepVarNum[d_indepvarscount-1]);
  for (int i = 0; i < d_allIndepVarNum[d_indepvarscount-1]; i++) {
    i1[i] = vector<double>(d_allIndepVarNum[0]);
  }
  //assign values (backwards)
  for (int i = d_indepvarscount-2; i>=0; i--) {
    for (int j = 0; j < d_allIndepVarNum[i+1] ; j++) {
      double v = getDouble( table_stream );
      indep_headers[i][j] = v;
    }
  }

  int size=1;
  //ND size
  for (int i = 0; i < d_indepvarscount; i++) {
    size = size*d_allIndepVarNum[i];
  }

#ifdef UINTAH_ENABLE_KOKKOS
  tempTableContainer table("ClassicMixingTable",d_varscount,size);
#else
  table = tempTableContainer (d_varscount , std::vector<double> (size,0.0));
#endif

  int size2 = size/d_allIndepVarNum[d_indepvarscount-1];
  proc0cout << "Table size " << size << endl;

  proc0cout << "Reading in the dependent variables: " << endl;
  bool read_assign = true;
  if (d_indepvarscount > 1) {
    for (int kk = 0; kk < d_varscount; kk++) {
      proc0cout << " loading ---> " << d_allDepVarNames[kk] << endl;

      for (int mm = 0; mm < d_allIndepVarNum[d_indepvarscount-1]; mm++) {
        if (read_assign) {
          for (int i = 0; i < d_allIndepVarNum[0]; i++) {
            double v = getDouble(table_stream);
            i1[mm][i] = v;
          }
        } else {
          //read but don't assign inbetween vals
          for (int i = 0; i < d_allIndepVarNum[0]; i++) {
            getDouble(table_stream);
          }
        }
        for (int j=0; j<size2; j++) {
          double v = getDouble(table_stream);
#ifdef UINTAH_ENABLE_KOKKOS
          table(kk,j + mm*size2) = v;
#else
          table[kk][j + mm*size2] = v;
#endif

        }
      }
      if ( read_assign ) { read_assign = false; }
    }
  } else {
    for (int kk = 0; kk < d_varscount; kk++) {
      proc0cout << "loading --->" << d_allDepVarNames[kk] << endl;
      if (read_assign) {
        for (int i=0; i<d_allIndepVarNum[0]; i++) {
          double v = getDouble(table_stream);
          i1[0][i] = v;
        }
      } else {
        for (int i=0; i<d_allIndepVarNum[0]; i++) {
          getDouble(table_stream);
        }
      }
      for (int j=0; j<size; j++) {
        double v = getDouble(table_stream);
#ifdef UINTAH_ENABLE_KOKKOS
        table(kk, j) = v;
#else
        table[kk][j] = v;
#endif
      }
      if (read_assign){read_assign = false;}
    }
  }


  if (d_indepvarscount == 1) {
    ND_interp = scinew Interp1(d_allIndepVarNum, table, i1);
  } else if (d_indepvarscount == 2) {
    ND_interp = scinew Interp2(d_allIndepVarNum, table, indep_headers, i1);
  } else if (d_indepvarscount == 3) {
    ND_interp = scinew Interp3(d_allIndepVarNum, table, indep_headers, i1);
  } else if (d_indepvarscount == 4) {
    ND_interp = scinew Interp4(d_allIndepVarNum, table, indep_headers, i1);
  } else {  //IV > 4
    ND_interp = scinew InterpN(d_allIndepVarNum, table, indep_headers, i1, d_indepvarscount);
  }

  proc0cout << "Table successfully loaded into memory!" << endl;

}
//---------------------------
double
ClassicTableInterface::getTableValue( std::vector<double> iv, std::string variable )
{
  int dep_index = findIndex( variable );
  _iv_transform->transform( iv, 0.0 );

  std::vector<int> varIndex (1, dep_index );
  std::vector<double> tabValue(1, 0.0);
  tabValue = ND_interp->find_val( iv, varIndex );
  double value = tabValue[0];

  return value;
}

//---------------------------
double
ClassicTableInterface::getTableValue( std::vector<double> iv, std::string depend_varname,
    MixingRxnModel::StringToCCVar inert_mixture_fractions,
    IntVector c )
{

  double total_inert_f = 0.0;

  int dep_index = findIndex( depend_varname );

  for (StringToCCVar::iterator inert_iter = inert_mixture_fractions.begin();
      inert_iter != inert_mixture_fractions.end(); inert_iter++ ){

    double inert_f = inert_iter->second.var[c];
    total_inert_f += inert_f;

  }

  _iv_transform->transform( iv, total_inert_f );

  std::vector<int> varIndex (1, dep_index );
  std::vector<double> tabValue(1, 0.0);
  tabValue = ND_interp->find_val( iv, varIndex );
  double table_value = tabValue[0];

  // for post look-up mixing
  for (StringToCCVar::iterator inert_iter = inert_mixture_fractions.begin();
      inert_iter != inert_mixture_fractions.end(); inert_iter++ ){

    double inert_f = inert_iter->second.var[c];
    doubleMap inert_species_map_list = d_inertMap.find( inert_iter->first )->second;

    post_mixing( table_value, inert_f, depend_varname, inert_species_map_list );

  }

  return table_value;

}
//---------------------------
double
ClassicTableInterface::getTableValue( std::vector<double> iv, std::string depend_varname,
    MixingRxnModel::doubleMap inert_mixture_fractions )
{

  double total_inert_f = 0.0;

  int dep_index = findIndex( depend_varname );

  for (MixingRxnModel::doubleMap::iterator inert_iter = inert_mixture_fractions.begin();
      inert_iter != inert_mixture_fractions.end(); inert_iter++ ){

    double inert_f = inert_iter->second;
    total_inert_f += inert_f;

  }

  _iv_transform->transform( iv, total_inert_f );

  std::vector<int> varIndex (1, dep_index );
  std::vector<double> tabValue(1, 0.0);
  tabValue = ND_interp->find_val( iv, varIndex );
  double table_value = tabValue[0];

  // for post look-up mixing
  for (MixingRxnModel::doubleMap::iterator inert_iter = inert_mixture_fractions.begin();
      inert_iter != inert_mixture_fractions.end(); inert_iter++ ){

    double inert_f = inert_iter->second;
    doubleMap inert_species_map_list = d_inertMap.find( inert_iter->first )->second;

    post_mixing( table_value, inert_f, depend_varname, inert_species_map_list );

  }

  return table_value;

}

//---------------------------
void
ClassicTableInterface::checkForConstants(gzFile &fp, const string & inputfile ) {

  proc0cout << "\n Looking for constants in the header... " << endl;

  bool look = true;
  while ( look ){

    char ch = gzgetc( fp );

    if ( ch == '#' ) {

      char key = gzgetc( fp );

      if ( key == 'K' ) {
        for (int i = 0; i < 3; i++ ){
          key = gzgetc( fp ); // reading the word KEY and space
        }

        string name;
        while ( true ) {
          key = gzgetc( fp );
          if ( key == '=' ) {
            break;
          }
          name.push_back( key );  // reading in the token's key name
        }

        string value_s;
        while ( true ) {
          key = gzgetc( fp );
          if ( key == '\n' || key == '\t' || key == ' ' ) {
            break;
          }
          value_s.push_back( key ); // reading in the token's value
        }

        double value;
        sscanf( value_s.c_str(), "%lf", &value );

        proc0cout << " KEY found: " << name << " = " << value << endl;

        d_constants.insert( make_pair( name, value ) );

      } else {
        while ( true ) {
          ch = gzgetc( fp ); // skipping this line
          if ( ch == '\n' || ch == '\t' ) {
            break;
          }
        }
      }

    } else {

      look = false;

    }
  }
}

void
ClassicTableInterface::checkForConstants(stringstream &table_stream,
    const string & inputfile )
{

  proc0cout << "\n Looking for constants in the header... " << endl;

  bool look = true;
  while ( look ){

    //    char ch = gzgetc( table_stream );
    char ch = table_stream.get();

    if ( ch == '#' ) {

      //  char key = gzgetc( table_stream );
      char key = table_stream.get() ;

      if ( key == 'K' ) {
        for (int i = 0; i < 3; i++ ){
          //  key = gzgetc( table_stream ); // reading the word KEY and space
          key = table_stream.get() ; // reading the word KEY and space
        }

        string name;
        while ( true ) {
          //  key = gzgetc( table_stream );
          key = table_stream.get();
          if ( key == '=' ) {
            break;
          }
          name.push_back( key );  // reading in the token's key name
        }

        string value_s;
        while ( true ) {
          //  key = gzgetc( table_stream );
          key = table_stream.get();
          if ( key == '\n' || key == '\t' || key == ' ' ) {
            break;
          }
          value_s.push_back( key ); // reading in the token's value
        }

        double value;
        sscanf( value_s.c_str(), "%lf", &value );

        proc0cout << " KEY found: " << name << " = " << value << endl;

        d_constants.insert( make_pair( name, value ) );

      } else {

        while ( true ) {
          // ch = gzgetc( table_stream ); // skipping this line
          ch = table_stream.get(); // skipping this line
          if ( ch == '\n' || ch == '\t' ) {
            break;
          }
        }
      }

    } else {

      look = false;

    }
  }
}
