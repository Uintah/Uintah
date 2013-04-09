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
#include <Core/IO/UintahZlibUtil.h>


using namespace std;
using namespace Uintah;

//--------------------------------------------------------------------------- 
// Default Constructor 
//--------------------------------------------------------------------------- 
ClassicTableInterface::ClassicTableInterface( ArchesLabel* labels, const MPMArchesLabel* MAlabels ) :
  MixingRxnModel( labels, MAlabels ), depVarIndexMapLock("ARCHES d_depVarIndexMap lock"),
  enthalpyVarIndexMapLock("ARCHES d_enthalpyVarIndexMap lock")
{
  _boundary_condition = scinew BoundaryCondition_new( labels->d_sharedState->getArchesMaterial(0)->getDWIndex() ); 
}

//--------------------------------------------------------------------------- 
// Default Destructor
//--------------------------------------------------------------------------- 
ClassicTableInterface::~ClassicTableInterface()
{
  delete _boundary_condition; 
}

//--------------------------------------------------------------------------- 
// Problem Setup
//--------------------------------------------------------------------------- 
  void
ClassicTableInterface::problemSetup( const ProblemSpecP& propertiesParameters )
{
  // Create sub-ProblemSpecP object
  string tableFileName;
  ProblemSpecP db_classic = propertiesParameters->findBlock("ClassicTable");

  // Obtain object parameters
  db_classic->require( "inputfile", tableFileName );
  db_classic->getWithDefault( "cold_flow", d_coldflow, false); 

  // READ TABLE: 
  proc0cout << "----------Mixing Table Information---------------  " << endl;
  loadMixingTable( tableFileName );
  checkForConstants( tableFileName );
  proc0cout << "-------------------------------------------------  " << endl;

  // Extract independent and dependent variables from input file
  ProblemSpecP db_rootnode = propertiesParameters;
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

  // Confirm that table has been loaded into memory
  d_table_isloaded = true;

  problemSetupCommon( db_classic ); 

  proc0cout << "--- End Classic Arches table information --- " << endl;
  proc0cout << endl;

  // Match the requested dependent variables with their table index:
  getIndexInfo(); 
  if (!d_coldflow) 
    getEnthalpyIndexInfo();

  // *** HACKISHNESS *** 
  // Check for heat loss as a property model 
  // If found, add sensible and adiabatic enthalpy to the lookup 
  // Some of this is a repeat of what is happening already in HeatLoss.cc
  const ProblemSpecP db_root = db_classic->getRootNode(); 
  if ( db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("PropertyModels") ){ 
   const ProblemSpecP db_prop_models = 
     db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("PropertyModels");

    for ( ProblemSpecP model_db = db_prop_models->findBlock("model");
        model_db != 0; model_db = model_db->findNextBlock("model") ){

      std::string type; 
      model_db->getAttribute( "type", type ); 

      std::string adiab_name, sens_name; 

      model_db->getWithDefault( "adiabatic_enthalpy_label" , adiab_name  , string( "adiabaticenthalpy" ) );
      model_db->getWithDefault( "sensible_enthalpy_label"  , sens_name   , string( "sensibleenthalpy" ) );

      if ( type == "heat_loss" ){ 
        insertIntoMap( adiab_name ); 
        insertIntoMap( sens_name ); 
      } 

    }
  } 
  //**** END HACKISHNESS ***
  //setting varlabels to roles:
  d_lab->setVarlabelToRole( "temperature", "temperature" );
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
    const TimeIntegratorLabel* time_labels, 
    const bool initialize_me,
    const bool with_energy_exch, 
    const bool modify_ref_den )

{
  string taskname = "ClassicTableInterface::getState"; 
  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &ClassicTableInterface::getState, time_labels, initialize_me, with_energy_exch, modify_ref_den );

  // independent variables :: these must have been computed previously 
  for ( MixingRxnModel::VarMap::iterator i = d_ivVarMap.begin(); i != d_ivVarMap.end(); ++i ) {

    tsk->requires( Task::NewDW, i->second, gn, 0 ); 

  }

  // ensure that dependent variables are matched to their index. 
  getIndexInfo(); 

  // dependent variables
  if ( initialize_me ) {

    for ( MixingRxnModel::VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ) {
      tsk->computes( i->second ); 
    }

    tsk->computes( d_lab->d_drhodfCPLabel ); // I don't think this is used anywhere...maybe in coldflow? 
    if (!d_coldflow) { 
      // other dependent vars:
      tsk->computes( d_lab->d_tempINLabel ); // lame ... fix me
      tsk->computes( d_lab->d_cpINLabel ); 
      tsk->computes( d_lab->d_co2INLabel ); 
      tsk->computes( d_lab->d_h2oINLabel ); 
      tsk->computes( d_lab->d_sootFVINLabel ); 
    }

    if (d_MAlab)
      tsk->computes( d_lab->d_densityMicroLabel ); 

  } else {

    for ( MixingRxnModel::VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ) {
      tsk->modifies( i->second ); 
    }

    tsk->modifies( d_lab->d_drhodfCPLabel ); // I don't think this is used anywhere...maybe in coldflow? 
    if (!d_coldflow) { 
      // other dependent vars:
      tsk->modifies( d_lab->d_tempINLabel );     // lame .... fix me
      tsk->modifies( d_lab->d_cpINLabel ); 
      tsk->modifies( d_lab->d_co2INLabel ); 
      tsk->modifies( d_lab->d_h2oINLabel ); 
      tsk->modifies( d_lab->d_sootFVINLabel ); 
    }

    if (d_MAlab)
      tsk->modifies( d_lab->d_densityMicroLabel ); 

  }

  // other variables 
  tsk->modifies( d_lab->d_densityCPLabel );  // lame .... fix me
  if ( modify_ref_den ) {
    tsk->computes(time_labels->ref_density); 
  }
  tsk->requires( Task::NewDW, d_lab->d_volFractionLabel, gn, 0 ); 
  tsk->requires( Task::NewDW, d_lab->d_cellTypeLabel, gn, 0 ); 

  // for inert mixing 
  for ( InertMasterMap::iterator iter = d_inertMap.begin(); iter != d_inertMap.end(); iter++ ){ 
    const VarLabel* label = VarLabel::find( iter->first ); 
    tsk->requires( Task::NewDW, label, gn, 0 ); 
  } 

  sched->addTask( tsk, level->eachPatch(), d_lab->d_sharedState->allArchesMaterials() ); 
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
    const TimeIntegratorLabel* time_labels, 
    const bool initialize_me, 
    const bool with_energy_exch, 
    const bool modify_ref_den )
{
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType gn = Ghost::None; 
    const Patch* patch = patches->get(p); 
    int archIndex = 0; 
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    MixingRxnModel::InertMixing* inert_transform=0; 
    if ( d_does_post_mixing && d_has_transform ){ 
      inert_transform = dynamic_cast<MixingRxnModel::InertMixing*>(_iv_transform); 
    }

    constCCVariable<double> eps_vol; 
    constCCVariable<int> cell_type; 
    new_dw->get( eps_vol, d_lab->d_volFractionLabel, matlIndex, patch, gn, 0 ); 
    new_dw->get( cell_type, d_lab->d_cellTypeLabel, matlIndex, patch, gn, 0 ); 

    //independent variables:
    std::vector<constCCVariable<double> > indep_storage; 

    int size = (int)d_allIndepVarNames.size();
    for ( int i = 0; i < size; i++ ){

      VarMap::iterator ivar = d_ivVarMap.find( d_allIndepVarNames[i] ); 

cout << "ClassicTableInterface.cc: indep var names: " << i << ", " << d_allIndepVarNames[i] << "\n";

      constCCVariable<double> the_var; 
      new_dw->get( the_var, ivar->second, matlIndex, patch, gn, 0 );
      indep_storage.push_back( the_var ); 

    }

    // dependent variables:
    CCVariable<double> arches_temperature; 
    CCVariable<double> arches_cp; 
    CCVariable<double> arches_co2; 
    CCVariable<double> arches_h2o; 
    CCVariable<double> arches_soot; 
    CCVariable<double> mpmarches_denmicro; 

    DepVarMap depend_storage; 
    if ( initialize_me ) {

cout << "ClassicTableInterface.cc: initialize_me\n";

      for ( VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){

        DepVarCont storage;

        storage.var = new CCVariable<double>; 
        new_dw->allocateAndPut( *storage.var, i->second, matlIndex, patch ); 
        (*storage.var).initialize(0.0);

        IndexMap::iterator i_index = d_depVarIndexMap.find( i->first ); 
        storage.index = i_index->second; 

        depend_storage.insert( make_pair( i->first, storage ));

      }

      // others: 
      CCVariable<double> drho_df; 

      new_dw->allocateAndPut( drho_df, d_lab->d_drhodfCPLabel, matlIndex, patch ); 
      if (!d_coldflow) { 
        new_dw->allocateAndPut( arches_temperature, d_lab->d_tempINLabel, matlIndex, patch ); 
        new_dw->allocateAndPut( arches_cp, d_lab->d_cpINLabel, matlIndex, patch ); 
        new_dw->allocateAndPut( arches_co2, d_lab->d_co2INLabel, matlIndex, patch ); 
        new_dw->allocateAndPut( arches_h2o, d_lab->d_h2oINLabel, matlIndex, patch ); 
        new_dw->allocateAndPut( arches_soot, d_lab->d_sootFVINLabel, matlIndex, patch ); 
      }
      if (d_MAlab) {
        new_dw->allocateAndPut( mpmarches_denmicro, d_lab->d_densityMicroLabel, matlIndex, patch ); 
        mpmarches_denmicro.initialize(0.0);
      }

      drho_df.initialize(0.0);  // this variable might not be actually used anywhere and may just be polution  
      if ( !d_coldflow ) { 
        arches_temperature.initialize(0.0); 
        arches_cp.initialize(0.0); 
        arches_co2.initialize(0.0); 
        arches_h2o.initialize(0.0);
        arches_soot.initialize(0.0); 
      }

    } else { 

      for ( VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){

        DepVarCont storage;

        storage.var = new CCVariable<double>; 
        new_dw->getModifiable( *storage.var, i->second, matlIndex, patch ); 

        IndexMap::iterator i_index = d_depVarIndexMap.find( i->first ); 
        storage.index = i_index->second; 

        depend_storage.insert( make_pair( i->first, storage ));

      }

      // others:
      CCVariable<double> drho_dw; 
      new_dw->getModifiable( drho_dw, d_lab->d_drhodfCPLabel, matlIndex, patch ); 
      if (!d_coldflow) { 
        new_dw->getModifiable( arches_temperature, d_lab->d_tempINLabel, matlIndex, patch ); 
        new_dw->getModifiable( arches_cp, d_lab->d_cpINLabel, matlIndex, patch ); 
        new_dw->getModifiable( arches_co2, d_lab->d_co2INLabel, matlIndex, patch ); 
        new_dw->getModifiable( arches_h2o, d_lab->d_h2oINLabel, matlIndex, patch ); 
        new_dw->getModifiable( arches_soot, d_lab->d_sootFVINLabel, matlIndex, patch ); 
      }
      if (d_MAlab) 
        new_dw->getModifiable( mpmarches_denmicro, d_lab->d_densityMicroLabel, matlIndex, patch ); 
    }

    // for inert mixing 
    StringToCCVar inert_mixture_fractions; 
    inert_mixture_fractions.clear(); 
    for ( InertMasterMap::iterator iter = d_inertMap.begin(); iter != d_inertMap.end(); iter++ ){ 
      const VarLabel* label = VarLabel::find( iter->first ); 
      constCCVariable<double> variable; 
      new_dw->get( variable, label, matlIndex, patch, gn, 0 ); 
      ConstVarContainer container; 
      container.var = variable; 

      inert_mixture_fractions.insert( std::make_pair( iter->first, container) ); 

    } 

    CCVariable<double> arches_density; 
    new_dw->getModifiable( arches_density, d_lab->d_densityCPLabel, matlIndex, patch ); 

cout << "ClassicTableInterface.cc: cell iter size: " << patch->getCellIterator().size() << "\n";

    // Go through the patch and populate the requested state variables
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 

      // fill independent variables
      std::vector<double> iv; 
      for ( std::vector<constCCVariable<double> >::iterator i = indep_storage.begin(); i != indep_storage.end(); ++i ) {
        iv.push_back( (*i)[c] );
      }

      // do a variable transform (if specified)
      double total_inert_f = 0.0; 
      for (StringToCCVar::iterator inert_iter = inert_mixture_fractions.begin(); 
          inert_iter != inert_mixture_fractions.end(); inert_iter++ ){

        double inert_f = inert_iter->second.var[c];
        total_inert_f += inert_f; 
      }

      if ( d_does_post_mixing && d_has_transform ) { 
        inert_transform->transform( iv, total_inert_f ); 
      } else { 
        _iv_transform->transform( iv ); 
      }

      // retrieve all depenedent variables from table
      for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){

				double table_value = ND_interp->find_val( iv, i->second.index );

        // for post look-up mixing
        for (StringToCCVar::iterator inert_iter = inert_mixture_fractions.begin(); 
            inert_iter != inert_mixture_fractions.end(); inert_iter++ ){

          double inert_f = inert_iter->second.var[c];
          doubleMap inert_species_map_list = d_inertMap.find( inert_iter->first )->second; 

          double temp_table_value = table_value; 
          if ( i->first == "density" ){ 
            temp_table_value = 1.0/table_value; 
          } 

          post_mixing( temp_table_value, inert_f, i->first, inert_species_map_list ); 

          if ( i->first == "density" ){ 
            table_value = 1.0 / temp_table_value; 
          } else { 
            table_value = temp_table_value; 
          } 

        }

        table_value *= eps_vol[c]; 
        (*i->second.var)[c] = table_value;


        if (i->first == "density") {

          arches_density[c] = table_value; 

          if (d_MAlab)
            mpmarches_denmicro[c] = table_value; 

        } else if (i->first == "temperature" && !d_coldflow) {

          arches_temperature[c] = table_value; 

        } else if (i->first == "specificheat" && !d_coldflow) {

          arches_cp[c] = table_value; 

        } else if (i->first == "CO2" && !d_coldflow) {

          arches_co2[c] = table_value; 

        } else if (i->first == "H2O" && !d_coldflow) {

          arches_h2o[c] = table_value; 

        }

      }
    }

    // set boundary property values: 
    vector<Patch::FaceType> bf;
    vector<Patch::FaceType>::const_iterator bf_iter;
    patch->getBoundaryFaces(bf);

cout << "ClassicTableInterface.cc: faces: " << bf.size() << "\n";

    // Loop over all boundary faces on this patch
    for (bf_iter = bf.begin(); bf_iter != bf.end(); bf_iter++){

      Patch::FaceType face = *bf_iter; 
      IntVector insideCellDir = patch->faceDirection(face);

      int numChildren = patch->getBCDataArray(face)->getNumberChildren(matlIndex);
      for (int child = 0; child < numChildren; child++){

        std::vector<double> iv; 
        Iterator nu;
        Iterator bound_ptr; 

        std::vector<double> bc_values;

        // look to make sure every variable has a BC set:
        // stuff the bc values into a container for use later
        for ( int i = 0; i < (int) d_allIndepVarNames.size(); i++ ){

          std::string variable_name = d_allIndepVarNames[i]; 
          string face_name; 
          string bc_kind="NotSet"; 
          double bc_value = 0.0; 
          bool foundIterator = "false"; 

          getBCKind( patch, face, child, variable_name, matlIndex, bc_kind, face_name ); 
//qwerty bound pointer is different... 

          // The template parameter needs to be generalized here to handle strings, etc...
          foundIterator = 
            getIteratorBCValue<double>( patch, face, child, variable_name, matlIndex, bc_value, bound_ptr ); 

// FIXME: debug
cout << "ClassicTableInterface.cc: bc_value: " << bc_value << "\n";
 for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++){ cout << *bound_ptr << ", "; }

cout << "\n";
// FIXME: end debug


          if ( foundIterator ) { 
            bc_values.push_back( bc_value ); 
          } else { 
            // FIXME: This might not be right...????
            throw InvalidValue( "Error: Boundary condition not found for: "+variable_name, __FILE__, __LINE__ ); 
          } 

        }

        // FIXME: debug
        cout << "ClassicTableInterface.cc: bound_ptr size: " << bound_ptr.size() << "\n";
        // FIXME: end debug

cout << "ClassicTableInterface.cc: 1) here:" << iv.size() <<": ";
for( int cntr = 0; cntr < iv.size(); cntr++ ) { cout << iv[cntr] << ", "; }
cout << "\n";

        // now use the last bound_ptr to loop over all boundary cells: 
        for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++){

          IntVector c   =   *bound_ptr; 
          IntVector cp1 = ( *bound_ptr - insideCellDir ); 

cout << "ClassicTableInterface.cc: c, cp1: " << c << ", " << cp1 <<"\n";

          // again loop over iv's and fill iv vector
          for ( int i = 0; i < (int) d_allIndepVarNames.size(); i++ ){

cout << "ClassicTableInterface.cc: indep_storage(i,c), indep_storage(i,cp1): " << indep_storage[i][c] << ", " << indep_storage[i][cp1] <<"\n";

            iv.push_back( 0.5 * ( indep_storage[i][c] + indep_storage[i][cp1]) );
          }

cout << "ClassicTableInterface.cc: 2) here:" << iv.size() <<": ";
for( int cntr = 0; cntr < iv.size(); cntr++ ) { cout << iv[cntr] << ", "; }
cout << "\n";

          double total_inert_f = 0.0; 
          for (StringToCCVar::iterator inert_iter = inert_mixture_fractions.begin(); 
              inert_iter != inert_mixture_fractions.end(); inert_iter++ ){

            total_inert_f += 0.5 * ( inert_iter->second.var[c] + inert_iter->second.var[cp1] );

          }

          if ( d_does_post_mixing && d_has_transform ) { 
            inert_transform->transform( iv, total_inert_f ); 
          } else { 
            _iv_transform->transform( iv ); 
          }

          // now get state for boundary cell: 
cout << "ClassicTableInterface.cc: 3) here:" << iv.size() <<": ";
for( int cntr = 0; cntr < iv.size(); cntr++ ) { cout << iv[cntr] << ", "; }
cout << "\n";

          for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){

            //  double table_value = tableLookUp( iv, i->second.index ); 
            double table_value = ND_interp->find_val( iv, i->second.index );

            // for post look-up mixing
            for (StringToCCVar::iterator inert_iter = inert_mixture_fractions.begin(); 
                inert_iter != inert_mixture_fractions.end(); inert_iter++ ){

              double inert_f = inert_iter->second.var[c];
              doubleMap inert_species_map_list = d_inertMap.find( inert_iter->first )->second; 

              double temp_table_value = table_value; 
              if ( i->first == "density" ){ 
                temp_table_value = 1.0/table_value; 
              } 

              post_mixing( temp_table_value, inert_f, i->first, inert_species_map_list ); 

              if ( i->first == "density" ){ 
                table_value = 1.0 / temp_table_value; 
              } else { 
                table_value = temp_table_value; 
              } 
            } 

            table_value *= eps_vol[c]; 
            (*i->second.var)[c] = table_value;

            if (i->first == "density") {
              // Two ways of setting density.  Note that the old ARCHES code used the table value directly and not the ghost_value as defined below. 
              // This gets density = bc value on face:
              //double ghost_value = 2.0*table_value - arches_density[cp1];
              //arches_density[c] = ghost_value; 
              // This gets density = bc value in extra cell 
              arches_density[c] = table_value; 

              if (d_MAlab)
                mpmarches_denmicro[c] = table_value; 

            } else if (i->first == "temperature" && !d_coldflow) {
              arches_temperature[c] = table_value; 
            } else if (i->first == "specificheat" && !d_coldflow) {
              arches_cp[c] = table_value; 
            } else if (i->first == "CO2" && !d_coldflow) {
              arches_co2[c] = table_value; 
cout << "ClassicTableInterface.cc: co2: " << table_value << "\n";
            } else if (i->first == "H2O" && !d_coldflow) {
              arches_h2o[c] = table_value; 
            }
          }
          iv.resize(0);
        }
        bc_values.resize(0);
          
        //_______________________________________
        //correct for solid wall temperatures
        //Q: do we want to do this here? 
        std::string T_name = "SolidWallTemperature"; 
        string face_name; 
        string bc_kind="NotSet"; 
        double bc_value = 0.0; 
        bool foundIterator = "false"; 
        getBCKind( patch, face, child, T_name, matlIndex, bc_kind, face_name ); 

        cout << "ClassicTableInterface.cc: bc_kind is '" << bc_kind << "', ''" << face_name << ", child: " << child << "\n";

        if ( bc_kind == "Dirichlet" || bc_kind == "Neumann" ) { 
          foundIterator = 
            getIteratorBCValue<double>( patch, face, child, T_name, matlIndex, bc_value, bound_ptr ); 
          //it is possible that this wasn't even set for a face, thus we can't really do 
          // any error checking here. 
        }

        const DepVarMap::iterator iter = depend_storage.find(_temperature_label_name); 

        if ( foundIterator ) {

cout << "ClassicTableInterface.cc: found Iterator: " << bc_kind << "\n";
          if ( iter == depend_storage.end() ) { 
            throw InternalError("Error: SolidWallTemperature was specified in the <BoundaryCondition> section yet I could not find a temperature variable (default label=temperature). Consider setting/checking <temperature_label_name> in the input file. " ,__FILE__,__LINE__);
          } 

          double dx = 0.0;
          double the_sign = 1.0; 

          if ( bc_kind == "Dirichlet" ) { 

cout << "ClassicTableInterface.cc: Dirichlet\n";
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++){

              IntVector c   =   *bound_ptr; 
              if ( cell_type[c] == BoundaryCondition_new::WALL ){ 
                (*iter->second.var)[c] = bc_value; 
              }

            }  

          } else if ( bc_kind == "Neumann" ) { 

cout << "ClassicTableInterface.cc: Neumann\n";
            Vector Dx = patch->dCell(); 
            switch (face) {
              case Patch::xminus:
                dx = Dx.x(); 
                the_sign = -1.0;
                break; 
              case Patch::xplus:
                dx = Dx.x(); 
                break; 
              case Patch::yminus:
                dx = Dx.y(); 
                the_sign = -1.0; 
                break; 
              case Patch::yplus:
                dx = Dx.y(); 
                break; 
              case Patch::zminus:
                dx = Dx.z(); 
                the_sign = -1.0; 
                break; 
              case Patch::zplus:
                dx = Dx.z(); 
                break; 
              default: 
                throw InvalidValue("Error: Face type not recognized.",__FILE__,__LINE__); 
                break; 
            }

            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++){

              IntVector c   =   *bound_ptr; 
              IntVector cp1 = ( *bound_ptr - insideCellDir ); 
        
              if ( cell_type[c] == BoundaryCondition_new::WALL ){ 
                (*iter->second.var)[c] = (*iter->second.var)[cp1] + the_sign * dx * bc_value; 
              }

            }  
          } 
        } 
      }
    }

    for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){
      delete i->second.var;
    }

    // reference density modification 
    if ( modify_ref_den ) {

cout << "ClassicTableInterface.cc: modify_ref_den\n";

      double den_ref = 0.0;

      if (patch->containsCell(d_ijk_den_ref)) {

        den_ref = arches_density[d_ijk_den_ref];
        cout << "ClassicTableInterface.cc: Modified reference density to: density_ref = " << den_ref << endl;

      }
      new_dw->put(sum_vartype(den_ref),time_labels->ref_density);
    }
  }
}


//--------------------------------------------------------------------------- 
// Old Table Hack -- to be removed with Properties.cc
//--------------------------------------------------------------------------- 
  void 
ClassicTableInterface::oldTableHack( const InletStream& inStream, Stream& outStream, bool calcEnthalpy, const string bc_type )
{

  cout_tabledbg << " In method ClassicTableInterface::OldTableHack " << endl;

  //This is a temporary hack to get the table stuff working with the new interface
  std::vector<double> iv(d_allIndepVarNames.size());

  for ( int i = 0; i < (int) d_allIndepVarNames.size(); i++){

    if ( (d_allIndepVarNames[i] == "mixture_fraction") || (d_allIndepVarNames[i] == "coal_gas_mix_frac") || (d_allIndepVarNames[i] == "MixtureFraction")){
      iv[i] = inStream.d_mixVars[0]; 
    } else if (d_allIndepVarNames[i] == "mixture_fraction_variance") {
      iv[i] = 0.0;
    } else if (d_allIndepVarNames[i] == "mixture_fraction_2") {
      iv[i] = inStream.d_f2; // set below if there is one...just want to make sure it is initialized properly
    } else if (d_allIndepVarNames[i] == "mixture_fraction_variance_2") {
      iv[i] = 0.0; 
    } else if (d_allIndepVarNames[i] == "heat_loss" || d_allIndepVarNames[i] == "HeatLoss") {
      iv[i] = inStream.d_heatloss; 
      if (!calcEnthalpy) {
        iv[i] = 0.0; // override any user input because case is adiabatic
      }
    }
  }

  if ( d_does_post_mixing && d_has_transform ) { 
    throw ProblemSetupException("ERROR! I shouldn't be in this part of the code.", __FILE__, __LINE__); 
  } else { 
    _iv_transform->transform( iv ); 
  }

  double f                 = 0.0; 
  double adiab_enthalpy    = 0.0; 
  double current_heat_loss = 0.0;
  double init_enthalpy     = 0.0; 

  f  = inStream.d_mixVars[0]; 

  if (calcEnthalpy) {

    // non-adiabatic case
    double enthalpy          = 0.0; 
    double sensible_enthalpy = 0.0; 

    IndexMap::iterator i_index = d_enthalpyVarIndexMap.find( "sensibleenthalpy" ); 
//    sensible_enthalpy    = tableLookUp( iv, i_index->second ); 
		sensible_enthalpy    = ND_interp->find_val( iv, i_index->second );
		
    i_index = d_enthalpyVarIndexMap.find( "adiabaticenthalpy" ); 
//    adiab_enthalpy = tableLookUp( iv, i_index->second ); 
		adiab_enthalpy = ND_interp->find_val( iv, i_index->second );

    enthalpy          = inStream.d_enthalpy; 

    if ( inStream.d_initEnthalpy || ((abs(adiab_enthalpy - enthalpy)/abs(adiab_enthalpy) < 1.0e-4 ) && f < 1.0e-4) ) {

      current_heat_loss = inStream.d_heatloss; 

      init_enthalpy = adiab_enthalpy - current_heat_loss * sensible_enthalpy; 

    } else {

      throw ProblemSetupException("ERROR! I shouldn't be in this part of the code.", __FILE__, __LINE__); 

    }
  } else {

    // adiabatic case
    init_enthalpy = 0.0;
    current_heat_loss = 0.0; 

  }

  IndexMap::iterator i_index = d_depVarIndexMap.find( "density" );
  //  outStream.d_density = tableLookUp( iv, i_index->second ); 
	outStream.d_density = ND_interp->find_val( iv, i_index->second );
	
  if (!d_coldflow) { 
    i_index = d_depVarIndexMap.find( "temperature" );
   // outStream.d_temperature = tableLookUp( iv, i_index->second ); 
		outStream.d_temperature = ND_interp->find_val( iv, i_index->second );
		
    i_index = d_depVarIndexMap.find( "specificheat" );
    // outStream.d_cp          = tableLookUp( iv, i_index->second ); 
		outStream.d_cp          = ND_interp->find_val( iv, i_index->second ); 
		
    i_index = d_depVarIndexMap.find( "H2O" );
    // outStream.d_h2o         = tableLookUp( iv, i_index->second );
		outStream.d_h2o         = ND_interp->find_val( iv, i_index->second );
	 	
    i_index = d_depVarIndexMap.find( "CO2" );
    //outStream.d_co2         = tableLookUp( iv, i_index->second ); 
		outStream.d_co2         = ND_interp->find_val( iv, i_index->second ); 
		
    outStream.d_heatLoss    = current_heat_loss; 
    if (inStream.d_initEnthalpy) outStream.d_enthalpy = init_enthalpy; 
  }

  cout_tabledbg << " Leaving method ClassicTableInterface::OldTableHack " << endl;
}

//--------------------------------------------------------------------------- 
// schedule Dummy Init
//--------------------------------------------------------------------------- 
  void 
ClassicTableInterface::sched_dummyInit( const LevelP& level, 
    SchedulerP& sched )

{
}

//--------------------------------------------------------------------------- 
// Dummy Init
//--------------------------------------------------------------------------- 
  void 
ClassicTableInterface::dummyInit( const ProcessorGroup* pc, 
    const PatchSubset* patches, 
    const MaterialSubset* matls, 
    DataWarehouse* old_dw, 
    DataWarehouse* new_dw )
{
}

//-------------------------------------
  void 
ClassicTableInterface::getIndexInfo()
{
  for ( MixingRxnModel::VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){ 

    std::string name = i->first; 
    int index = findIndex( name ); 

    depVarIndexMapLock.readLock();
    IndexMap::iterator iter = d_depVarIndexMap.find( name );
    depVarIndexMapLock.readUnlock();

    // Only insert variable if it isn't already there. 
    if ( iter == d_depVarIndexMap.end() ) {
      cout_tabledbg << " Inserting " << name << " index information into storage." << endl;

      depVarIndexMapLock.writeLock();
      iter = d_depVarIndexMap.insert( make_pair( name, index ) ).first; 
      depVarIndexMapLock.writeUnlock();
    }
  }
}

//-------------------------------------
  void 
ClassicTableInterface::getEnthalpyIndexInfo()
{
  enthalpyVarIndexMapLock.writeLock();
  cout_tabledbg << "ClassicTableInterface::getEnthalpyIndexInfo(): Looking up sensible enthalpy" << endl;
  int index = findIndex( "sensibleenthalpy" ); 

  d_enthalpyVarIndexMap.insert( make_pair( "sensibleenthalpy", index ));

  cout_tabledbg << "ClassicTableInterface::getEnthalpyIndexInfo(): Looking up adiabatic enthalpy" << endl;
  index = findIndex( "adiabaticenthalpy" ); 
  d_enthalpyVarIndexMap.insert( make_pair( "adiabaticenthalpy", index ));

  index = findIndex( "density" ); 
  d_enthalpyVarIndexMap.insert( make_pair( "density", index ));
  enthalpyVarIndexMapLock.writeUnlock();
}

//-------------------------------------



//-----------------------------------------
  void
ClassicTableInterface::loadMixingTable( const string & inputfile )
{

  proc0cout << " Preparing to read the table inputfile:   " << inputfile << "\n";
  gzFile gzFp = gzopen( inputfile.c_str(), "r" );

  if( gzFp == NULL ) {
    // If errno is 0, then not enough memory to uncompress file.
    proc0cout << "Error with gz in opening file: " << inputfile << ". Errno: " << errno << "\n"; 
    throw ProblemSetupException("Unable to open the given input file: " + inputfile, __FILE__, __LINE__);
  }

  d_indepvarscount = getInt( gzFp );

  proc0cout << " Total number of independent variables: " << d_indepvarscount << endl;

  d_allIndepVarNames = vector<std::string>(d_indepvarscount);
  int index_is_hl = -1; 
  int hl_grid_size = -1; 
 
  for (int ii = 0; ii < d_indepvarscount; ii++) {
    string varname = getString( gzFp );
    d_allIndepVarNames[ii] =  varname ;
    if ( varname == "heat_loss" ) 
      index_is_hl = ii;  
  }

  d_allIndepVarNum = vector<int>(d_indepvarscount);
  for (int ii = 0; ii < d_indepvarscount; ii++) {
    int grid_size = getInt( gzFp ); 
    d_allIndepVarNum[ii] =  grid_size ; 

    if ( ii == index_is_hl ) 
      hl_grid_size = grid_size - 1; 
  }

  for (int ii = 0; ii < d_indepvarscount; ii++){
    proc0cout << " Independent variable: " << d_allIndepVarNames[ii] << " has a grid size of: " << d_allIndepVarNum[ii] << endl;
  }

  // Total number of variables in the table: non-adaibatic table has sensibile enthalpy too
  d_varscount = getInt( gzFp );
  proc0cout << " Total dependent variables in table: " << d_varscount << endl;

  d_allDepVarNames = vector<std::string>(d_varscount); 
  for (int ii = 0; ii < d_varscount; ii++) {

    std::string variable; 
    variable = getString( gzFp );
    d_allDepVarNames[ii] = variable ; 
  }

  // Units
  d_allDepVarUnits = vector<std::string>(d_varscount); 
  for (int ii = 0; ii < d_varscount; ii++) {
    std::string units = getString( gzFp );
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
	    double v = getDouble( gzFp );
	    indep_headers[i][j] = v;
	  }
  }
	
  int size=1;
  //ND size
  for (int i = 0; i < d_indepvarscount; i++) {
	  size = size*d_allIndepVarNum[i];
  }

  // getting heat loss bounds: 
  if ( index_is_hl != -1 ) { 
    if ( index_is_hl == 0 ) {
      d_hl_lower_bound = (i1[0][0]);
      d_hl_upper_bound = (i1[hl_grid_size][0]);
    } else { 
	    d_hl_lower_bound = indep_headers[index_is_hl-1][0];
	    d_hl_upper_bound = indep_headers[index_is_hl-1][d_allIndepVarNum[index_is_hl]-1];
    }  
    proc0cout << " Lower bounds on heat loss = " << d_hl_lower_bound << endl;
    proc0cout << " Upper bounds on heat loss = " << d_hl_upper_bound << endl;
  } 

  table = vector<vector<double> >(d_varscount); 
  for ( int i = 0; i < d_varscount; i++ ){ 
    table[i] = vector<double>(size);
  }
	
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
					  double v = getDouble(gzFp);
					  i1[mm][i] = v;
				  }
			  } else {
				  //read but don't assign inbetween vals
				  for (int i = 0; i < d_allIndepVarNum[0]; i++) {
					  double v = getDouble(gzFp);
					  v += 0.0;
				  }
			  }
			    for (int j=0; j<size2; j++) {
				    double v = getDouble(gzFp);
				    table[kk][j + mm*size2] = v;
				  }
			  }
		  if ( read_assign ) { read_assign = false; }
	  } 
	} else {
		for (int kk = 0; kk < d_varscount; kk++) {
			proc0cout << "loading --->" << d_allDepVarNames[kk] << endl;
			if (read_assign) {
				for (int i=0; i<d_allIndepVarNum[0]; i++) {
					double v = getDouble(gzFp);
					i1[0][i] = v;
				}
			} else { 
			  for (int i=0; i<d_allIndepVarNum[0]; i++) {
					double v = getDouble(gzFp);
					v += 0.0;
				}	
			}
			for (int j=0; j<size; j++) {
				double v = getDouble(gzFp);
				table[kk][j] = v;
			}
			if (read_assign){read_assign = false;}
	  }
	}
  
  // Closing the file pointer
  gzclose( gzFp );

	proc0cout << "creating object " << endl;
	
	if (d_indepvarscount == 1) {
		ND_interp = new Interp1(d_allIndepVarNum, table, i1);
  }	else if (d_indepvarscount == 2) {
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
//---------------------------
double 
ClassicTableInterface::getTableValue( std::vector<double> iv, std::string variable )
{
  IndexMap::iterator i_index = d_enthalpyVarIndexMap.find( variable ); 
  double value = ND_interp->find_val(iv, i_index->second );
	return value; 
}

//---------------------------
double 
ClassicTableInterface::getTableValue( std::vector<double> iv, std::string depend_varname, 
                                      MixingRxnModel::StringToCCVar inert_mixture_fractions, 
                                      IntVector c )
{ 

  double total_inert_f = 0.0; 
  MixingRxnModel::InertMixing* inert_transform=0; 
  if ( d_does_post_mixing && d_has_transform ){ 
    inert_transform = dynamic_cast<MixingRxnModel::InertMixing*>(_iv_transform); 
  }

	int dep_index = findIndex( depend_varname ); 

  for (StringToCCVar::iterator inert_iter = inert_mixture_fractions.begin(); 
      inert_iter != inert_mixture_fractions.end(); inert_iter++ ){

    double inert_f = inert_iter->second.var[c];
    total_inert_f += inert_f; 

  }

  if ( d_does_post_mixing && d_has_transform ) { 
    inert_transform->transform( iv, total_inert_f ); 
  } else { 
    _iv_transform->transform( iv ); 
  }

	double table_value = ND_interp->find_val( iv, dep_index ); 

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
  MixingRxnModel::InertMixing* inert_transform=0; 
  if ( d_does_post_mixing && d_has_transform ){ 
    inert_transform = dynamic_cast<MixingRxnModel::InertMixing*>(_iv_transform); 
  }

	int dep_index = findIndex( depend_varname ); 

  for (MixingRxnModel::doubleMap::iterator inert_iter = inert_mixture_fractions.begin(); 
      inert_iter != inert_mixture_fractions.end(); inert_iter++ ){

    double inert_f = inert_iter->second;
    total_inert_f += inert_f; 

  }

  if ( d_does_post_mixing && d_has_transform ) { 
    inert_transform->transform( iv, total_inert_f ); 
  } else { 
    _iv_transform->transform( iv ); 
  }

	double table_value = ND_interp->find_val( iv, dep_index ); 

  // for post look-up mixing
  for (MixingRxnModel::doubleMap::iterator inert_iter = inert_mixture_fractions.begin(); 
      inert_iter != inert_mixture_fractions.end(); inert_iter++ ){

    double inert_f = inert_iter->second;
    doubleMap inert_species_map_list = d_inertMap.find( inert_iter->first )->second; 

    post_mixing( table_value, inert_f, depend_varname, inert_species_map_list ); 

  }

	return table_value; 

}

void ClassicTableInterface::checkForConstants( const string & inputfile ) { 

  proc0cout << "\n Looking for constants in the header... " << endl;

  gzFile gzFp = gzopen( inputfile.c_str(), "r" );

  if( gzFp == NULL ) {
    // If errno is 0, then not enough memory to uncompress file.
    proc0cout << "Error with gz in opening file: " << inputfile << ". Errno: " << errno << "\n"; 
    throw ProblemSetupException("Unable to open the given input file: " + inputfile, __FILE__, __LINE__);
  }

  bool look = true; 
  while ( look ){ 

    char ch = gzgetc( gzFp ); 

    if ( ch == '#' ) { 

      char key = gzgetc( gzFp );

      if ( key == 'K' ) {
        for (int i = 0; i < 3; i++ ){
          key = gzgetc( gzFp ); // reading the word KEY and space
        }

        string name; 
        while ( true ) {
          key = gzgetc( gzFp ); 
          if ( key == '=' ) { 
            break; 
          }
          name.push_back( key );  // reading in the token's key name
        }

        string value_s; 
        while ( true ) { 
          key = gzgetc( gzFp ); 
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
          ch = gzgetc( gzFp ); // skipping this line
          if ( ch == '\n' || ch == '\t' ) { 
            break; 
          }
        }
      }

    } else {

      look = false; 

    }
  }

  // Closing the file pointer
  gzclose( gzFp );
}
//---------------------

