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

//----- ClassicTableInterface.cc --------------------------------------------------

// includes for Arches
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <CCA/Components/Arches/ChemMix/ClassicTableInterface.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelFactory.h>
#include <CCA/Components/Arches/PropertyModels/HeatLoss.h>
#include <CCA/Components/Arches/ChemMixV2/ClassicTableUtility.h>

#include <Core/Parallel/MasterLock.h>

#include <sci_defs/kokkos_defs.h>

#define OLD_TABLE 1
#undef OLD_TABLE

using namespace std;
using namespace Uintah;

namespace {

Uintah::MasterLock dependency_map_mutex{};
Uintah::MasterLock enthalpy_map_mutex{};

}

//---------------------------------------------------------------------------
// Default Constructor
//---------------------------------------------------------------------------
ClassicTableInterface::ClassicTableInterface( MaterialManagerP& materialManager )
  : MixingRxnModel( materialManager )
{
  m_matl_index = 0;
}

//---------------------------------------------------------------------------
// Default Destructor
//---------------------------------------------------------------------------
ClassicTableInterface::~ClassicTableInterface()
{
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

   //READ TABLE:
  ND_interp=SCINEW_ClassicTable(tableFileName); // requires a delete on ND_interp object by host class
         
  d_allDepVarNames=ND_interp->tableInfo.d_savedDep_var;
  d_allIndepVarNames=ND_interp->tableInfo.d_allIndepVarNames;
  d_allIndepVarNum=ND_interp->tableInfo.d_allIndepVarNum;

  d_constants=(ND_interp->tableInfo.d_constants);     ///< List of constants in table header
  d_indepvarscount=d_allIndepVarNames.size();       ///< Number of independent variables
  d_varscount=d_allDepVarNames.size();            ///< Total dependent variables


  // Confirm that table has been loaded into memory
  d_table_isloaded = true;


  // Extract independent and dependent variables from input file
  ProblemSpecP db_rootnode = db;
  db_rootnode = db_rootnode->getRootNode();

  proc0cout << "---------------------------------------------------------------  " << std::endl;
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


  // Match the requested dependent variables with their table index:
  getIndexInfo();
  if (!d_coldflow)
    getEnthalpyIndexInfo();

  problemSetupCommon( db_classic, this );

  d_hl_upper_bound = 1;
  d_hl_lower_bound = -1;

  if ( _iv_transform->has_heat_loss() ){

    const vector<double> hl_bounds = _iv_transform->get_hl_bounds(ND_interp->tableInfo.indep_headers , d_allIndepVarNum);

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
        // int timeStep = m_materialManager->getCurrentTopLevelTimeStep();

        timeStep_vartype timeStep(0);
        if( sched->get_dw(0) && sched->get_dw(0)->exists( m_timeStepLabel ) )
          sched->get_dw(0)->get( timeStep, m_timeStepLabel );
        else if( sched->get_dw(1) && sched->get_dw(1)->exists( m_timeStepLabel ) )
          sched->get_dw(1)->get( timeStep, m_timeStepLabel );

        if ( timeStep != 0 ){
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

  sched->addTask( tsk, level->eachPatch(), m_materialManager->allMaterials( "Arches" ) );
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


    std::vector<CCVariable<double> >CCVar_vec_lookup (d_dvVarMap.size()); // needs to be expanded newTable
    std::vector<int> depVarIndexes(d_nDepVars);
    if ( initialize_me ) {
       int  ixx =0;
      for ( VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){

           new_dw->allocateAndPut( CCVar_vec_lookup[ixx], i->second, m_matl_index, patch );
           (CCVar_vec_lookup[ixx]).initialize(0.0);

           IndexMap::iterator i_index = d_depVarIndexMap.find( i->first );
           depVarIndexes[ixx]=i_index->second;

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
       ixx++;
      }

    } else {

       int  ixx =0;
      for ( VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){

   
        new_dw->getModifiable(CCVar_vec_lookup[ixx],i->second  , m_matl_index, patch );

        IndexMap::iterator i_index = d_depVarIndexMap.find( i->first );
        depVarIndexes[ixx]=i_index->second;

   
        std::string name = i->first+"_old";
        VarMap::iterator i_old = d_oldDvVarMap.find(name);


        if ( i_old != d_oldDvVarMap.end() ){
          //copy current value into old
          CCVariable<double> old_value;
          new_dw->getModifiable( old_value, i_old->second, m_matl_index, patch );
          old_value.copy(CCVar_vec_lookup[ixx] );
        }

       ixx++;
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

    IntVector domLo = patch->getCellLowIndex();
    IntVector domHi = patch->getCellHighIndex();
    Uintah::BlockRange range(domLo,domHi);



   std::vector<CCVariable<double> > IVs_transformed(nIndVars);
   for (int  ix = 0; ix< nIndVars; ix++){
     IVs_transformed[ix].allocate(domLo,domHi);
   }


    Uintah::parallel_for(range,  [&]( int i,  int j, int k){

    std::vector<double> iv1(nIndVars);
    int ixxx=0;
    for ( std::vector<constCCVariable<double> >::iterator iter = indep_storage.begin(); iter != indep_storage.end(); ++iter ) {

      iv1[ixxx]=(*iter)(i,j,k);
      ixxx++;
    }


    double total_inert_f = 0.0;
    for (StringToCCVar::iterator inert_iter = inert_mixture_fractions.begin();
        inert_iter != inert_mixture_fractions.end(); inert_iter++ ){
        double inert_f = inert_iter->second.var(i,j,k);
        total_inert_f += inert_f;
    }

    _iv_transform->transform( iv1, total_inert_f );


    for (int  ix = 0; ix< nIndVars; ix++){
      IVs_transformed[ix](i,j,k)=iv1[ix];
    }

    });


                             //need a getState where the 
    ND_interp->getState( IVs_transformed, CCVar_vec_lookup   ,d_allIndepVarNames ,patch,depVarIndexes);
  



////////    now deal with the mixing and density checks same as before
    int density_index=0;

    for ( unsigned int depVar_i=0; depVar_i < d_dvVarMap.size(); depVar_i++){ 
          if ( d_allDepVarNames[depVarIndexes[depVar_i]] == "density" ){
          density_index=depVarIndexes[depVar_i];
        Uintah::parallel_for(range,  [&]( int i,  int j, int k){

        for (StringToCCVar::iterator inert_iter = inert_mixture_fractions.begin();
            inert_iter != inert_mixture_fractions.end(); inert_iter++ ){
          double inert_f = inert_iter->second.var(i,j,k);
          doubleMap inert_species_map_list = d_inertMap.find( inert_iter->first )->second;

          double temp_table_value= 1.0/CCVar_vec_lookup[depVar_i](i,j,k);
          post_mixing( temp_table_value, inert_f, d_allDepVarNames[depVarIndexes[depVar_i]], inert_species_map_list );
          CCVar_vec_lookup[depVar_i](i,j,k)  = 1.0/temp_table_value;
          }

          CCVar_vec_lookup[depVar_i](i,j,k)  *= eps_vol(i,j,k);
          arches_density(i,j,k) = CCVar_vec_lookup[depVar_i](i,j,k);
          });
        }else{
          Uintah::parallel_for(range,  [&]( int i,  int j, int k){
        for (StringToCCVar::iterator inert_iter = inert_mixture_fractions.begin();
            inert_iter != inert_mixture_fractions.end(); inert_iter++ ){
          double inert_f = inert_iter->second.var(i,j,k);
          doubleMap inert_species_map_list = d_inertMap.find( inert_iter->first )->second;

          double temp_table_value=CCVar_vec_lookup[depVar_i](i,j,k);
          post_mixing( temp_table_value, inert_f, d_allDepVarNames[depVarIndexes[depVar_i]], inert_species_map_list );
          CCVar_vec_lookup[depVar_i](i,j,k)  = temp_table_value;

        }


          CCVar_vec_lookup[depVar_i] (i,j,k)  *= eps_vol(i,j,k);
          });
        }
    }


    // TODO: Move this to parallel for.  How do you do this for a boundary condition?
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

          std::vector<double> depVarValues;
          depVarValues = ND_interp->find_val(iv, depVarIndexes );

          //take care of the mixing and density the same
          int depVarCount = 0;
          // now get state for boundary cell:
          for ( unsigned int depVar_i=0; depVar_i < d_dvVarMap.size(); depVar_i++){ 

            // for post look-up mixing
            for (StringToCCVar::iterator inert_iter = inert_mixture_fractions.begin();
                inert_iter != inert_mixture_fractions.end(); inert_iter++ ){

              doubleMap inert_species_map_list = d_inertMap.find( inert_iter->first )->second;

              double temp_table_value = depVarValues[depVarCount];
              if ( depVarIndexes[depVar_i] == density_index ){
                temp_table_value = 1.0/depVarValues[depVarCount];
              }

              post_mixing( temp_table_value, total_inert_f, d_allDepVarNames[depVarIndexes[depVar_i]], inert_species_map_list );

              if ( depVarIndexes[depVar_i] == density_index ){
                depVarValues[depVarCount] = 1.0 / temp_table_value;
              } else {
                depVarValues[depVarCount] = temp_table_value;
              }
            }

            depVarValues[depVarCount] *= eps_vol[c];
            double ghost_value = 2.0*depVarValues[depVarCount] -  CCVar_vec_lookup[ depVar_i][cp1];
            CCVar_vec_lookup[ depVar_i][c] = ghost_value;

            if (depVarIndexes[depVar_i] == density_index)
              arches_density[c] = ghost_value;
            depVarCount++;
          } 
          iv.resize(0);
        }
      }
    }

    // reference density modification
    if ( modify_ref_den ) {

      //actually modify the reference density value:
      std::vector<double> iv = _iv_transform->get_reference_iv();

      std::vector<int> varIndex (1, density_index  );
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

