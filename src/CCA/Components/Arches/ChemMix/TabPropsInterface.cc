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

using namespace std;
using namespace Uintah;

//--------------------------------------------------------------------------- 
// Default Constructor 
//--------------------------------------------------------------------------- 
TabPropsInterface::TabPropsInterface( const ArchesLabel* labels, const MPMArchesLabel* MAlabels ) :
MixingRxnModel( labels, MAlabels )
{}

//--------------------------------------------------------------------------- 
// Default Destructor
//--------------------------------------------------------------------------- 
TabPropsInterface::~TabPropsInterface()
{}

//--------------------------------------------------------------------------- 
// Problem Setup
//--------------------------------------------------------------------------- 
void
TabPropsInterface::problemSetup( const ProblemSpecP& propertiesParameters )
{
  // Create sub-ProblemSpecP object
  string tableFileName;
  ProblemSpecP db_tabprops = propertiesParameters->findBlock("TabProps");
  
  // Obtain object parameters
  db_tabprops->require( "inputfile", tableFileName );

  db_tabprops->getWithDefault( "hl_pressure", d_hl_pressure, 0.0); 
  db_tabprops->getWithDefault( "hl_outlet",   d_hl_outlet,   0.0); 
  db_tabprops->getWithDefault( "hl_scalar_init", d_hl_scalar_init, 0.0); 
  db_tabprops->getWithDefault( "cold_flow", d_coldflow, false); 
  db_tabprops->getWithDefault( "adiabatic", d_adiabatic, false); 

  // need the reference denisty point: (also in PhysicalPropteries object but this was easier than passing it around)
  const ProblemSpecP db_root = db_tabprops->getRootNode(); 
  db_root->findBlock("PhysicalConstants")->require("reference_point", d_ijk_den_ref);  
  

  // Check for and deal with filename extension
  // - if table file name has .h5 extension, remove it
  // - otherwise, assume it is an .h5 file but no extension was given
  string extension (tableFileName.end()-3,tableFileName.end());
  if( extension == ".h5" || extension == ".H5" ) {
    tableFileName = tableFileName.substr( 0, tableFileName.size() - 3 );
  }
  
  // Load data from HDF5 file into StateTable
  d_statetbl.read_hdf5(tableFileName);

  // Extract independent and dependent variables from input file
  ProblemSpecP db_rootnode = propertiesParameters;
  db_rootnode = db_rootnode->getRootNode();

  proc0cout << endl;
  proc0cout << "--- TabProps information --- " << endl;
  proc0cout << endl;

  setMixDVMap( db_rootnode ); 

  // Extract independent and dependent variables from the table
  d_allIndepVarNames = d_statetbl.get_indepvar_names();
  d_allDepVarNames   = d_statetbl.get_depvar_names();
 
  proc0cout << "  Now matching user-defined IV's with table IV's" << endl;
  proc0cout << "     Note: If sus crashes here, check to make sure your" << endl;
  proc0cout << "           <TransportEqns><eqn> names match those in the table. " << endl;

  cout_tabledbg << " Creating the independent variable map " << endl;
  for ( unsigned int i = 0; i < d_allIndepVarNames.size(); ++i ){

    //put the right labels in the label map
    string varName = d_allIndepVarNames[i];  

    // !! need to add support for variance !!
    if (varName == "heat_loss") {

      cout_tabledbg << " Heat loss being inserted into the indep. var map. " << endl;

      d_ivVarMap.insert(make_pair(varName, d_lab->d_heatLossLabel)).first; 

    } else if ( varName == "DissipationRate") {

      cout_tabledbg << " Scalar dissipation rate being inserted into the indep. var map. " << endl;

      // dissipation rate comes from a property model 
      PropertyModelFactory& prop_factory = PropertyModelFactory::self(); 
      PropertyModelBase& prop = prop_factory.retrieve_property_model( "scalar_dissipation_rate");
      d_ivVarMap.insert( make_pair( varName, prop.getPropLabel()) ).first; 

    } else {

      cout_tabledbg << " Variable: " << varName << " being inserted into the indep. var map"<< endl; 

      // then it must be a mixture fraction 
      EqnFactory& eqn_factory = EqnFactory::self();
      EqnBase& eqn = eqn_factory.retrieve_scalar_eqn( varName );
      d_ivVarMap.insert(make_pair(varName, eqn.getTransportEqnLabel())).first; 

    }
  }

  proc0cout << "  Matching sucessful!" << endl;
  proc0cout << endl;

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
                                   const TimeIntegratorLabel* time_labels, 
                                   const bool initialize_me,
                                   const bool with_energy_exch, 
                                   const bool modify_ref_den )

{
  string taskname = "TabPropsInterface::getState"; 
  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &TabPropsInterface::getState, time_labels, initialize_me, with_energy_exch, modify_ref_den );

  // independent variables
  for ( MixingRxnModel::VarMap::iterator i = d_ivVarMap.begin(); i != d_ivVarMap.end(); ++i ) {

    tsk->requires( Task::NewDW, i->second, gn, 0 ); 

  }

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
  if ( modify_ref_den )
    tsk->computes(time_labels->ref_density); 
   

  sched->addTask( tsk, level->eachPatch(), d_lab->d_sharedState->allArchesMaterials() ); 
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

    //independent variables:
    std::vector<constCCVariable<double> > indep_storage; 
    const std::vector<string>& iv_names = getAllIndepVars();

    for ( int i = 0; i < (int) iv_names.size(); i++ ){

      VarMap::iterator ivar = d_ivVarMap.find( iv_names[i] ); 

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

    CCMap depend_storage; 
    if ( initialize_me ) {
    
      for ( VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){
       
        CCVariable<double>* the_var = new CCVariable<double>; 
        new_dw->allocateAndPut( *the_var, i->second, matlIndex, patch ); 
        (*the_var).initialize(0.0);

        depend_storage.insert( make_pair( i->first, the_var )).first; 
        
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

      drho_df.initialize(0.0);  // this variable might not be actually used anywhere any may just be polution  
      if ( !d_coldflow ) { 
        arches_temperature.initialize(0.0); 
        arches_cp.initialize(0.0); 
        arches_co2.initialize(0.0); 
        arches_h2o.initialize(0.0);
        arches_soot.initialize(0.0); 
      }

    } else { 

      for ( VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){
       
        CCVariable<double>* the_var = new CCVariable<double>; 
        new_dw->getModifiable( *the_var, i->second, matlIndex, patch ); 

        depend_storage.insert( make_pair( i->first, the_var )).first; 
        
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

    CCVariable<double> arches_density; 
    new_dw->getModifiable( arches_density, d_lab->d_densityCPLabel, matlIndex, patch ); 

    // Go through the patch and populate the requested state variables
    for (CellIterator iter=patch->getExtraCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 

      // fill independent variables
      std::vector<double> iv; 
      for ( std::vector<constCCVariable<double> >::iterator i = indep_storage.begin(); i != indep_storage.end(); ++i ) {
        iv.push_back( (*i)[c] );
      }

      // retrieve all depenedent variables from table
      for ( std::map< string, CCVariable<double>* >::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){
  
        double table_value = getSingleState( i->first, iv ); 
        (*i->second)[c] = table_value;

        if (i->first == "density") {
          arches_density[c] = table_value; 
          if (d_MAlab)
            mpmarches_denmicro[c] = table_value; 
        } else if (i->first == "temperature" && !d_coldflow) {
          arches_temperature[c] = table_value; 
        //} else if (i->first == "heat_capacity" && !d_coldflow) {
        } else if (i->first == "specificheat" && !d_coldflow) {
          arches_cp[c] = table_value; 
        } else if (i->first == "CO2" && !d_coldflow) {
          arches_co2[c] = table_value; 
        } else if (i->first == "H2O" && !d_coldflow) {
          arches_h2o[c] = table_value; 
        }

      }

    }

    for ( CCMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){
      delete i->second;
    }

    // reference density modification 
    if ( modify_ref_den ) {

      double den_ref = 0.0;

      if (patch->containsCell(d_ijk_den_ref)) {

        den_ref = arches_density[d_ijk_den_ref];
        cerr << "Modified reference density to: density_ref = " << den_ref << endl;

      }
      new_dw->put(sum_vartype(den_ref),time_labels->ref_density);
    }
  }
}

//--------------------------------------------------------------------------- 
// schedule Compute Heat Loss
//--------------------------------------------------------------------------- 
void 
TabPropsInterface::sched_computeHeatLoss( const LevelP& level, SchedulerP& sched, const bool initialize_me, const bool calcEnthalpy )
{
  string taskname = "TabPropsInterface::computeHeatLoss"; 
  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &TabPropsInterface::computeHeatLoss, initialize_me, calcEnthalpy );

  // independent variables
  for (MixingRxnModel::VarMap::iterator i = d_ivVarMap.begin(); i != d_ivVarMap.end(); ++i) {

    const VarLabel* the_label = i->second;
    if (i->first != "heat_loss") 
      tsk->requires( Task::NewDW, the_label, gn, 0 ); 
  }

  // heat loss must be computed if this is the first FE step 
  if (initialize_me)
    tsk->computes( d_lab->d_heatLossLabel );
  else 
    tsk->modifies( d_lab->d_heatLossLabel ); 

  if ( calcEnthalpy )
    tsk->requires( Task::NewDW, d_lab->d_enthalpySPLabel, gn, 0 ); 

  sched->addTask( tsk, level->eachPatch(), d_lab->d_sharedState->allArchesMaterials() ); 
}

//--------------------------------------------------------------------------- 
// Compute Heat Loss
//--------------------------------------------------------------------------- 
void 
TabPropsInterface::computeHeatLoss( const ProcessorGroup* pc, 
                                    const PatchSubset* patches, 
                                    const MaterialSubset* matls, 
                                    DataWarehouse* old_dw, 
                                    DataWarehouse* new_dw, 
                                    const bool initialize_me,
                                    const bool calcEnthalpy )
{
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType gn = Ghost::None; 
    const Patch* patch = patches->get(p); 
    int archIndex = 0; 
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> heat_loss; 
    if ( initialize_me )
      new_dw->allocateAndPut( heat_loss, d_lab->d_heatLossLabel, matlIndex, patch ); 
    else 
      new_dw->getModifiable ( heat_loss, d_lab->d_heatLossLabel, matlIndex, patch ); 
    heat_loss.initialize(0.0); 

    constCCVariable<double> enthalpy; 
    if ( calcEnthalpy ) 
      new_dw->get(enthalpy, d_lab->d_enthalpySPLabel, matlIndex, patch, gn, 0 ); 

    std::vector<constCCVariable<double> > the_variables; 
    const std::vector<string>& iv_names = getAllIndepVars();

    // exceptions for cold flow or adiabatic cases
    bool compute_heatloss = true; 
    if ( d_coldflow ) 
      compute_heatloss = false; 
    if ( d_adiabatic ) 
      compute_heatloss = false; 

    if ( compute_heatloss ) { 

      for ( int i = 0; i < (int) iv_names.size(); i++ ){

        VarMap::iterator ivar = d_ivVarMap.find( iv_names[i] ); 
        if ( ivar->first != "heat_loss" ){
          constCCVariable<double> test_Var; 
          new_dw->get( test_Var, ivar->second, matlIndex, patch, gn, 0 );  

          the_variables.push_back( test_Var ); 
        } else {
          constCCVariable<double> a_null_var; 
          the_variables.push_back( a_null_var ); // to preserve the total number of IV otherwise you will have problems below
        }
      }

      for (CellIterator iter=patch->getCellIterator(0); !iter.done(); iter++){
        IntVector c = *iter; 

        vector<double> iv; 
        int index = 0; 
        for ( std::vector<constCCVariable<double> >::iterator i = the_variables.begin(); i != the_variables.end(); i++){

          if ( d_allIndepVarNames[index] != "heat_loss" ) 
            iv.push_back( (*i)[c] );
          else 
            iv.push_back( 0.0 ); 

          index++; 
        }

        // actually compute the heat loss: 
        double sensible_enthalpy  = getSingleState( "sensibleenthalpy", iv ); 
        double adiabatic_enthalpy = getSingleState( "adiabaticenthalpy", iv );  
        double current_heat_loss  = 0.0;
        double small = 1e-10; 
        if ( calcEnthalpy )
          current_heat_loss = ( adiabatic_enthalpy - enthalpy[c] ) / ( sensible_enthalpy + small ); 

        if ( current_heat_loss < -1.0 )
          current_heat_loss = -1.0; 
        else if ( current_heat_loss > 1.0 ) 
          current_heat_loss = 1.0; 

        heat_loss[c] = current_heat_loss; 

      }
    }
  }
}

//--------------------------------------------------------------------------- 
// schedule Compute First Enthalpy
//--------------------------------------------------------------------------- 
void 
TabPropsInterface::sched_computeFirstEnthalpy( const LevelP& level, SchedulerP& sched )
{
  string taskname = "TabPropsInterface::computeFirstEnthalpy"; 
  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &TabPropsInterface::computeFirstEnthalpy );

  tsk->modifies( d_lab->d_enthalpySPLabel ); 

  // independent variables
  for (MixingRxnModel::VarMap::iterator i = d_ivVarMap.begin(); i != d_ivVarMap.end(); ++i) {

    const VarLabel* the_label = i->second;
    if (i->first != "heat_loss") 
      tsk->requires( Task::NewDW, the_label, gn, 0 ); 
  }

  sched->addTask( tsk, level->eachPatch(), d_lab->d_sharedState->allArchesMaterials() ); 

}

//--------------------------------------------------------------------------- 
// Compute First Enthalpy
//--------------------------------------------------------------------------- 
void 
TabPropsInterface::computeFirstEnthalpy( const ProcessorGroup* pc, 
                                         const PatchSubset* patches, 
                                         const MaterialSubset* matls, 
                                         DataWarehouse* old_dw, 
                                         DataWarehouse* new_dw ) 
{

  for (int p=0; p < patches->size(); p++){

    cout_tabledbg << " In TabPropsInterface::getFirstEnthalpy " << endl;

    Ghost::GhostType gn = Ghost::None; 
    const Patch* patch = patches->get(p); 
    int archIndex = 0; 
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> enthalpy; 
    new_dw->getModifiable( enthalpy, d_lab->d_enthalpySPLabel, matlIndex, patch ); 
      
    std::vector<constCCVariable<double> > the_variables; 
    for ( VarMap::iterator i = d_ivVarMap.begin(); i != d_ivVarMap.end(); i++ ){

      if ( i->first != "heat_loss" ){ // heat loss hasn't been computed yet so this is why we have an "if" here. 
        
        cout_tabledbg << " Found label =  " << i->first <<  endl;
        constCCVariable<double> test_Var; 
        new_dw->get( test_Var, i->second, matlIndex, patch, gn, 0 );  

        the_variables.push_back( test_Var ); 
      } else {
        constCCVariable<double> a_null_var;
        the_variables.push_back( a_null_var ); // to preserve the total number of IV otherwise you will have problems below
      }
    }

    for (CellIterator iter=patch->getCellIterator(0); !iter.done(); iter++){
      IntVector c = *iter; 

      vector<double> iv; 
      int index = 0; 
      for ( std::vector<constCCVariable<double> >::iterator i = the_variables.begin(); i != the_variables.end(); i++){

        if ( d_allIndepVarNames[index] != "heat_loss" ) 
          //iv.push_back( (*i)[c] ); // <--- I am not sure how this worked before. 
          iv.push_back(0.0);
        else 
          iv.push_back( d_hl_scalar_init ); 

        index++; 
      }

      double current_heat_loss = d_hl_scalar_init; // may want to make this more sophisticated later(?)
      double sensible_enthalpy = getSingleState( "sensibleenthalpy", iv ); 
      double adiab_enthalpy    = getSingleState( "adiabaticenthalpy", iv ); 
      enthalpy[c]     = adiab_enthalpy - current_heat_loss * sensible_enthalpy; 

    }
  }
}

//--------------------------------------------------------------------------- 
// Old Table Hack -- to be removed with Properties.cc
//--------------------------------------------------------------------------- 
void 
TabPropsInterface::oldTableHack( const InletStream& inStream, Stream& outStream, bool calcEnthalpy, const string bc_type )
{

  cout_tabledbg << " In method TabPropsInterface::OldTableHack " << endl;

  //This is a temporary hack to get the table stuff working with the new interface
  const std::vector<string>& iv_names = getAllIndepVars();
  std::vector<double> iv(iv_names.size());

  // notes: 
  // mixture fraction 2 is being set to zero always!...fix it!

  for ( int i = 0; i < (int) iv_names.size(); i++){

    if ( (iv_names[i] == "mixture_fraction") || (iv_names[i] == "coal_gas_mix_frac") || (iv_names[i] == "MixtureFraction")){
      iv[i] = inStream.d_mixVars[0]; 
    } else if (iv_names[i] == "mixture_fraction_variance") {
      iv[i] = 0.0;
    } else if (iv_names[i] == "mixture_fraction_2") {
      iv[i] = 0.0; 
    } else if (iv_names[i] == "mixture_fraction_variance_2") {
      iv[i] = 0.0; 
    } else if (iv_names[i] == "heat_loss") {
      if ( bc_type == "pressure" )
        iv[i] = d_hl_pressure; 
      else if ( bc_type == "outlet" )
        iv[i] = d_hl_outlet; 
      else if ( bc_type == "scalar_init" )
        iv[i] = d_hl_scalar_init; 
      else
        iv[i] = 0.0; 
      if (!calcEnthalpy) {
        iv[i] = 0.0; // override any user input because case is adiabatic
        if ( d_hl_scalar_init > 0.0 || d_hl_outlet > 0.0 || d_hl_pressure > 0.0 )
          proc0cout << "NOTICE!: Case is adiabatic so we will ignore your heat loss initialization." << endl;
      }
    }
  }

  double f                 = 0.0; 
  double f_2               = 0.0; 
  double adiab_enthalpy    = 0.0; 
  double current_heat_loss = 0.0;
  double init_enthalpy     = 0.0; 

  f  = inStream.d_mixVars[0]; 
  if (inStream.d_has_second_mixfrac){
    f_2 = inStream.d_f2; 
  }

  if (calcEnthalpy) {

    // non-adiabatic case
    double enthalpy          = 0.0; 
    double sensible_enthalpy = 0.0; 

    sensible_enthalpy = getSingleState( "sensibleenthalpy", iv ); 
    adiab_enthalpy    = getSingleState( "adiabaticenthalpy", iv ); 

    enthalpy          = inStream.d_enthalpy; 

    if ( inStream.d_initEnthalpy || ((abs(adiab_enthalpy - enthalpy)/abs(adiab_enthalpy) < 1.0e-4 ) && f < 1.0e-4) ) {

      if ( bc_type == "pressure" )
        current_heat_loss = d_hl_pressure; 
      else if ( bc_type == "outlet" )
        current_heat_loss = d_hl_outlet; 
      else if ( bc_type == "scalar_init" )
        current_heat_loss = d_hl_scalar_init; 
      else
        current_heat_loss = 0.0; 

      init_enthalpy = adiab_enthalpy - current_heat_loss * sensible_enthalpy; 

    } else {

      throw ProblemSetupException("ERROR! I shouldn't be in this part of the code.", __FILE__, __LINE__); 

    }
  } else {

    // adiabatic case
    init_enthalpy = 0.0;
    current_heat_loss = 0.0; 

  }

  outStream.d_density     = getSingleState( "density", iv ); 
  if (!d_coldflow) { 
    outStream.d_temperature = getSingleState( "temperature", iv ); 
    //outStream.d_cp          = getSingleState( "heat_capacity", iv ); 
    outStream.d_cp          = getSingleState( "specificheat", iv ); 
    outStream.d_h2o         = getSingleState( "H2O", iv); 
    outStream.d_co2         = getSingleState( "CO2", iv);
    outStream.d_heatLoss    = current_heat_loss; 
    if (inStream.d_initEnthalpy) outStream.d_enthalpy = init_enthalpy; 
  }

  cout_tabledbg << " Leaving method TabPropsInterface::OldTableHack " << endl;
}

//--------------------------------------------------------------------------- 
// Get all Dependent variables 
//--------------------------------------------------------------------------- 
/** @details

This method will first check to see if the table is loaded; if it is, it
will return a reference to d_allDepVarNames, which is a private vector<string>
of the TabPropsInterface class
*/
const vector<string> &
TabPropsInterface::getAllDepVars()
{
  if( d_table_isloaded == true ) {
    vector<string>& d_allDepVarNames_ref(d_allDepVarNames);
    return d_allDepVarNames_ref;
  } else {
    ostringstream exception;
    exception << "Error: You requested a list of dependent variables " <<
                 "before specifying the table that you were using. " << endl;
    throw InternalError(exception.str(),__FILE__,__LINE__);
  }
}

//--------------------------------------------------------------------------- 
// Get all independent Variables
//--------------------------------------------------------------------------- 
/** @details
This method will first check to see if the table is loaded; if it is, it
will return a reference to d_allIndepVarNames, which is a private
vector<string> of the TabPropsInterface class
*/
const vector<string> &
TabPropsInterface::getAllIndepVars()
{
  if( d_table_isloaded == true ) {
    vector<string>& d_allIndepVarNames_ref(d_allIndepVarNames);
    return d_allIndepVarNames_ref;
  } 
  else {
    ostringstream exception;
    exception << "Error: You requested a list of independent variables " <<
                 "before specifying the table that you were using. " << endl;
    throw InternalError(exception.str(),__FILE__,__LINE__);
  }
}

//--------------------------------------------------------------------------- 
// schedule Dummy Init
//--------------------------------------------------------------------------- 
void 
TabPropsInterface::sched_dummyInit( const LevelP& level, 
                                    SchedulerP& sched )

{
  string taskname = "TabPropsInterface::dummyInit"; 
  //Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &TabPropsInterface::dummyInit ); 

  // dependent variables
  for ( MixingRxnModel::VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ) {
      tsk->computes( i->second ); 
      tsk->requires( Task::OldDW, i->second, Ghost::None, 0 ); 
  }

  sched->addTask( tsk, level->eachPatch(), d_lab->d_sharedState->allArchesMaterials() ); 
}

//--------------------------------------------------------------------------- 
// Dummy Init
//--------------------------------------------------------------------------- 
void 
TabPropsInterface::dummyInit( const ProcessorGroup* pc, 
                              const PatchSubset* patches, 
                              const MaterialSubset* matls, 
                              DataWarehouse* old_dw, 
                              DataWarehouse* new_dw )
{
  for (int p=0; p < patches->size(); p++){

    //Ghost::GhostType gn = Ghost::None; 
    const Patch* patch = patches->get(p); 
    int archIndex = 0; 
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 


    // dependent variables:
    for ( VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){
    
      cout_tabledbg << " In TabProps::dummyInit, getting " << i->first << " for initializing. " << endl;
      CCVariable<double>* the_var = new CCVariable<double>; 
      new_dw->allocateAndPut( *the_var, i->second, matlIndex, patch ); 
      (*the_var).initialize(0.0);
      constCCVariable<double> old_var; 
      old_dw->get(old_var, i->second, matlIndex, patch, Ghost::None, 0 ); 
      
    }
  }
}

