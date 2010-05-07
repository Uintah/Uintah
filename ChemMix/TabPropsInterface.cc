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
#include <CCA/Components/Arches/ChemMix/TabProps/StateTable.h>
#include <CCA/Components/Arches/ChemMix/TabPropsInterface.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>

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
using namespace SCIRun;



/**************************************************************************
 TabPropsInterface.cc

INPUT FILE TAGS
This code checks for the following tags/attributes in the input file:
    <Properties>
        <TabProps>
            <table_file_name>THIS FIELD IS REQUIRED</table_file_name>
            <strict_mode>true/false (not required)</strict_mode>
            <diagnostic_mode>true/false (not required)</diagnostic_mode>
        </TabProps>
    </Properties>

    <DataArchiver>
        <save name="BlahBlahBlah" table_lookup="true/false">
    </DataArchiver>

This is used to construct the vector of user-requested dependent variables.

WARNINGS
The code will throw exceptions for the following reasons:
- no <table_file_name> specified in the intput file
- the getState method is run without the problemSetup method being run (b/c the problemSetup sets 
  the boolean d_table_isloaded to true when a table is loaded, and you can't run getState without 
  first loading a table)
- if bool strictMode is true, and a dependent variable specified in the input file does not
  match the names of any of the dependent variables in the table
- the getDepVars or getIndepVars methods are run on a TabPropsInterface object which hasn't loaded
  a table yet (i.e. hasn't run problemSetup method yet) 

***************************************************************************/



//****************************************************************************
// Default constructor for TabPropsInterface
//****************************************************************************
TabPropsInterface::TabPropsInterface( const ArchesLabel* labels ) :
MixingRxnModel( labels )
{
}

//****************************************************************************
// Destructor
//****************************************************************************
TabPropsInterface::~TabPropsInterface()
{
}

//****************************************************************************
// TabPropsInterface problemSetup
//
// Obtain parameters from the input file
// Construct lists of independent and dependent variables (from user and from table)
// Verify that these variables match
// 
//****************************************************************************
void
TabPropsInterface::problemSetup( const ProblemSpecP& propertiesParameters )
{
  // Create sub-ProblemSpecP object
  string tableFileName;
  ProblemSpecP db_tabprops = propertiesParameters->findBlock("TabProps");
  
  // Obtain object parameters
  db_tabprops->require( "inputfile", tableFileName );
  db_tabprops->getWithDefault( "strict_mode", d_strict_mode, false );
  db_tabprops->getWithDefault( "diagnostic_mode", d_diagnostic_mode, false );

  db_tabprops->getWithDefault( "hl_pressure", d_hl_pressure, 0.0); 
  db_tabprops->getWithDefault( "hl_outlet",   d_hl_outlet,   0.0); 
  db_tabprops->getWithDefault( "hl_scalar_init", d_hl_scalar_init, 0.0); 

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

  setMixDVMap( db_rootnode ); 

  // Extract independent and dependent variables from the table
  d_allIndepVarNames = d_statetbl.get_indepvar_names();
  d_allDepVarNames   = d_statetbl.get_depvar_names();

  for ( unsigned int i = 0; i < d_allIndepVarNames.size(); ++i ){

    //put the right labels in the label map
    string varName = d_allIndepVarNames[i];  

    if (varName == "mixture_fraction") {

      d_ivVarMap.insert(make_pair(varName, d_lab->d_scalarSPLabel)).first;

    } else if (varName == "coal_gas_mix_frac"){

      d_ivVarMap.insert(make_pair(varName, d_lab->d_scalarSPLabel)).first;

    } else if (varName == "mixture_fraction_2") {

      //second mixture fraction coming from the extra transport mechanism 
      EqnFactory& eqn_factory = EqnFactory::self();
      EqnBase& eqn = eqn_factory.retrieve_scalar_eqn( varName );
      d_ivVarMap.insert(make_pair(varName, eqn.getTransportEqnLabel())).first; 


    } else if (varName == "mixture_fraction_variance") {

      d_ivVarMap.insert(make_pair(varName, d_lab->d_scalarVarSPLabel)).first;

    } else if (varName == "mixture_fraction_variance_2") {

      //second mixture fraction variance coming from the extra transport mechanism 
      EqnFactory& eqn_factory = EqnFactory::self();
      EqnBase& eqn = eqn_factory.retrieve_scalar_eqn( varName );
      d_ivVarMap.insert(make_pair(varName, eqn.getTransportEqnLabel())).first; 

    } else if (varName == "heat_loss") {

      d_ivVarMap.insert(make_pair(varName, d_lab->d_heatLossLabel)).first; 

    } else {
      proc0cout << "For table indep. variable: " <<  varName << endl;
      throw ProblemSetupException("ERROR! I currently don't know how to deal with this variable (ie, I can't map it to a transported variable.)", __FILE__, __LINE__); 
    }
  }

  // Confirm that table has been loaded into memory
  d_table_isloaded = true;

  // Verify table -- probably do this in MixingRxnTable.cc
  // - diagnostic mode - checks variables requested saved in the input file to variables found in the table
  // - strict mode - if a dependent variable is requested saved in the input file, but NOT found in the table, throw error
  //verifyTable( d_diagnostic_mode, d_strict_mode );
}



void const 
TabPropsInterface::verifyTable(  bool diagnosticMode,
                             bool strictMode )
{
  //verifyIV(diagnosticMode, strictMode);
  //verifyDV(diagnosticMode, strictMode);
}


void const
TabPropsInterface::verifyDV( bool diagnosticMode, bool strictMode )
{
// To be done later.
}

void const 
TabPropsInterface::verifyIV( bool diagnosticMode, bool strictMode ) 
{
// To be done later.
}

// --------------------------------------------------------------------------------
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

    // other dependent vars:
    tsk->computes( d_lab->d_drhodfCPLabel ); // I don't think this is used anywhere...maybe in coldflow? 
    tsk->computes( d_lab->d_tempINLabel ); // lame ... fix me
    tsk->computes( d_lab->d_cpINLabel ); 
    tsk->computes( d_lab->d_co2INLabel ); 
    tsk->computes( d_lab->d_h2oINLabel ); 
    tsk->computes( d_lab->d_sootFVINLabel ); 

  } else {

    for ( MixingRxnModel::VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ) {
      tsk->modifies( i->second ); 
    }

    // other dependent vars:
    tsk->modifies( d_lab->d_drhodfCPLabel ); // I don't think this is used anywhere...maybe in coldflow? 
    tsk->modifies( d_lab->d_tempINLabel );     // lame .... fix me
    tsk->modifies( d_lab->d_cpINLabel ); 
    tsk->modifies( d_lab->d_co2INLabel ); 
    tsk->modifies( d_lab->d_h2oINLabel ); 
    tsk->modifies( d_lab->d_sootFVINLabel ); 

  }

  // other variables 
  tsk->modifies( d_lab->d_densityCPLabel );  // lame .... fix me
  if ( modify_ref_den )
    tsk->computes(time_labels->ref_density); 
   

  sched->addTask( tsk, level->eachPatch(), d_lab->d_sharedState->allArchesMaterials() ); 
}

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

    CCMap depend_storage; 
    if ( initialize_me ) {
    
      for ( VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){
       
        CCVariable<double>* the_var = new CCVariable<double>; 
        new_dw->allocateAndPut( *the_var, i->second, matlIndex, patch ); 

        depend_storage.insert( make_pair( i->first, the_var )).first; 
        
      }

      // others: 
      CCVariable<double> drho_df; 

      new_dw->allocateAndPut( drho_df, d_lab->d_drhodfCPLabel, matlIndex, patch ); 
      new_dw->allocateAndPut( arches_temperature, d_lab->d_tempINLabel, matlIndex, patch ); 
      new_dw->allocateAndPut( arches_cp, d_lab->d_cpINLabel, matlIndex, patch ); 
      new_dw->allocateAndPut( arches_co2, d_lab->d_co2INLabel, matlIndex, patch ); 
      new_dw->allocateAndPut( arches_h2o, d_lab->d_h2oINLabel, matlIndex, patch ); 
      new_dw->allocateAndPut( arches_soot, d_lab->d_sootFVINLabel, matlIndex, patch ); 

      drho_df.initialize(0.0);  // this variable might not be actually used anywhere any may just be polution  
      arches_temperature.initialize(0.0); 
      arches_cp.initialize(0.0); 
      arches_co2.initialize(0.0); 
      arches_h2o.initialize(0.0);
      arches_soot.initialize(0.0); 

    } else { 

      for ( VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){
       
        CCVariable<double>* the_var = new CCVariable<double>; 
        new_dw->getModifiable( *the_var, i->second, matlIndex, patch ); 

        depend_storage.insert( make_pair( i->first, the_var )).first; 
        
      }

      // others:
      CCVariable<double> drho_dw; 
      new_dw->getModifiable( drho_dw, d_lab->d_drhodfCPLabel, matlIndex, patch ); 
      new_dw->getModifiable( arches_temperature, d_lab->d_tempINLabel, matlIndex, patch ); 
      new_dw->getModifiable( arches_cp, d_lab->d_cpINLabel, matlIndex, patch ); 
      new_dw->getModifiable( arches_co2, d_lab->d_co2INLabel, matlIndex, patch ); 
      new_dw->getModifiable( arches_h2o, d_lab->d_h2oINLabel, matlIndex, patch ); 
      new_dw->getModifiable( arches_soot, d_lab->d_sootFVINLabel, matlIndex, patch ); 
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
        } else if (i->first == "temperature") {
          arches_temperature[c] = table_value; 
        } else if (i->first == "heat_capacity") {
          arches_cp[c] = table_value; 
        } else if (i->first == "CO2") {
          arches_co2[c] = table_value; 
        } else if (i->first == "H2O") {
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

void 
TabPropsInterface::sched_computeHeatLoss( const LevelP& level, SchedulerP& sched, const bool initialize_me )
{
  string taskname = "TabPropsInterface::computeHeatLoss"; 
  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &TabPropsInterface::computeHeatLoss, initialize_me );

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

  tsk->requires( Task::NewDW, d_lab->d_enthalpySPLabel, gn, 0 ); 

  sched->addTask( tsk, level->eachPatch(), d_lab->d_sharedState->allArchesMaterials() ); 
}

void 
TabPropsInterface::computeHeatLoss( const ProcessorGroup* pc, 
                                    const PatchSubset* patches, 
                                    const MaterialSubset* matls, 
                                    DataWarehouse* old_dw, 
                                    DataWarehouse* new_dw, 
                                    const bool initialize_me )
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
    new_dw->get(enthalpy, d_lab->d_enthalpySPLabel, matlIndex, patch, gn, 0 ); 

    std::vector<constCCVariable<double> > the_variables; 
    const std::vector<string>& iv_names = getAllIndepVars();

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
      double sensible_enthalpy  = getSingleState( "sensible_heat", iv ); 
      double adiabatic_enthalpy = getSingleState( "adiabatic_enthalpy", iv );  
      double small = 1e-10; 
      double current_heat_loss = ( adiabatic_enthalpy - enthalpy[c] ) / ( sensible_enthalpy + small ); 

      if ( current_heat_loss < -1.0 )
        current_heat_loss = -1.0; 
      else if ( current_heat_loss > 1.0 ) 
        current_heat_loss = 1.0; 

      heat_loss[c] = current_heat_loss; 

    }
  }
}

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
void 
TabPropsInterface::computeFirstEnthalpy( const ProcessorGroup* pc, 
                                         const PatchSubset* patches, 
                                         const MaterialSubset* matls, 
                                         DataWarehouse* old_dw, 
                                         DataWarehouse* new_dw ) 
{

  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType gn = Ghost::None; 
    const Patch* patch = patches->get(p); 
    int archIndex = 0; 
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> enthalpy; 
    new_dw->getModifiable( enthalpy, d_lab->d_enthalpySPLabel, matlIndex, patch ); 
      
    std::vector<constCCVariable<double> > the_variables; 
    for ( VarMap::iterator i = d_ivVarMap.begin(); i != d_ivVarMap.end(); i++ ){

      if ( i->first != "heat_loss" ){ // heat loss hasn't been computed yet so this is why we have an "if" here. 
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
          iv.push_back( (*i)[c] );
        else 
          iv.push_back( d_hl_scalar_init ); 

        index++; 
      }

      double current_heat_loss = d_hl_scalar_init; // may want to make this more sophisticated later(?)
      double sensible_enthalpy = getSingleState( "sensible_heat", iv ); 
      double adiab_enthalpy    = getSingleState( "adiabatic_enthalpy", iv ); 
      enthalpy[c]     = adiab_enthalpy - current_heat_loss * sensible_enthalpy; 

    }
  }
}

//****************************************************************************
// TabPropsInterface getState 
//
// Call the StateTable::query method for each cell on a given patch
// Called from Properties::computeProps
//
//****************************************************************************
/*
void
TabPropsInterface::getState( MixingRxnModel::ConstVarMap ivVar, MixingRxnModel::VarMap dvVar, const Patch* patch )
{
  if( d_table_isloaded == false ) {
    throw InternalError("ERROR:Arches:TabPropsInterface - You requested a thermodynamic state, but no "
                        "table has been loaded. You must specify a table filename in your input file.",__FILE__,__LINE__);
  }
  
  for (CellIterator iCell = patch->getCellIterator(); !iCell.done(); ++iCell) {
    IntVector currCell = *iCell;
    
    // loop over all independent variables to extract IV values at currCell
    for (unsigned int i=0; i<ivVar.size(); ++i) { 
      MixingRxnModel::ConstVarMap::iterator iVar = ivVar.find(i);
      d_indepVarValues[i] = (*iVar->second)[currCell];
    }
 
    // loop over all dependent variables to query table and record values
    for (unsigned int i=0; i<d_allDepVarNames.size(); ++i) {
      MixingRxnModel::VarMap::iterator iVar = dvVar.find(i);
      (*iVar->second)[currCell] = d_statetbl.query(d_allDepVarNames[i], &d_indepVarValues[0]);
    }

  }
}
*/

void 
TabPropsInterface::oldTableHack( const InletStream& inStream, Stream& outStream, bool calcEnthalpy, const string bc_type )
{

  //This is a temporary hack to get the table stuff working with the new interface
  const std::vector<string>& iv_names = getAllIndepVars();
  std::vector<double> iv(iv_names.size());

  // notes: 
  // mixture fraction 2 is being set to zero always!...fix it!

  for ( int i = 0; i < (int) iv_names.size(); i++){

    if ( (iv_names[i] == "mixture_fraction") || (iv_names[i] == "coal_gas_mix_frac")){
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

    double enthalpy          = 0.0; 
    double sensible_enthalpy = 0.0; 

    sensible_enthalpy = getSingleState( "sensible_heat", iv ); 
    adiab_enthalpy    = getSingleState( "adiabatic_enthalpy", iv ); 

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

    outStream.d_temperature = getSingleState( "temperature", iv ); 
    outStream.d_density     = getSingleState( "density", iv ); 
    outStream.d_cp          = getSingleState( "heat_capacity", iv ); 
    outStream.d_h2o         = getSingleState( "H2O", iv); 
    outStream.d_co2         = getSingleState( "CO2", iv);
    outStream.d_heatLoss    = current_heat_loss; 
    if (inStream.d_initEnthalpy) outStream.d_enthalpy = init_enthalpy; 

  }
}


//****************************************************************************
// TabPropsInterface getDepVars
//
// This method will first check to see if the table is loaded; if it is, it
// will return a reference to d_allDepVarNames, which is a private vector<string>
// of the TabPropsInterface class
//
// (If it is a reference to a private class variable, can the reciever of
//  the reference vector<string> still access it?)
//
//****************************************************************************
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



//****************************************************************************
// TabPropsInterface getIndepVars
//
// This method will first check to see if the table is loaded; if it is, it
// will return a reference to d_allIndepVarNames, which is a private
// vector<string> of the TabPropsInterface class
// 
// (If it is a reference to a private class variable, can the reciever of
//  the reference vector<string> still access it?)
// 
//****************************************************************************
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

