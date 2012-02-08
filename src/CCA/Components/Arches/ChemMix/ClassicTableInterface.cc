/*

   The MIT License

   Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


//----- ClassicTableInterface.cc --------------------------------------------------

// includes for Arches
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <CCA/Components/Arches/ChemMix/ClassicTableInterface.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelFactory.h>

// includes for Uintah
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
ClassicTableInterface::ClassicTableInterface( const ArchesLabel* labels, const MPMArchesLabel* MAlabels ) :
  MixingRxnModel( labels, MAlabels )
{
  _boundary_condition = scinew BoundaryCondition_new( labels ); 
  _use_mf_for_hl = false;
 
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
  ProblemSpecP db_properties_root = propertiesParameters; 

  // Obtain object parameters
  db_classic->require( "inputfile", tableFileName );
  db_classic->getWithDefault( "hl_scalar_init", d_hl_scalar_init, 0.0); 
  db_classic->getWithDefault( "cold_flow", d_coldflow, false); 
  db_properties_root->getWithDefault( "use_mixing_model", d_use_mixing_model, false ); 
  db_classic->getWithDefault( "enthalpy_label", d_enthalpy_name, "enthalpySP" ); 
 
  // Developer only for now. 
  if ( db_classic->findBlock("mf_for_hl") ){ 
    _use_mf_for_hl =  true; 
  } 

  d_noisy_hl_warning = false; 
  if ( ProblemSpecP temp = db_classic->findBlock("noisy_hl_warning") ) 
    d_noisy_hl_warning = true; 

  // only solve for heat loss if a working radiation model is found
  const ProblemSpecP params_root = db_classic->getRootNode();
  ProblemSpecP db_enthalpy  =  params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ExplicitSolver")->findBlock("EnthalpySolver");
  ProblemSpecP db_sources   =  params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources"); 
  d_allocate_soot = false; 
  if ( db_sources ) { 
    for (ProblemSpecP src_db = db_sources->findBlock("src");
        src_db !=0; src_db = src_db->findNextBlock("src")){
      std::string type="null";
      src_db->getAttribute("type",type); 
      if ( type == "do_radiation" ){ 
        d_allocate_soot = true; 
      } 
    }
  } 
  if (db_enthalpy) { 
    ProblemSpecP db_radiation = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ExplicitSolver")->findBlock("EnthalpySolver")->findBlock("DORadiationModel");
    d_adiabatic = true; 
    if (db_radiation) { 
      proc0cout << "Found a working radiation model -- will implement case with heat loss" << endl;
      d_adiabatic = false; 
      d_allocate_soot = false; // needed for new DORadiation source term 
    }
  } else {
    d_adiabatic = true; 
  }

  // Developer's switch
  if ( db_classic->findBlock("NOT_ADIABATIC") ) { 
    d_adiabatic = false; 
  } 

  // need the reference denisty point: (also in PhysicalPropteries object but this was easier than passing it around)
  const ProblemSpecP db_root = db_classic->getRootNode(); 
  db_root->findBlock("PhysicalConstants")->require("reference_point", d_ijk_den_ref);  

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

  for ( unsigned int i = 0; i < d_allIndepVarNames.size(); ++i ){

    //put the right labels in the label map
    string varName = d_allIndepVarNames[i];  

    // !! need to add support for variance !!
    if (varName == "heat_loss" || varName == "HeatLoss") {

      cout_tabledbg << " Heat loss being inserted into the indep. var map. " << endl;

      d_ivVarMap.insert(make_pair(varName, d_lab->d_heatLossLabel)).first; 

    } else if ( varName == "scalar_variance" || varName == "MixtureFractionVariance" ) {

      cout_tabledbg << " Scalar variance being inserted into the indep. var map. " << endl;

      d_ivVarMap.insert(make_pair(varName, d_lab->d_normalizedScalarVarLabel)).first; 

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

  problemSetupCommon( db_classic ); 

  proc0cout << "--- End Classic Arches table information --- " << endl;
  proc0cout << endl;

  // Match the requested dependent variables with their table index:
  getIndexInfo(); 
  if (!d_coldflow) 
    getEnthalpyIndexInfo();

  d_enthalpy_label = VarLabel::find( d_enthalpy_name ); 

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
  if ( modify_ref_den )
    tsk->computes(time_labels->ref_density); 
  tsk->requires( Task::NewDW, d_lab->d_volFractionLabel, gn, 0 ); 
  tsk->requires( Task::NewDW, d_lab->d_cellTypeLabel, gn, 0 ); 

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

    constCCVariable<double> eps_vol; 
    constCCVariable<int> cell_type; 
    new_dw->get( eps_vol, d_lab->d_volFractionLabel, matlIndex, patch, gn, 0 ); 
    new_dw->get( cell_type, d_lab->d_cellTypeLabel, matlIndex, patch, gn, 0 ); 

    //independent variables:
    std::vector<constCCVariable<double> > indep_storage; 

    for ( int i = 0; i < (int) d_allIndepVarNames.size(); i++ ){

      VarMap::iterator ivar = d_ivVarMap.find( d_allIndepVarNames[i] ); 

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

      for ( VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){

        DepVarCont storage;

        storage.var = new CCVariable<double>; 
        new_dw->allocateAndPut( *storage.var, i->second, matlIndex, patch ); 
        (*storage.var).initialize(0.0);

        IndexMap::iterator i_index = d_depVarIndexMap.find( i->first ); 
        storage.index = i_index->second; 

        depend_storage.insert( make_pair( i->first, storage )).first; 

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

        depend_storage.insert( make_pair( i->first, storage )).first; 

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
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 

      // fill independent variables
      std::vector<double> iv; 
      for ( std::vector<constCCVariable<double> >::iterator i = indep_storage.begin(); i != indep_storage.end(); ++i ) {
        iv.push_back( (*i)[c] );
      }

      // do a variable transform (if specified)
      _iv_transform->transform( iv ); 

      // retrieve all depenedent variables from table
      for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){

      //    double table_value = tableLookUp( iv, i->second.index ); 
				double table_value = ND_interp->find_val( iv, i->second.index );
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

    // Loop over all boundary faces on this patch
    for (bf_iter = bf.begin(); bf_iter != bf.end(); bf_iter++){

      Patch::FaceType face = *bf_iter; 
      IntVector insideCellDir = patch->faceDirection(face);

      int numChildren = patch->getBCDataArray(face)->getNumberChildren(matlIndex);
      for (int child = 0; child < numChildren; child++){

        std::vector<double> iv; 
        Iterator nu;
        Iterator bound_ptr; 

        std::vector<ClassicTableInterface::BoundaryType> which_bc;
        std::vector<double> bc_values;

        // look to make sure every variable has a BC set:
        for ( int i = 0; i < (int) d_allIndepVarNames.size(); i++ ){
          std::string variable_name = d_allIndepVarNames[i]; 

          const BoundCondBase* bc = patch->getArrayBCValues( face, matlIndex,
              variable_name, bound_ptr,
              nu, child );

          const BoundCond<double> *new_bcs =  dynamic_cast<const BoundCond<double> *>(bc);
          if ( new_bcs == 0 ) {
            cout << "Error: For variable named " << variable_name << endl;
            throw InvalidValue( "Error: When trying to compute properties at a boundary, found boundary specification missing in the <Grid> section of the input file.", __FILE__, __LINE__); 
          }

          double bc_value     = new_bcs->getValue();
          std::string bc_kind = new_bcs->getBCType__NEW(); 

          if ( bc_kind == "Dirichlet" ) {
            which_bc.push_back(ClassicTableInterface::DIRICHLET); 
          } else if (bc_kind == "Neumann" ) { 
            which_bc.push_back(ClassicTableInterface::NEUMANN); 
          } else if (bc_kind == "FromFile") { 
            which_bc.push_back(ClassicTableInterface::FROMFILE);
          } else { 
            throw InvalidValue( "Error: BC type not supported for property calculation", __FILE__, __LINE__ ); 
          }

          // currently assuming a constant value across the boundary
          bc_values.push_back( bc_value ); 

          delete bc; 

        }

        // now use the last bound_ptr to loop over all boundary cells: 
        for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++){

          IntVector c   =   *bound_ptr; 
          IntVector cp1 = ( *bound_ptr - insideCellDir ); 

          // again loop over iv's and fill iv vector
          for ( int i = 0; i < (int) d_allIndepVarNames.size(); i++ ){

            switch (which_bc[i]) { 

              case ClassicTableInterface::DIRICHLET:
                iv.push_back( bc_values[i] );
                break; 

              case ClassicTableInterface::NEUMANN:
                iv.push_back( 0.5 * (indep_storage[i][c] + indep_storage[i][cp1]) );
                break; 

              case ClassicTableInterface::FROMFILE: 
                iv.push_back( 0.5 * (indep_storage[i][c] + indep_storage[i][cp1]) );
                break; 

              default: 
                throw InvalidValue( "Error: BC type not supported for property calculation", __FILE__, __LINE__ ); 

            }
          }

          _iv_transform->transform( iv ); 

          // now get state for boundary cell: 
          for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){

            //  double table_value = tableLookUp( iv, i->second.index ); 
						double table_value = ND_interp->find_val( iv, i->second.index );
            table_value *= eps_vol[c]; 
            //double orig_save = (*i->second.var)[c]; 
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

        const BoundCondBase* bc = patch->getArrayBCValues( face, matlIndex,
            T_name, bound_ptr,
            nu, child );

        const DepVarMap::iterator iter = depend_storage.find(_temperature_label_name); 

        const BoundCond<double> *new_bcs =  dynamic_cast<const BoundCond<double> *>(bc);

        delete bc; 

        if ( new_bcs != 0 ) {

          if ( iter == depend_storage.end() ) { 
            throw InternalError("Error: SolidWallTemperature was specified in the <BoundaryCondition> section yet I could not find a temperature variable (default label=temperature). Consider setting/checking <temperature_label_name> in the input file. " ,__FILE__,__LINE__);
          } 

          // if new_bcs == 0, then it assumes you intelligently set the temperature some other way. 
          double bc_value     = new_bcs->getValue();
          std::string bc_kind = new_bcs->getBCType__NEW(); 

          double dx = 0.0;
          double the_sign = 1.0; 

          if ( bc_kind == "Dirichlet" ) { 

            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++){

              IntVector c   =   *bound_ptr; 
              if ( cell_type[c] == BoundaryCondition_new::WALL ){ 
                (*iter->second.var)[c] = bc_value; 
              }

            }  

          } else if ( bc_kind == "Neumann" ) { 

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
ClassicTableInterface::sched_computeHeatLoss( const LevelP& level, SchedulerP& sched, const bool initialize_me, const bool calcEnthalpy )
{
  string taskname = "ClassicTableInterface::computeHeatLoss"; 
  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &ClassicTableInterface::computeHeatLoss, initialize_me, calcEnthalpy );

  // independent variables
  for (MixingRxnModel::VarMap::iterator i = d_ivVarMap.begin(); i != d_ivVarMap.end(); ++i) {

    const VarLabel* the_label = i->second;
    if (i->first != "heat_loss" && i->first != "HeatLoss" ) 
      tsk->requires( Task::NewDW, the_label, gn, 0 ); 
  }

  // heat loss must be computed if this is the first FE step 
  if (initialize_me)
    tsk->computes( d_lab->d_heatLossLabel );
  else 
    tsk->modifies( d_lab->d_heatLossLabel ); 

  if ( calcEnthalpy )
    tsk->requires( Task::NewDW, d_enthalpy_label, gn, 0 ); 

  sched->addTask( tsk, level->eachPatch(), d_lab->d_sharedState->allArchesMaterials() ); 
}

//--------------------------------------------------------------------------- 
// Compute Heat Loss
//--------------------------------------------------------------------------- 
  void 
ClassicTableInterface::computeHeatLoss( const ProcessorGroup* pc, 
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
    std::string heat_loss_string; 

    CCVariable<double> heat_loss; 
    if ( initialize_me )
      new_dw->allocateAndPut( heat_loss, d_lab->d_heatLossLabel, matlIndex, patch ); 
    else 
      new_dw->getModifiable ( heat_loss, d_lab->d_heatLossLabel, matlIndex, patch ); 
    heat_loss.initialize(0.0); 

    constCCVariable<double> enthalpy; 
    if ( calcEnthalpy ) 
      new_dw->get(enthalpy, d_enthalpy_label, matlIndex, patch, gn, 0 ); 

    std::vector<constCCVariable<double> > the_variables; 

    // exceptions for cold flow or adiabatic cases
    bool compute_heatloss = true; 
    if ( d_coldflow ) 
      compute_heatloss = false; 
    if ( d_adiabatic ) 
      compute_heatloss = false; 

    if ( compute_heatloss ) { 

      for ( int i = 0; i < (int) d_allIndepVarNames.size(); i++ ){

        VarMap::iterator ivar = d_ivVarMap.find( d_allIndepVarNames[i] ); 
        if ( ivar->first != "heat_loss" && ivar->first != "HeatLoss" ){
          constCCVariable<double> test_Var; 
          new_dw->get( test_Var, ivar->second, matlIndex, patch, gn, 0 );  

          the_variables.push_back( test_Var ); 
        } else {
          heat_loss_string = ivar->first; 
          constCCVariable<double> a_null_var; 
          the_variables.push_back( a_null_var ); // to preserve the total number of IV otherwise you will have problems below
        }
      }

      bool lower_hl_exceeded = false; 
      bool upper_hl_exceeded = false;

      for (CellIterator iter=patch->getCellIterator(0); !iter.done(); iter++){
        IntVector c = *iter; 

        vector<double> iv; 
        int index = 0; 
        for ( std::vector<constCCVariable<double> >::iterator i = the_variables.begin(); i != the_variables.end(); i++){

          if ( d_allIndepVarNames[index] != "heat_loss" && d_allIndepVarNames[index] != "HeatLoss" ) 
            iv.push_back( (*i)[c] );
          else 
            iv.push_back( 0.0 ); 

          index++; 
        }

        _iv_transform->transform( iv ); 

        // actually compute the heat loss: 
        IndexMap::iterator i_index = d_enthalpyVarIndexMap.find( "sensibleenthalpy" ); 
       // double sensible_enthalpy    = tableLookUp( iv, i_index->second ); 
				double sensible_enthalpy    = ND_interp->find_val( iv, i_index->second );
				
				
				
				
        i_index = d_enthalpyVarIndexMap.find( "adiabaticenthalpy" ); 

        double adiabatic_enthalpy = 0.0;
        if ( !_use_mf_for_hl ) { 
  //        adiabatic_enthalpy = tableLookUp( iv, i_index->second ); 
					adiabatic_enthalpy = ND_interp->find_val( iv, i_index->second ); 
					
        } else { 
          // WARNING: Hardcoded index for development testing
          // only works for "coal" tables
          // requires that you have _H_fuel defined in the table
          adiabatic_enthalpy = _H_fuel * iv[2] + _H_ox * (1.0-iv[2]);
        }
        double current_heat_loss  = 0.0;
        double small = 1.0; 
        if ( calcEnthalpy ) {
          
          double numerator = adiabatic_enthalpy - enthalpy[c]; 
          current_heat_loss = ( numerator ) / ( sensible_enthalpy + small ); 

        }

        if ( current_heat_loss < d_hl_lower_bound ) {

          current_heat_loss = d_hl_lower_bound; 
          lower_hl_exceeded = true; 

        } else if ( current_heat_loss > d_hl_upper_bound ) { 

          current_heat_loss = d_hl_upper_bound; 
          upper_hl_exceeded = true; 

        }

        heat_loss[c] = current_heat_loss; 

      }

      if ( d_noisy_hl_warning ) { 
       
        if ( upper_hl_exceeded || lower_hl_exceeded ) {  
          cout << "Patch with bounds: " << patch->getCellLowIndex() << " to " << patch->getCellHighIndex()  << endl;
          if ( lower_hl_exceeded ) 
            cout << "   --> lower heat loss exceeded. " << endl;
          if ( upper_hl_exceeded ) 
            cout << "   --> upper heat loss exceeded. " << endl;
        } 
      } 

      //boundary conditions
      _boundary_condition->setScalarValueBC( pc, patch, heat_loss, heat_loss_string );

    }
  }
}

//--------------------------------------------------------------------------- 
// schedule Compute First Enthalpy
//--------------------------------------------------------------------------- 
  void 
ClassicTableInterface::sched_computeFirstEnthalpy( const LevelP& level, SchedulerP& sched )
{
  string taskname = "ClassicTableInterface::computeFirstEnthalpy"; 
  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &ClassicTableInterface::computeFirstEnthalpy );


  tsk->modifies( d_enthalpy_label ); 

  // independent variables
  for (MixingRxnModel::VarMap::iterator i = d_ivVarMap.begin(); i != d_ivVarMap.end(); ++i) {

    const VarLabel* the_label = i->second;
    if (i->first != "heat_loss" && i->first != "HeatLoss") 
      tsk->requires( Task::NewDW, the_label, gn, 0 ); 
  }

  sched->addTask( tsk, level->eachPatch(), d_lab->d_sharedState->allArchesMaterials() ); 

}

//--------------------------------------------------------------------------- 
// Compute First Enthalpy
//--------------------------------------------------------------------------- 
  void 
ClassicTableInterface::computeFirstEnthalpy( const ProcessorGroup* pc, 
    const PatchSubset* patches, 
    const MaterialSubset* matls, 
    DataWarehouse* old_dw, 
    DataWarehouse* new_dw ) 
{

  for (int p=0; p < patches->size(); p++){

    cout_tabledbg << " In ClassicTableInterface::getFirstEnthalpy " << endl;

    Ghost::GhostType gn = Ghost::None; 
    const Patch* patch = patches->get(p); 
    int archIndex = 0; 
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> enthalpy; 
    new_dw->getModifiable( enthalpy, d_enthalpy_label, matlIndex, patch ); 

    std::vector<constCCVariable<double> > the_variables; 

    for ( vector<string>::iterator i = d_allIndepVarNames.begin(); i != d_allIndepVarNames.end(); i++){

      const VarMap::iterator iv_iter = d_ivVarMap.find( *i ); 

      if ( iv_iter == d_ivVarMap.end() ) {
        cout << " For variable named: " << *i << endl;
        throw InternalError("Error: Could not map this label to the correct Uintah grid variable." ,__FILE__,__LINE__);
      }

      if ( *i != "heat_loss" && *i != "HeatLoss" ) { // heat loss hasn't been computed yet so this is why 
        // we have an "if" here.
        cout_tabledbg << " Found label = " << iv_iter->first << endl;
        constCCVariable<double> test_Var;
        new_dw->get( test_Var, iv_iter->second, matlIndex, patch, gn, 0 ); 

        the_variables.push_back( test_Var ); 

      } else {

        constCCVariable<double> a_null_var;
        the_variables.push_back( a_null_var ); // to preserve the total number of 
        // IV otherwise you will have problems below

      }
    }

    for (CellIterator iter=patch->getCellIterator(0); !iter.done(); iter++){
      IntVector c = *iter; 

      vector<double> iv; 
      int index = 0; 
      for ( std::vector<constCCVariable<double> >::iterator i = the_variables.begin(); i != the_variables.end(); i++){

        if ( d_allIndepVarNames[index] != "heat_loss" && d_allIndepVarNames[index] != "HeatLoss") 
          iv.push_back( (*i)[c] );
        else 
          iv.push_back( d_hl_scalar_init ); 

        index++; 
      }

      _iv_transform->transform( iv ); 

      double current_heat_loss = d_hl_scalar_init; // may want to make this more sophisticated later(?)
      IndexMap::iterator i_index = d_enthalpyVarIndexMap.find( "sensibleenthalpy" ); 
 //     double sensible_enthalpy    = tableLookUp( iv, i_index->second ); 
			double sensible_enthalpy    = ND_interp->find_val( iv, i_index->second );
			
      i_index = d_enthalpyVarIndexMap.find( "adiabaticenthalpy" ); 
      double adiabatic_enthalpy = 0.0; 
      if ( !_use_mf_for_hl ){ 
        //adiabatic_enthalpy = tableLookUp( iv, i_index->second ); 
				adiabatic_enthalpy = ND_interp->find_val( iv, i_index->second );
      } else { 
        //WARNING: Development only
        adiabatic_enthalpy = _H_fuel * iv[2] + _H_ox * (1.0-iv[2]);
      } 

      enthalpy[c]     = adiabatic_enthalpy - current_heat_loss * sensible_enthalpy; 

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

  _iv_transform->transform( iv ); 

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
  string taskname = "ClassicTableInterface::dummyInit"; 
  //Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &ClassicTableInterface::dummyInit ); 

  // dependent variables
  for ( MixingRxnModel::VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ) {
    tsk->computes( i->second ); 
    tsk->requires( Task::OldDW, i->second, Ghost::None, 0 ); 
  }

  if ( d_allocate_soot ) { 
    tsk->computes( d_lab->d_sootFVINLabel ); 
    tsk->requires( Task::OldDW, d_lab->d_sootFVINLabel, Ghost::None, 0 );  
  }

  sched->addTask( tsk, level->eachPatch(), d_lab->d_sharedState->allArchesMaterials() ); 
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

      the_var->copyData( old_var ); 

    }

    if ( d_allocate_soot ) { 
      CCVariable<double> soot; 
      constCCVariable<double> old_soot; 
      new_dw->allocateAndPut( soot, d_lab->d_sootFVINLabel, matlIndex, patch ); 
      old_dw->get( old_soot, d_lab->d_sootFVINLabel, matlIndex, patch, Ghost::None, 0 ); 
      soot.initialize(0.0);
      soot.copyData( old_soot );
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

    IndexMap::iterator iter = d_depVarIndexMap.find( name );   
    // Only insert variable if it isn't already there. 
    if ( iter == d_depVarIndexMap.end() ) {

      cout_tabledbg << " Inserting " << name << " index information into storage." << endl;
      iter = d_depVarIndexMap.insert( make_pair( name, index ) ).first; 

    }
  }
}

//-------------------------------------
  void 
ClassicTableInterface::getEnthalpyIndexInfo()
{
  cout_tabledbg << "ClassicTableInterface::getEnthalpyIndexInfo(): Looking up sensible enthalpy" << endl;
  int index = findIndex( "sensibleenthalpy" ); 

  d_enthalpyVarIndexMap.insert( make_pair( "sensibleenthalpy", index )).first; 

  cout_tabledbg << "ClassicTableInterface::getEnthalpyIndexInfo(): Looking up adiabatic enthalpy" << endl;
  index = findIndex( "adiabaticenthalpy" ); 
  d_enthalpyVarIndexMap.insert( make_pair( "adiabaticenthalpy", index )).first; 

  index = findIndex( "density" ); 
  d_enthalpyVarIndexMap.insert( make_pair( "density", index )).first; 
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
double ClassicTableInterface::getTableValue( std::vector<double> iv, std::string variable )
{
  IndexMap::iterator i_index = d_enthalpyVarIndexMap.find( variable ); 
  //double value    = tableLookUp( iv, i_index->second ); 
  double value = ND_interp->find_val(iv, i_index->second );
	return value; 
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

