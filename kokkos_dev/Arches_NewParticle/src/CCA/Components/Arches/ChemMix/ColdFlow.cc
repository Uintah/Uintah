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


//----- ColdFlow.cc --------------------------------------------------

// includes for Arches
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <CCA/Components/Arches/ChemMix/ColdFlow.h>
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

using namespace std;
using namespace Uintah;

//--------------------------------------------------------------------------- 
// Default Constructor 
//--------------------------------------------------------------------------- 
ColdFlow::ColdFlow( const ArchesLabel* labels, const MPMArchesLabel* MAlabels ) :
  MixingRxnModel( labels, MAlabels )
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
ColdFlow::problemSetup( const ProblemSpecP& propertiesParameters )
{
  // Create sub-ProblemSpecP object
  string tableFileName;
  ProblemSpecP db_coldflow = propertiesParameters->findBlock("ColdFlow");
  ProblemSpecP db_properties_root = propertiesParameters; 

  // Need the reference denisty point: (also in PhysicalPropteries object but this was easier than passing it around)
  const ProblemSpecP db_root = db_coldflow->getRootNode(); 
  db_root->findBlock("PhysicalConstants")->require("reference_point", d_ijk_den_ref);  

  // d_stream[ kind index (density, temperature) ][ stream index ( 0,1) ]
  db_coldflow->findBlock( "Stream_1" )->require( "density", d_stream[0][0] ); 
  db_coldflow->findBlock( "Stream_1" )->require( "temperature", d_stream[1][0] ); 
  db_coldflow->findBlock( "Stream_2" )->require( "density", d_stream[0][1] ); 
  db_coldflow->findBlock( "Stream_2" )->require( "temperature", d_stream[1][1] ); 
  db_coldflow->require( "mixture_fraction_label", d_cold_flow_mixfrac ); 

  // Extract independent and dependent variables from input file
  ProblemSpecP db_rootnode = propertiesParameters;
  db_rootnode = db_rootnode->getRootNode();

  proc0cout << endl;
  proc0cout << "--- Cold Flow information --- " << endl;
  proc0cout << endl;

  // This sets the table lookup variables and saves them in a map
  // Map<string name, Label>
  insertIntoMap( "density" ); 
  insertIntoMap( "temperature" ); 

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
  d_ivVarMap.insert(make_pair(varName, eqn.getTransportEqnLabel())).first; 

  proc0cout << "  Matching successful!" << endl;
  proc0cout << endl;

  proc0cout << "--- End Cold Flow information --- " << endl;
  proc0cout << endl;
}

//--------------------------------------------------------------------------- 
// schedule get State
//--------------------------------------------------------------------------- 
void 
ColdFlow::sched_getState( const LevelP& level, 
                          SchedulerP& sched, 
                          const TimeIntegratorLabel* time_labels, 
                          const bool initialize_me,
                          const bool with_energy_exch, 
                          const bool modify_ref_den )
{
  string taskname = "ColdFlow::getState"; 
  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &ColdFlow::getState, time_labels, initialize_me, with_energy_exch, modify_ref_den );

  // independent variables :: these must have been computed previously 
  for ( MixingRxnModel::VarMap::iterator i = d_ivVarMap.begin(); i != d_ivVarMap.end(); ++i ) {

    tsk->requires( Task::NewDW, i->second, gn, 0 ); 

  }

  if ( initialize_me ) {

    for ( MixingRxnModel::VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ) {
      tsk->computes( i->second ); 
    }

    tsk->computes( d_lab->d_drhodfCPLabel ); // I don't think this is used anywhere...maybe in coldflow? 

    if (d_MAlab)
      tsk->computes( d_lab->d_densityMicroLabel ); 

  } else {

    for ( MixingRxnModel::VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ) {
      tsk->modifies( i->second ); 
    }

    tsk->modifies( d_lab->d_drhodfCPLabel ); // I don't think this is used anywhere...maybe in coldflow? 

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
ColdFlow::getState( const ProcessorGroup* pc, 
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

    // dependent variables
    CCVariable<double> mpmarches_denmicro; 

    for ( int i = 0; i < (int) d_allIndepVarNames.size(); i++ ){

      VarMap::iterator ivar = d_ivVarMap.find( d_allIndepVarNames[i] ); 

      constCCVariable<double> the_var; 
      new_dw->get( the_var, ivar->second, matlIndex, patch, gn, 0 );
      indep_storage.push_back( the_var ); 

    }

    DepVarMap depend_storage; 
    if ( initialize_me ) {

      for ( VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){

        DepVarCont storage;

        storage.var = new CCVariable<double>; 
        new_dw->allocateAndPut( *storage.var, i->second, matlIndex, patch ); 
        (*storage.var).initialize(0.0);

        depend_storage.insert( make_pair( i->first, storage )).first; 

      }

      // others: 
      CCVariable<double> drho_df; 

      new_dw->allocateAndPut( drho_df, d_lab->d_drhodfCPLabel, matlIndex, patch ); 

      if (d_MAlab) {
        new_dw->allocateAndPut( mpmarches_denmicro, d_lab->d_densityMicroLabel, matlIndex, patch ); 
        mpmarches_denmicro.initialize(0.0);
      }

      drho_df.initialize(0.0);  // this variable might not be actually used anywhere and may just be polution  

    } else { 

      for ( VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){

        DepVarCont storage;

        storage.var = new CCVariable<double>; 
        new_dw->getModifiable( *storage.var, i->second, matlIndex, patch ); 

        depend_storage.insert( make_pair( i->first, storage )).first; 

      }

      // others:
      CCVariable<double> drho_dw; 
      new_dw->getModifiable( drho_dw, d_lab->d_drhodfCPLabel, matlIndex, patch ); 

      if (d_MAlab) 
        new_dw->getModifiable( mpmarches_denmicro, d_lab->d_densityMicroLabel, matlIndex, patch ); 
    }

    std::map< std::string, int> iter_to_index;  
    for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){

      // this just maps the iterator to an index so that density and temperature can be 
      // easily identified: 
      if ( i->first == "density" ) {
        iter_to_index.insert( make_pair( i->first, 0 )).first;
      } else if ( i->first == "temperature" ) {
        iter_to_index.insert( make_pair( i->first, 1 )).first;
      }
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

      // retrieve all depenedent variables from table
      for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){

        std::map< std::string, int>::iterator i_to_i = iter_to_index.find( i->first ); 
        double table_value = coldFlowMixing( iv, i_to_i->second ); 

        (*i->second.var)[c] = table_value;

        if (i->first == "density") {
          arches_density[c] = table_value; 
          if (d_MAlab) {
            mpmarches_denmicro[c] = table_value; 
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

      int numChildren = patch->getBCDataArray(face)->getNumberChildren(matlIndex);
      for (int child = 0; child < numChildren; child++){

        std::vector<double> iv; 
        Iterator nu;
        Iterator bound_ptr; 

        std::vector<ColdFlow::BoundaryType> which_bc;
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
            which_bc.push_back(ColdFlow::DIRICHLET); 
          } else if (bc_kind == "Neumann" ) { 
            which_bc.push_back(ColdFlow::NEUMANN); 
          } else {
            throw InvalidValue( "Error: BC type not supported for property calculation", __FILE__, __LINE__ ); 
          }

          // currently assuming a constant value across the mesh. 
          bc_values.push_back( bc_value ); 

          bc_values.push_back(0.0);
          which_bc.push_back(ColdFlow::DIRICHLET);

          delete bc; 

        }

        // now use the last bound_ptr to loop over all boundary cells: 
        for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++){

          IntVector c   =   *bound_ptr; 
          IntVector cp1 = ( *bound_ptr - insideCellDir ); 

          // again loop over iv's and fill iv vector
          for ( int i = 0; i < (int) d_allIndepVarNames.size(); i++ ){

            switch (which_bc[i]) { 
              case ColdFlow::DIRICHLET:
                iv.push_back( bc_values[i] ); 
                break; 
              case ColdFlow::NEUMANN:
                iv.push_back(0.5*(indep_storage[i][c] + indep_storage[i][cp1]));  
                break; 
              default: 
                throw InvalidValue( "Error: BC type not supported for property calculation", __FILE__, __LINE__ ); 
            }
          }

          // now get state for boundary cell: 
          for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){

            std::map< std::string, int>::iterator i_to_i = iter_to_index.find( i->first ); 
            double table_value = coldFlowMixing( iv, i_to_i->second ); 

            (*i->second.var)[c] = table_value;

            if (i->first == "density") {
              //double ghost_value = 2.0*table_value - arches_density[cp1];
              arches_density[c] = table_value;
              //arches_density[c] = ghost_value; 
              if (d_MAlab) {
                mpmarches_denmicro[c] = table_value; 
              }
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
ColdFlow::sched_computeHeatLoss( const LevelP& level, SchedulerP& sched, const bool initialize_me, const bool calcEnthalpy )
{
  // nothing to do here. 
}

//--------------------------------------------------------------------------- 
// schedule Compute First Enthalpy
//--------------------------------------------------------------------------- 
  void 
ColdFlow::sched_computeFirstEnthalpy( const LevelP& level, SchedulerP& sched )
{
  // nothing to do here. 
}

//--------------------------------------------------------------------------- 
// schedule Dummy Init
//--------------------------------------------------------------------------- 
  void 
ColdFlow::sched_dummyInit( const LevelP& level, 
    SchedulerP& sched )

{
  string taskname = "ColdFlow::dummyInit"; 
  //Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &ColdFlow::dummyInit ); 

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
ColdFlow::dummyInit( const ProcessorGroup* pc, 
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

      cout_tabledbg << " In ColdFlow::dummyInit, getting " << i->first << " for initializing. " << endl;
      CCVariable<double>* the_var = new CCVariable<double>; 
      new_dw->allocateAndPut( *the_var, i->second, matlIndex, patch ); 
      (*the_var).initialize(0.0);
      constCCVariable<double> old_var; 
      old_dw->get(old_var, i->second, matlIndex, patch, Ghost::None, 0 ); 

      the_var->copyData( old_var ); 

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
