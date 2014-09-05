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

//----- ConstantProps.cc --------------------------------------------------

// includes for Arches
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <CCA/Components/Arches/ChemMix/ConstantProps.h>

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
ConstantProps::ConstantProps( ArchesLabel* labels, const MPMArchesLabel* MAlabels ) :
  MixingRxnModel( labels, MAlabels )
{}

//--------------------------------------------------------------------------- 
// Default Destructor
//--------------------------------------------------------------------------- 
ConstantProps::~ConstantProps()
{}

//--------------------------------------------------------------------------- 
// Problem Setup
//--------------------------------------------------------------------------- 
  void
ConstantProps::problemSetup( const ProblemSpecP& propertiesParameters )
{
  // Create sub-ProblemSpecP object
  ProblemSpecP db_coldflow = propertiesParameters->findBlock("ConstantProps");
  ProblemSpecP db_properties_root = propertiesParameters; 

  // Need the reference denisty point: (also in PhysicalPropteries object but this was easier than passing it around)
  const ProblemSpecP db_root = db_coldflow->getRootNode(); 
  db_root->findBlock("PhysicalConstants")->require("reference_point", d_ijk_den_ref);  

  db_coldflow->require( "density", _density ); 
  db_coldflow->require( "temperature", _temperature ); 

  insertIntoMap("density");
  insertIntoMap("temperature"); 

  proc0cout << "   ------ Using constant density and temperature -------   " << endl;

  //setting varlabels to roles: 
  d_lab->setVarlabelToRole( "temperature", "temperature" ); 
  d_lab->setVarlabelToRole( "density", "density" ); 

  problemSetupCommon( db_coldflow ); 

}

//--------------------------------------------------------------------------- 
// schedule get State
//--------------------------------------------------------------------------- 
  void 
ConstantProps::sched_getState( const LevelP& level, 
    SchedulerP& sched, 
    const TimeIntegratorLabel* time_labels, 
    const bool initialize_me,
    const bool with_energy_exch, 
    const bool modify_ref_den )
{
  string taskname = "ConstantProps::getState"; 
  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &ConstantProps::getState, time_labels, initialize_me, with_energy_exch, modify_ref_den );

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
  tsk->modifies( d_lab->d_densityCPLabel );  

  if ( modify_ref_den ){
    tsk->computes(time_labels->ref_density); 
  }

  tsk->requires( Task::NewDW, d_lab->d_volFractionLabel, gn, 0 ); 

  sched->addTask( tsk, level->eachPatch(), d_lab->d_sharedState->allArchesMaterials() ); 
}

//--------------------------------------------------------------------------- 
// get State
//--------------------------------------------------------------------------- 
  void 
ConstantProps::getState( const ProcessorGroup* pc, 
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
    new_dw->get( eps_vol, d_lab->d_volFractionLabel, matlIndex, patch, gn, 0 ); 

    // dependent variables
    CCVariable<double> mpmarches_denmicro; 

    DepVarMap depend_storage; 
    if ( initialize_me ) {

      for ( VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){

        DepVarCont storage;

        storage.var = new CCVariable<double>; 
        new_dw->allocateAndPut( *storage.var, i->second, matlIndex, patch ); 
        (*storage.var).initialize(0.0);

        depend_storage.insert( make_pair( i->first, storage ));

      }

      // others: 
      CCVariable<double> drho_df; 

      new_dw->allocateAndPut( drho_df, d_lab->d_drhodfCPLabel, matlIndex, patch ); 

      if (d_MAlab) {
        new_dw->allocateAndPut( mpmarches_denmicro, d_lab->d_densityMicroLabel, matlIndex, patch ); 
        mpmarches_denmicro.initialize(0.0);
      }

      drho_df.initialize(0.0);  // this variable might not be actually used anywhere and may just be pollution  

    } else { 

      for ( VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ){

        DepVarCont storage;

        storage.var = new CCVariable<double>; 
        new_dw->getModifiable( *storage.var, i->second, matlIndex, patch ); 

        depend_storage.insert( make_pair( i->first, storage ));

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
      if ( i->first == "density" )
        iter_to_index.insert( make_pair( i->first, 0 ));
      else if ( i->first == "temperature" )
        iter_to_index.insert( make_pair( i->first, 1 ));

    }


    CCVariable<double> arches_density; 
    new_dw->getModifiable( arches_density, d_lab->d_densityCPLabel, matlIndex, patch ); 


    for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){
      if ( i->first == "density" ){ 
        (*i->second.var).initialize(_density); 
        arches_density.copyData( (*i->second.var) ); 
      } else if ( i->first == "temperature" ){ 
        (*i->second.var).initialize(_temperature); 
      } 
    }

    // Go through the patch and populate the requested state variables
    for (CellIterator iter=patch->getExtraCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 

      // retrieve all depenedent variables from table
      for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){

        std::map< std::string, int>::iterator i_to_i = iter_to_index.find( i->first ); 

        if ( i->first == "density" ){ 
          (*i->second.var)[c] *= eps_vol[c]; 
          arches_density[c] = (*i->second.var)[c];
        } else if ( i->first == "temperature" ){ 
          (*i->second.var)[c] = _temperature * eps_vol[c]; 
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

      }
      new_dw->put(sum_vartype(den_ref),time_labels->ref_density);
    }
  }
}

void ConstantProps::oldTableHack( const InletStream& inStream, Stream& outStream, bool calcEnthalpy, const string bc_type )
{
}

//--------------------------------------------------------------------------- 
// schedule Dummy Init
//--------------------------------------------------------------------------- 
  void 
ConstantProps::sched_dummyInit( const LevelP& level, 
    SchedulerP& sched )

{
}

//--------------------------------------------------------------------------- 
// Dummy Init
//--------------------------------------------------------------------------- 
  void 
ConstantProps::dummyInit( const ProcessorGroup* pc, 
    const PatchSubset* patches, 
    const MaterialSubset* matls, 
    DataWarehouse* old_dw, 
    DataWarehouse* new_dw )
{
}

double ConstantProps::getTableValue( std::vector<double> iv, std::string variable )
{

  if ( variable == "density" ){ 

    return _density; 

  } else if ( variable == "temperature" ) { 

    return _temperature; 

  } else { 

    // a bit dangerous?
    //
    double value = 0;
    return value; 

  }
}
