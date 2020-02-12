/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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
#include <Core/Grid/MaterialManager.h>
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
ConstantProps::ConstantProps( MaterialManagerP& materialManager ) :
  MixingRxnModel( materialManager )
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
ConstantProps::problemSetup( const ProblemSpecP& db )
{
  // Create sub-ProblemSpecP object
  ProblemSpecP db_coldflow = db;

  d_allDepVarNames.push_back("density");
  d_allDepVarNames.push_back("temperature");

  // Need the reference denisty point: (also in PhysicalPropteries object but this was easier than passing it around)
  const ProblemSpecP db_root = db_coldflow->getRootNode();
  db_root->findBlock("PhysicalConstants")->require("reference_point", d_ijk_den_ref);

  db_coldflow->require( "density", _density );
  bool test = insertIntoMap("density");
  if ( !test ){
    throw InvalidValue("Error: Could not insert the following into the table lookup: density",
                       __FILE__,__LINE__);
  }


  //Automatically adding density_old to the table lookup because this
  //is needed for scalars that aren't solved on stage 1:
  ChemHelper& helper = ChemHelper::self();
  helper.add_lookup_species( "density", ChemHelper::OLD );

  proc0cout << "   ------ Using constant density and temperature -------   " << endl;

  problemSetupCommon( db_coldflow, this );


  _includeTemp=false;
  if(db_coldflow->findBlock("temperature")) {
    db_coldflow->require( "temperature", _temperature );
    test = insertIntoMap("temperature");
     if ( !test ){
       throw InvalidValue("Error: Could not insert the following into the table lookup: temperature",
                          __FILE__,__LINE__);
     }
    _includeTemp=true;
  }

}

//---------------------------------------------------------------------------
// schedule get State
//---------------------------------------------------------------------------
  void
ConstantProps::sched_getState( const LevelP& level,
    SchedulerP& sched,
    const int time_substep,
    const bool initialize_me,
    const bool modify_ref_den )
{
  string taskname = "ConstantProps::getState";
  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &ConstantProps::getState, time_substep, initialize_me, modify_ref_den );

  if ( initialize_me ) {

    for ( MixingRxnModel::VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ) {
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

  // other variables
  tsk->modifies( m_densityLabel );

  tsk->requires( Task::NewDW, m_volFractionLabel, gn, 0 );

  sched->addTask( tsk, level->eachPatch(), m_materialManager->allMaterials( "Arches" ) );
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
    const int time_substep,
    const bool initialize_me,
    const bool modify_ref_den )
{
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType gn = Ghost::None;
    const Patch* patch = patches->get(p);

    constCCVariable<double> eps_vol;
    new_dw->get( eps_vol, m_volFractionLabel, m_matl_index, patch, gn, 0 );

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
      else if ( i->first == "temperature" && _includeTemp )
        iter_to_index.insert( make_pair( i->first, 1 ));

    }


    CCVariable<double> arches_density;
    new_dw->getModifiable( arches_density, m_densityLabel, m_matl_index, patch );


    for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){
      if ( i->first == "density" ){
        (*i->second.var).initialize(_density);
        arches_density.copyData( (*i->second.var) );
      } else if ( i->first == "temperature" && _includeTemp){
        (*i->second.var).initialize(_temperature);
      }
    }

    // Go through the patch and populate the requested state variables
    for (CellIterator iter=patch->getExtraCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;

      // retrieve all depenedent variables from table
      for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){

        //std::map< std::string, int>::iterator i_to_i = iter_to_index.find( i->first ); //unused 

        if ( i->first == "density" ){
          (*i->second.var)[c] *= eps_vol[c];
          arches_density[c] = (*i->second.var)[c];
        } else if ( i->first == "temperature" && _includeTemp){
          (*i->second.var)[c] = _temperature * eps_vol[c];
        }

      }
    }

    for ( DepVarMap::iterator i = depend_storage.begin(); i != depend_storage.end(); ++i ){
      delete i->second.var;
    }

    // reference density modification
    if ( modify_ref_den ) {

      //actually modify the reference density value:
      if ( time_substep == 0 ){
        CCVariable<double> den_ref_array;
        new_dw->allocateAndPut(den_ref_array, m_denRefArrayLabel, m_matl_index, patch );

        for (CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++ ){
          IntVector c = *iter;
          den_ref_array[c] = _density;
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

double ConstantProps::getTableValue( std::vector<double> iv, std::string variable )
{

  if ( variable == "density" ){

    return _density;

  } else if ( variable == "temperature" && _includeTemp) {

    return _temperature;

  } else {

    // a bit dangerous?
    //
    double value = 0;
    return value;

  }
}
