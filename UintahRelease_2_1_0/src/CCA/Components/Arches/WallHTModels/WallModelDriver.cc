/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
  is hereby granted, free of charge, to any person obtaining a copy
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

#include <Core/Grid/Task.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/WallHTModels/WallModelDriver.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
#include <Core/Grid/Box.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>

#include <Core/Parallel/Parallel.h>
#include <iostream>
#include <fstream>
#include <iomanip>


using namespace std;
using namespace Uintah;

//_________________________________________
WallModelDriver::WallModelDriver( SimulationStateP& shared_state ) :
_shared_state( shared_state )
{

  const TypeDescription* CC_double = CCVariable<double>::getTypeDescription();

  _matl_index = _shared_state->getArchesMaterial( 0 )->getDWIndex();

  _T_copy_label = VarLabel::create( "T_copy", CC_double );
  _True_T_Label = VarLabel::create( "true_wall_temperature", CC_double);

}

//_________________________________________
WallModelDriver::~WallModelDriver()
{

  std::vector<WallModelDriver::HTModelBase*>::iterator iter;
  for ( iter = _all_ht_models.begin(); iter != _all_ht_models.end(); iter++ ){

    delete *iter;

  }

  VarLabel::destroy( _T_copy_label );
  VarLabel::destroy( _True_T_Label );
  if (do_coal_region){
    VarLabel::destroy( _deposit_thickness_label );
    VarLabel::destroy( _deposit_thickness_sb_s_label );
    VarLabel::destroy( _deposit_thickness_sb_l_label );
    VarLabel::destroy( _emissivity_label );
    VarLabel::destroy( _thermal_cond_en_label );
    VarLabel::destroy( _thermal_cond_sb_s_label );
    VarLabel::destroy( _thermal_cond_sb_l_label );
  }
}

//_________________________________________
void
WallModelDriver::problemSetup( const ProblemSpecP& input_db )
{

  ProblemSpecP db = input_db;

  //db->getWithDefault( "temperature_label", _T_label_name, "radiation_temperature" );
  _T_label_name = "radiation_temperature";

  bool found_radiation_model = false;
  if ( db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources") ){

    ProblemSpecP sources_db = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources");

    for( ProblemSpecP src_db = sources_db->findBlock("src"); src_db != nullptr; src_db = src_db->findNextBlock("src") ) {

      string type;
      src_db->getAttribute("type", type);

      if ( type == "do_radiation"  || type == "rmcrt_radiation" ){
        src_db->getWithDefault("calc_frequency", _calc_freq,3);  //default matches the default of the DOradiation solve
        found_radiation_model = true;
      }

    }

  } else {

    // no sources found
    throw InvalidValue("Error: No validate radiation model found for the wall heat transfer model (no <Sources> found in <TransportEqn>).", __FILE__, __LINE__);

  }

  if ( !found_radiation_model ){
    throw InvalidValue("Error: No validate radiation model found for the wall heat transfer model (no src attribute matched a recognized radiation model).", __FILE__, __LINE__);
  }


  do_coal_region = false;
  
  db->getWithDefault("relaxation_coef", _relax, 1.0);


  for ( ProblemSpecP db_model = db->findBlock( "model" ); db_model != nullptr; db_model = db_model->findNextBlock( "model" ) ){

    std::string type = "not_assigned";

    db_model->getAttribute("type", type);

    if ( type == "simple_ht" ) {

      HTModelBase* simple_ht = scinew SimpleHT();

      simple_ht->problemSetup( db_model );

      _all_ht_models.push_back( simple_ht );

    } else if ( type == "region_ht" ){

      HTModelBase* alstom_ht = scinew RegionHT();

      alstom_ht->problemSetup( db_model );

      _all_ht_models.push_back( alstom_ht );

    } else if ( type == "coal_region_ht" ){

      do_coal_region = true;
      HTModelBase* coal_ht = scinew CoalRegionHT();

      coal_ht->problemSetup( db_model );

      _all_ht_models.push_back( coal_ht );

      _dep_vel_name = get_dep_vel_name( db_model );

    } else {

      throw InvalidValue("Error: Wall Heat Transfer model not recognized.", __FILE__, __LINE__);

    }
  }

  if (do_coal_region){
    const TypeDescription* CC_double = CCVariable<double>::getTypeDescription();
    _deposit_thickness_label = VarLabel::create( "deposit_thickness", CC_double );
    _deposit_thickness_sb_s_label = VarLabel::create( "deposit_thickness_sb_s", CC_double );
    _deposit_thickness_sb_l_label = VarLabel::create( "deposit_thickness_sb_l", CC_double );
    _emissivity_label = VarLabel::create( "emissivity", CC_double );
    _thermal_cond_en_label = VarLabel::create( "thermal_cond_en", CC_double );
    _thermal_cond_sb_s_label = VarLabel::create( "thermal_cond_sb_l", CC_double );
    _thermal_cond_sb_l_label = VarLabel::create( "thermal_cond_sb_s", CC_double );
  }

}

//_________________________________________
void
WallModelDriver::sched_doWallHT( const LevelP& level, SchedulerP& sched, const int time_subset )
{

  Task* task = scinew Task( "WallModelDriver::doWallHT", this,
                           &WallModelDriver::doWallHT, time_subset );

  _T_label        = VarLabel::find( _T_label_name );
  _cellType_label = VarLabel::find( "cellType" );
  _cc_vel_label   = VarLabel::find( "CCVelocity" );

  _HF_E_label     = VarLabel::find( "radiationFluxE" );
  _HF_W_label     = VarLabel::find( "radiationFluxW" );
  _HF_N_label     = VarLabel::find( "radiationFluxN" );
  _HF_S_label     = VarLabel::find( "radiationFluxS" );
  _HF_T_label     = VarLabel::find( "radiationFluxT" );
  _HF_B_label     = VarLabel::find( "radiationFluxB" );

  if (do_coal_region){
    _ave_dep_vel_label = VarLabel::find(_dep_vel_name);
    _d_vol_ave_label = VarLabel::find("d_vol_ave");
  }

  if ( !check_varlabels() ){
    throw InvalidValue("Error: One of the varlabels for the wall model was not found.", __FILE__, __LINE__);
  }

  task->modifies(_T_label);

  if ( time_subset == 0 ) {

    task->computes( _T_copy_label );
    task->computes( _True_T_Label );
    task->requires( Task::OldDW , _cc_vel_label   , Ghost::None , 0 );
    task->requires( Task::OldDW , _T_label        , Ghost::None , 0 );
    //Use the restart information from the gas temperature label since the
    //True wall temperature may not exisit.
    //This is a band-aid for cases that were run previously without the
    //true wall temperature variable.
    task->requires( Task::OldDW, VarLabel::find("temperature"), Ghost::None, 0 );
    if (do_coal_region){
      task->computes( _deposit_thickness_label );
      task->computes( _deposit_thickness_sb_s_label );
      task->computes( _deposit_thickness_sb_l_label );
      task->computes( _emissivity_label );
      task->computes( _thermal_cond_en_label );
      task->computes( _thermal_cond_sb_s_label );
      task->computes( _thermal_cond_sb_l_label );
      task->requires( Task::OldDW, _deposit_thickness_label, Ghost::None, 0 );
      task->requires( Task::OldDW, _deposit_thickness_sb_s_label, Ghost::None, 0 );
      task->requires( Task::OldDW, _deposit_thickness_sb_l_label, Ghost::None, 0 );
      task->requires( Task::OldDW, _emissivity_label, Ghost::None, 0 );
      task->requires( Task::OldDW, _thermal_cond_en_label, Ghost::None, 0 );
      task->requires( Task::OldDW, _thermal_cond_sb_s_label, Ghost::None, 0 );
      task->requires( Task::OldDW, _thermal_cond_sb_l_label, Ghost::None, 0 );
      task->requires( Task::OldDW, _ave_dep_vel_label, Ghost::None, 0 );
      task->requires( Task::OldDW, _d_vol_ave_label, Ghost::None, 0 );
    }
    //task->requires( Task::OldDW , _True_T_Label   , Ghost::None , 0 );

    task->requires( Task::NewDW , _cellType_label , Ghost::AroundCells , 1 );


      task->requires( Task::OldDW, _HF_E_label, Ghost::AroundCells, 1 );
      task->requires( Task::OldDW, _HF_W_label, Ghost::AroundCells, 1 );
      task->requires( Task::OldDW, _HF_N_label, Ghost::AroundCells, 1 );
      task->requires( Task::OldDW, _HF_S_label, Ghost::AroundCells, 1 );
      task->requires( Task::OldDW, _HF_T_label, Ghost::AroundCells, 1 );
      task->requires( Task::OldDW, _HF_B_label, Ghost::AroundCells, 1 );

  } else {


    task->requires( Task::NewDW, _True_T_Label, Ghost::None, 0 );
    task->requires( Task::NewDW, _T_copy_label, Ghost::None, 0 );
    task->requires( Task::NewDW , _cellType_label , Ghost::AroundCells , 1 );

  }

  task->requires( Task::OldDW,_shared_state->get_delt_label(), Ghost::None, 0);
  sched->addTask(task, level->eachPatch(), _shared_state->allArchesMaterials());

}

//_________________________________________
void
WallModelDriver::doWallHT( const ProcessorGroup* my_world,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw,
                           const int time_subset )
{

  //patch loop
  for (int p=0; p < patches->size(); p++){

    int timestep = _shared_state->getCurrentTopLevelTimeStep();

    const Patch* patch = patches->get(p);
    HTVariables vars;
    vars.relax = _relax;
    vars.time = _shared_state->getElapsedSimTime();
    delt_vartype DT;
    old_dw->get(DT, _shared_state->get_delt_label());
    vars.delta_t = DT;

    // Note: The local T_copy is necessary because boundary conditions are being applied
    // in the table lookup to T based on the conditions for the independent variables. These
    // BCs are being applied regardless of the type of wall temperature model.

    if( time_subset == 0 && timestep % _calc_freq == 0 ){

      // actually compute the wall HT model

      old_dw->get( vars.T_old      , _T_label      , _matl_index , patch , Ghost::None , 0 );
      old_dw->get( vars.cc_vel     , _cc_vel_label , _matl_index , patch , Ghost::None , 0 );
      old_dw->get( vars.T_real_old , VarLabel::find("temperature"), _matl_index, patch, Ghost::None, 0 );
      //old_dw->get( vars.T_real_old , _True_T_Label , _matl_index , patch , Ghost::None , 0 );

      new_dw->getModifiable(  vars.T, _T_label, _matl_index   , patch );
      new_dw->allocateAndPut( vars.T_copy     , _T_copy_label , _matl_index, patch );
      new_dw->allocateAndPut( vars.T_real     , _True_T_Label , _matl_index, patch );
      new_dw->get(   vars.celltype , _cellType_label , _matl_index , patch , Ghost::AroundCells, 1 );

      vars.T_real.copyData(vars.T_real_old);

        old_dw->get(   vars.incident_hf_e     , _HF_E_label     , _matl_index , patch, Ghost::AroundCells, 1 );
        old_dw->get(   vars.incident_hf_w     , _HF_W_label     , _matl_index , patch, Ghost::AroundCells, 1 );
        old_dw->get(   vars.incident_hf_n     , _HF_N_label     , _matl_index , patch, Ghost::AroundCells, 1 );
        old_dw->get(   vars.incident_hf_s     , _HF_S_label     , _matl_index , patch, Ghost::AroundCells, 1 );
        old_dw->get(   vars.incident_hf_t     , _HF_T_label     , _matl_index , patch, Ghost::AroundCells, 1 );
        old_dw->get(   vars.incident_hf_b     , _HF_B_label     , _matl_index , patch, Ghost::AroundCells, 1 );

      if (do_coal_region){
        old_dw->get( vars.ave_deposit_velocity , _ave_dep_vel_label, _matl_index, patch, Ghost::None, 0 ); // from particle model
        old_dw->get( vars.d_vol_ave , _d_vol_ave_label, _matl_index, patch, Ghost::None, 0 ); // from particle model
        old_dw->get( vars.deposit_thickness_old , _deposit_thickness_label, _matl_index, patch, Ghost::None, 0 );
        old_dw->get( vars.deposit_thickness_sb_s_old , _deposit_thickness_sb_s_label, _matl_index, patch, Ghost::None, 0 );
        old_dw->get( vars.deposit_thickness_sb_l_old , _deposit_thickness_sb_l_label, _matl_index, patch, Ghost::None, 0 );
        old_dw->get( vars.emissivity_old , _emissivity_label, _matl_index, patch, Ghost::None, 0 );
        old_dw->get( vars.thermal_cond_en_old , _thermal_cond_en_label, _matl_index, patch, Ghost::None, 0 );
        old_dw->get( vars.thermal_cond_sb_s_old , _thermal_cond_sb_s_label, _matl_index, patch, Ghost::None, 0 );
        old_dw->get( vars.thermal_cond_sb_l_old , _thermal_cond_sb_l_label, _matl_index, patch, Ghost::None, 0 );
        new_dw->allocateAndPut( vars.deposit_thickness, _deposit_thickness_label , _matl_index, patch ); // this isn't getModifiable because it hasn't been computed in DepositionVelocity yet.
        new_dw->allocateAndPut( vars.deposit_thickness_sb_s, _deposit_thickness_sb_s_label , _matl_index, patch ); // this isn't getModifiable because it hasn't been computed in DepositionVelocity yet.
        new_dw->allocateAndPut( vars.deposit_thickness_sb_l, _deposit_thickness_sb_l_label , _matl_index, patch ); // this isn't getModifiable because it hasn't been computed in DepositionVelocity yet.
        new_dw->allocateAndPut( vars.emissivity, _emissivity_label , _matl_index, patch ); // this isn't getModifiable because it hasn't been computed in DepositionVelocity yet.
        new_dw->allocateAndPut( vars.thermal_cond_en, _thermal_cond_en_label , _matl_index, patch ); // this isn't getModifiable because it hasn't been computed in DepositionVelocity yet.
        new_dw->allocateAndPut( vars.thermal_cond_sb_s, _thermal_cond_sb_s_label , _matl_index, patch ); // this isn't getModifiable because it hasn't been computed in DepositionVelocity yet.
        new_dw->allocateAndPut( vars.thermal_cond_sb_l, _thermal_cond_sb_l_label , _matl_index, patch ); // this isn't getModifiable because it hasn't been computed in DepositionVelocity yet.
        CellIterator c = patch->getExtraCellIterator();
        for (; !c.done(); c++ ){
          if ( vars.celltype[*c] > 7 && vars.celltype[*c] < 11 ){
            vars.deposit_thickness[*c] = vars.deposit_thickness_old[*c];
            vars.deposit_thickness_sb_s[*c] = vars.deposit_thickness_sb_s_old[*c];
            vars.deposit_thickness_sb_l[*c] = vars.deposit_thickness_sb_l_old[*c];
            vars.emissivity[*c] = vars.emissivity_old[*c];
            vars.thermal_cond_en[*c] = vars.thermal_cond_en_old[*c];
            vars.thermal_cond_sb_s[*c] = vars.thermal_cond_sb_s_old[*c];
            vars.thermal_cond_sb_l[*c] = vars.thermal_cond_sb_l_old[*c];
          } else {
            vars.deposit_thickness[*c] = 0.0;
            vars.deposit_thickness_sb_s[*c] = 0.0;
            vars.deposit_thickness_sb_l[*c] = 0.0;
            vars.emissivity[*c] = 0.0;
            vars.thermal_cond_en[*c] = 0.0;
            vars.thermal_cond_sb_s[*c] = 0.0;
            vars.thermal_cond_sb_l[*c] = 0.0;
          }
        }
      }

      std::vector<WallModelDriver::HTModelBase*>::iterator iter;

      for ( iter = _all_ht_models.begin(); iter != _all_ht_models.end(); iter++ ){

        (*iter)->computeHT( patch, vars, vars.T );

      }

      //Do a wholesale copy of T -> T_copy.  Note that we could force the derived models to do this
      //but that creates a danger of a developer forgeting to perform the operation. For now, do it
      //here for saftey and simplicity. Maybe rethink this if efficiency becomes an issue.
      vars.T_copy.copyData( vars.T );

    } else if ( time_subset == 0 && timestep % _calc_freq != 0 ) {

      // no ht solve this step:
      // 1) copy T_old (from OldDW) -> T   (to preserve BCs)
      // 2) copy T -> T_copy  (for future RK steps)

      CCVariable<double> T;
      CCVariable<double> T_copy;
      CCVariable<double> T_real;
      constCCVariable<double> T_real_old;
      constCCVariable<double> T_old;
      constCCVariable<int> cell_type;

      old_dw->get( T_old             , _T_label        , _matl_index , patch    , Ghost::None , 0 );
      //if ( !doing_restart )
        //old_dw->get( T_real_old      , _True_T_Label   , _matl_index , patch    , Ghost::None , 0 );
      old_dw->get( T_real_old , VarLabel::find("temperature"), _matl_index, patch, Ghost::None, 0 );
      new_dw->get( cell_type         , _cellType_label , _matl_index , patch    , Ghost::AroundCells , 1 );
      new_dw->getModifiable(  T      , _T_label        , _matl_index , patch );
      new_dw->allocateAndPut( T_copy , _T_copy_label   , _matl_index , patch );
      new_dw->allocateAndPut( T_real , _True_T_Label   , _matl_index , patch );

      std::vector<WallModelDriver::HTModelBase*>::iterator iter;

      for ( iter = _all_ht_models.begin(); iter != _all_ht_models.end(); iter++ ){

        // Note: This only copies the previous T bounary conditions into T and does not
        // change the gas temperature field.
        (*iter)->copySolution( patch, T, T_old, cell_type );

      }

      //Do a wholesale copy of T -> T_copy.  Note that we could force the derived models to do this
      //but that creates a danger of a developer forgeting to perform the operation. For now, do it
      //here for saftey and simplicity. Maybe rethink this if efficiency becomes an issue.
      T_copy.copyData( T );

      //T_real.copyData( T_real_old );
      CellIterator c = patch->getExtraCellIterator();
      for (; !c.done(); c++ ){
        if ( cell_type[*c] > 7 && cell_type[*c] < 11 ){
          T_real[*c] = T_real_old[*c];
        } else {
          T_real[*c] = 0.0;
        }
      }

      if (do_coal_region){
        CCVariable<double> deposit_thickness;
        CCVariable<double> deposit_thickness_sb_s;
        CCVariable<double> deposit_thickness_sb_l;
        CCVariable<double> emissivity;
        CCVariable<double> thermal_cond_en;
        CCVariable<double> thermal_cond_sb_s;
        CCVariable<double> thermal_cond_sb_l;
        constCCVariable<double> deposit_thickness_old;
        constCCVariable<double> deposit_thickness_sb_s_old;
        constCCVariable<double> deposit_thickness_sb_l_old;
        constCCVariable<double> emissivity_old;
        constCCVariable<double> thermal_cond_en_old;
        constCCVariable<double> thermal_cond_sb_s_old;
        constCCVariable<double> thermal_cond_sb_l_old;
        old_dw->get( deposit_thickness_old , _deposit_thickness_label, _matl_index, patch, Ghost::None, 0 );
        old_dw->get( deposit_thickness_sb_s_old , _deposit_thickness_sb_s_label, _matl_index, patch, Ghost::None, 0 );
        old_dw->get( deposit_thickness_sb_l_old , _deposit_thickness_sb_l_label, _matl_index, patch, Ghost::None, 0 );
        old_dw->get( emissivity_old , _emissivity_label, _matl_index, patch, Ghost::None, 0 );
        old_dw->get( thermal_cond_en_old , _thermal_cond_en_label, _matl_index, patch, Ghost::None, 0 );
        old_dw->get( thermal_cond_sb_s_old , _thermal_cond_sb_s_label, _matl_index, patch, Ghost::None, 0 );
        old_dw->get( thermal_cond_sb_l_old , _thermal_cond_sb_l_label, _matl_index, patch, Ghost::None, 0 );
        new_dw->allocateAndPut( deposit_thickness, _deposit_thickness_label, _matl_index , patch );
        new_dw->allocateAndPut( deposit_thickness_sb_s, _deposit_thickness_sb_s_label, _matl_index , patch );
        new_dw->allocateAndPut( deposit_thickness_sb_l, _deposit_thickness_sb_l_label, _matl_index , patch );
        new_dw->allocateAndPut( emissivity, _emissivity_label, _matl_index , patch );
        new_dw->allocateAndPut( thermal_cond_en, _thermal_cond_en_label, _matl_index , patch );
        new_dw->allocateAndPut( thermal_cond_sb_s, _thermal_cond_sb_s_label, _matl_index , patch );
        new_dw->allocateAndPut( thermal_cond_sb_l, _thermal_cond_sb_l_label, _matl_index , patch );
        CellIterator c = patch->getExtraCellIterator();
        for (; !c.done(); c++ ){
          if ( cell_type[*c] > 7 && cell_type[*c] < 11 ){
            deposit_thickness[*c] = deposit_thickness_old[*c];
            deposit_thickness_sb_s[*c] = deposit_thickness_sb_s_old[*c];
            deposit_thickness_sb_l[*c] = deposit_thickness_sb_l_old[*c];
            emissivity[*c] = emissivity_old[*c];
            thermal_cond_en[*c] = thermal_cond_en_old[*c];
            thermal_cond_sb_s[*c] = thermal_cond_sb_s_old[*c];
            thermal_cond_sb_l[*c] = thermal_cond_sb_l_old[*c];
          } else {
            deposit_thickness[*c] = 0.0;
            deposit_thickness_sb_s[*c] = 0.0;
            deposit_thickness_sb_l[*c] = 0.0;
            emissivity[*c] = 0.0;
            thermal_cond_en[*c] = 0.0;
            thermal_cond_sb_s[*c] = 0.0;
            thermal_cond_sb_l[*c] = 0.0;
          }
        }
      }


    } else {

      // no ht solve for RK steps > 0:
      // 1) T_copy (NewDW) should have the BC's from previous solution
      // 2) copy BC information from T_copy (NewDW) -> T to preserve BCs

      CCVariable<double> T;
      constCCVariable<double> T_old;
      constCCVariable<int> cell_type;

      new_dw->getModifiable( T , _T_label        , _matl_index , patch );
      new_dw->get( T_old       , _T_copy_label   , _matl_index , patch    , Ghost::None , 0 );
      new_dw->get( cell_type   , _cellType_label , _matl_index , patch    , Ghost::AroundCells , 1 );

      std::vector<WallModelDriver::HTModelBase*>::iterator iter;

      for ( iter = _all_ht_models.begin(); iter != _all_ht_models.end(); iter++ ){

        (*iter)->copySolution( patch, T, T_old, cell_type );

      }
    }
  }
}

//_________________________________________
void
WallModelDriver::sched_copyWallTintoT( const LevelP& level, SchedulerP& sched )
{

  Task* task = scinew Task( "WallModelDriver::copyWallTintoT", this,
                            &WallModelDriver::copyWallTintoT );

  //WARNING: Hardcoding temperature for now:
  task->modifies( VarLabel::find("temperature") );
  task->requires( Task::NewDW, _True_T_Label, Ghost::None, 0 );
  task->requires( Task::NewDW, _cellType_label, Ghost::None, 0 );

  sched->addTask( task, level->eachPatch(), _shared_state->allArchesMaterials() );

}

//_________________________________________
void
WallModelDriver::copyWallTintoT( const ProcessorGroup* my_world,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw )
{

  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);

    constCCVariable<double> T_real;
    constCCVariable<int> cell_type;
    CCVariable<double> T;

    new_dw->get( T_real, _True_T_Label, _matl_index, patch, Ghost::None, 0 );
    new_dw->get( cell_type, _cellType_label, _matl_index, patch, Ghost::None, 0 );
    new_dw->getModifiable( T, VarLabel::find("temperature"), _matl_index, patch );
    CellIterator c = patch->getExtraCellIterator();
    for (; !c.done(); c++ ){

      if ( cell_type[*c] > 7 && cell_type[*c] < 11 ){

        T[*c] = T_real[*c];

      }
    }


  }
}


// ********----- DERIVED HT MODELS --------********
//

// Simple HT model
//----------------------------------
WallModelDriver::SimpleHT::SimpleHT()
: _sigma_constant( 5.670e-8 )
{};

//----------------------------------
WallModelDriver::SimpleHT::~SimpleHT(){};

//----------------------------------
void
WallModelDriver::SimpleHT::problemSetup( const ProblemSpecP& input_db ){

  ProblemSpecP db = input_db;

  db->require("k", _k);
  db->require("wall_thickness", _dy);
  db->require("tube_side_T", _T_inner);
  db->getWithDefault( "T_wall_min", _T_min, 373 );
  db->getWithDefault( "T_wall_max", _T_max, 3000);

}

//----------------------------------
void
WallModelDriver::SimpleHT::computeHT( const Patch* patch, HTVariables& vars, CCVariable<double>& T ){

  double T_wall, net_q;
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);

  for( vector<Patch::FaceType>::const_iterator face_iter = bf.begin(); face_iter != bf.end(); ++face_iter ){

    Patch::FaceType face = *face_iter;
    IntVector offset = patch->faceDirection(face);
    CellIterator cell_iter = patch->getFaceIterator(face, Patch::InteriorFaceCells);

    constCCVariable<double>* q = NULL;
    switch (face) {
      case Patch::xminus:
        q = &(vars.incident_hf_w);
        break;
      case Patch::xplus:
        q = &(vars.incident_hf_e);
        break;
      case Patch::yminus:
        q = &(vars.incident_hf_s);
        break;
      case Patch::yplus:
        q = &(vars.incident_hf_n);
        break;
      case Patch::zminus:
        q = &(vars.incident_hf_b);
        break;
      case Patch::zplus:
        q = &(vars.incident_hf_t);
        break;
      default:
        break;
    }

    for(;!cell_iter.done(); cell_iter++){

      IntVector c = *cell_iter;        //this is the interior cell
      IntVector adj = c + offset;      //this is the cell IN the wall

      if ( vars.celltype[c + offset] == BoundaryCondition_new::WALL ){

        net_q = (*q)[c] - _sigma_constant * pow( vars.T_old[adj], 4 );
        net_q = net_q > 0 ? net_q : 0;
        T_wall = _T_inner + net_q * _dy / _k;

        T_wall = T_wall > _T_max ? _T_max : T_wall;
        T_wall = T_wall < _T_min ? _T_min : T_wall;

        T[adj] = ( 1.0 - vars.relax ) * vars.T_old[adj] + ( vars.relax ) * T_wall;

      }
    }
  }
}
//----------------------------------
void
WallModelDriver::SimpleHT::copySolution( const Patch* patch, CCVariable<double>& T, constCCVariable<double>& T_old, constCCVariable<int>& cell_type )
{
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);

  for( vector<Patch::FaceType>::const_iterator face_iter = bf.begin(); face_iter != bf.end(); ++face_iter ){

    Patch::FaceType face = *face_iter;
    IntVector offset = patch->faceDirection(face);
    CellIterator cell_iter = patch->getFaceIterator(face, Patch::InteriorFaceCells);

    for(;!cell_iter.done(); cell_iter++){

      IntVector c = *cell_iter;        //this is the interior cell
      IntVector adj = c + offset;      //this is the cell IN the wall

      if ( cell_type[c + offset] == BoundaryCondition_new::WALL ){

        T[adj] = T_old[adj];

      }
    }
  }
}

//--------------------------------------------
// RegionHT HT model
//----------------------------------
WallModelDriver::RegionHT::RegionHT()
: _sigma_constant( 5.670e-8 )
{

  _d.push_back(IntVector(1,0,0));
  _d.push_back(IntVector(-1,0,0));
  _d.push_back(IntVector(0,1,0));
  _d.push_back(IntVector(0,-1,0));
  _d.push_back(IntVector(0,0,1));
  _d.push_back(IntVector(0,0,-1));

};

//----------------------------------
WallModelDriver::RegionHT::~RegionHT(){
};

//----------------------------------
void
WallModelDriver::RegionHT::problemSetup( const ProblemSpecP& input_db ){

  ProblemSpecP db = input_db;
  db->getWithDefault( "max_it", _max_it, 50 );
  db->getWithDefault( "initial_tol", _init_tol, 1e-3 );
  db->getWithDefault( "tol", _tol, 1e-5 );

  for( ProblemSpecP r_db = db->findBlock("region"); r_db != nullptr; r_db = r_db->findNextBlock("region") ) {

    WallInfo info;
    ProblemSpecP geometry_db = r_db->findBlock("geom_object");
    GeometryPieceFactory::create( geometry_db, info.geometry );
    r_db->require("k", info.k);
    r_db->require("wall_thickness", info.dy);
    r_db->getWithDefault("wall_emissivity", info.emissivity, 1.0);
    r_db->require("tube_side_T", info.T_inner);
    //r_db->getWithDefault("max_TW", info.max_TW, 3500.0); //may need to revisit this later for cases when the wall tried to give energy back
    _regions.push_back( info );

  }
}

//----------------------------------
void
WallModelDriver::RegionHT::computeHT( const Patch* patch, HTVariables& vars, CCVariable<double>& T ){

  double net_q, rad_q, total_area_face;
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  Vector Dx = patch->dCell(); // cell spacing
  double DxDy = Dx.x() * Dx.y();
  double DxDz = Dx.x() * Dx.z();
  double DyDz = Dx.y() * Dx.z();
  Box patchBox = patch->getExtraBox();
  vector<int> container_flux_ind;
  vector<double> area_face;
  int total_flux_ind = 0;
  const int FLOW = -1;
  area_face.push_back(DyDz);
  area_face.push_back(DyDz);
  area_face.push_back(DxDz);
  area_face.push_back(DxDz);
  area_face.push_back(DxDy);
  area_face.push_back(DxDy);

  IntVector lowPindex = patch->getCellLowIndex();
  IntVector highPindex = patch->getCellHighIndex();
  //Pad for ghosts
  lowPindex -= IntVector(1,1,1);
  highPindex += IntVector(1,1,1);

  for ( std::vector<WallInfo>::iterator region_iter = _regions.begin(); region_iter != _regions.end(); region_iter++ ){

    WallInfo wi = *region_iter;

    for ( std::vector<GeometryPieceP>::iterator geom_iter = wi.geometry.begin(); geom_iter != wi.geometry.end(); geom_iter++ ){

      GeometryPieceP geom = *geom_iter;
      Box geomBox = geom->getBoundingBox();
      Box box     = geomBox.intersect( patchBox );

      //does the patchbox and geometry box overlap?
      if ( !box.degenerate() ){

        CellIterator iter = patch->getCellCenterIterator( box );
        constCCVariable<double>* q;

        for (; !iter.done(); iter++){

          IntVector c = *iter;
          total_flux_ind = 0;

          // is the point inside of region as defined by geom?
          if ( in_or_out( c, geom, patch ) ){

            container_flux_ind.clear();
            total_area_face=0;

            // Loop over all faces and find which faces have flow cells next to them
            for ( int i = 0; i < 6; i++ )
            {
              // is the current cell a wall or intrusion cell?
              if ( vars.celltype[c] == BoundaryCondition_new::WALL ||
                   vars.celltype[c] == BoundaryCondition_new::INTRUSION ){

                // is the neighbor in the current direction i a flow cell?
                //if ( patch->containsCell( c + _d[i] ) )
                if ( patch->containsIndex(lowPindex, highPindex, c + _d[i] ) )
                {
                  if ( vars.celltype[c + _d[i]] == FLOW )
                  {
                    container_flux_ind.push_back(i);
                    total_flux_ind+=1;
                  }
                }
              }
            }
            // if total_flux_ind is larger than 0 then there is incoming an incoming heatflux from the flow cell
            if ( total_flux_ind>0 )
            {

              rad_q=0;

              // get the total incoming heat flux from radiation:
              for ( int pp = 0; pp < total_flux_ind; pp++ )
              {

                q = get_flux( container_flux_ind[pp], vars );

                // The total watts contribution:
                rad_q += (*q)[c+_d[container_flux_ind[pp]]] * area_face[container_flux_ind[pp]];

                // The total cell surface area exposed to radiation:
                total_area_face += area_face[container_flux_ind[pp]];

              }

              rad_q /= total_area_face; // representative radiative flux to the cell.

              double d_tol    = 1e-15;
              double delta    = 1;
              int NIter       = 15;
              double f0       = 0.0;
              double f1       = 0.0;
              double TW       = vars.T_old[c];
              double T_max    = pow( rad_q/_sigma_constant, 0.25); // if k = 0.0;
              double TW_guess, TW_tmp, TW_old, TW_new;

              TW_guess = vars.T_old[c];
              TW_old = TW_guess-delta;
              net_q = rad_q - _sigma_constant * pow( TW_old, 4 );
              net_q = net_q > 0 ? net_q : 0;
              net_q *= wi.emissivity;
              f0 = - TW_old + wi.T_inner + net_q * wi.dy / wi.k;

              TW_new = TW_guess+delta;
              net_q = rad_q - _sigma_constant * pow( TW_new, 4 );
              net_q = net_q>0 ? net_q : 0;
              net_q *= wi.emissivity;
              f1 = - TW_new + wi.T_inner + net_q * wi.dy / wi.k;

              for ( int iterT=0; iterT < NIter; iterT++) {

                TW_tmp = TW_old;
                TW_old = TW_new;

                TW_new = TW_tmp - ( TW_new - TW_tmp )/( f1 - f0 ) * f0;
                TW_new = max( wi.T_inner , min( T_max, TW_new ) );

                if (std::abs(TW_new-TW_old) < d_tol){

                  TW    =  TW_new;
                  net_q =  rad_q - _sigma_constant * pow( TW_new, 4 );
                  net_q =  net_q > 0 ? net_q : 0;
                  net_q *= wi.emissivity;

                  break;

                }

                f0    =  f1;
                net_q =  rad_q - _sigma_constant * pow( TW_new, 4 );
                net_q =  net_q>0 ? net_q : 0;
                net_q *= wi.emissivity;
                f1    = - TW_new + wi.T_inner + net_q * wi.dy / wi.k;

              }

              // now to make consistent with assumed emissivity of 1 in radiation model:
              // q_radiation - 1 * sigma Tw' ^ 4 = emissivity * ( q_radiation - sigma Tw ^ 4 )
              // q_radiation - sigma Tw' ^ 4 = net_q

              vars.T_real[c] = (1 - vars.relax) * vars.T_real_old[c] + vars.relax * TW_new;

              TW = pow( (rad_q-net_q) / _sigma_constant, 0.25);

              T[c] = ( 1 - vars.relax ) * vars.T_old[c] + vars.relax * TW;

            }
          }
        }
      }
    }
  }
}
//----------------------------------
void
WallModelDriver::RegionHT::copySolution( const Patch* patch, CCVariable<double>& T, constCCVariable<double>& T_old, constCCVariable<int>& celltype )
{
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  Box patchBox = patch->getExtraBox();
  const int FLOW = -1;

  for ( std::vector<WallInfo>::iterator region_iter = _regions.begin(); region_iter != _regions.end(); region_iter++ ){

    WallInfo wi = *region_iter;

    for ( std::vector<GeometryPieceP>::iterator geom_iter = wi.geometry.begin(); geom_iter != wi.geometry.end(); geom_iter++ ){

      GeometryPieceP geom = *geom_iter;
      Box geomBox = geom->getBoundingBox();
      Box box     = geomBox.intersect( patchBox );

      //does the patchbox and geometry box overlap?
      if ( !box.degenerate() ){

        CellIterator iter = patch->getCellCenterIterator( box );

        for (; !iter.done(); iter++){

          IntVector c = *iter;

          for ( int i = 0; i < 6; i++ ){

            // is the current cell a solid?
            if ( celltype[c] == BoundaryCondition_new::WALL ||
                celltype[c] == BoundaryCondition_new::INTRUSION ){

              // is the neighbor in the current direction a flow?
              if ( patch->containsCell( c + _d[i] ) ){
                if ( celltype[c + _d[i]] == FLOW ){

                  T[c] = T_old[c];

                }
              }
            }
          }
        }
      }
    }
  }
}

//--------------------------------------------
// CoalRegionHT HT model
//----------------------------------
WallModelDriver::CoalRegionHT::CoalRegionHT()
: _sigma_constant( 5.670e-8 )
{

  _d.push_back(IntVector(1,0,0));
  _d.push_back(IntVector(-1,0,0));
  _d.push_back(IntVector(0,1,0));
  _d.push_back(IntVector(0,-1,0));
  _d.push_back(IntVector(0,0,1));
  _d.push_back(IntVector(0,0,-1));

};

//----------------------------------
WallModelDriver::CoalRegionHT::~CoalRegionHT(){
  delete m_em_model;
  delete m_tc_model;
};

//----------------------------------
void
WallModelDriver::CoalRegionHT::problemSetup( const ProblemSpecP& input_db ){

  ProblemSpecP db = input_db;
  db->getWithDefault( "max_it", _max_it, 50 );
  db->getWithDefault( "initial_tol", _init_tol, 1e-3 );
  db->getWithDefault( "tol", _tol, 1e-5 );

  int emissivity_model_type = get_emissivity_model_type( db );
  if (emissivity_model_type == 1){
    m_em_model = scinew constant_e();
  } else if (emissivity_model_type == 2){
    m_em_model = scinew dynamic_e(db);
  } else if (emissivity_model_type == 3){
    m_em_model = scinew pokluda_e(db);
  } else {
    throw InvalidValue("ERROR: WallModelDriver: No emissivity model selected.",__FILE__,__LINE__);
  }

  int thermal_cond_model_type = get_thermal_cond_model_type( db );
  if (thermal_cond_model_type == 1){
    m_tc_model = scinew constant_tc();
  } else if (thermal_cond_model_type == 2){
    m_tc_model = scinew hadley_tc(db);
  } else {
    throw InvalidValue("ERROR: WallModelDriver: No thermal conductivity model selected.",__FILE__,__LINE__);
  }
  
  std::vector<double> default_comp = {39.36,25.49, 7.89,  10.12, 2.46, 0.0, 1.09, 4.10};
  std::vector<double> y_ash;
  db->getWithDefault( "sb_ash_composition", y_ash, default_comp);
  double sum_y_ash = 0.0;
  std::for_each(y_ash.begin(), y_ash.end(), [&] (double n) { sum_y_ash += n;});
  for(std::vector<int>::size_type i = 0; i != y_ash.size(); i++) {
    y_ash[i]=y_ash[i]/sum_y_ash;
  } 
  //                      0      1       2        3       4        5       6      7
  //                      sio2,  al2o3,  cao,     fe2o3,  na2o,    bao,    tio2,  mgo.  
  std::vector<double> MW={60.08, 101.96, 56.0774, 159.69, 61.9789, 153.33, 79.866,40.3044};
  std::vector<double> x_ash;
  double sum_x_ash = 0.0;
  for(std::vector<int>::size_type i = 0; i != y_ash.size(); i++) {
       sum_x_ash += y_ash[i]*MW[i];
  } 
  for(std::vector<int>::size_type i = 0; i != y_ash.size(); i++) {
       x_ash.push_back( y_ash[i]*MW[i] / sum_x_ash );
  } 

  double rho_ash_bulk;
  if ( db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")){
    ProblemSpecP db_part_properties =  db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");
    db_part_properties->getWithDefault( "rho_ash_bulk",rho_ash_bulk,2300.0);
  } else {
    throw InvalidValue("Error: WallHT type coal_region requires ParticleProperties rho_ash_bulk to be specified.", __FILE__, __LINE__);
  }
  double p_void0;
  db->getWithDefault( "sb_deposit_porosity",p_void0,0.6); // note here we are using the sb layer to estimate the wall density no the enamel layer.
  double deposit_density = rho_ash_bulk * p_void0;  

  for ( ProblemSpecP r_db = db->findBlock("coal_region"); r_db != nullptr; r_db = r_db->findNextBlock("coal_region") ) {

    WallInfo info;
    ProblemSpecP geometry_db = r_db->findBlock("geom_object");
    GeometryPieceFactory::create( geometry_db, info.geometry );
    r_db->require("erosion_thickness", info.dy_erosion);
    info.T_slag = ParticleTools::getAshFluidTemperature(r_db);
    r_db->require("tscale_dep", info.t_sb);
    r_db->require("k", info.k);
    r_db->require("wall_thickness", info.dy);
    r_db->getWithDefault("k_deposit_enamel", info.k_dep_en,1.0);
    if (info.k_dep_en <= 0.0)
    {
      throw InvalidValue("ERROR: k_deposit_enamel must be greater than 0.0.",__FILE__,__LINE__);
    }
    r_db->getWithDefault("k_deposit", info.k_dep_sb,1.0);
    if (info.k_dep_sb <= 0.0)
    {
      throw InvalidValue("ERROR: k_deposit must be greater than 0.0.",__FILE__,__LINE__);
    }
    r_db->getWithDefault("enamel_deposit_thickness", info.dy_dep_en,0.0);
    r_db->getWithDefault("wall_emissivity", info.emissivity, 1.0);
    r_db->require("tube_side_T", info.T_inner);
    info.deposit_density = deposit_density; 
    info.x_ash = x_ash; 
    _regions.push_back( info );

  }
}

//----------------------------------
void
WallModelDriver::CoalRegionHT::computeHT( const Patch* patch, HTVariables& vars, CCVariable<double>& T ){

  double residual, TW_new, T_old, net_q, rad_q, total_area_face, average_area_face, R_wall, R_en, R_sb, R_tot, Emiss, dp_arrival, tau_sint;
  double T_i, T_en, T_metal, T_sb_l, T_sb_s, dy_dep_sb_s, dy_dep_sb_l, dy_dep_sb_l_old, dy_dep_en, k_en, k_sb_s, k_sb_l, k_en_old, k_sb_s_old, k_sb_l_old;
  const std::string enamel_name = "enamel"; 
  const std::string sb_name = "sb";
  const std::string sb_l_name = "sb_liquid";
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  Vector Dx = patch->dCell(); // cell spacing
  double DxDy = Dx.x() * Dx.y();
  double DxDz = Dx.x() * Dx.z();
  double DyDz = Dx.y() * Dx.z();
  Box patchBox = patch->getExtraBox();
  vector<int> container_flux_ind;
  vector<double> area_face;
  int total_flux_ind = 0;
  const int FLOW = -1;
  area_face.push_back(DyDz);
  area_face.push_back(DyDz);
  area_face.push_back(DxDz);
  area_face.push_back(DxDz);
  area_face.push_back(DxDy);
  area_face.push_back(DxDy);

  IntVector lowPindex = patch->getCellLowIndex();
  IntVector highPindex = patch->getCellHighIndex();
  //Pad for ghosts
  lowPindex -= IntVector(1,1,1);
  highPindex += IntVector(1,1,1);

  for ( std::vector<WallInfo>::iterator region_iter = _regions.begin(); region_iter != _regions.end(); region_iter++ ){

    WallInfo wi = *region_iter;

    for ( std::vector<GeometryPieceP>::iterator geom_iter = wi.geometry.begin(); geom_iter != wi.geometry.end(); geom_iter++ ){

      GeometryPieceP geom = *geom_iter;
      Box geomBox = geom->getBoundingBox();
      Box box     = geomBox.intersect( patchBox );

      //does the patchbox and geometry box overlap?
      if ( !box.degenerate() ){

        CellIterator iter = patch->getCellCenterIterator( box );
        constCCVariable<double>* q;

        for (; !iter.done(); iter++){

          IntVector c = *iter;
          total_flux_ind = 0;

          // is the point inside of region as defined by geom?
          if ( in_or_out( c, geom, patch ) ){

            container_flux_ind.clear();
            total_area_face=0;

            // Loop over all faces and find which faces have flow cells next to them
            for ( int i = 0; i < 6; i++ )
            {
              // is the current cell a wall or intrusion cell?
              if ( vars.celltype[c] == BoundaryCondition_new::WALL ||
                   vars.celltype[c] == BoundaryCondition_new::INTRUSION ){

                // is the neighbor in the current direction i a flow cell?
                //if ( patch->containsCell( c + _d[i] ) )
                if ( patch->containsIndex(lowPindex, highPindex, c + _d[i] ) )
                {
                  if ( vars.celltype[c + _d[i]] == FLOW )
                  {
                    container_flux_ind.push_back(i);
                    total_flux_ind+=1;
                  }
                }
              }
            }
            // if total_flux_ind is larger than 0 then there is incoming an incoming heatflux from the flow cell
            if ( total_flux_ind>0 )
            {

              rad_q=0;

              // get the total incoming heat flux from radiation:
              for ( int pp = 0; pp < total_flux_ind; pp++ )
              {

                q = get_flux( container_flux_ind[pp], vars );

                // The total watts contribution:
                rad_q += (*q)[c+_d[container_flux_ind[pp]]] * area_face[container_flux_ind[pp]];

                // The total cell surface area exposed to radiation:
                total_area_face += area_face[container_flux_ind[pp]];

              }

              rad_q /= total_area_face; // representative radiative flux to the cell.
              average_area_face = total_area_face/total_flux_ind;
              
              // We are using the old emissivity, and thermal conductivities for the calculation of the deposition regime
              // and the new temperature. We then update the emissivity and thermal conductivies using the new temperature.                   
              // This effectivly time-lags the emissivity and thermal conductivies by one radiation-solve. We chose to do this                
              // because we didn't want to make the temperature solve more expensive.                                
              k_en=vars.thermal_cond_en_old[c];
              k_sb_s=vars.thermal_cond_sb_s_old[c];
              k_sb_l=vars.thermal_cond_sb_l_old[c];
              Emiss=vars.emissivity_old[c]; 
              TW_new =  vars.T_real_old[c];
              T_old =  vars.T_real_old[c];
                                            
              dp_arrival=vars.d_vol_ave[c];
              tau_sint=min(dp_arrival/max(vars.ave_deposit_velocity[c],1e-50),1e10); // [s]
              dy_dep_sb_s = std::min(vars.ave_deposit_velocity[c]*wi.t_sb, wi.dy_erosion);// This includes our crude erosion model. If the deposit wants to grow above a certain size it will erode. 

              // here we computed quantaties to find which deposition regime we are in.
              dy_dep_en = wi.dy_dep_en;
              R_wall = wi.dy / wi.k;
              R_en = dy_dep_en/k_en;
              R_sb = dy_dep_sb_s / k_sb_s;
              double visc = 0; // Pa-s
              urbain_viscosity(visc, wi.T_slag, wi.x_ash);
              double kin_visc = visc/wi.deposit_density;
              double a_p = 0.8434; //
              double b_p = 1./3.; //
              double cell_width = std::pow(average_area_face,0.5); // m
              double mdot = vars.ave_deposit_velocity[c] * wi.deposit_density * average_area_face; // liquid mass flow rate kg/s
              double dsb_l_max = a_p*std::pow(kin_visc*kin_visc/9.8,1./3.)*std::pow(4.*(mdot/cell_width)/visc,b_p);
              double rad_q_melt = (wi.T_slag-wi.T_inner)/((R_en+R_sb+R_wall)*Emiss) + _sigma_constant*std::pow(wi.T_slag,4.0);
              double qnet_max =  (wi.T_slag-wi.T_inner)/(R_wall+R_en);
              double Ts_max =  qnet_max*(dsb_l_max/k_sb_l)+wi.T_slag;
              double rad_q_max = qnet_max/Emiss+_sigma_constant*std::pow( Ts_max,4.0);

              if (rad_q < rad_q_melt){
                // regime 1 
                // dy_dep_sb_s = vars.ave_deposit_velocity[c] * wi.t_sb; // this has already been computed above 
                dy_dep_sb_l = 0.0;
                R_tot = R_wall + R_en + dy_dep_sb_s/k_sb_s;
                newton_solve( TW_new, wi.T_inner, T_old, R_tot, rad_q, Emiss );
              } else if (rad_q>=rad_q_melt && rad_q<=rad_q_max){
                // regime 2 
                dy_dep_sb_l = (dsb_l_max/(rad_q_max-rad_q_melt))*(rad_q-rad_q_melt);
                R_tot = dy_dep_sb_l/k_sb_l;
                newton_solve( TW_new, wi.T_slag, T_old, R_tot, rad_q, Emiss );
                // now we can solve for the solid layer thickness given the new surface temperature. 
                double qnet = rad_q - _sigma_constant * std::pow( TW_new, 4.0 );
                qnet *= Emiss;
                qnet = qnet > 1e-8 ? qnet : 1e-8; // to avoid div by zero we set min at 1e-8
                dy_dep_sb_s = k_sb_s*((TW_new-wi.T_inner)/qnet - R_wall - R_en - dy_dep_sb_l/k_sb_l);
              } else { // rad_q > rad_q_max
                // regime 3 
                dy_dep_sb_s = 0.0;
                dy_dep_sb_l = dsb_l_max;
                residual = 0.0;
                for ( int iterT=0; iterT < 100; iterT++) { // iterate tc until we reach a consistent solution.
                  dy_dep_sb_l_old = dy_dep_sb_l;  
                  double qnet = rad_q - _sigma_constant * std::pow( TW_new, 4.0 );
                  qnet *= Emiss;
                  qnet = qnet > 1e-8 ? qnet : 1e-8; // to avoid div by zero we set min at 1e-8
                  R_tot = R_wall + R_en + dy_dep_sb_l/k_sb_l;
                  newton_solve( TW_new, wi.T_inner, T_old, R_tot, rad_q, Emiss );
                  T_i = TW_new - qnet * dy_dep_sb_l / k_sb_l;// this is the interface temperature between the liquid and the enamel. 
                  urbain_viscosity(visc, T_i, wi.x_ash);
                  kin_visc = visc/wi.deposit_density;
                  dy_dep_sb_l = a_p*std::pow(kin_visc*kin_visc/9.8,1./3.)*std::pow(4.*(mdot/cell_width)/visc,b_p);
                  residual = std::abs(dy_dep_sb_l_old - dy_dep_sb_l)/(dy_dep_sb_l+1e-100); 
                  if (residual < 1e-4){
                    break;
                  }
                }
              }
              vars.deposit_thickness_sb_l[c] = (1-vars.relax) * vars.deposit_thickness_sb_l_old[c] + vars.relax * dy_dep_sb_l;
              vars.deposit_thickness_sb_s[c] = (1-vars.relax) * vars.deposit_thickness_sb_s_old[c] + vars.relax * dy_dep_sb_s;
              dy_dep_sb_s = vars.deposit_thickness_sb_s[c];
              dy_dep_sb_l = vars.deposit_thickness_sb_l[c];
              vars.deposit_thickness[c] = dy_dep_sb_s + dy_dep_sb_l + dy_dep_en; // this is the total deposit thickness on the wall.

              vars.T_real[c] = (1 - vars.relax) * vars.T_real_old[c] + vars.relax * TW_new; // this is the real wall temperature, vars.T_real_old is the old solution for "temperature".
              // update the net heat flux, emissivity, and thermal conductivies with the new time-averaged wall temperature and thickness
              m_em_model->model(Emiss,wi.emissivity,vars.T_real[c],dp_arrival, tau_sint);
              vars.emissivity[c]=Emiss;
              net_q = rad_q - _sigma_constant * std::pow( vars.T_real[c], 4.0 );
              net_q = net_q > 0 ? net_q : 0; 
              net_q *= Emiss;
              residual = 0.0;
              for ( int iterT=0; iterT < 100; iterT++) { // iterate tc until we reach a consistent solution.
                k_sb_s_old = k_sb_s; 
                k_sb_l_old = k_sb_l; 
                k_en_old = k_en; 
                T_sb_s = vars.T_real[c] - net_q * dy_dep_sb_l / k_sb_l; // Temperature between the liquid and solid sb layer. 
                T_en = T_sb_s - net_q * dy_dep_sb_s / k_sb_s; // Temperature between the solid sb and enamel layer. 
                T_metal = T_en - net_q * dy_dep_en / k_en; // Temperature between the enamel and sb layer. 
                T_sb_l = std::max(wi.T_inner,(vars.T_real[c] + T_sb_s)/2.0);//Temperature at the center of liquid sb layer. std::max is needed because of the negative temperatures during initialization. 
                T_sb_s = std::max(wi.T_inner,(T_sb_s + T_en)/2.0);//Temperature at the center of the solid sb layer. std::max is needed because of the negative temperatures during initialization.
                T_en = std::max(wi.T_inner,(T_en + T_metal)/2.0);//Temperature at the center of the enamel layer. std::max is needed because of the negative temperatures during initialization.
                m_tc_model->model(k_en,wi.k_dep_en,T_en,enamel_name);//enamel layer
                m_tc_model->model(k_sb_s,wi.k_dep_sb,T_sb_s,sb_name);//solid sb layer
                m_tc_model->model(k_sb_l,wi.k_dep_sb,T_sb_l,sb_l_name);//liquid sb layer
                residual = std::abs(k_sb_s - k_sb_s_old)/(k_sb_s_old+1e-100) + std::abs(k_sb_l - k_sb_l_old)/(k_sb_l_old+1e-100) + std::abs(k_en - k_en_old)/(k_en_old+1e-100); 
                if (residual < 1e-4){
                  break;
                }
              }
              vars.thermal_cond_en[c]=k_en;
              vars.thermal_cond_sb_s[c]=k_sb_s;
              vars.thermal_cond_sb_l[c]=k_sb_l;
              // now to make consistent with assumed emissivity of 1 in radiation model:
              // q_radiation - 1 * sigma Tw' ^ 4 = emissivity * ( q_radiation - sigma Tw ^ 4 )
              // q_radiation - sigma Tw' ^ 4 = net_q
              // Tw' = ((q_radiation - net_q)/sigma)^0.25
              TW_new = std::pow( (rad_q-net_q) / _sigma_constant, 0.25); // correct TW_new for radiation model
              T[c] = ( 1 - vars.relax ) * vars.T_old[c] + vars.relax * TW_new; // this is the radiation wall temperature, var.T_old[c] is the old solution for radiation "temperature".
            }
          }
        }
      }
    }
  }
}
//----------------------------------
void
WallModelDriver::CoalRegionHT::copySolution( const Patch* patch, CCVariable<double>& T, constCCVariable<double>& T_old, constCCVariable<int>& celltype )
{
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  Box patchBox = patch->getExtraBox();
  const int FLOW = -1;

  for ( std::vector<WallInfo>::iterator region_iter = _regions.begin(); region_iter != _regions.end(); region_iter++ ){

    WallInfo wi = *region_iter;

    for ( std::vector<GeometryPieceP>::iterator geom_iter = wi.geometry.begin(); geom_iter != wi.geometry.end(); geom_iter++ ){

      GeometryPieceP geom = *geom_iter;
      Box geomBox = geom->getBoundingBox();
      Box box     = geomBox.intersect( patchBox );

      //does the patchbox and geometry box overlap?
      if ( !box.degenerate() ){

        CellIterator iter = patch->getCellCenterIterator( box );

        for (; !iter.done(); iter++){

          IntVector c = *iter;

          for ( int i = 0; i < 6; i++ ){

            // is the current cell a solid?
            if ( celltype[c] == BoundaryCondition_new::WALL ||
                celltype[c] == BoundaryCondition_new::INTRUSION ){

              // is the neighbor in the current direction a flow?
              if ( patch->containsCell( c + _d[i] ) ){
                if ( celltype[c + _d[i]] == FLOW ){

                  T[c] = T_old[c];

                }
              }
            }
          }
        }
      }
    }
  }
}
//----------------------------------
void
WallModelDriver::CoalRegionHT::newton_solve(double &TW_new, double &T_shell, double &T_old, double &R_tot, double &rad_q, double &Emiss )
{
  // solver constants
  double d_tol    = 1e-15;
  double delta    = 1;
  int NIter       = 15;
  double f0       = 0.0;
  double f1       = 0.0;
  double T_max    = pow( rad_q/_sigma_constant, 0.25); // if k = 0.0;
  double net_q, TW_guess, TW_tmp, TW_old;
  // new solve
  TW_guess = T_old;
  TW_old = TW_guess-delta;
  net_q = rad_q - _sigma_constant * pow( TW_old, 4 );
  net_q = net_q > 0 ? net_q : 0;
  net_q *= Emiss;
  f0 = - TW_old + T_shell + net_q * R_tot;
  TW_new = TW_guess+delta;
  net_q = rad_q - _sigma_constant * pow( TW_new, 4 );
  net_q *= Emiss;
  net_q = net_q>0 ? net_q : 0;
  f1 = - TW_new + T_shell + net_q * R_tot;
  for ( int iterT=0; iterT < NIter; iterT++) {
    TW_tmp = TW_old;
    TW_old = TW_new;
    TW_new = TW_tmp - ( TW_new - TW_tmp )/( f1 - f0 ) * f0;
    TW_new = max( T_shell , min( T_max, TW_new ) );
    if (std::abs(TW_new-TW_old) < d_tol){
      net_q =  rad_q - _sigma_constant * pow( TW_new, 4 );
      net_q =  net_q > 0 ? net_q : 0;
      net_q *= Emiss;
      break;
    }
    f0    =  f1;
    net_q =  rad_q - _sigma_constant * pow( TW_new, 4 );
    net_q =  net_q>0 ? net_q : 0;
    net_q *= Emiss;
    f1    = - TW_new + T_shell + net_q * R_tot;
  }
}
void WallModelDriver::CoalRegionHT::urbain_viscosity(double &visc, double &T, std::vector<double> &x_ash)
{  // Urbain model 1981
  //0      1       2        3       4        5       6      7
  //sio2,  al2o3,  cao,     fe2o3,  na2o,    bao,    tio2,  mgo.  
  const double N = x_ash[0];
  const double al2o3=x_ash[1];

  // compute modifier
  const double cao=x_ash[2];
  const double mgo=x_ash[7];
  const double na2o=x_ash[4];
  const double k2o=0.0;
  const double feo=0.0;
  const double mno=0.0;
  const double nio=0.0;
  const double tio2=x_ash[6];
  const double zro2=0.0;
  
  const double M=1*(cao+mgo+na2o+k2o+feo+mno+nio)+2*(tio2+zro2);
  const double alpha=M/(M+al2o3);
  const double B0=13.8+39.9355*alpha-44.049*alpha*alpha;
  const double B1=30.481-117.1505*alpha+129.9978*alpha*alpha;
  const double B2=-40.9429+234.0486*alpha-300.04*alpha*alpha;
  const double B3=60.7619-153.9276*alpha+211.1616*alpha*alpha;
  
  const double B=B0+B1*N+B2*N*N+B3*N*N*N;
  const double A=std::exp(-(0.2693*B+11.6725));
  visc=0.1*A*T*std::exp((B*1000)/T);


};
