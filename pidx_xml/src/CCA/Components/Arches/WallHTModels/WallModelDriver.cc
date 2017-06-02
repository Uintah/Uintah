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
    VarLabel::destroy( _emissivity_label );
    VarLabel::destroy( _thermal_cond_en_label );
    VarLabel::destroy( _thermal_cond_sb_label );
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
    _emissivity_label = VarLabel::create( "emissivity", CC_double );
    _thermal_cond_en_label = VarLabel::create( "thermal_cond_en", CC_double );
    _thermal_cond_sb_label = VarLabel::create( "thermal_cond_sb", CC_double );
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
      task->computes( _emissivity_label );
      task->computes( _thermal_cond_en_label );
      task->computes( _thermal_cond_sb_label );
      task->requires( Task::OldDW, _deposit_thickness_label, Ghost::None, 0 );
      task->requires( Task::OldDW, _emissivity_label, Ghost::None, 0 );
      task->requires( Task::OldDW, _thermal_cond_en_label, Ghost::None, 0 );
      task->requires( Task::OldDW, _thermal_cond_sb_label, Ghost::None, 0 );
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
        old_dw->get( vars.emissivity_old , _emissivity_label, _matl_index, patch, Ghost::None, 0 );
        old_dw->get( vars.thermal_cond_en_old , _thermal_cond_en_label, _matl_index, patch, Ghost::None, 0 );
        old_dw->get( vars.thermal_cond_sb_old , _thermal_cond_sb_label, _matl_index, patch, Ghost::None, 0 );
        new_dw->allocateAndPut( vars.deposit_thickness, _deposit_thickness_label , _matl_index, patch ); // this isn't getModifiable because it hasn't been computed in DepositionVelocity yet.
        new_dw->allocateAndPut( vars.emissivity, _emissivity_label , _matl_index, patch ); // this isn't getModifiable because it hasn't been computed in DepositionVelocity yet.
        new_dw->allocateAndPut( vars.thermal_cond_en, _thermal_cond_en_label , _matl_index, patch ); // this isn't getModifiable because it hasn't been computed in DepositionVelocity yet.
        new_dw->allocateAndPut( vars.thermal_cond_sb, _thermal_cond_sb_label , _matl_index, patch ); // this isn't getModifiable because it hasn't been computed in DepositionVelocity yet.
        CellIterator c = patch->getExtraCellIterator();
        for (; !c.done(); c++ ){
          if ( vars.celltype[*c] > 7 && vars.celltype[*c] < 11 ){
            vars.deposit_thickness[*c] = vars.deposit_thickness_old[*c];
            vars.emissivity[*c] = vars.emissivity_old[*c];
            vars.thermal_cond_en[*c] = vars.thermal_cond_en_old[*c];
            vars.thermal_cond_sb[*c] = vars.thermal_cond_sb_old[*c];
          } else {
            vars.deposit_thickness[*c] = 0.0;
            vars.emissivity[*c] = 0.0;
            vars.thermal_cond_en[*c] = 0.0;
            vars.thermal_cond_sb[*c] = 0.0;
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
        CCVariable<double> emissivity;
        CCVariable<double> thermal_cond_en;
        CCVariable<double> thermal_cond_sb;
        constCCVariable<double> deposit_thickness_old;
        constCCVariable<double> emissivity_old;
        constCCVariable<double> thermal_cond_en_old;
        constCCVariable<double> thermal_cond_sb_old;
        old_dw->get( deposit_thickness_old , _deposit_thickness_label, _matl_index, patch, Ghost::None, 0 );
        old_dw->get( emissivity_old , _emissivity_label, _matl_index, patch, Ghost::None, 0 );
        old_dw->get( thermal_cond_en_old , _thermal_cond_en_label, _matl_index, patch, Ghost::None, 0 );
        old_dw->get( thermal_cond_sb_old , _thermal_cond_sb_label, _matl_index, patch, Ghost::None, 0 );
        new_dw->allocateAndPut( deposit_thickness, _deposit_thickness_label, _matl_index , patch );
        new_dw->allocateAndPut( emissivity, _emissivity_label, _matl_index , patch );
        new_dw->allocateAndPut( thermal_cond_en, _thermal_cond_en_label, _matl_index , patch );
        new_dw->allocateAndPut( thermal_cond_sb, _thermal_cond_sb_label, _matl_index , patch );
        CellIterator c = patch->getExtraCellIterator();
        for (; !c.done(); c++ ){
          if ( cell_type[*c] > 7 && cell_type[*c] < 11 ){
            deposit_thickness[*c] = deposit_thickness_old[*c];
            emissivity[*c] = emissivity_old[*c];
            thermal_cond_en[*c] = thermal_cond_en_old[*c];
            thermal_cond_sb[*c] = thermal_cond_sb_old[*c];
          } else {
            deposit_thickness[*c] = 0.0;
            emissivity[*c] = 0.0;
            thermal_cond_en[*c] = 0.0;
            thermal_cond_sb[*c] = 0.0;
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
    _regions.push_back( info );

  }
}

//----------------------------------
void
WallModelDriver::CoalRegionHT::computeHT( const Patch* patch, HTVariables& vars, CCVariable<double>& T ){

  double TW_new, T_old, net_q, rad_q, total_area_face, R_wall, R_en, Emiss, dp_arrival, tau_sint;
  double T_en, T_metal, Tsb, Ten, dy_dep_sb, dy_dep_en, k_en, k_sb, k_en_old, k_sb_old;
  const std::string enamel_name = "enamel"; 
  const std::string sb_name = "sb";
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

              // We are using the old emissivity, and thermal conductivities for the calculation of the deposition regime
              // and the new temperature. We then update the emissivity and thermal conductivies using the new temperature.                   
              // This effectivly time-lags the emissivity and thermal conductivies by one radiation-solve. We chose to do this                
              // because we didn't want to make the temperature solve more expensive.                                
              k_en=vars.thermal_cond_en_old[c];
              k_sb=vars.thermal_cond_sb_old[c];
              Emiss=vars.emissivity_old[c]; 
              TW_new =  vars.T_real_old[c];
                                            
                                            
              dp_arrival=vars.d_vol_ave[c];
              tau_sint=min(dp_arrival/max(vars.ave_deposit_velocity[c],1e-50),1e10); // [s]
              vars.deposit_thickness[c] = vars.ave_deposit_velocity[c] * wi.t_sb;
              vars.deposit_thickness[c] = min(vars.deposit_thickness[c],wi.dy_erosion);// Here is our crude erosion model. If the deposit wants to grow above a certain size it will erode.

              dy_dep_en = wi.dy_dep_en;
              // here we computed quantaties to find which deposition regime we are in.
              R_wall = wi.dy / wi.k;
              R_en = wi.dy_dep_en/k_en;
              double qnet_max = rad_q - _sigma_constant * std::pow( wi.T_slag, 4.0 );
              qnet_max *= Emiss;
              qnet_max = qnet_max > 1e-8 ? qnet_max : 1e-8; // to avoid div by zero we min is 1e-8
              double dy_dep_sb_max = k_sb * ( (wi.T_slag-wi.T_inner)/qnet_max - R_wall - R_en);
              double dy_dep_en_max = k_en * ( (wi.T_slag-wi.T_inner)/qnet_max - R_wall);
              double rad_q_max = (wi.T_slag-wi.T_inner)/(R_wall*Emiss) + _sigma_constant*std::pow(wi.T_slag,4.0);
              double rad_q_en = (wi.T_slag-wi.T_inner)/((R_wall + R_en)*Emiss) + _sigma_constant*std::pow(wi.T_slag,4.0);

              if (vars.deposit_thickness[c] < dy_dep_sb_max) {
                // Regime 1
                // the resulting temperature will be less than T_slag.
                // deposit_thickness doesn't need to be modified.
              } else if (rad_q >= rad_q_max) {
                // Regime 3
                // the resulting temperature will be higher than Tslag.
                vars.deposit_thickness[c] = 0.0;
                dy_dep_en = 0.0;
              } else {
                // Regime 2
                vars.deposit_thickness[c] = (rad_q >= rad_q_en) ? 0.0 : dy_dep_sb_max;
                dy_dep_en = (rad_q >= rad_q_en) ? dy_dep_en_max : dy_dep_en;
              }

              vars.deposit_thickness[c] = (1-vars.relax) * vars.deposit_thickness_old[c] + vars.relax * vars.deposit_thickness[c];  // here we time average the deposit thickness so that it doesn't vary when we switch regimes.
              dy_dep_sb = vars.deposit_thickness[c];

              T_old =  vars.T_real_old[c];
              newton_solve( wi, TW_new, k_sb, k_en, dy_dep_sb, dy_dep_en, T_old, rad_q, R_wall, Emiss );
              vars.T_real[c] = (1 - vars.relax) * vars.T_real_old[c] + vars.relax * TW_new; // this is the real wall temperature, vars.T_real_old is the old solution for "temperature".
              // update the emissivity model with the new wall temperature
              m_em_model->model(Emiss,wi.emissivity,vars.T_real[c],dp_arrival, tau_sint);
              vars.emissivity[c]=Emiss;
              // update the thermal conductivies with the new wall temperature
              net_q = rad_q - _sigma_constant * std::pow( vars.T_real[c], 4.0 );
              net_q = net_q > 0 ? net_q : 0; 
              net_q *= Emiss;
              double residual = 0.0;
              for ( int iterT=0; iterT < 100; iterT++) {
                k_sb_old = k_sb; 
                k_en_old = k_en; 
                T_en = vars.T_real[c] - net_q * dy_dep_sb / k_sb; // Temperature at the surface between enamel and sb layer. 
                T_metal = T_en - net_q * dy_dep_en / k_en; // Temperature at the surface between enamel and metal/refractory. 
                Tsb = std::max(wi.T_inner,(vars.T_real[c] + T_en)/2.0);//Temperature at the center of the sb layer. std::max is needed because of the negative temperatures during initialization.
                Ten = std::max(wi.T_inner,(T_en + T_metal)/2.0);//Temperature at the center of the enamel layer. std::max is needed because of the negative temperatures during initialization.
                m_tc_model->model(k_en,wi.k_dep_en,Ten,enamel_name);//enamel layer
                m_tc_model->model(k_sb,wi.k_dep_sb,Tsb,sb_name);//sb layer
                residual = std::abs(k_sb - k_sb_old)/(k_sb_old+1e-100) + std::abs(k_en - k_en_old)/(k_en_old+1e-100); 
                if (residual < 1e-4){
                  break;
                }
              }
              vars.thermal_cond_en[c]=k_en;
              vars.thermal_cond_sb[c]=k_sb;
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
WallModelDriver::CoalRegionHT::newton_solve(WallInfo& wi, double &TW_new, double &k_sb, double &k_en, double &dy_dep_sb, double &dy_dep_en, double &T_old, double &rad_q, double &R_wall, double &Emiss )
{
  // solver constants
  double d_tol    = 1e-15;
  double delta    = 1;
  int NIter       = 15;
  double f0       = 0.0;
  double f1       = 0.0;
  double T_max    = pow( rad_q/_sigma_constant, 0.25); // if k = 0.0;
  double net_q, TW_guess, TW_tmp, TW_old, R_sb, R_en, R_tot;
  R_sb = dy_dep_sb / k_sb;
  R_en = dy_dep_en / k_en;
  R_tot = R_wall + R_sb + R_en; // total thermal resistance
  // new solve
  TW_guess = T_old;
  TW_old = TW_guess-delta;
  net_q = rad_q - _sigma_constant * pow( TW_old, 4 );
  net_q = net_q > 0 ? net_q : 0;
  net_q *= Emiss;
  f0 = - TW_old + wi.T_inner + net_q * R_tot;
  TW_new = TW_guess+delta;
  net_q = rad_q - _sigma_constant * pow( TW_new, 4 );
  net_q *= Emiss;
  net_q = net_q>0 ? net_q : 0;
  f1 = - TW_new + wi.T_inner + net_q * R_tot;
  for ( int iterT=0; iterT < NIter; iterT++) {
    TW_tmp = TW_old;
    TW_old = TW_new;
    TW_new = TW_tmp - ( TW_new - TW_tmp )/( f1 - f0 ) * f0;
    TW_new = max( wi.T_inner , min( T_max, TW_new ) );
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
    f1    = - TW_new + wi.T_inner + net_q * R_tot;
  }
}
