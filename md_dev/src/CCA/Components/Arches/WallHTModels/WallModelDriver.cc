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

  _T_copy_label          = VarLabel::create( "T_copy", CC_double ); 

}

//_________________________________________
WallModelDriver::~WallModelDriver()
{

  std::vector<WallModelDriver::HTModelBase*>::iterator iter; 
  for ( iter = _all_ht_models.begin(); iter != _all_ht_models.end(); iter++ ){

    delete *iter;

  }

  VarLabel::destroy( _T_copy_label ); 

}

//_________________________________________
void
WallModelDriver::problemSetup( const ProblemSpecP& input_db ) 
{

  ProblemSpecP db = input_db; 

  db->getWithDefault( "temperature_label", _T_label_name, "temperature" ); 

  bool found_radiation_model = false;  
  if ( db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources") ){ 

    ProblemSpecP sources_db = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources");

    for (ProblemSpecP src_db = sources_db->findBlock("src");
          src_db !=0; src_db = src_db->findNextBlock("src")){

      string type; 
      src_db->getAttribute("type", type); 

      if ( type == "do_radiation" ){ 

        src_db->getWithDefault("calc_frequency", _calc_freq,3);;  //default matches the default of the radiation solvers
        found_radiation_model = true; 

      } else if ( type == "rmcrt" ) { 

        src_db->getWithDefault("calc_frequency", _calc_freq,3);;  //default matches the default of the radiation solvers
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


  for ( ProblemSpecP db_model = db->findBlock( "model" ); db_model != 0; db_model = db_model->findNextBlock( "model" ) ){

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

    } else { 

      throw InvalidValue("Error: Wall Heat Transfer model not recognized.", __FILE__, __LINE__);

    } 
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
  _HF_E_label     = VarLabel::find( "new_radiationFluxE" );
  _HF_W_label     = VarLabel::find( "new_radiationFluxW" );
  _HF_N_label     = VarLabel::find( "new_radiationFluxN" );
  _HF_S_label     = VarLabel::find( "new_radiationFluxS" );
  _HF_T_label     = VarLabel::find( "new_radiationFluxT" );
  _HF_B_label     = VarLabel::find( "new_radiationFluxB" );
  _cc_vel_label   = VarLabel::find( "CCVelocity" ); 

  if ( !check_varlabels() ){ 
    throw InvalidValue("Error: One of the varlabels for the wall model was not found.", __FILE__, __LINE__);
  } 

  task->modifies(_T_label);

  if ( time_subset == 0 ) { 

    task->computes( _T_copy_label ); 

    task->requires( Task::OldDW, _cc_vel_label, Ghost::None, 0 ); 
    task->requires( Task::OldDW , _T_label        , Ghost::None , 0 ); 

    task->requires( Task::NewDW , _cellType_label , Ghost::None , 0 );
    task->requires( Task::NewDW , _HF_E_label     , Ghost::None , 0 );
    task->requires( Task::NewDW , _HF_W_label     , Ghost::None , 0 );
    task->requires( Task::NewDW , _HF_N_label     , Ghost::None , 0 );
    task->requires( Task::NewDW , _HF_S_label     , Ghost::None , 0 );
    task->requires( Task::NewDW , _HF_T_label     , Ghost::None , 0 );
    task->requires( Task::NewDW , _HF_B_label     , Ghost::None , 0 );

  } else { 

    task->requires( Task::NewDW, _T_copy_label, Ghost::None, 0 );

  } 
  
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

    const Level* level = getLevel(patches);
    
    const Patch* patch = patches->get(p);
    HTVariables vars;

    // Note: The local T_copy is necessary because boundary conditions are being applied 
    // in the table lookup to T based on the conditions for the independent variables. These 
    // BCs are being applied regardless of the type of wall temperature model. 

    if( time_subset == 0 && timestep % _calc_freq == 0 ){

      // actually compute the wall HT model 

      old_dw->get( vars.T_old  , _T_label      , _matl_index , patch , Ghost::None , 0 );
      old_dw->get( vars.cc_vel , _cc_vel_label , _matl_index , patch , Ghost::None , 0 );

      new_dw->getModifiable(  vars.T, _T_label, _matl_index   , patch ); 
      new_dw->allocateAndPut( vars.T_copy     , _T_copy_label , _matl_index, patch ); 

      new_dw->get(   vars.celltype , _cellType_label , _matl_index , patch , Ghost::None, 0 );
      new_dw->get(   vars.incident_hf_e     , _HF_E_label     , _matl_index , patch , Ghost::None, 0 );
      new_dw->get(   vars.incident_hf_w     , _HF_W_label     , _matl_index , patch , Ghost::None, 0 );
      new_dw->get(   vars.incident_hf_n     , _HF_N_label     , _matl_index , patch , Ghost::None, 0 );
      new_dw->get(   vars.incident_hf_s     , _HF_S_label     , _matl_index , patch , Ghost::None, 0 );
      new_dw->get(   vars.incident_hf_t     , _HF_T_label     , _matl_index , patch , Ghost::None, 0 );
      new_dw->get(   vars.incident_hf_b     , _HF_B_label     , _matl_index , patch , Ghost::None, 0 );

      std::vector<WallModelDriver::HTModelBase*>::iterator iter; 

      for ( iter = _all_ht_models.begin(); iter != _all_ht_models.end(); iter++ ){

        (*iter)->computeHT( patch, vars );

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
      constCCVariable<double> T_old; 
      constCCVariable<int> cell_type; 

      old_dw->get( T_old             , _T_label        , _matl_index , patch    , Ghost::None , 0 );
      new_dw->get( cell_type         , _cellType_label , _matl_index , patch    , Ghost::None , 0 );
      new_dw->getModifiable(  T      , _T_label        , _matl_index , patch );
      new_dw->allocateAndPut( T_copy , _T_copy_label   , _matl_index , patch );

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

    } else { 

      // no ht solve for RK steps > 0: 
      // 1) T_copy (NewDW) should have the BC's from previous solution
      // 2) copy BC information from T_copy (NewDW) -> T to preserve BCs

      CCVariable<double> T; 
      constCCVariable<double> T_old; 
      constCCVariable<int> cell_type; 

      new_dw->getModifiable( T , _T_label        , _matl_index , patch );
      new_dw->get( T_old       , _T_copy_label   , _matl_index , patch    , Ghost::None , 0 );
      new_dw->get( cell_type   , _cellType_label , _matl_index , patch    , Ghost::None , 0 );

      std::vector<WallModelDriver::HTModelBase*>::iterator iter; 

      for ( iter = _all_ht_models.begin(); iter != _all_ht_models.end(); iter++ ){

        (*iter)->copySolution( patch, T, T_old, cell_type ); 

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
  db->getWithDefault( "relaxation_coef", _relax, 1.0); 

} 

//----------------------------------
void 
WallModelDriver::SimpleHT::computeHT( const Patch* patch, HTVariables& vars ){ 

  double T_wall, net_q;
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  
  for( vector<Patch::FaceType>::const_iterator face_iter = bf.begin(); face_iter != bf.end(); ++face_iter ){

    Patch::FaceType face = *face_iter;
    IntVector offset = patch->faceDirection(face);
    CellIterator cell_iter = patch->getFaceIterator(face, Patch::InteriorFaceCells);

    constCCVariable<double> q; 
    switch (face) {
      case Patch::xminus:
          q = vars.incident_hf_w;
        break; 
      case Patch::xplus:
          q = vars.incident_hf_e;
        break; 
      case Patch::yminus:
          q = vars.incident_hf_s; 
        break; 
      case Patch::yplus:
          q = vars.incident_hf_n;
        break; 
      case Patch::zminus:
          q = vars.incident_hf_b;
        break; 
      case Patch::zplus:
          q = vars.incident_hf_t;
        break; 
      default: 
        break; 
    }
    
    for(;!cell_iter.done(); cell_iter++){

      IntVector c = *cell_iter;        //this is the interior cell 
      IntVector adj = c + offset;      //this is the cell IN the wall 

      if ( vars.celltype[c + offset] == BoundaryCondition_new::WALL ){ 

          net_q = q[c] - _sigma_constant * pow( vars.T_old[adj], 4 );
          net_q = net_q > 0 ? net_q : 0;
          T_wall = _T_inner + net_q * _dy / _k;

          T_wall = T_wall > _T_max ? _T_max : T_wall;
          T_wall = T_wall < _T_min ? _T_min : T_wall;

          vars.T[adj] = ( 1.0 - _relax ) * vars.T_old[adj] + ( _relax ) * T_wall;

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

  for (ProblemSpecP r_db = db->findBlock("region");
      r_db !=0; r_db = r_db->findNextBlock("region")){

    WallInfo info; 
    ProblemSpecP geometry_db = r_db->findBlock("geom_object");
    GeometryPieceFactory::create( geometry_db, info.geometry ); 
    r_db->require("k", info.k);
    r_db->require("wall_thickness", info.dy);
    r_db->require("tube_side_T", info.T_inner); 
    r_db->require("max_TW", info.max_TW);
    r_db->require("min_TW", info.min_TW);
    r_db->getWithDefault("relaxation_coef", info.relax, 1.0);
    _regions.push_back( info ); 

  }
} 

//----------------------------------
void 
WallModelDriver::RegionHT::computeHT( const Patch* patch, HTVariables& vars ){ 

  int num;
  double TW1, TW0, Tmax, Tmin, net_q, error, initial_error;
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
        constCCVariable<double> q; 

        for (; !iter.done(); iter++){ 

          IntVector c = *iter; 

          // is the point inside of region as defined by geom? 
          if ( in_or_out( c, geom, patch ) ){ 

            for ( int i = 0; i < 6; i++ ){ 

              // is the current cell a solid? 
              if ( vars.celltype[c] == BoundaryCondition_new::WALL ||
                   vars.celltype[c] == BoundaryCondition_new::INTRUSION ){ 

                // is the neighbor in the current direction a flow? 
                if ( patch->containsCell( c + _d[i] ) ){ 
                  if ( vars.celltype[c + _d[i]] == FLOW ){ 
                    
                    q = get_flux( i, vars );  
                    TW0 = vars.T_old[c];
                    net_q = q[c+_d[i]] - _sigma_constant * pow( TW0 , 4 );
                    net_q = net_q > 0 ? net_q : 0;
                    TW1 = wi.T_inner + net_q * wi.dy / wi.k;
                   
                    if( TW1 < TW0 ){
                        Tmax = TW0>wi.max_TW ? wi.max_TW : TW0;
                        Tmin = TW1<wi.min_TW ? wi.min_TW : TW1;
                    }
                    else{
                        Tmax = TW1>wi.max_TW ? wi.max_TW : TW1;
                        Tmin = TW0<wi.min_TW ? wi.min_TW : TW0;
                    }

                    initial_error = fabs( TW0 - TW1 ) / TW1;
                    
                    vars.T[c] = TW0;

                    error = 1;
                    num = 0;

                    while ( initial_error > _init_tol && error > _tol && num < _max_it ){

                        TW0 = ( Tmax + Tmin ) / 2.0; 
                        net_q = q[c+_d[i]] - _sigma_constant * pow( TW0, 4 );
                        net_q = net_q>0 ? net_q : 0;
                        TW1 = wi.T_inner + net_q * wi.dy / wi.k;

                        if( TW1 < TW0 ) {

                            Tmax = TW0;

                        } else {

                            Tmin = TW0;

                        }
                        
                        error = fabs( Tmax - Tmin ) / TW0;

                        num++;

                    }

                    vars.T[c] = (1-wi.relax)*vars.T_old[c]+wi.relax*TW0; 

                  } 
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

// removing for the time being
////_________________________________________
//void 
//WallModelDriver::sched_doWallHT_alltoall( const LevelP& level, SchedulerP& sched, const int time_subset )
//{
//
//  Task* task = scinew Task( "WallModelDriver::doWallHT_alltoall", this, 
//                           &WallModelDriver::doWallHT_alltoall, time_subset ); 
//
//  _T_label        = VarLabel::find( _T_label_name );
//  _cellType_label = VarLabel::find( "cellType" );
//  _HF_E_label     = VarLabel::find( "new_radiationFluxE" );
//  _HF_W_label     = VarLabel::find( "new_radiationFluxW" );
//  _HF_N_label     = VarLabel::find( "new_radiationFluxN" );
//  _HF_S_label     = VarLabel::find( "new_radiationFluxS" );
//  _HF_T_label     = VarLabel::find( "new_radiationFluxT" );
//  _HF_B_label     = VarLabel::find( "new_radiationFluxB" );
//
//  if ( !check_varlabels() ){ 
//    throw InvalidValue("Error: One of the varlabels for the wall model was not found.", __FILE__, __LINE__);
//  } 
//
//  task->modifies(_T_label);
//
//  if ( time_subset == 0 ){ 
//
//    task->requires(Task::OldDW , _cellType_label , Ghost::AroundNodes , SHRT_MAX);
//    task->requires(Task::OldDW , _T_label        , Ghost::AroundNodes , SHRT_MAX);
//    task->requires(Task::OldDW , _HF_E_label     , Ghost::AroundNodes , SHRT_MAX);
//    task->requires(Task::OldDW , _HF_W_label     , Ghost::AroundNodes , SHRT_MAX);
//    task->requires(Task::OldDW , _HF_N_label     , Ghost::AroundNodes , SHRT_MAX);
//    task->requires(Task::OldDW , _HF_S_label     , Ghost::AroundNodes , SHRT_MAX);
//    task->requires(Task::OldDW , _HF_T_label     , Ghost::AroundNodes , SHRT_MAX);
//    task->requires(Task::OldDW , _HF_B_label     , Ghost::AroundNodes , SHRT_MAX);
//
//  } else { 
//
//    task->requires(Task::NewDW , _cellType_label , Ghost::AroundNodes , SHRT_MAX);
//    task->requires(Task::NewDW , _T_label        , Ghost::AroundNodes , SHRT_MAX);
//    task->requires(Task::NewDW , _HF_E_label     , Ghost::AroundNodes , SHRT_MAX);
//    task->requires(Task::NewDW , _HF_W_label     , Ghost::AroundNodes , SHRT_MAX);
//    task->requires(Task::NewDW , _HF_N_label     , Ghost::AroundNodes , SHRT_MAX);
//    task->requires(Task::NewDW , _HF_S_label     , Ghost::AroundNodes , SHRT_MAX);
//    task->requires(Task::NewDW , _HF_T_label     , Ghost::AroundNodes , SHRT_MAX);
//    task->requires(Task::NewDW , _HF_B_label     , Ghost::AroundNodes , SHRT_MAX);
//
//  } 
//
//  vector<const Patch*>my_patches;
//
//  my_patches.push_back(level->getPatchFromPoint(Point(0.0,0.0,0.0), false));
//
//  PatchSet *my_each_patch = scinew PatchSet();
//  my_each_patch->addReference();
//  my_each_patch->addEach(my_patches);
//  
//  sched->addTask(task, my_each_patch, _shared_state->allArchesMaterials());
//  
//}
//
////_________________________________________
//void 
//WallModelDriver::doWallHT_alltoall( const ProcessorGroup* my_world,
//                                    const PatchSubset* patches, 
//                                    const MaterialSubset* matls, 
//                                    DataWarehouse* old_dw, 
//                                    DataWarehouse* new_dw, 
//                                    const int time_subset )
//{
//  const Level* level = getLevel(patches);
//
//  // Determine the size of the domain.
//  IntVector domainLo, domainHi;
//  IntVector domainLo_EC, domainHi_EC;
//  
//  level->findInteriorCellIndexRange(domainLo, domainHi);     // excluding extraCells
//  level->findCellIndexRange(domainLo_EC, domainHi_EC);       // including extraCells
//  
//  CCVariable<double> T; 
//  constCCVariable<int>    celltype;
//  constCCVariable<double> const_T;
//  constCCVariable<double> hf_e;
//  constCCVariable<double> hf_w;
//  constCCVariable<double> hf_n;
//  constCCVariable<double> hf_s;
//  constCCVariable<double> hf_t;
//  constCCVariable<double> hf_b;
//
//  DataWarehouse* which_dw; 
//  if ( time_subset == 0 ) { 
//    which_dw = old_dw; 
//  } else { 
//    which_dw = new_dw; 
//  }
//
//  which_dw->getRegion(   celltype , _cellType_label , _matl_index , level , domainLo_EC , domainHi_EC);
//  which_dw->getRegion(   const_T  , _T_label        , _matl_index , level , domainLo_EC , domainHi_EC);
//  which_dw->getRegion(   hf_e     , _HF_E_label     , _matl_index , level , domainLo_EC , domainHi_EC);
//  which_dw->getRegion(   hf_w     , _HF_W_label     , _matl_index , level , domainLo_EC , domainHi_EC);
//  which_dw->getRegion(   hf_n     , _HF_N_label     , _matl_index , level , domainLo_EC , domainHi_EC);
//  which_dw->getRegion(   hf_s     , _HF_S_label     , _matl_index , level , domainLo_EC , domainHi_EC);
//  which_dw->getRegion(   hf_t     , _HF_T_label     , _matl_index , level , domainLo_EC , domainHi_EC);
//  which_dw->getRegion(   hf_b     , _HF_B_label     , _matl_index , level , domainLo_EC , domainHi_EC);
//
//  //patch loop
//  for (int p=0; p < patches->size(); p++){
//
//    const Patch* patch = patches->get(p);
//
//    new_dw->getModifiable( T, _T_label, _matl_index, patch ); 
//
//    // actually perform the ht calculation 
//    // pass fluxes, T, const_T
//    // return new T
//
//
//  }
//}
