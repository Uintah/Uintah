#include <Core/Grid/Task.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/WallHTModels/WallModelDriver.h>

using namespace Uintah; 
using namespace std; 

//_________________________________________
WallModelDriver::WallModelDriver( SimulationStateP& shared_state ) :
  _shared_state( shared_state )
{

  _matl_index = _shared_state->getArchesMaterial( 0 )->getDWIndex(); 

}

//_________________________________________
WallModelDriver::~WallModelDriver()
{
}

//_________________________________________
void
WallModelDriver::problemSetup( const ProblemSpecP& input_db ) 
{

  ProblemSpecP db = input_db; 

  db->getWithDefault( "temperature_label", _T_label_name, string( "temperature" ) ); 

  for ( ProblemSpecP db_model = db->findBlock( "model" ); db_model != 0; db_model = db_model->findNextBlock( "model" ) ){

    std::string type = "not_assigned"; 
    db_model->getAttribute("type", type); 

    if ( type == "simple_wall" ) {

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

  if ( !check_varlabels() ){ 
    throw InvalidValue("Error: One of the varlabels for the wall model was not found.", __FILE__, __LINE__);
  } 

  task->modifies(_T_label);

  if ( time_subset == 0 ){ 

    task->requires(Task::OldDW , _cellType_label , Ghost::None , 0 );
    task->requires(Task::OldDW , _HF_E_label     , Ghost::None , 0 );
    task->requires(Task::OldDW , _HF_W_label     , Ghost::None , 0 );
    task->requires(Task::OldDW , _HF_N_label     , Ghost::None , 0 );
    task->requires(Task::OldDW , _HF_S_label     , Ghost::None , 0 );
    task->requires(Task::OldDW , _HF_T_label     , Ghost::None , 0 );
    task->requires(Task::OldDW , _HF_B_label     , Ghost::None , 0 );
    task->modifies(_T_label); 

  } else { 

    task->requires(Task::NewDW , _cellType_label , Ghost::None , 0 );
    task->requires(Task::NewDW , _HF_E_label     , Ghost::None , 0 );
    task->requires(Task::NewDW , _HF_W_label     , Ghost::None , 0 );
    task->requires(Task::NewDW , _HF_N_label     , Ghost::None , 0 );
    task->requires(Task::NewDW , _HF_S_label     , Ghost::None , 0 );
    task->requires(Task::NewDW , _HF_T_label     , Ghost::None , 0 );
    task->requires(Task::NewDW , _HF_B_label     , Ghost::None , 0 );
    task->modifies(_T_label); 

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
  const Level* level = getLevel(patches);
  
  DataWarehouse* which_dw; 
  if ( time_subset == 0 ) { 
    which_dw = old_dw; 
  } else { 
    which_dw = new_dw; 
  }

  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    HTVariables* vars;

    new_dw->getModifiable( vars->T, _T_label, _matl_index, patch ); 
    which_dw->get(   vars->celltype , _cellType_label , _matl_index , patch , Ghost::None, 0 );
    which_dw->get(   vars->hf_e     , _HF_E_label     , _matl_index , patch , Ghost::None, 0 );
    which_dw->get(   vars->hf_w     , _HF_W_label     , _matl_index , patch , Ghost::None, 0 );
    which_dw->get(   vars->hf_n     , _HF_N_label     , _matl_index , patch , Ghost::None, 0 );
    which_dw->get(   vars->hf_s     , _HF_S_label     , _matl_index , patch , Ghost::None, 0 );
    which_dw->get(   vars->hf_t     , _HF_T_label     , _matl_index , patch , Ghost::None, 0 );
    which_dw->get(   vars->hf_b     , _HF_B_label     , _matl_index , patch , Ghost::None, 0 );

    std::vector<WallModelDriver::HTModelBase>::iterator iter; 

    //loop over all HT models
    for ( iter = _all_ht_models.begin(); iter != _all_ht_models.end(); iter++ ){

      iter->computeHT( patch, vars );

    }
  }
}

//_________________________________________
void 
WallModelDriver::sched_doWallHT_alltoall( const LevelP& level, SchedulerP& sched, const int time_subset )
{

  Task* task = scinew Task( "WallModelDriver::doWallHT_alltoall", this, 
                           &WallModelDriver::doWallHT_alltoall, time_subset ); 

  _T_label        = VarLabel::find( _T_label_name );
  _cellType_label = VarLabel::find( "cellType" );
  _HF_E_label     = VarLabel::find( "new_radiationFluxE" );
  _HF_W_label     = VarLabel::find( "new_radiationFluxW" );
  _HF_N_label     = VarLabel::find( "new_radiationFluxN" );
  _HF_S_label     = VarLabel::find( "new_radiationFluxS" );
  _HF_T_label     = VarLabel::find( "new_radiationFluxT" );
  _HF_B_label     = VarLabel::find( "new_radiationFluxB" );

  if ( !check_varlabels() ){ 
    throw InvalidValue("Error: One of the varlabels for the wall model was not found.", __FILE__, __LINE__);
  } 

  task->modifies(_T_label);

  if ( time_subset == 0 ){ 

    task->requires(Task::OldDW , _cellType_label , Ghost::AroundNodes , SHRT_MAX);
    task->requires(Task::OldDW , _T_label        , Ghost::AroundNodes , SHRT_MAX);
    task->requires(Task::OldDW , _HF_E_label     , Ghost::AroundNodes , SHRT_MAX);
    task->requires(Task::OldDW , _HF_W_label     , Ghost::AroundNodes , SHRT_MAX);
    task->requires(Task::OldDW , _HF_N_label     , Ghost::AroundNodes , SHRT_MAX);
    task->requires(Task::OldDW , _HF_S_label     , Ghost::AroundNodes , SHRT_MAX);
    task->requires(Task::OldDW , _HF_T_label     , Ghost::AroundNodes , SHRT_MAX);
    task->requires(Task::OldDW , _HF_B_label     , Ghost::AroundNodes , SHRT_MAX);

  } else { 

    task->requires(Task::NewDW , _cellType_label , Ghost::AroundNodes , SHRT_MAX);
    task->requires(Task::NewDW , _T_label        , Ghost::AroundNodes , SHRT_MAX);
    task->requires(Task::NewDW , _HF_E_label     , Ghost::AroundNodes , SHRT_MAX);
    task->requires(Task::NewDW , _HF_W_label     , Ghost::AroundNodes , SHRT_MAX);
    task->requires(Task::NewDW , _HF_N_label     , Ghost::AroundNodes , SHRT_MAX);
    task->requires(Task::NewDW , _HF_S_label     , Ghost::AroundNodes , SHRT_MAX);
    task->requires(Task::NewDW , _HF_T_label     , Ghost::AroundNodes , SHRT_MAX);
    task->requires(Task::NewDW , _HF_B_label     , Ghost::AroundNodes , SHRT_MAX);

  } 

  vector<const Patch*>my_patches;

  my_patches.push_back(level->getPatchFromPoint(Point(0.0,0.0,0.0), false));

  PatchSet *my_each_patch = scinew PatchSet();
  my_each_patch->addReference();
  my_each_patch->addEach(my_patches);
  
  sched->addTask(task, my_each_patch, _shared_state->allArchesMaterials());
  
}

//_________________________________________
void 
WallModelDriver::doWallHT_alltoall( const ProcessorGroup* my_world,
                                    const PatchSubset* patches, 
                                    const MaterialSubset* matls, 
                                    DataWarehouse* old_dw, 
                                    DataWarehouse* new_dw, 
                                    const int time_subset )
{
  const Level* level = getLevel(patches);

  // Determine the size of the domain.
  IntVector domainLo, domainHi;
  IntVector domainLo_EC, domainHi_EC;
  
  level->findInteriorCellIndexRange(domainLo, domainHi);     // excluding extraCells
  level->findCellIndexRange(domainLo_EC, domainHi_EC);       // including extraCells
  
  CCVariable<double> T; 
  constCCVariable<int>    celltype;
  constCCVariable<double> const_T;
  constCCVariable<double> hf_e;
  constCCVariable<double> hf_w;
  constCCVariable<double> hf_n;
  constCCVariable<double> hf_s;
  constCCVariable<double> hf_t;
  constCCVariable<double> hf_b;

  DataWarehouse* which_dw; 
  if ( time_subset == 0 ) { 
    which_dw = old_dw; 
  } else { 
    which_dw = new_dw; 
  }

  which_dw->getRegion(   celltype , _cellType_label , _matl_index , level , domainLo_EC , domainHi_EC);
  which_dw->getRegion(   const_T  , _T_label        , _matl_index , level , domainLo_EC , domainHi_EC);
  which_dw->getRegion(   hf_e     , _HF_E_label     , _matl_index , level , domainLo_EC , domainHi_EC);
  which_dw->getRegion(   hf_w     , _HF_W_label     , _matl_index , level , domainLo_EC , domainHi_EC);
  which_dw->getRegion(   hf_n     , _HF_N_label     , _matl_index , level , domainLo_EC , domainHi_EC);
  which_dw->getRegion(   hf_s     , _HF_S_label     , _matl_index , level , domainLo_EC , domainHi_EC);
  which_dw->getRegion(   hf_t     , _HF_T_label     , _matl_index , level , domainLo_EC , domainHi_EC);
  which_dw->getRegion(   hf_b     , _HF_B_label     , _matl_index , level , domainLo_EC , domainHi_EC);

  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);

    new_dw->getModifiable( T, _T_label, _matl_index, patch ); 

    // actually perform the ht calculation 
    // pass fluxes, T, const_T
    // return new T


  }
}

// ********----- DERIVED HT MODELS --------********
//
// Simple HT model
//----------------------------------
WallModelDriver::SimpleHT::SimpleHT(){}; 

//----------------------------------
WallModelDriver::SimpleHT::~SimpleHT(){}; 

//----------------------------------
void 
WallModelDriver::SimpleHT::problemSetup( const ProblemSpecP& input_db ){ 

  ProblemSpecP db = input_db; 

  db->require("k", _k);
  db->require("wall_thickness", _dy);
  db->require("tube_size_T", _T_inner); 

} 

//----------------------------------
void 
WallModelDriver::SimpleHT::computeHT( const Patch* patch, HTVariables* vars ){ 

  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  
  for( vector<Patch::FaceType>::const_iterator face_iter = bf.begin(); face_iter != bf.end(); ++face_iter ){

    Patch::FaceType face = *face_iter;
    IntVector offset = patch->faceDirection(face);
    CellIterator cell_iter = patch->getFaceIterator(face, Patch::InteriorFaceCells);

    constCCVariable<double> q; 
    switch (face) {
      case Patch::xminus:
        q = vars->hf_e;
        break; 
      case Patch::xplus:
        q = vars->hf_w;
        break; 
      case Patch::yminus:
        q = vars->hf_n;
        break; 
      case Patch::yplus:
        q = vars->hf_s;
        break; 
      case Patch::zminus:
        q = vars->hf_t;
        break; 
      case Patch::zplus:
        q = vars->hf_b;
        break; 
      default: 
        break; 
    }
    
    for(;!cell_iter.done(); cell_iter++){

      IntVector c = *cell_iter;        //this is the interior cell 
      IntVector adj = c + offset;      //this is the cell IN the wall 

      vars->T[c] = _T_inner - q[c] * _dy/_k; 

    }
  }
}


