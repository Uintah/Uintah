#include <CCA/Components/Arches/SourceTerms/MomentumDragSrc.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/DragModel.h>
#include <CCA/Components/Arches/CoalModels/HeatTransfer.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>

//===========================================================================

using namespace std;
using namespace Uintah;

MomentumDragSrc::MomentumDragSrc( std::string src_name, MaterialManagerP& materialManager,
                                 vector<std::string> req_label_names, std::string type )
: SourceTermBase(src_name, materialManager, req_label_names, type)
{
  _src_label = VarLabel::create( src_name, CCVariable<Vector>::getTypeDescription() );

  _source_grid_type = CCVECTOR_SRC;
}

MomentumDragSrc::~MomentumDragSrc()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
MomentumDragSrc::problemSetup(const ProblemSpecP& inputdb)
{
  ProblemSpecP db = inputdb;

  db->get("N", _N); //number of quad nodes
  db->getWithDefault( "base_x_label", _base_x_drag, "none" );
  db->getWithDefault( "base_y_label", _base_y_drag, "none" );
  db->getWithDefault( "base_z_label", _base_z_drag, "none" );
}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term
//---------------------------------------------------------------------------
void
MomentumDragSrc::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "MomentumDragSrc::eval";
  Task* tsk = scinew Task(taskname, this, &MomentumDragSrc::computeSource, timeSubStep);

  if (timeSubStep == 0 ) {
    tsk->computes(_src_label);
  } else {
    tsk->modifies(_src_label);
  }

  for ( int i = 0; i<_N; i++ ) {
    std::stringstream out;
    out << i;
    std::string node = out.str();

    if ( _base_x_drag != "none" ) {
      std::string model_name;
      model_name = "gas_";
      model_name += _base_x_drag;
      model_name += "_";
      model_name += node;
      const VarLabel* tempLabel = VarLabel::find( model_name );
      tsk->requires( Task::OldDW, tempLabel, Ghost::None, 0);
    }

    if ( _base_y_drag != "none" ) {
      std::string model_name;
      model_name = "gas_";
      model_name += _base_y_drag;
      model_name += "_";
      model_name += node;
      const VarLabel* tempLabel = VarLabel::find( model_name );
      tsk->requires( Task::OldDW, tempLabel, Ghost::None, 0);
    }

    if ( _base_z_drag != "none" ) {
      std::string model_name;
      model_name = "gas_";
      model_name += _base_z_drag;
      model_name += "_";
      model_name += node;
      const VarLabel* tempLabel = VarLabel::find( model_name );
      tsk->requires( Task::OldDW, tempLabel, Ghost::None, 0);
    }
  }

  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));
}

struct sumGasDrag{
       sumGasDrag( std::string _base_x_drag,
                   std::string _base_y_drag,
                   std::string _base_z_drag,
                   constCCVariable<double> &_gas_xdrag,
                   constCCVariable<double> &_gas_ydrag,
                   constCCVariable<double> &_gas_zdrag,
                   CCVariable<Vector> &_dragSrc) :
                   gas_xdrag(_gas_xdrag),
                   gas_ydrag(_gas_ydrag),
                   gas_zdrag(_gas_zdrag),
                   dragSrc(_dragSrc)
                   { sumX=(_base_x_drag != "none");
                     sumY=(_base_y_drag != "none");
                     sumZ=(_base_z_drag != "none");}

      void operator()(int i , int j, int k ) const {
         dragSrc(i,j,k) += Vector(sumX ? gas_xdrag(i,j,k) : 0.0,sumY ? gas_ydrag(i,j,k): 0.0, sumZ ? gas_zdrag(i,j,k): 0.0);
      }

     private:
      bool sumX;
      bool sumY;
      bool sumZ;
      constCCVariable<double> &gas_xdrag;
      constCCVariable<double> &gas_ydrag;
      constCCVariable<double> &gas_zdrag;
      CCVariable<Vector> &dragSrc;
};
//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
void
MomentumDragSrc::computeSource( const ProcessorGroup* pc,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw,
                               int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<Vector> dragSrc;
    if ( new_dw->exists(_src_label, matlIndex, patch ) ){
      new_dw->getModifiable( dragSrc, _src_label, matlIndex, patch );
      dragSrc.initialize(Vector(0.,0.,0.));
    } else {
      new_dw->allocateAndPut( dragSrc, _src_label, matlIndex, patch );
      dragSrc.initialize(Vector(0.,0.,0.));
    }

    for (int i = 0; i < _N; i++ ) {
      constCCVariable<double> gas_xdrag;
      constCCVariable<double> gas_ydrag;
      constCCVariable<double> gas_zdrag;
      Vector gas_drag;
      std::stringstream out;
      out << i;
      std::string node = out.str();

      if ( _base_x_drag != "none" ) {
        std::string model_name;
        model_name = "gas_";
        model_name += _base_x_drag;
        model_name += "_";
        model_name += node;
        const VarLabel* tempLabel = VarLabel::find( model_name );
        old_dw->get( gas_xdrag, tempLabel, matlIndex, patch, gn, 0);
      }

      if ( _base_y_drag != "none" ) {
        std::string model_name;
        model_name = "gas_";
        model_name += _base_y_drag;
        model_name += "_";
        model_name += node;
        const VarLabel* tempLabel = VarLabel::find( model_name );
        old_dw->get( gas_ydrag, tempLabel, matlIndex, patch, gn, 0);
      }

      if ( _base_z_drag != "none" ) {
        std::string model_name;
        model_name = "gas_";
        model_name += _base_z_drag;
        model_name += "_";
        model_name += node;
        const VarLabel* tempLabel = VarLabel::find( model_name );
        old_dw->get( gas_zdrag, tempLabel, matlIndex, patch, gn, 0);
      }

  Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

  sumGasDrag doSumGas(_base_x_drag,
                      _base_y_drag,
                      _base_z_drag,
                      gas_xdrag,
                      gas_ydrag,
                      gas_zdrag,
                      dragSrc);

  Uintah::parallel_for(range, doSumGas);
    }
  }
}
//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
MomentumDragSrc::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "MomentumDragSrc::initialize";

  Task* tsk = scinew Task(taskname, this, &MomentumDragSrc::initialize);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter);
  }

  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));

}
//---------------------------------------------------------------------------
// Method: initialization
//---------------------------------------------------------------------------
void
MomentumDragSrc::initialize( const ProcessorGroup* pc,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<Vector> src;

    new_dw->allocateAndPut( src, _src_label, matlIndex, patch );

    src.initialize(Vector(0.0,0.0,0.0));

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
      CCVariable<double> tempVar;
      new_dw->allocateAndPut(tempVar, *iter, matlIndex, patch );
    }
  }
}




