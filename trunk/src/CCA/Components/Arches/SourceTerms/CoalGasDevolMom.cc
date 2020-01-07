#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/CoalGasDevolMom.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/CoalModels/PartVel.h>

//===========================================================================

using namespace std;
using namespace Uintah;

CoalGasDevolMom::CoalGasDevolMom( std::string src_name, vector<std::string> label_names, ArchesLabel* field_labels,  MaterialManagerP& materialManager, std::string type )
: SourceTermBase( src_name, field_labels->d_materialManager, label_names, type ),
  _field_labels(field_labels)
{
  _src_label = VarLabel::create( src_name, CCVariable<Vector>::getTypeDescription() );

  _source_grid_type = CCVECTOR_SRC;
}

CoalGasDevolMom::~CoalGasDevolMom()
{
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
CoalGasDevolMom::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb;

  db->require( "devol_model_name", _devol_model_name );

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term
//---------------------------------------------------------------------------
void
CoalGasDevolMom::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "CoalGasDevolMom::eval";
  Task* tsk = scinew Task(taskname, this, &CoalGasDevolMom::computeSource, timeSubStep);

  if (timeSubStep == 0) {
    tsk->computes(_src_label);
  } else {
    tsk->modifies(_src_label);
  }

  DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self();
  CoalModelFactory& modelFactory = CoalModelFactory::self();

  for (int iqn = 0; iqn < dqmomFactory.get_quad_nodes(); iqn++){

    std::string model_name = _devol_model_name;
    std::string node;
    std::stringstream out;
    out << iqn;
    node = out.str();
    model_name += "_qn";
    model_name += node;

    ModelBase& model = modelFactory.retrieve_model( model_name );

    const VarLabel* tempgasLabel_m = model.getGasSourceLabel();
    tsk->requires( Task::NewDW, tempgasLabel_m, Ghost::None, 0 );

    ArchesLabel::PartVelMap::const_iterator i = _field_labels->partVel.find(iqn);
    tsk->requires( Task::NewDW, i->second, Ghost::None, 0 );
  }

  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));

}
struct sumDevolGasSourceMom{
       sumDevolGasSourceMom(constCCVariable<double>& _qn_gas_devol,
                           constCCVariable<Vector> &_part_vel,
                           CCVariable<Vector>& _devolSrc) :
                           qn_gas_devol(_qn_gas_devol),
                           part_vel(_part_vel),
                           devolSrc(_devolSrc)
                           {  }

  void operator()(int i , int j, int k ) const {
   Vector part_vel_t = part_vel(i,j,k);
   devolSrc(i,j,k) += Vector(qn_gas_devol(i,j,k)*part_vel_t.x(),qn_gas_devol(i,j,k)*part_vel_t.y(),qn_gas_devol(i,j,k)*part_vel_t.z());
  }

  private:
   constCCVariable<double>& qn_gas_devol;
   constCCVariable<Vector>& part_vel;
   Vector part_vel_t;
   CCVariable<Vector>& devolSrc;


};
//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
void
CoalGasDevolMom::computeSource( const ProcessorGroup* pc,
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

    DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self();
    CoalModelFactory& modelFactory = CoalModelFactory::self();

    CCVariable<Vector> devolSrc;
    if ( timeSubStep == 0 ){
      new_dw->allocateAndPut( devolSrc, _src_label, matlIndex, patch );
      devolSrc.initialize(Vector(0.0,0.0,0.0));
    } else {
      new_dw->getModifiable( devolSrc, _src_label, matlIndex, patch );
      devolSrc.initialize(Vector(0.0,0.0,0.0));
    }


    for (int iqn = 0; iqn < dqmomFactory.get_quad_nodes(); iqn++){
      std::string model_name = _devol_model_name;
      std::string node;
      std::stringstream out;
      out << iqn;
      node = out.str();
      model_name += "_qn";
      model_name += node;

      ModelBase& model = modelFactory.retrieve_model( model_name );

      constCCVariable<double> qn_gas_devol;
      Vector momentum_devol_tmp;
      constCCVariable<Vector> partVel;
      const VarLabel* gasModelLabel = model.getGasSourceLabel();

      new_dw->get( qn_gas_devol, gasModelLabel, matlIndex, patch, gn, 0 );
      ArchesLabel::PartVelMap::const_iterator iter = _field_labels->partVel.find(iqn);
      new_dw->get(partVel, iter->second, matlIndex, patch, gn, 0);

      Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

      sumDevolGasSourceMom doSumDevolGasMom(qn_gas_devol, partVel, devolSrc);

      Uintah::parallel_for(range, doSumDevolGasMom);
    }
  }
}
//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
CoalGasDevolMom::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "CoalGasDevolMom::initialize";

  Task* tsk = scinew Task(taskname, this, &CoalGasDevolMom::initialize);

  tsk->computes(_src_label);

  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));

}
void
CoalGasDevolMom::initialize( const ProcessorGroup* pc,
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

  }
}




