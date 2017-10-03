#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/CoalGasDevol.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
//#include <CCA/Components/Arches/CoalModels/KobayashiSarofimDevol.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>

//===========================================================================

using namespace std;
using namespace Uintah;

CoalGasDevol::CoalGasDevol( std::string src_name, vector<std::string> label_names, SimulationStateP& shared_state, std::string type )
: SourceTermBase( src_name, shared_state, label_names, type )
{
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() );
}

CoalGasDevol::~CoalGasDevol()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
CoalGasDevol::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb;

  db->require( "devol_model_name", _devol_model_name );

  _source_grid_type = CC_SRC;

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term
//---------------------------------------------------------------------------
void
CoalGasDevol::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "CoalGasDevol::eval";
  Task* tsk = scinew Task(taskname, this, &CoalGasDevol::computeSource, timeSubStep);

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

  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
struct sumDevolGasSource{
       sumDevolGasSource(constCCVariable<double>& _qn_gas_devol,
                           CCVariable<double>& _devolSrc) :
#ifdef UINTAH_ENABLE_KOKKOS
                           qn_gas_devol(_qn_gas_devol.getKokkosView()),
                           devolSrc(_devolSrc.getKokkosView())
#else
                           qn_gas_devol(_qn_gas_devol),
                           devolSrc(_devolSrc)
#endif
                           {  }

  void operator()(int i , int j, int k ) const {
   devolSrc(i,j,k) += qn_gas_devol(i,j,k);
  }

  private:
#ifdef UINTAH_ENABLE_KOKKOS
   KokkosView3<const double> qn_gas_devol;
   KokkosView3<double>  devolSrc;
#else
   constCCVariable<double>& qn_gas_devol;
   CCVariable<double>& devolSrc;
#endif
};
//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
void
CoalGasDevol::computeSource( const ProcessorGroup* pc,
                   const PatchSubset* patches,
                   const MaterialSubset* matls,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw,
                   int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    //Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex();

    DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self();
    CoalModelFactory& modelFactory = CoalModelFactory::self();

    CCVariable<double> devolSrc;
    if ( timeSubStep == 0 ){
      new_dw->allocateAndPut( devolSrc, _src_label, matlIndex, patch );
      devolSrc.initialize(0.0);
    } else {
      new_dw->getModifiable( devolSrc, _src_label, matlIndex, patch );
      devolSrc.initialize(0.0);
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
      const VarLabel* gasModelLabel = model.getGasSourceLabel();

      new_dw->get( qn_gas_devol, gasModelLabel, matlIndex, patch, gn, 0 );

      Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

      sumDevolGasSource doSumDevolGas(qn_gas_devol,
                                      devolSrc);

      Uintah::parallel_for(range, doSumDevolGas);
    }
  }
}
//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
CoalGasDevol::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "CoalGasDevol::initialize";

  Task* tsk = scinew Task(taskname, this, &CoalGasDevol::initialize);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter);
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void
CoalGasDevol::initialize( const ProcessorGroup* pc,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex();

    CCVariable<double> src;

    new_dw->allocateAndPut( src, _src_label, matlIndex, patch );

    src.initialize(0.0);

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
      CCVariable<double> tempVar;
      new_dw->allocateAndPut(tempVar, *iter, matlIndex, patch );
    }
  }
}




