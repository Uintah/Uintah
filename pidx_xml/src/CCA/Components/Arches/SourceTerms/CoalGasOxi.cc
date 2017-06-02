#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/CoalGasOxi.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CharOxidationShaddix.h>
#include <CCA/Components/Arches/CoalModels/CharOxidationSmith.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>

//===========================================================================

using namespace std;
using namespace Uintah;

CoalGasOxi::CoalGasOxi( std::string src_name, vector<std::string> label_names, SimulationStateP& shared_state, std::string type )
: SourceTermBase( src_name, shared_state, label_names, type )
{
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() );
}

CoalGasOxi::~CoalGasOxi()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
CoalGasOxi::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb;

  db->require( "char_oxidation_model_name", _oxi_model_name );

  _source_grid_type = CC_SRC;

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term
//---------------------------------------------------------------------------
void
CoalGasOxi::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "CoalGasOxi::eval";
  Task* tsk = scinew Task(taskname, this, &CoalGasOxi::computeSource, timeSubStep);

  if (timeSubStep == 0) {
    tsk->computes(_src_label);
  } else {
    tsk->modifies(_src_label);
  }

  DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self();
  CoalModelFactory& modelFactory = CoalModelFactory::self();

  for (int iqn = 0; iqn < dqmomFactory.get_quad_nodes(); iqn++){

    std::string model_name = _oxi_model_name;
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

struct sumCharOxyGasSource{
       sumCharOxyGasSource(constCCVariable<double>& _qn_gas_oxi,
                           CCVariable<double>& _oxiSrc) :
#ifdef UINTAH_ENABLE_KOKKOS
                           qn_gas_oxi(_qn_gas_oxi.getKokkosView()),
                           oxiSrc(_oxiSrc.getKokkosView())
#else
                           qn_gas_oxi(_qn_gas_oxi),
                           oxiSrc(_oxiSrc)
#endif
                           {  }

  void operator()(int i , int j, int k ) const {
   oxiSrc(i,j,k) += qn_gas_oxi(i,j,k); // All the work is performed in Char Oxidation model
  }

  private:
#ifdef UINTAH_ENABLE_KOKKOS
   KokkosView3<const double> qn_gas_oxi;
   KokkosView3<double>  oxiSrc;
#else
   constCCVariable<double>& qn_gas_oxi;
   CCVariable<double>& oxiSrc;
#endif
};


//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
void
CoalGasOxi::computeSource( const ProcessorGroup* pc,
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

    CCVariable<double> oxiSrc;
    if ( timeSubStep == 0 ){
      new_dw->allocateAndPut( oxiSrc, _src_label, matlIndex, patch );
      oxiSrc.initialize(0.0);
    } else {
      new_dw->getModifiable( oxiSrc, _src_label, matlIndex, patch );
      oxiSrc.initialize(0.0);
    }


    for (int iqn = 0; iqn < dqmomFactory.get_quad_nodes(); iqn++){
      std::string model_name = _oxi_model_name;
      std::string node;
      std::stringstream out;
      out << iqn;
      node = out.str();
      model_name += "_qn";
      model_name += node;

      ModelBase& model = modelFactory.retrieve_model( model_name );

      constCCVariable<double> qn_gas_oxi;
      const VarLabel* gasModelLabel = model.getGasSourceLabel();

      new_dw->get( qn_gas_oxi, gasModelLabel, matlIndex, patch, gn, 0 );

      Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

      sumCharOxyGasSource doSumOxyGas(qn_gas_oxi,
                                      oxiSrc);

      Uintah::parallel_for(range, doSumOxyGas);
    }
  }
}


//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
CoalGasOxi::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "CoalGasOxi::initialize";

  Task* tsk = scinew Task(taskname, this, &CoalGasOxi::initialize);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter);
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void
CoalGasOxi::initialize( const ProcessorGroup* pc,
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




