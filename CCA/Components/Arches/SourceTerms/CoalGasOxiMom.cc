#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/CoalGasOxiMom.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CharOxidationShaddix.h>
#include <CCA/Components/Arches/CoalModels/CharOxidationSmith.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/CoalModels/PartVel.h>

//===========================================================================

#include <CCA/Components/Arches/FunctorSwitch.h>

#ifdef USE_FUNCTOR
#  include <Core/Grid/Variables/BlockRange.h>
#  ifdef UINTAH_ENABLE_KOKKOS
#    include <Kokkos_Core.hpp>
#  endif //UINTAH_ENABLE_KOKKOS
#endif

using namespace std;
using namespace Uintah; 

CoalGasOxiMom::CoalGasOxiMom( std::string src_name, vector<std::string> label_names, ArchesLabel* field_labels, SimulationStateP& shared_state, std::string type ) 
: SourceTermBase( src_name, field_labels->d_sharedState, label_names, type ),
  _field_labels(field_labels)
{
  _src_label = VarLabel::create( src_name, CCVariable<Vector>::getTypeDescription() ); 
  
  _source_grid_type = CCVECTOR_SRC; 
}

CoalGasOxiMom::~CoalGasOxiMom()
{
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
CoalGasOxiMom::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  db->require( "char_oxidation_model_name", _oxi_model_name ); 
  
}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
CoalGasOxiMom::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "CoalGasOxiMom::eval";
  Task* tsk = scinew Task(taskname, this, &CoalGasOxiMom::computeSource, timeSubStep);

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
    
    ArchesLabel::PartVelMap::const_iterator i = _field_labels->partVel.find(iqn);
    tsk->requires( Task::NewDW, i->second, Ghost::None, 0 );
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
CoalGasOxiMom::computeSource( const ProcessorGroup* pc, 
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

    CCVariable<Vector> oxiSrc; 
    if ( timeSubStep == 0 ){ 
      new_dw->allocateAndPut( oxiSrc, _src_label, matlIndex, patch );
      oxiSrc.initialize(Vector(0.0,0.0,0.0));
    } else { 
      new_dw->getModifiable( oxiSrc, _src_label, matlIndex, patch ); 
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
      Vector momentum_oxi_tmp;
      constCCVariable<Vector> partVel;
      const VarLabel* gasModelLabel = model.getGasSourceLabel(); 
 
      new_dw->get( qn_gas_oxi, gasModelLabel, matlIndex, patch, gn, 0 );
      ArchesLabel::PartVelMap::const_iterator iter = _field_labels->partVel.find(iqn);
      new_dw->get(partVel, iter->second, matlIndex, patch, gn, 0);
      
#ifdef USE_FUNCTOR
      Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

      sumCharOxyGasSource doSumOxyGas(qn_gas_oxi, 
                                      oxiSrc);

      Uintah::parallel_for(range, doSumOxyGas);
#else
      for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        Vector part_vel = partVel[c]; 
        momentum_oxi_tmp = Vector(part_vel.x()*qn_gas_oxi[c],part_vel.y()*qn_gas_oxi[c],part_vel.z()*qn_gas_oxi[c]);
        oxiSrc[c] += momentum_oxi_tmp;
      }
#endif
    }
  }
}
//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
CoalGasOxiMom::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "CoalGasOxiMom::initialize"; 

  Task* tsk = scinew Task(taskname, this, &CoalGasOxiMom::initialize);

  tsk->computes(_src_label);

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void 
CoalGasOxiMom::initialize( const ProcessorGroup* pc, 
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

    CCVariable<Vector> src;

    new_dw->allocateAndPut( src, _src_label, matlIndex, patch ); 

    src.initialize(Vector(0.0,0.0,0.0)); 

  }
}




