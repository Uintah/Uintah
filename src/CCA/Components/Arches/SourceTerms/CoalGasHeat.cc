#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/CoalGasHeat.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/SimpleHeatTransfer.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>

#include <sci_defs/kokkos_defs.h>

//===========================================================================
//
using namespace std;
using namespace Uintah;

CoalGasHeat::CoalGasHeat( std::string src_name, vector<std::string> label_names, SimulationStateP& shared_state, std::string type )
: SourceTermBase( src_name, shared_state, label_names, type )
{
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() );
}

CoalGasHeat::~CoalGasHeat()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
CoalGasHeat::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb;

  db->require( "heat_model_name", _heat_model_name );
   
  m_dest_flag = false;
  if (db->findBlock("heat_BirthDeath")) {
    ProblemSpecP db_bd = db->findBlock("heat_BirthDeath");
    db_bd->getWithDefault("retained_deposit_factor", m_retained_deposit_factor, 1.0);
    m_dest_flag = true;
    m_enthalpy_root = ParticleTools::parse_for_role_to_label(db, "enthalpy");
  }

  _source_grid_type = CC_SRC;

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term
//---------------------------------------------------------------------------
void
CoalGasHeat::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "CoalGasHeat::eval";
  Task* tsk = scinew Task(taskname, this, &CoalGasHeat::computeSource, timeSubStep);

  if (timeSubStep == 0) {
    tsk->computes(_src_label);
  } else {
    tsk->modifies(_src_label);
  }

  DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self();
  CoalModelFactory& modelFactory = CoalModelFactory::self();

  for (int iqn = 0; iqn < dqmomFactory.get_quad_nodes(); iqn++){
    std::string weight_name = "w_qn";
    std::string model_name = _heat_model_name;
    std::string node;
    std::stringstream out;
    out << iqn;
    node = out.str();
    weight_name += node;
    model_name  += "_qn";
    model_name  += node;

    EqnBase& eqn = dqmomFactory.retrieve_scalar_eqn( weight_name );

    ModelBase& model = modelFactory.retrieve_model( model_name );

    const VarLabel* tempgasLabel_m = model.getGasSourceLabel();
    tsk->requires( Task::NewDW, tempgasLabel_m, Ghost::None, 0 );
  
    if (m_dest_flag){
      // require enthalpy birth/death   
      std::string enthalpyqn_name = ParticleTools::append_qn_env(m_enthalpy_root, iqn );
      EqnBase& temp_enthalpy_eqn = dqmomFactory.retrieve_scalar_eqn(enthalpyqn_name);
      DQMOMEqn& enthalpy_eqn = dynamic_cast<DQMOMEqn&>(temp_enthalpy_eqn);
      const std::string enthalpy_birth_name = enthalpy_eqn.get_model_by_type( "BirthDeath" );
      std::string enthalpy_birth_qn_name = ParticleTools::append_qn_env(enthalpy_birth_name, iqn);
      const VarLabel* enthalpy_birthdeath_varlabel=VarLabel::find(enthalpy_birth_qn_name);
      tsk->requires( Task::NewDW, enthalpy_birthdeath_varlabel, Ghost::None, 0 );
    }
    

  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}

struct sumHeatGasDestSource{
       sumHeatGasDestSource(constCCVariable<double>& _qn_gas_dest,
                           CCVariable<double>& _enthalpySrc,
                           double& _retained_deposit_factor,
                           double& _w_scaling_constant,
                           double& _enthalpy_scaling_constant ) :
#ifdef UINTAH_ENABLE_KOKKOS
                           qn_gas_dest(_qn_gas_dest.getKokkosView()),
                           enthalpySrc(_enthalpySrc.getKokkosView()),
                           enthalpy_scaling_constant(_enthalpy_scaling_constant),
                           w_scaling_constant(_w_scaling_constant),
                           retained_deposit_factor(_retained_deposit_factor)
#else
                           qn_gas_dest(_qn_gas_dest),
                           enthalpySrc(_enthalpySrc),
                           enthalpy_scaling_constant(_enthalpy_scaling_constant),
                           w_scaling_constant(_w_scaling_constant),
                           retained_deposit_factor(_retained_deposit_factor)
#endif
                           {  }

  void operator()(int i , int j, int k ) const {
   enthalpySrc(i,j,k) += - (1.0-retained_deposit_factor)*qn_gas_dest(i,j,k)*w_scaling_constant*enthalpy_scaling_constant; // minus sign because it is applied to the gas  
  }

  private:
#ifdef UINTAH_ENABLE_KOKKOS
   KokkosView3<const double> qn_gas_dest;
   KokkosView3<double>  enthalpySrc;
#else
   constCCVariable<double>& qn_gas_dest;
   CCVariable<double>& enthalpySrc;
#endif
  double enthalpy_scaling_constant;
  double w_scaling_constant;
  double retained_deposit_factor;
};
struct sumEnthalpyGasSource{
       sumEnthalpyGasSource(constCCVariable<double>& _qn_gas_enthalpy,
                           CCVariable<double>& _enthalpySrc) :
#ifdef UINTAH_ENABLE_KOKKOS
                           qn_gas_enthalpy(_qn_gas_enthalpy.getKokkosView()),
                           enthalpySrc(_enthalpySrc.getKokkosView())
#else
                           qn_gas_enthalpy(_qn_gas_enthalpy),
                           enthalpySrc(_enthalpySrc)
#endif
                           {  }

  void operator()(int i , int j, int k ) const {
   enthalpySrc(i,j,k) += qn_gas_enthalpy(i,j,k);
  }

  private:
#ifdef UINTAH_ENABLE_KOKKOS
   KokkosView3<const double> qn_gas_enthalpy;
   KokkosView3<double>  enthalpySrc;
#else
   constCCVariable<double>& qn_gas_enthalpy;
   CCVariable<double>& enthalpySrc;
#endif
};


//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
void
CoalGasHeat::computeSource( const ProcessorGroup* pc,
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

    CCVariable<double> heatSrc;
    if ( timeSubStep == 0 ){
      new_dw->allocateAndPut( heatSrc, _src_label, matlIndex, patch );
      heatSrc.initialize(0.0);
    } else {
      new_dw->getModifiable( heatSrc, _src_label, matlIndex, patch );
      heatSrc.initialize(0.0);
    }

    for (int iqn = 0; iqn < dqmomFactory.get_quad_nodes(); iqn++){
      std::string model_name = _heat_model_name;
      std::string node;
      std::stringstream out;
      out << iqn;
      node = out.str();
      model_name += "_qn";
      model_name += node;

      ModelBase& model = modelFactory.retrieve_model( model_name );

      constCCVariable<double> qn_gas_heat;
      const VarLabel* gasModelLabel = model.getGasSourceLabel();

      new_dw->get( qn_gas_heat, gasModelLabel, matlIndex, patch, gn, 0 );
      Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
      sumEnthalpyGasSource doSumEnthalpySource(qn_gas_heat,heatSrc);
      Uintah::parallel_for(range, doSumEnthalpySource);
      
      if (m_dest_flag){
        // get enthalpy birth death, enthalpy scaling constant and equation handle   
        std::string enthalpyqn_name = ParticleTools::append_qn_env(m_enthalpy_root, iqn );
        EqnBase& temp_enthalpy_eqn = dqmomFactory.retrieve_scalar_eqn(enthalpyqn_name);
        DQMOMEqn& enthalpy_eqn = dynamic_cast<DQMOMEqn&>(temp_enthalpy_eqn);
        double enthalpy_scaling_constant = enthalpy_eqn.getScalingConstant(iqn);
        const std::string enthalpy_birth_name = enthalpy_eqn.get_model_by_type( "BirthDeath" );
        std::string enthalpy_birth_qn_name = ParticleTools::append_qn_env(enthalpy_birth_name, iqn);
        const VarLabel* enthalpy_birthdeath_varlabel=VarLabel::find(enthalpy_birth_qn_name);
        // get weight scaling constant and equation handle 
        std::string weightqn_name = ParticleTools::append_qn_env("w", iqn);
        EqnBase& temp_weight_eqn = dqmomFactory.retrieve_scalar_eqn(weightqn_name);
        DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(temp_weight_eqn);
        double w_scaling_constant = weight_eqn.getScalingConstant(iqn);
        
        constCCVariable<double> qn_gas_dest;
        new_dw->get( qn_gas_dest, enthalpy_birthdeath_varlabel, matlIndex, patch, gn, 0 );
        // sum the dest sources
        sumHeatGasDestSource doSumHeatDestGas(qn_gas_dest,heatSrc,m_retained_deposit_factor,w_scaling_constant,enthalpy_scaling_constant);  
        Uintah::parallel_for(range, doSumHeatDestGas);
      }
    }
  }
}
//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
CoalGasHeat::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "CoalGasHeat::initialize";

  Task* tsk = scinew Task(taskname, this, &CoalGasHeat::initialize);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter);
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void
CoalGasHeat::initialize( const ProcessorGroup* pc,
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




