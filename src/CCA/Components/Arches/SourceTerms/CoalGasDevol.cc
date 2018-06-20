#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/CoalGasDevol.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
//#include <CCA/Components/Arches/CoalModels/KobayashiSarofimDevol.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>

#include <sci_defs/kokkos_defs.h>

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

   m_dest_flag = false;
  if (db->findBlock("devol_BirthDeath")) {
    ProblemSpecP db_bd = db->findBlock("devol_BirthDeath");
    m_dest_flag = true;
    m_rcmass_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_RAWCOAL);
  }

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
  
    if (m_dest_flag){
      // require RCmass birth/death   
      std::string rcmassqn_name = ArchesCore::append_qn_env(m_rcmass_root, iqn );
      EqnBase& temp_rcmass_eqn = dqmomFactory.retrieve_scalar_eqn(rcmassqn_name);
      DQMOMEqn& rcmass_eqn = dynamic_cast<DQMOMEqn&>(temp_rcmass_eqn);
      const std::string rawcoal_birth_name = rcmass_eqn.get_model_by_type( "BirthDeath" );
      std::string rawcoal_birth_qn_name = ArchesCore::append_qn_env(rawcoal_birth_name, iqn);
      const VarLabel* rcmass_birthdeath_varlabel=VarLabel::find(rawcoal_birth_qn_name);
      tsk->requires( Task::NewDW, rcmass_birthdeath_varlabel, Ghost::None, 0 );
    }
    

  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
struct sumDevolGasDestSource{
       sumDevolGasDestSource(constCCVariable<double>& _qn_gas_dest,
                           CCVariable<double>& _devolSrc,
                           double& _w_scaling_constant,
                           double& _rc_scaling_constant ) :
#if defined( KOKKOS_ENABLE_OPENMP )
                           qn_gas_dest(_qn_gas_dest.getKokkosView()),
                           devolSrc(_devolSrc.getKokkosView()),
                           rc_scaling_constant(_rc_scaling_constant),
                           w_scaling_constant(_w_scaling_constant)
#else
                           qn_gas_dest(_qn_gas_dest),
                           devolSrc(_devolSrc),
                           rc_scaling_constant(_rc_scaling_constant),
                           w_scaling_constant(_w_scaling_constant)
#endif
                           {  }

  void operator()(int i , int j, int k ) const {
   devolSrc(i,j,k) += - qn_gas_dest(i,j,k)*w_scaling_constant*rc_scaling_constant; // minus sign because it is applied to the gas  
  }

  private:
#if defined( KOKKOS_ENABLE_OPENMP )
   KokkosView3<const double, Kokkos::HostSpace> qn_gas_dest;
   KokkosView3<double, Kokkos::HostSpace>  devolSrc;
#else
   constCCVariable<double>& qn_gas_dest;
   CCVariable<double>& devolSrc;
#endif
  double rc_scaling_constant;
  double w_scaling_constant;
};
struct sumDevolGasSource{
       sumDevolGasSource(constCCVariable<double>& _qn_gas_devol,
                           CCVariable<double>& _devolSrc) :
#if defined( KOKKOS_ENABLE_OPENMP )
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
#if defined( KOKKOS_ENABLE_OPENMP )
   KokkosView3<const double, Kokkos::HostSpace> qn_gas_devol;
   KokkosView3<double, Kokkos::HostSpace>  devolSrc;
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
      sumDevolGasSource doSumDevolGas(qn_gas_devol,devolSrc); 
      Uintah::parallel_for(range, doSumDevolGas);

      if (m_dest_flag){
        // get RCmass birth death, RC scaling constant and equation handle   
        std::string rcmassqn_name = ArchesCore::append_qn_env(m_rcmass_root, iqn );
        EqnBase& temp_rcmass_eqn = dqmomFactory.retrieve_scalar_eqn(rcmassqn_name);
        DQMOMEqn& rcmass_eqn = dynamic_cast<DQMOMEqn&>(temp_rcmass_eqn);
        double rc_scaling_constant = rcmass_eqn.getScalingConstant(iqn);
        const std::string rawcoal_birth_name = rcmass_eqn.get_model_by_type( "BirthDeath" );
        std::string rawcoal_birth_qn_name = ArchesCore::append_qn_env(rawcoal_birth_name, iqn);
        const VarLabel* rcmass_birthdeath_varlabel=VarLabel::find(rawcoal_birth_qn_name);
        // get weight scaling constant and equation handle 
        std::string weightqn_name = ArchesCore::append_qn_env("w", iqn);
        EqnBase& temp_weight_eqn = dqmomFactory.retrieve_scalar_eqn(weightqn_name);
        DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(temp_weight_eqn);
        double w_scaling_constant = weight_eqn.getScalingConstant(iqn);
        
        constCCVariable<double> qn_gas_dest;
        new_dw->get( qn_gas_dest, rcmass_birthdeath_varlabel, matlIndex, patch, gn, 0 );
        // sum the dest sources
        sumDevolGasDestSource doSumDevolDestGas(qn_gas_dest,devolSrc,w_scaling_constant,rc_scaling_constant);  
        Uintah::parallel_for(range, doSumDevolDestGas);
      }
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
