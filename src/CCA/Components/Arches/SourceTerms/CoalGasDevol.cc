#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
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
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>

#include <sci_defs/kokkos_defs.h>

//===========================================================================

using namespace std;
using namespace Uintah;

CoalGasDevol::CoalGasDevol( std::string src_name, vector<std::string> label_names, MaterialManagerP& materialManager, std::string type )
: SourceTermBase( src_name, materialManager, label_names, type )
{
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() );
}

CoalGasDevol::~CoalGasDevol()
{
  VarLabel::destroy(m_tar_src_label);
  VarLabel::destroy(m_devol_for_nox_src_label);
  VarLabel::destroy(m_devol_bd_src_label);
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
CoalGasDevol::problemSetup(const ProblemSpecP& inputdb)
{

  CoalHelper& coal_helper = CoalHelper::self();
  ProblemSpecP db = inputdb;

  db->require( "devol_model_name", _devol_model_name );
  
  db->getWithDefault( "tar_creation_src_label", m_tar_src_name, "Tar_source" );
  m_tar_src_label = VarLabel::create( m_tar_src_name, CCVariable<double>::getTypeDescription() );
  _mult_srcs.push_back( m_tar_src_name ); // this makes the source term available as a second source term within the implemenation.
  
  db->getWithDefault( "devol_src_label_for_nox", m_devol_for_nox_src_name, "Devol_NOx_source" );
  m_devol_for_nox_src_label = VarLabel::create( m_devol_for_nox_src_name, CCVariable<double>::getTypeDescription() );
  _mult_srcs.push_back( m_devol_for_nox_src_name ); // this makes the source term available as a second source term within the implemenation.
  
  db->getWithDefault( "bd_devol_src_label", m_devol_bd_src_name, "birth_death_devol_source" );
  m_devol_bd_src_label = VarLabel::create( m_devol_bd_src_name, CCVariable<double>::getTypeDescription() );
  _mult_srcs.push_back( m_devol_bd_src_name ); // this makes the source term available as a second source term within the implemenation.
  
  m_tarFrac = coal_helper.get_coal_db().Tar_fraction;
  m_lightFrac = 1.0 - m_tarFrac;

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
    tsk->computes(m_tar_src_label);
    tsk->computes(m_devol_for_nox_src_label);
    tsk->computes(m_devol_bd_src_label);
  } else {
    tsk->modifies(_src_label);
    tsk->modifies(m_tar_src_label);
    tsk->modifies(m_devol_for_nox_src_label);
    tsk->modifies(m_devol_bd_src_label);
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

  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));

}

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

    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self();
    CoalModelFactory& modelFactory = CoalModelFactory::self();

    CCVariable<double> devolSrc;
    CCVariable<double> tarSrc;
    CCVariable<double> devolNOxSrc;
    CCVariable<double> bd_devolSrc;
    if ( timeSubStep == 0 ){
      new_dw->allocateAndPut( devolSrc, _src_label, matlIndex, patch );
      devolSrc.initialize(0.0);
      new_dw->allocateAndPut( tarSrc, m_tar_src_label, matlIndex, patch );
      tarSrc.initialize(0.0);
      new_dw->allocateAndPut( devolNOxSrc, m_devol_for_nox_src_label, matlIndex, patch );
      devolNOxSrc.initialize(0.0);
      new_dw->allocateAndPut( bd_devolSrc, m_devol_bd_src_label, matlIndex, patch );
      bd_devolSrc.initialize(0.0);
    } else {
      new_dw->getModifiable( devolSrc, _src_label, matlIndex, patch );
      devolSrc.initialize(0.0);
      new_dw->getModifiable( tarSrc, m_tar_src_label, matlIndex, patch );
      tarSrc.initialize(0.0);
      new_dw->getModifiable( devolNOxSrc, m_devol_for_nox_src_label, matlIndex, patch );
      devolNOxSrc.initialize(0.0);
      new_dw->getModifiable( bd_devolSrc, m_devol_bd_src_label, matlIndex, patch );
      bd_devolSrc.initialize(0.0);
    }

    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

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
      constCCVariable<double> qn_gas_dest;
      double rc_scaling_constant = 0.0;
      double w_scaling_constant = 0.0;
      const VarLabel* gasModelLabel = model.getGasSourceLabel();

      new_dw->get( qn_gas_devol, gasModelLabel, matlIndex, patch, gn, 0 );
      if (m_dest_flag){
        std::string rcmassqn_name = ArchesCore::append_qn_env(m_rcmass_root, iqn );
        EqnBase& temp_rcmass_eqn = dqmomFactory.retrieve_scalar_eqn(rcmassqn_name);
        DQMOMEqn& rcmass_eqn = dynamic_cast<DQMOMEqn&>(temp_rcmass_eqn);
        rc_scaling_constant = rcmass_eqn.getScalingConstant(iqn);
        const std::string rawcoal_birth_name = rcmass_eqn.get_model_by_type( "BirthDeath" );
        std::string rawcoal_birth_qn_name = ArchesCore::append_qn_env(rawcoal_birth_name, iqn);
        const VarLabel* rcmass_birthdeath_varlabel=VarLabel::find(rawcoal_birth_qn_name);
        // get weight scaling constant and equation handle 
        std::string weightqn_name = ArchesCore::append_qn_env("w", iqn);
        EqnBase& temp_weight_eqn = dqmomFactory.retrieve_scalar_eqn(weightqn_name);
        DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(temp_weight_eqn);
        w_scaling_constant = weight_eqn.getScalingConstant(iqn);
        new_dw->get( qn_gas_dest, rcmass_birthdeath_varlabel, matlIndex, patch, gn, 0 );
      }
      
      // devolSrc = sum_i( rxn_devol_i - b/d_rc )       -> used for coal_gas_mix_frac 
      // tarSrc = sum_i( f_T*rxn_devol_i )              -> used for Tar 
      // devolNOxSrc = sum_i( (1-f_T)*rxn_devol_i )     -> used for part of devol rate in NOx 
      // bd_devolSrc = sum_i( - b/d_rc )                -> used in nox to compute remaining piece of devol rate
      //
      // f_T is the fraction of devol products that are tar (heavy gas instead of light).
      // sum_i is the sum over all particle environments
      // b/d_rc is the birth death term from the perspective of the particles (thus a - sign for the gas)
      
      Uintah::parallel_for( range, [&](int i, int j, int k){
        // compute the contribution of eta_source1 from the reactions
        devolSrc(i,j,k) += qn_gas_devol(i,j,k);
        // compute the tar source term
        tarSrc(i,j,k) +=  m_tarFrac * qn_gas_devol(i,j,k); 
        // compute the devol source term for the nox model
        devolNOxSrc(i,j,k) += m_lightFrac*qn_gas_devol(i,j,k);

        if (m_dest_flag){
          devolSrc(i,j,k) += - qn_gas_dest(i,j,k)*w_scaling_constant*rc_scaling_constant; // minus sign because it is applied to the gas  
          bd_devolSrc(i,j,k) += - qn_gas_dest(i,j,k)*w_scaling_constant*rc_scaling_constant; // minus sign because it is applied to the gas  
        }
      });

    } // end environment loop
  } // end patch loop
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
  tsk->computes(m_tar_src_label);
  tsk->computes(m_devol_for_nox_src_label);
  tsk->computes(m_devol_bd_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter);
  }

  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));

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
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> src;
    CCVariable<double> tarsrc;
    CCVariable<double> devolnoxsrc;
    CCVariable<double> bd_devolsrc;

    new_dw->allocateAndPut( src, _src_label, matlIndex, patch );
    new_dw->allocateAndPut( tarsrc, m_tar_src_label, matlIndex, patch );
    new_dw->allocateAndPut( devolnoxsrc, m_devol_for_nox_src_label, matlIndex, patch );
    new_dw->allocateAndPut( bd_devolsrc, m_devol_bd_src_label, matlIndex, patch );

    src.initialize(0.0);
    tarsrc.initialize(0.0);
    devolnoxsrc.initialize(0.0);
    bd_devolsrc.initialize(0.0);

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
      CCVariable<double> tempVar;
      new_dw->allocateAndPut(tempVar, *iter, matlIndex, patch );
      tempVar.initialize(0.0);
    }
  }
}
