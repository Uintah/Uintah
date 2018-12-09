#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/CoalGasOxi.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CharOxidationShaddix.h>
#include <CCA/Components/Arches/CoalModels/CharOxidationSmith.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>

#include <sci_defs/kokkos_defs.h>

//===========================================================================

using namespace std;
using namespace Uintah;

CoalGasOxi::CoalGasOxi( std::string src_name, vector<std::string> label_names, MaterialManagerP& materialManager, std::string type )
: SourceTermBase( src_name, materialManager, label_names, type )
{
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() );
}

CoalGasOxi::~CoalGasOxi()
{
  VarLabel::destroy(m_char_for_nox_src_label);
  VarLabel::destroy(m_char_bd_src_label);
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
CoalGasOxi::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb;

  db->require( "char_oxidation_model_name", _oxi_model_name );
  
  db->getWithDefault( "char_src_label_for_nox", m_char_for_nox_src_name, "Char_NOx_source" );
  m_char_for_nox_src_label = VarLabel::create( m_char_for_nox_src_name, CCVariable<double>::getTypeDescription() );
  _mult_srcs.push_back( m_char_for_nox_src_name ); // this makes the source term available as a second source term within the implemenation.
  
  db->getWithDefault( "bd_char_src_label", m_char_bd_src_name, "birth_death_char_source" );
  m_char_bd_src_label = VarLabel::create( m_char_bd_src_name, CCVariable<double>::getTypeDescription() );
  _mult_srcs.push_back( m_char_bd_src_name ); // this makes the source term available as a second source term within the implemenation.

   m_dest_flag = false;
  if (db->findBlock("char_BirthDeath")) {
    ProblemSpecP db_bd = db->findBlock("char_BirthDeath");
    m_dest_flag = true;
    m_charmass_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_CHAR);
  }

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
    tsk->computes(m_char_for_nox_src_label);
    tsk->computes(m_char_bd_src_label);
  } else {
    tsk->modifies(_src_label);
    tsk->modifies(m_char_for_nox_src_label);
    tsk->modifies(m_char_bd_src_label);
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
  
    if (m_dest_flag){
      // require Charmass birth/death   
      std::string charmassqn_name = ArchesCore::append_qn_env(m_charmass_root, iqn );
      EqnBase& temp_charmass_eqn = dqmomFactory.retrieve_scalar_eqn(charmassqn_name);
      DQMOMEqn& charmass_eqn = dynamic_cast<DQMOMEqn&>(temp_charmass_eqn);
      const std::string char_birth_name = charmass_eqn.get_model_by_type( "BirthDeath" );
      std::string char_birth_qn_name = ArchesCore::append_qn_env(char_birth_name, iqn);
      const VarLabel* charmass_birthdeath_varlabel=VarLabel::find(char_birth_qn_name);
      tsk->requires( Task::NewDW, charmass_birthdeath_varlabel, Ghost::None, 0 );
    }
    

  }

  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));

}

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
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self();
    CoalModelFactory& modelFactory = CoalModelFactory::self();

    CCVariable<double> oxiSrc;
    CCVariable<double> charNOxSrc;
    CCVariable<double> bd_charSrc;
    if ( timeSubStep == 0 ){
      new_dw->allocateAndPut( oxiSrc, _src_label, matlIndex, patch );
      oxiSrc.initialize(0.0);
      new_dw->allocateAndPut( charNOxSrc, m_char_for_nox_src_label, matlIndex, patch );
      charNOxSrc.initialize(0.0);
      new_dw->allocateAndPut( bd_charSrc, m_char_bd_src_label, matlIndex, patch );
      bd_charSrc.initialize(0.0);
    } else {
      new_dw->getModifiable( oxiSrc, _src_label, matlIndex, patch );
      oxiSrc.initialize(0.0);
      new_dw->getModifiable( charNOxSrc, m_char_for_nox_src_label, matlIndex, patch );
      charNOxSrc.initialize(0.0);
      new_dw->getModifiable( bd_charSrc, m_char_bd_src_label, matlIndex, patch );
      bd_charSrc.initialize(0.0);
    }
    
    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

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
      constCCVariable<double> qn_gas_dest;
      double char_scaling_constant = 0.0;
      double w_scaling_constant = 0.0;
      const VarLabel* gasModelLabel = model.getGasSourceLabel();


      new_dw->get( qn_gas_oxi, gasModelLabel, matlIndex, patch, gn, 0 );
      if (m_dest_flag){
        // get Charmass birth death, CH scaling constant and equation handle   
        std::string charmassqn_name = ArchesCore::append_qn_env(m_charmass_root, iqn );
        EqnBase& temp_charmass_eqn = dqmomFactory.retrieve_scalar_eqn(charmassqn_name);
        DQMOMEqn& charmass_eqn = dynamic_cast<DQMOMEqn&>(temp_charmass_eqn);
        char_scaling_constant = charmass_eqn.getScalingConstant(iqn);
        const std::string char_birth_name = charmass_eqn.get_model_by_type( "BirthDeath" );
        std::string char_birth_qn_name = ArchesCore::append_qn_env(char_birth_name, iqn);
        const VarLabel* charmass_birthdeath_varlabel=VarLabel::find(char_birth_qn_name);
        // get weight scaling constant and equation handle 
        std::string weightqn_name = ArchesCore::append_qn_env("w", iqn);
        EqnBase& temp_weight_eqn = dqmomFactory.retrieve_scalar_eqn(weightqn_name);
        DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(temp_weight_eqn);
        w_scaling_constant = weight_eqn.getScalingConstant(iqn);
        new_dw->get( qn_gas_dest, charmass_birthdeath_varlabel, matlIndex, patch, gn, 0 );
      }
      
      // charSrc = sum_i( rxn_char_i - b/d_ch )       -> used for coal_gas_mix_frac 
      // charNOxSrc = sum_i( (1-f_T)*rxn_char_i )     -> used for part of char rate in NOx 
      // bd_charSrc = sum_i( - b/d_ch )                -> used in nox to compute remaining piece of char rate
      // sum_i is the sum over all particle environments
      // b/d_ch is the birth death term from the perspective of the particles (thus a - sign for the gas)
      
      Uintah::parallel_for( range, [&](int i, int j, int k){
        // compute the contribution of eta_source1 from the reactions
        oxiSrc(i,j,k) += qn_gas_oxi(i,j,k);
        // compute the char source term for the nox model
        charNOxSrc(i,j,k) += qn_gas_oxi(i,j,k);

        if (m_dest_flag){
          oxiSrc(i,j,k) += - qn_gas_dest(i,j,k)*w_scaling_constant*char_scaling_constant; // minus sign because it is applied to the gas  
          bd_charSrc(i,j,k) += - qn_gas_dest(i,j,k)*w_scaling_constant*char_scaling_constant; // minus sign because it is applied to the gas  
        }
      });
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
  tsk->computes(m_char_for_nox_src_label);
  tsk->computes(m_char_bd_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter);
  }

  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));

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
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> src;
    CCVariable<double> charnoxsrc;
    CCVariable<double> bd_charsrc;

    new_dw->allocateAndPut( src, _src_label, matlIndex, patch );
    new_dw->allocateAndPut( charnoxsrc, m_char_for_nox_src_label, matlIndex, patch );
    new_dw->allocateAndPut( bd_charsrc, m_char_bd_src_label, matlIndex, patch );

    src.initialize(0.0);
    charnoxsrc.initialize(0.0);
    bd_charsrc.initialize(0.0);

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
      CCVariable<double> tempVar;
      new_dw->allocateAndPut(tempVar, *iter, matlIndex, patch );
    }
  }
}
