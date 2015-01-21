#include <CCA/Components/Arches/CoalModels/BTDevol.h>
#include <CCA/Components/Arches/ParticleModels/ParticleHelper.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Directives.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

//---------------------------------------------------------------------------
// Builder:
BTDevolBuilder::BTDevolBuilder( const std::string         & modelName,
                                                            const vector<std::string> & reqICLabelNames,
                                                            const vector<std::string> & reqScalarLabelNames,
                                                            ArchesLabel         * fieldLabels,
                                                            SimulationStateP          & sharedState,
                                                            int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

BTDevolBuilder::~BTDevolBuilder(){}

ModelBase* BTDevolBuilder::build() {
  return scinew BTDevol( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

BTDevol::BTDevol( std::string modelName, 
                                              SimulationStateP& sharedState,
                                              ArchesLabel* fieldLabels,
                                              vector<std::string> icLabelNames, 
                                              vector<std::string> scalarLabelNames,
                                              int qn ) 
: Devolatilization(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  pi = acos(-1.0);
}

BTDevol::~BTDevol()
{
}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
  void 
BTDevol::problemSetup(const ProblemSpecP& params, int qn)
{

  ProblemSpecP db = params;
  const ProblemSpecP params_root = db->getRootNode();

  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();
  
  ProblemSpecP db_coal_props = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal")->findBlock("Properties");
  
  // create raw coal mass var label and get scaling constant
  std::string rcmass_root = ParticleHelper::parse_for_role_to_label(db, "raw_coal"); 
  std::string rcmass_name = ParticleHelper::append_env( rcmass_root, d_quadNode ); 
  std::string rcmassqn_name = ParticleHelper::append_qn_env( rcmass_root, d_quadNode ); 
  _rcmass_varlabel = VarLabel::find(rcmass_name);
  EqnBase& temp_rcmass_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(rcmassqn_name);
  DQMOMEqn& rcmass_eqn = dynamic_cast<DQMOMEqn&>(temp_rcmass_eqn);
   _rc_scaling_constant = rcmass_eqn.getScalingConstant(d_quadNode);

  // create char mass var label
  std::string char_root = ParticleHelper::parse_for_role_to_label(db, "char"); 
  std::string char_name = ParticleHelper::append_env( char_root, d_quadNode ); 
  _char_varlabel = VarLabel::find(char_name); 
  
  // create particle temperature label
  std::string temperature_root = ParticleHelper::parse_for_role_to_label(db, "temperature"); 
  std::string temperature_name = ParticleHelper::append_env( temperature_root, d_quadNode ); 
  _particle_temperature_varlabel = VarLabel::find(temperature_name);
 
  // Look for required scalars
  if (db_coal_props->findBlock("BTDevol")) {
    ProblemSpecP db_BT = db_coal_props->findBlock("BTDevol");
    db_BT->require("Tig", _Tig);
    db_BT->require("v_hiT", _v_hiT);
    db_BT->require("Ta", _Ta);
    db_BT->require("A", _A);
    db_BT->require("c", _c);
    db_BT->require("d", _d);
  } else { 
    throw ProblemSetupException("Error: BT_coefficients missing in <CoalProperties>.", __FILE__, __LINE__); 
  }
  if ( db_coal_props->findBlock("density")){ 
    db_coal_props->require("density", rhop);
  } else { 
    throw ProblemSetupException("Error: You must specify density in <CoalProperties>.", __FILE__, __LINE__); 
  }
  if ( db_coal_props->findBlock("diameter_distribution")){ 
    db_coal_props->require("diameter_distribution", particle_sizes);
  } else { 
    throw ProblemSetupException("Error: You must specify diameter_distribution in <CoalProperties>.", __FILE__, __LINE__); 
  }
  if ( db_coal_props->findBlock("ultimate_analysis")){ 
    ProblemSpecP db_ua = db_coal_props->findBlock("ultimate_analysis"); 
    CoalAnalysis coal; 
    db_ua->require("C",coal.C);
    db_ua->require("H",coal.H);
    db_ua->require("O",coal.O);
    db_ua->require("N",coal.N);
    db_ua->require("S",coal.S);
    db_ua->require("H2O",coal.H2O);
    db_ua->require("ASH",coal.ASH);
    db_ua->require("CHAR",coal.CHAR);
    total_rc=coal.C+coal.H+coal.O+coal.N+coal.S; // (C+H+O+N+S) dry ash free total
    total_dry=coal.C+coal.H+coal.O+coal.N+coal.S+coal.ASH+coal.CHAR; // (C+H+O+N+S+char+ash)  moisture free total
    rc_mass_frac=total_rc/total_dry; // mass frac of rc (dry) 
    char_mass_frac=coal.CHAR/total_dry; // mass frac of char (dry)
    ash_mass_frac=coal.ASH/total_dry; // mass frac of ash (dry)
    int p_size=particle_sizes.size();
    for (int n=0; n<p_size; n=n+1)
      {
        vol_dry.push_back((pi/6)*pow(particle_sizes[n],3)); // m^3/particle
        mass_dry.push_back(vol_dry[n]*rhop); // kg/particle
        ash_mass_init.push_back(mass_dry[n]*ash_mass_frac); // kg_ash/particle (initial) 
        char_mass_init.push_back(mass_dry[n]*char_mass_frac); // kg_char/particle (initial)
        rc_mass_init.push_back(mass_dry[n]*rc_mass_frac); // kg_ash/particle (initial)
      }
  } else { 
    throw ProblemSetupException("Error: You must specify ultimate_analysis in <CoalProperties>.", __FILE__, __LINE__); 
  }

  // get weight scaling constant
  std::string weightqn_name = ParticleHelper::append_qn_env("w", d_quadNode); 
  std::string weight_name = ParticleHelper::append_env("w", d_quadNode); 
  _weight_varlabel = VarLabel::find(weight_name); 
  EqnBase& temp_weight_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(weightqn_name);
  DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(temp_weight_eqn);
  _weight_small = weight_eqn.getSmallClipPlusTol();
  _weight_scaling_constant = weight_eqn.getScalingConstant(d_quadNode);

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model 
//---------------------------------------------------------------------------
void 
BTDevol::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "BTDevol::computeModel";
  Task* tsk = scinew Task(taskname, this, &BTDevol::computeModel, timeSubStep);

  Ghost::GhostType gn = Ghost::None;

  Task::WhichDW which_dw; 

  if (timeSubStep == 0 ) { 
    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel);
    tsk->computes(d_charLabel);
    which_dw = Task::OldDW; 
  } else {
    tsk->modifies(d_modelLabel); 
    tsk->modifies(d_gasLabel);
    tsk->modifies(d_charLabel); 
    which_dw = Task::NewDW; 
  }

  tsk->requires( which_dw, _particle_temperature_varlabel, gn, 0 ); 
  tsk->requires( which_dw, _rcmass_varlabel, gn, 0 ); 
  tsk->requires( which_dw, _char_varlabel, gn, 0 );
  tsk->requires( which_dw, _weight_varlabel, gn, 0 ); 
  tsk->requires( Task::OldDW, d_fieldLabels->d_sharedState->get_delt_label()); 

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}

//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
BTDevol::computeModel( const ProcessorGroup * pc, 
                                     const PatchSubset    * patches, 
                                     const MaterialSubset * matls, 
                                     DataWarehouse        * old_dw, 
                                     DataWarehouse        * new_dw,
                                     const int timeSubStep )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    delt_vartype DT;
    old_dw->get(DT, d_fieldLabels->d_sharedState->get_delt_label());
    double dt = DT;

    CCVariable<double> devol_rate;
    CCVariable<double> gas_devol_rate; 
    CCVariable<double> char_rate;
    DataWarehouse* which_dw; 

    if ( timeSubStep == 0 ){ 
      which_dw = old_dw; 
      new_dw->allocateAndPut( devol_rate, d_modelLabel, matlIndex, patch );
      devol_rate.initialize(0.0);
      new_dw->allocateAndPut( gas_devol_rate, d_gasLabel, matlIndex, patch );
      gas_devol_rate.initialize(0.0);
      new_dw->allocateAndPut( char_rate, d_charLabel, matlIndex, patch );
      char_rate.initialize(0.0);
    } else { 
      which_dw = new_dw; 
      new_dw->getModifiable( devol_rate, d_modelLabel, matlIndex, patch ); 
      new_dw->getModifiable( gas_devol_rate, d_gasLabel, matlIndex, patch ); 
      new_dw->getModifiable( char_rate, d_charLabel, matlIndex, patch );
    }

    constCCVariable<double> temperature; 
    which_dw->get( temperature , _particle_temperature_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> rcmass; 
    which_dw->get( rcmass    , _rcmass_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> charmass; 
    which_dw->get( charmass , _char_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> weight; 
    which_dw->get( weight , _weight_varlabel , matlIndex , patch , gn , 0 );
    
    double rcmass_init = rc_mass_init[d_quadNode];

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;
      if (weight[c]/_weight_scaling_constant > _weight_small) {

        double rcmassph=rcmass[c];
        double temperatureph=temperature[c];
        double charmassph=charmass[c];
        double weightph=weight[c];

        //VERIFICATION
        //rcmassph=1;
        //temperatureph=300;
        //charmassph=0.0;
        //weightph=_rc_scaling_constant*_weight_scaling_constant;
        //rcmass_init = 1;


        // m_init = m_residual_solid + m_h_off_gas + m_vol
        // m_vol = m_init - m_residual_solid - m_h_off_gas
        // but m_h_off_gas is negative by construction so 
        // m_vol = m_init - m_residual_solid + m_h_off_gas
        v_inf = 0.5*_v_hiT*(1.0 - tanh(_c*(_Tig-temperatureph)/temperatureph + _d));
        m_vol = rcmass_init - rcmassph + charmassph;
        f_drive = max(rcmass_init*v_inf - m_vol, 0.0);
        rateMax = max((rcmassph + min(0.0,charmassph))*weightph/dt,0.0);
        rate = _A * exp(-_Ta/temperatureph) * f_drive;
        devol_rate[c] = -rate*weightph/(_rc_scaling_constant*_weight_scaling_constant); //rate of consumption of raw coal mass
        gas_devol_rate[c] = rate*weightph; // rate of creation of coal off gas
        char_rate[c] = 0; // rate of creation of char
        if( devol_rate[c] < (-rateMax/(_rc_scaling_constant*_weight_scaling_constant))) {
          devol_rate[c] = -rateMax/(_rc_scaling_constant*_weight_scaling_constant);
          gas_devol_rate[c] = rateMax;
          char_rate[c] = 0;
        }
        //additional check to make sure we have positive rates when we have small amounts of rc and char.. 
        if( (devol_rate[c] > -1e-16) && ((rcmassph+min(0.0,charmassph)) < 1e-16)) {
          devol_rate[c] = 0;
          gas_devol_rate[c] = 0;
          char_rate[c] = 0;
        }
   
      } else {
        devol_rate[c] = 0;
        gas_devol_rate[c] = 0;
        char_rate[c] = 0;
      }
    }//end cell loop
  }//end patch loop
}
