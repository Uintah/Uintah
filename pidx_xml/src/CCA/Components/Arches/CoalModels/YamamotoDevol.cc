#include <CCA/Components/Arches/CoalModels/YamamotoDevol.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
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
YamamotoDevolBuilder::YamamotoDevolBuilder( const std::string         & modelName,
                                                            const vector<std::string> & reqICLabelNames,
                                                            const vector<std::string> & reqScalarLabelNames,
                                                            ArchesLabel         * fieldLabels,
                                                            SimulationStateP          & sharedState,
                                                            int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

YamamotoDevolBuilder::~YamamotoDevolBuilder(){}

ModelBase* YamamotoDevolBuilder::build() {
  return scinew YamamotoDevol( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

YamamotoDevol::YamamotoDevol( std::string modelName, 
                                              SimulationStateP& sharedState,
                                              ArchesLabel* fieldLabels,
                                              vector<std::string> icLabelNames, 
                                              vector<std::string> scalarLabelNames,
                                              int qn ) 
: Devolatilization(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  _pi = acos(-1.0);
  _R = 8.314; 
}

YamamotoDevol::~YamamotoDevol()
{
}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
  void 
YamamotoDevol::problemSetup(const ProblemSpecP& params, int qn)
{

  ProblemSpecP db = params;
  const ProblemSpecP params_root = db->getRootNode();

  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();
  
  ProblemSpecP db_coal_props = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");
  
  // create raw coal mass var label and get scaling constant
  std::string rcmass_root = ParticleTools::parse_for_role_to_label(db, "raw_coal"); 
  std::string rcmass_name = ParticleTools::append_env( rcmass_root, d_quadNode ); 
  std::string rcmassqn_name = ParticleTools::append_qn_env( rcmass_root, d_quadNode ); 
  _rcmass_varlabel = VarLabel::find(rcmass_name);
  EqnBase& temp_rcmass_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(rcmassqn_name);
  DQMOMEqn& rcmass_eqn = dynamic_cast<DQMOMEqn&>(temp_rcmass_eqn);
   _rc_scaling_constant = rcmass_eqn.getScalingConstant(d_quadNode);
  std::string ic_RHS = rcmassqn_name+"_RHS";
  _RHS_source_varlabel = VarLabel::find(ic_RHS);

  // create char mass var label
  std::string char_root = ParticleTools::parse_for_role_to_label(db, "char"); 
  std::string char_name = ParticleTools::append_env( char_root, d_quadNode ); 
  _char_varlabel = VarLabel::find(char_name); 
  
  // create particle temperature label
  std::string temperature_root = ParticleTools::parse_for_role_to_label(db, "temperature"); 
  std::string temperature_name = ParticleTools::append_env( temperature_root, d_quadNode ); 
  _particle_temperature_varlabel = VarLabel::find(temperature_name);
 
  // Look for required scalars
  if (db_coal_props->findBlock("Yamamoto_coefficients")) {
    db_coal_props->require("Yamamoto_coefficients", Yamamoto_coefficients);
    _Av=Yamamoto_coefficients[0];
    _Ev=Yamamoto_coefficients[1];
    _Yv=Yamamoto_coefficients[2];
    _c0=Yamamoto_coefficients[3];
    _c1=Yamamoto_coefficients[4];
    _c2=Yamamoto_coefficients[5];
    _c3=Yamamoto_coefficients[6];
    _c4=Yamamoto_coefficients[7];
    _c5=Yamamoto_coefficients[8];
  } else { 
    throw ProblemSetupException("Error: Yamamoto_coefficients missing in <CoalProperties>.", __FILE__, __LINE__); 
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
        vol_dry.push_back((_pi/6.0)*std::pow(particle_sizes[n],3.0)); // m^3/particle
        mass_dry.push_back(vol_dry[n]*rhop); // kg/particle
        ash_mass_init.push_back(mass_dry[n]*ash_mass_frac); // kg_ash/particle (initial) 
        char_mass_init.push_back(mass_dry[n]*char_mass_frac); // kg_char/particle (initial)
        rc_mass_init.push_back(mass_dry[n]*rc_mass_frac); // kg_ash/particle (initial)
      }
  } else { 
    throw ProblemSetupException("Error: You must specify ultimate_analysis in <CoalProperties>.", __FILE__, __LINE__); 
  }

  // get weight scaling constant
  std::string weightqn_name = ParticleTools::append_qn_env("w", d_quadNode); 
  std::string weight_name = ParticleTools::append_env("w", d_quadNode); 
  _weight_varlabel = VarLabel::find(weight_name); 
  EqnBase& temp_weight_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(weightqn_name);
  DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(temp_weight_eqn);
  _weight_small = weight_eqn.getSmallClipPlusTol();
  _weight_scaling_constant = weight_eqn.getScalingConstant(d_quadNode);

}
//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void 
YamamotoDevol::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  string taskname = "YamamotoDevol::initVars"; 
  Task* tsk = scinew Task(taskname, this, &YamamotoDevol::initVars);

  tsk->computes(d_modelLabel);
  tsk->computes(d_gasLabel);
  tsk->computes(d_charLabel);

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
YamamotoDevol::initVars( const ProcessorGroup * pc, 
                              const PatchSubset    * patches, 
                              const MaterialSubset * matls, 
                              DataWarehouse        * old_dw, 
                              DataWarehouse        * new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> devol_rate;
    CCVariable<double> gas_devol_rate; 
    CCVariable<double> char_rate;
    
    new_dw->allocateAndPut( devol_rate, d_modelLabel, matlIndex, patch );
    devol_rate.initialize(0.0);
    new_dw->allocateAndPut( gas_devol_rate, d_gasLabel, matlIndex, patch );
    gas_devol_rate.initialize(0.0);
    new_dw->allocateAndPut( char_rate, d_charLabel, matlIndex, patch );
    char_rate.initialize(0.0);


  }
}

//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model 
//---------------------------------------------------------------------------
void 
YamamotoDevol::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "YamamotoDevol::computeModel";
  Task* tsk = scinew Task(taskname, this, &YamamotoDevol::computeModel, timeSubStep);

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
  tsk->requires( Task::NewDW, _RHS_source_varlabel, gn, 0 ); 
  tsk->requires( Task::OldDW, d_fieldLabels->d_sharedState->get_delt_label()); 

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}

//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
YamamotoDevol::computeModel( const ProcessorGroup * pc, 
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

    Vector Dx = patch->dCell(); 
    double vol = Dx.x()* Dx.y()* Dx.z(); 
    
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
    constCCVariable<double> RHS_source; 
    new_dw->get( RHS_source , _RHS_source_varlabel , matlIndex , patch , gn , 0 );
    
    double rcmass_init = rc_mass_init[d_quadNode];
    double kv;        ///< Rate constant for devolatilization reaction 1
    double Xv;
    double Fv;
    double rateMax;
    double rate;

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;
      if (weight[c]/_weight_scaling_constant > _weight_small) {

        double rcmassph=rcmass[c];
        double RHS_sourceph=RHS_source[c];
        double temperatureph=temperature[c];
        double charmassph=charmass[c];
        double weightph=weight[c];

        //VERIFICATION
        //rcmassph=6.76312e-11;
        //temperatureph=1434.86;
        //charmassph=3.32801e-10;
        //weightph=2.29063e+06;
        
        Xv = (rcmass_init-rcmassph)/rcmass_init;
        Xv = min(max(Xv,0.0),1.0);// make sure Xv is between 0 and 1
        Fv = _c5*std::pow(Xv,5.0) + _c4*std::pow(Xv,4.0) + _c3*std::pow(Xv,3.0) + _c2*std::pow(Xv,2.0) + _c1*Xv +_c0;
        kv = exp(Fv)*_Av*exp(-_Ev/(_R*temperatureph));
        
        rateMax = 0.5 * max( (rcmassph+min(0.0,charmassph))/(dt) + RHS_sourceph/(vol*weightph) , 0.0 );
        rate = kv*(rcmassph+min(0.0,charmassph));
        devol_rate[c] = -kv*weightph/(_rc_scaling_constant*_weight_scaling_constant); //rate of consumption of raw coal mass
        gas_devol_rate[c] = (_Yv*kv)*rcmassph*weightph; // rate of creation of coal off gas
        char_rate[c] = (1.0-_Yv)*kv*rcmassph*weightph; // rate of creation of char
        if( rateMax < rate ) {
          devol_rate[c] = -rateMax*weightph/(_rc_scaling_constant*_weight_scaling_constant);
          gas_devol_rate[c] = _Yv*rateMax*weightph;
          char_rate[c] = (1.0-_Yv)*rateMax*weightph;
        }

        //additional check to make sure we have positive rates when we have small amounts of rc and char.. 
        if( devol_rate[c]>0.0 || (rcmassph+min(0.0,charmassph)) < 1e-20 ) {
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
