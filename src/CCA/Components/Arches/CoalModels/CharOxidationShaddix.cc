#include <CCA/Components/Arches/CoalModels/CharOxidationShaddix.h>
#include <CCA/Components/Arches/CoalModels/Devolatilization.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace Uintah; 

//---------------------------------------------------------------------------
// Builder:
CharOxidationShaddixBuilder::CharOxidationShaddixBuilder( const std::string         & modelName,
                                                      const vector<std::string> & reqICLabelNames,
                                                      const vector<std::string> & reqScalarLabelNames,
                                                      ArchesLabel         * fieldLabels,
                                                      SimulationStateP          & sharedState,
                                                      int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

CharOxidationShaddixBuilder::~CharOxidationShaddixBuilder(){}

ModelBase* CharOxidationShaddixBuilder::build() {
  return scinew CharOxidationShaddix( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

CharOxidationShaddix::CharOxidationShaddix( std::string modelName, 
                                        SimulationStateP& sharedState,
                                        ArchesLabel* fieldLabels,
                                        vector<std::string> icLabelNames, 
                                        vector<std::string> scalarLabelNames,
                                        int qn ) 
: CharOxidation(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  // Set constants
  pi = 3.14159265358979;
  R = 8.314; // J/K/mol
  WC = 12.0e-3;  //kg/mol
  WO2 = 32.0; //g/mol
  WCO2 = 44.0;
  WH2O = 18.0;
  WN2 = 28.0;
  D1 = 0.153e-4; // Binary diffusion coef. O2/CO2
  D2 = 0.24e-4;  // Binary diffusion coef. O2/H2O
  D3 = 0.219e-4; // Binary diffusion coef. O2/N2
  T0 = 293;

  // Eastern bituminous coal, non-linear regression
  As = 344.0;  // mol/s.m^2.atm^n
  Es = 45.5e3; // J/mol
  n = 0.18;

  // Eastern bituminous coal, Hurt & Mitchell
  //As = 94.0;  // mol/s.m^2.atm^n
  //Es = 10.4e3; // J/mol
  //n = 0.5;

  // Eastern bituminous coal, non-linear regression, LH expression
  //A1 = 61.0;
  //E1 = 0.5e3;
  //n = 0.1;
  //A2 = 20.0;
  //E2 = 107.4e3;

  // Enthalpy of formation (J/mol)
  HF_CO2 = -393509.0;
  HF_CO  = -110525.0;

  part_temp_from_enth = false;
}

CharOxidationShaddix::~CharOxidationShaddix()
{
}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
CharOxidationShaddix::problemSetup(const ProblemSpecP& params, int qn)
{
  CharOxidation::problemSetup( params, qn );

  ProblemSpecP db = params; 

  //std::string s = "N2";

  //d_fieldLabels->add_species(s);

  std::string N2name = "N2";
  d_fieldLabels->add_species(N2name);
  std::string O2name = "O2";
  d_fieldLabels->add_species(O2name);
  std::string MWname = "mixture_molecular_weight";
  d_fieldLabels->add_species(MWname);
  
  // check for viscosity
  const ProblemSpecP params_root = db->getRootNode(); 

  string label_name;
  string role_name;
  string temp_label_name;

  string temp_ic_name;
  string temp_ic_name_full;

  // Look for required internal coordinates
  ProblemSpecP db_icvars = params->findBlock("ICVars");
  if (db_icvars) {
    for (ProblemSpecP variable = db_icvars->findBlock("variable"); variable != 0; variable = variable->findNextBlock("variable") ) {
    
      variable->getAttribute("label",label_name);
      variable->getAttribute("role",role_name);

      temp_label_name = label_name;
      
      std::stringstream out;
      out << qn;
      string node = out.str();
      temp_label_name += "_qn";
      temp_label_name += node;

      // user specifies "role" of each internal coordinate
      // if it isn't an internal coordinate or a scalar, it's required explicitly
      // ( see comments in Arches::registerModels() for details )
      if ( role_name == "particle_length" 
               || role_name == "raw_coal_mass"
               || role_name == "char_mass"
               || role_name == "particle_temperature" ) {
        LabelToRoleMap[temp_label_name] = role_name;
      } else if( role_name == "particle_temperature_from_enthalpy" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        part_temp_from_enth = true;
      } else {
        std::string errmsg = "ERROR: CharOxidationShaddix: problemSetup(): Invalid variable role for Char Oxidation model!";
        throw InvalidValue(errmsg,__FILE__,__LINE__);
      }

      // set model clipping
      db->getWithDefault( "low_clip",  d_lowModelClip,  1.0e-6 );
      db->getWithDefault( "high_clip", d_highModelClip, 999999 );
    }
  }

  // fix the d_icLabels to point to the correct quadrature node (since there is 1 model per quad node)
  for ( vector<std::string>::iterator iString = d_icLabels.begin(); 
        iString != d_icLabels.end(); ++iString) {
    
    temp_ic_name      = (*iString);
    temp_ic_name_full = temp_ic_name;

    std::stringstream out;
    out << qn;
    string node = out.str();
    temp_ic_name_full += "_qn";
    temp_ic_name_full += node;

    std::replace( d_icLabels.begin(), d_icLabels.end(), temp_ic_name, temp_ic_name_full);
  }

  // fix the d_scalarLabels to point to the correct quadrature node (since there is 1 model per quad node)
  for ( vector<std::string>::iterator iString = d_scalarLabels.begin(); 
        iString != d_scalarLabels.end(); ++iString) {

    temp_ic_name      = (*iString);
    temp_ic_name_full = temp_ic_name;

    std::stringstream out;
    out << qn;
    string node = out.str();
    temp_ic_name_full += "_qn";
    temp_ic_name_full += node;

    std::replace( d_scalarLabels.begin(), d_scalarLabels.end(), temp_ic_name, temp_ic_name_full);
  }

  std::stringstream out;
  out << qn; 
  string node = out.str();
}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void 
CharOxidationShaddix::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "CharOxidationShaddix::initVars";
  Task* tsk = scinew Task(taskname, this, &CharOxidationShaddix::initVars);

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
CharOxidationShaddix::initVars( const ProcessorGroup * pc, 
                              const PatchSubset    * patches, 
                              const MaterialSubset * matls, 
                              DataWarehouse        * old_dw, 
                              DataWarehouse        * new_dw )
{
  /*
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
  }
  */
}

//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model 
//---------------------------------------------------------------------------
void 
CharOxidationShaddix::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "CharOxidationShaddix::sched_computeModel";
  Task* tsk = scinew Task(taskname, this, &CharOxidationShaddix::computeModel);

  d_timeSubStep = timeSubStep; 

  if (d_timeSubStep == 0 && !d_labelSchedInit) {
    // Every model term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;

    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel); 
    tsk->computes(d_particletempLabel);
    tsk->computes(d_surfacerateLabel);
    tsk->computes(d_PO2surfLabel);
  } else {
    tsk->modifies(d_modelLabel);
    tsk->modifies(d_gasLabel);  
    tsk->modifies(d_particletempLabel);
    tsk->modifies(d_surfacerateLabel);
    tsk->modifies(d_PO2surfLabel);
  }

  tsk->requires( Task::OldDW, d_fieldLabels->d_sharedState->get_delt_label(), Ghost::None, 0);
 
  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();
  CoalModelFactory& modelFactory = CoalModelFactory::self();
  DevolModelMap devolmodels_ = modelFactory.retrieve_devol_models();
  for( DevolModelMap::iterator iModel = devolmodels_.begin(); iModel != devolmodels_.end(); ++iModel ) {
    int modelNode = iModel->second->getquadNode();
    if( modelNode == d_quadNode) { 
      d_devolCharLabel = iModel->second->getCharSourceLabel();
      tsk->requires( Task::OldDW, d_devolCharLabel, Ghost::None, 0 );
    } 
  }


  // construct the weight label corresponding to this quad node
  std::string temp_weight_name = "w_qn";
  std::string node;
  std::stringstream out;
  out << d_quadNode;
  node = out.str();
  temp_weight_name += node;
  EqnBase& t_weight_eqn = dqmom_eqn_factory.retrieve_scalar_eqn( temp_weight_name );
  DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(t_weight_eqn);
  d_weight_label = weight_eqn.getTransportEqnLabel();
  tsk->requires(Task::OldDW, d_weight_label, Ghost::None, 0);

  // always require the gas-phase temperature
  tsk->requires(Task::OldDW, d_fieldLabels->d_tempINLabel, Ghost::None, 0);
  tsk->requires(Task::OldDW, d_fieldLabels->d_densityCPLabel, Ghost::None, 0);
  tsk->requires(Task::OldDW, d_PO2surfLabel, Ghost::None, 0);

  const VarLabel* d_O2_label = VarLabel::find("O2");
  tsk->requires(Task::OldDW, d_O2_label, Ghost::None, 0 );
  const VarLabel* d_CO2_label = VarLabel::find("CO2");
  tsk->requires(Task::OldDW, d_CO2_label, Ghost::None, 0 );
  const VarLabel* d_H2O_label = VarLabel::find("H2O");
  tsk->requires(Task::OldDW, d_H2O_label, Ghost::None, 0 );
  const VarLabel* d_N2_label = VarLabel::find("N2");
  tsk->requires(Task::OldDW, d_N2_label, Ghost::None, 0 );
  const VarLabel* d_MWmix_label = VarLabel::find("mixture_molecular_weight");
  tsk->requires(Task::OldDW, d_MWmix_label, Ghost::None, 0 );

  // for each required internal coordinate:
  for (vector<std::string>::iterator iter = d_icLabels.begin(); 
       iter != d_icLabels.end(); ++iter) { 

    map<string, string>::iterator iMap = LabelToRoleMap.find(*iter);

    if( iMap != LabelToRoleMap.end() ) {
      if ( iMap->second == "particle_temperature") {
        if( dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_particle_temperature_label = current_eqn.getTransportEqnLabel();
          d_pt_scaling_constant = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_particle_temperature_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: CharOxidationShaddix: sched_computeModel(): Invalid variable given in <ICVars> block, for <variable> tag for CharOxidationShaddix model.";
          errmsg += "\nCould not find given particle temperature variable \"";
          errmsg += *iter;
          errmsg += "\" in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }

      } else if ( iMap->second == "particle_temperature_from_enthalpy") {
        //std::string pt_temp_name = iMap->first;
        d_particle_temperature_label = VarLabel::find(iMap->first);
        d_pt_scaling_constant = 1.0;
        tsk->requires(Task::OldDW, d_particle_temperature_label, Ghost::None, 0);

      } else if( iMap->second == "particle_length" ) {
        if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_particle_length_label = current_eqn.getTransportEqnLabel();
          d_pl_scaling_constant = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_particle_length_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: CharOxidationShaddix: sched_computeModel(): Invalid variable given in <ICVars> block, for <variable> tag for CharOxidationShaddix model.";
          errmsg += "\nCould not find given particle length variable \"";
          errmsg += *iter;
          errmsg += "\" in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }

      } else if ( iMap->second == "raw_coal_mass") {
        if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_raw_coal_mass_label = current_eqn.getTransportEqnLabel();
          d_rc_scaling_constant = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_raw_coal_mass_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: CharOxidationShaddix: sched_computeModel(): Invalid variable given in <ICVars> block, for <variable> tag for CharOxidationShaddix model.";
          errmsg += "\nCould not find given coal mass  variable \"";
          errmsg += *iter;
          errmsg += "\" in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      
      } else if ( iMap->second == "char_mass") {
        if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_char_mass_label = current_eqn.getTransportEqnLabel();
          d_rh_scaling_constant = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_char_mass_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: CharOxidationShaddix: sched_computeModel(): Invalid variable given in <ICVars> block, for <variable> tag for CharOxidationShaddix model.";
          errmsg += "\nCould not find given coal mass  variable \"";
          errmsg += *iter;
          errmsg += "\" in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      }

    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ARCHES: CharOxidationShaddix: sched_computeModel(): You specified that the variable \"" + *iter + 
                           "\" was required, but you did not specify a role for it!\n";
      throw InvalidValue( errmsg, __FILE__, __LINE__);
    }
  }

  // for each required scalar variable:
  for( vector<std::string>::iterator iter = d_scalarLabels.begin();
       iter != d_scalarLabels.end(); ++iter) {
    map<string, string>::iterator iMap = LabelToRoleMap.find(*iter);
    
    /*
    if( iMap != LabelToRoleMap.end() ) {
      if( iMap->second == <insert role name here> ) {
        if( eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& current_eqn = eqn_factory.retrieve_scalar_eqn(*iter);
          d_<insert role name here>_label = current_eqn.getTransportEqnLabel();
          tsk->requires(Task::OldDW, d_<insert role name here>_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: CharOxidationShaddix: Invalid variable given in <scalarVars> block for <variable> tag for CharOxidationShaddix model.";
          errmsg += "\nCould not find given <insert role name here> variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      }
    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ARCHES: CharOxidationShaddix: You specified that the variable \"" + *iter + 
                           "\" was required, but you did not specify a role for it!\n";
      throw InvalidValue( errmsg, __FILE__, __LINE__);
    }
    */

  } //end for
  
   sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}

//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
CharOxidationShaddix::computeModel( const ProcessorGroup * pc, 
                                  const PatchSubset    * patches, 
                                  const MaterialSubset * matls, 
                                  DataWarehouse        * old_dw, 
                                  DataWarehouse        * new_dw )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    //Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> char_rate;
    if ( new_dw->exists( d_modelLabel, matlIndex, patch) ) {
      new_dw->getModifiable( char_rate, d_modelLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( char_rate, d_modelLabel, matlIndex, patch );
      char_rate.initialize(0.0);
    }
    
    CCVariable<double> gas_char_rate; 
    if( new_dw->exists( d_gasLabel, matlIndex, patch ) ) {
      new_dw->getModifiable( gas_char_rate, d_gasLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( gas_char_rate, d_gasLabel, matlIndex, patch );
      gas_char_rate.initialize(0.0);
    }

    CCVariable<double> particle_temp_rate;
    if( new_dw->exists( d_particletempLabel, matlIndex, patch ) ) {
      new_dw->getModifiable( particle_temp_rate, d_particletempLabel, matlIndex, patch );
    } else {
      new_dw->allocateAndPut( particle_temp_rate, d_particletempLabel, matlIndex, patch );
      particle_temp_rate.initialize(0.0);
    }

    CCVariable<double> surface_rate;
    if( new_dw->exists( d_surfacerateLabel, matlIndex, patch) ) {
      new_dw->getModifiable( surface_rate, d_surfacerateLabel, matlIndex, patch );
    } else {
      new_dw->allocateAndPut(surface_rate, d_surfacerateLabel, matlIndex, patch );
      surface_rate.initialize(0.0);
    }

    CCVariable<double> PO2surf_;
    if( new_dw->exists( d_PO2surfLabel, matlIndex, patch) ) {
      new_dw->getModifiable( PO2surf_, d_PO2surfLabel, matlIndex, patch );
    } else {
      new_dw->allocateAndPut(PO2surf_, d_PO2surfLabel, matlIndex, patch );
      PO2surf_.initialize(0.0);
    }

    constCCVariable<double> den;
    old_dw->get(den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0 ); 

    constCCVariable<double> temperature;
    constCCVariable<double> particle_temperature;
    constCCVariable<double> w_particle_length;
    constCCVariable<double> w_raw_coal_mass;
    constCCVariable<double> w_char_mass;
    constCCVariable<double> weight;
    constCCVariable<double> O2;
    constCCVariable<double> CO2;
    constCCVariable<double> H2O;
    constCCVariable<double> N2;
    constCCVariable<double> MWmix;
    constCCVariable<double> devolChar;
    constCCVariable<double> oldPO2surf_;

    old_dw->get( temperature, d_fieldLabels->d_tempINLabel, matlIndex, patch, gn, 0 );
    old_dw->get( particle_temperature, d_particle_temperature_label, matlIndex, patch, gn, 0 );
    old_dw->get( w_particle_length, d_particle_length_label, matlIndex, patch, gn, 0 );
    old_dw->get( w_raw_coal_mass, d_raw_coal_mass_label, matlIndex, patch, gn, 0 );
    old_dw->get( w_char_mass, d_char_mass_label, matlIndex, patch, gn, 0 );
    old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );
    old_dw->get( devolChar, d_devolCharLabel, matlIndex, patch, gn, 0 );
    old_dw->get( oldPO2surf_, d_PO2surfLabel, matlIndex, patch, gn, 0 );

    const VarLabel* d_O2_label = VarLabel::find("O2");
    old_dw->get( O2, d_O2_label, matlIndex, patch, gn, 0 );
    const VarLabel* d_CO2_label = VarLabel::find("CO2");
    old_dw->get( CO2, d_CO2_label, matlIndex, patch, gn, 0 );
    const VarLabel* d_H2O_label = VarLabel::find("H2O");
    old_dw->get( H2O, d_H2O_label, matlIndex, patch, gn, 0 );
    const VarLabel* d_N2_label = VarLabel::find("N2");
    old_dw->get( N2, d_N2_label, matlIndex, patch, gn, 0 );
    const VarLabel* d_MWmix_label = VarLabel::find("mixture_molecular_weight");
    old_dw->get( MWmix, d_MWmix_label, matlIndex, patch, gn, 0 );  // in kmol/kg_mix ?

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 
    
      // weight - check if small
      bool weight_is_small = (weight[c] < d_w_small);

      double scaled_weight;
      double unscaled_weight;
      // temperature - particle
      double unscaled_particle_temperature;
      // paticle length
      double unscaled_length;
      // particle raw coal mass
      double unscaled_raw_coal_mass;
      // particle char mass
      double unscaled_char_mass;
     
      if (weight_is_small && !d_unweighted) {
        char_production_rate_ = devolChar[c];
        char_rate[c] = char_production_rate_/(d_rh_scaling_constant*d_w_scaling_constant);
        gas_char_rate[c] = 0.0;
        particle_temp_rate[c] = 0.0;
        surface_rate[c] = 0.0;
      } else {

        if(d_unweighted){
          scaled_weight = weight[c];
          unscaled_weight = weight[c]*d_w_scaling_constant;
          if(part_temp_from_enth){
            unscaled_particle_temperature = particle_temperature[c];
          } else {
            unscaled_particle_temperature = particle_temperature[c]*d_pt_scaling_constant;
          }
          unscaled_length = w_particle_length[c]*d_pl_scaling_constant;
          unscaled_raw_coal_mass = w_raw_coal_mass[c]*d_rc_scaling_constant;
          unscaled_char_mass = w_char_mass[c]*d_rh_scaling_constant;

        } else {
          scaled_weight = weight[c];
          unscaled_weight = weight[c]*d_w_scaling_constant;
          if(part_temp_from_enth){
            unscaled_particle_temperature = particle_temperature[c];
            unscaled_particle_temperature = max(273.0, min(unscaled_particle_temperature,3000.0));
          } else {
            unscaled_particle_temperature = (particle_temperature[c]*d_pt_scaling_constant)/scaled_weight;
          }
          unscaled_length = (w_particle_length[c]*d_pl_scaling_constant)/scaled_weight;
          unscaled_raw_coal_mass = (w_raw_coal_mass[c]*d_rc_scaling_constant)/scaled_weight;
          unscaled_char_mass = (w_char_mass[c]*d_rh_scaling_constant)/scaled_weight;
        } 

        double small = 1e-16;
        //double MW_mix; // in g/mol
        //MW_mix = 1.0/(O2[c]/WO2 + CO2[c]/WCO2 + H2O[c]/WH2O);
 
        PO2_inf = O2[c]/WO2/MWmix[c];

        if((PO2_inf < 1e-10) || ((unscaled_raw_coal_mass+unscaled_char_mass) < small)) {
       //if((PO2_inf < 1e-10) || (unscaled_char_mass < small)) {
          PO2_surf = 0.0;
          CO2CO = 0.0;
          q = 0.0;
        } else {

          char_reaction_rate_ = 0.0;
          char_production_rate_ = 0.0;
          gas_char_rate_ = 0.0;     
          particle_temp_rate_ = 0.0;

          //PO2_surf = oldPO2surf_[c];

          PO2_surf = PO2_inf;

          d_totIter = 100;
          delta = PO2_inf/100.0;
          d_tol = 1e-15;
          f1 = 1.0;
          icount = 0;
     
          // Calculate O2 diffusion coefficient
          DO2 = (CO2[c]/WCO2 + H2O[c]/WH2O + N2[c]/WN2)/(CO2[c]/(WCO2*D1) + H2O[c]/(WH2O*D2) + N2[c]/(WN2*D3))*
                (pow((temperature[c]/T0),1.5));

          // Concentration C = P/RT
          Conc = MWmix[c]*den[c]*1000.0;

          // Solving diffusion of O2:
          // Newton_Raphson method - faster does not always converge 
           
          for ( int iter = 0; iter < 12; iter++) {
            icount++;
            CO2CO = 0.02*(pow(PO2_surf,0.21))*exp(3070.0/unscaled_particle_temperature);
            OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
            gamma = -(1.0-OF);
            ks = As*exp(-Es/(R*unscaled_particle_temperature));
            q = ks*(pow(PO2_surf,n));
            //k1 = A1*exp(-E1/(R*unscaled_particle_temperature));
            //k2 = A2*exp(-E2/(R*unscaled_particle_temperature));
            //q = k1*k2*(pow(PO2_surf,n))/(k1*(pow(PO2_surf,n))+k2);
            f1 = PO2_surf - gamma - (PO2_inf-gamma)*exp(-(q*unscaled_length)/(2*Conc*DO2));

            if (std::abs(f1) < d_tol)
              break;

            PO2_surf += delta;
            CO2CO = 0.02*(pow(PO2_surf,0.21))*exp(3070.0/unscaled_particle_temperature);
            OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
            gamma = -(1.0-OF);
            ks = As*exp(-Es/(R*unscaled_particle_temperature));
            q = ks*(pow(PO2_surf,n));
            //k1 = A1*exp(-E1/(R*unscaled_particle_temperature));
            //k2 = A2*exp(-E2/(R*unscaled_particle_temperature));
            //q = k1*k2*(pow(PO2_surf,n))/(k1*(pow(PO2_surf,n))+k2);
            f2 = PO2_surf - gamma - (PO2_inf-gamma)*exp(-(q*unscaled_length)/(2*Conc*DO2));

            PO2_surf -= delta + f1*delta/(f2-f1);
            PO2_surf = min(PO2_inf,max(0.0,PO2_surf));
          }
          
          if(std::abs(f1) > d_tol || isnan(f1)){ //switching to bisection technique
            lower_bound = 0.0;
            upper_bound = PO2_inf;
            for ( int iter = 0; iter < d_totIter; iter++) {
              icount++;
              PO2_surf = lower_bound;
              CO2CO = 0.02*(pow(PO2_surf,0.21))*exp(3070.0/unscaled_particle_temperature);
              OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
              gamma = -(1.0-OF);
              ks = As*exp(-Es/(R*unscaled_particle_temperature));
              //k1 = A1*exp(-E1/(R*unscaled_particle_temperature));
              //k2 = A2*exp(-E2/(R*unscaled_particle_temperature));
              //q = k1*k2*(pow(PO2_surf,n))/(k1*(pow(PO2_surf,n))+k2);
              q = ks*(pow(PO2_surf,n));
              f3 = PO2_surf - gamma - (PO2_inf-gamma)*exp(-(q*unscaled_length)/(2*Conc*DO2));

              PO2_surf = upper_bound;
              CO2CO = 0.02*(pow(PO2_surf,0.21))*exp(3070.0/unscaled_particle_temperature);
              OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
              gamma = -(1.0-OF);
              ks = As*exp(-Es/(R*unscaled_particle_temperature));
              q = ks*(pow(PO2_surf,n));
              //k1 = A1*exp(-E1/(R*unscaled_particle_temperature));
              //k2 = A2*exp(-E2/(R*unscaled_particle_temperature));
              //q = k1*k2*(pow(PO2_surf,n))/(k1*(pow(PO2_surf,n))+k2);
              f2 = PO2_surf - gamma - (PO2_inf-gamma)*exp(-(q*unscaled_length)/(2*Conc*DO2));

              if (std::abs(f2) < d_tol){
                break;
              }

              PO2_surf = 0.5*(lower_bound+upper_bound);
              CO2CO = 0.02*(pow(PO2_surf,0.21))*exp(3070.0/unscaled_particle_temperature);
              OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
              gamma = -(1.0-OF);
              ks = As*exp(-Es/(R*unscaled_particle_temperature));
              q = ks*(pow(PO2_surf,n));
              //k1 = A1*exp(-E1/(R*unscaled_particle_temperature));
              //k2 = A2*exp(-E2/(R*unscaled_particle_temperature));
              //q = k1*k2*(pow(PO2_surf,n))/(k1*(pow(PO2_surf,n))+k2);
              f1 = PO2_surf - gamma - (PO2_inf-gamma)*exp(-(q*unscaled_length)/(2*Conc*DO2));

              if (std::abs(f1) < d_tol){
                break;
              }

              if(icount > d_totIter+12-1) {
                //cout << "CharOxidationShaddix::computeModel : problem with bisection convergence, reaction rate set to zero" << endl;
                //cout << "icount " << icount << " f1 " << f1 << " f2 " << f2 << " f3 " << f3 << " PO2_inf " << PO2_inf << " PO2_surf " << PO2_surf << endl;
                //PO2_surf = 0.0;
                //CO2CO = 0.0;
                //q = 0.0;
                PO2_surf = PO2_inf;
                CO2CO = 0.02*(pow(PO2_surf,0.21))*exp(3070.0/unscaled_particle_temperature);
                OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
                gamma = -(1.0-OF);
                ks = As*exp(-Es/(R*unscaled_particle_temperature));
                q = ks*(pow(PO2_surf,n));
                break;
              }


              if(f2*f1<0){
                 lower_bound = PO2_surf;
              } else if(f1*f3<0){
                 upper_bound = PO2_surf;
              } else {
                //cout << "CharOxidationShaddix::computeModel : problem with bisection, reaction rate set to zero" << endl;
                //cout << "icount " << icount << " f1 " << f1 << " f2 " << f2 << " f3 " << f3 << " PO2_inf " << PO2_inf << " PO2_surf " << PO2_surf << endl;
                //cout << "gamma " << gamma << " q " << q << " unscaled_length " << unscaled_length << endl;
                //PO2_surf = 0.0;
                //CO2CO = 0.0;
                //q = 0.0;
                PO2_surf = PO2_inf;
                CO2CO = 0.02*(pow(PO2_surf,0.21))*exp(3070.0/unscaled_particle_temperature);
                OF = 0.5*(1.0 + CO2CO*(1+CO2CO));
                gamma = -(1.0-OF);
                ks = As*exp(-Es/(R*unscaled_particle_temperature));
                q = ks*(pow(PO2_surf,n));
                break;
              }

            }
          }              
        }

        char_production_rate_ = devolChar[c];
              
        char_reaction_rate_ = -pi*(pow(unscaled_length,2.0))*WC*q;
        particle_temp_rate_ = -pi*(pow(unscaled_length,2.0))*q/(1.0+CO2CO)*(CO2CO*HF_CO2 + HF_CO); // in J/s
  
          //cout << "O2 " << O2[c] << " CO2 " << CO2[c] << " H2O " << H2O[c] << " N2 " << N2[c] << " MW_mix " << MW_mix << " Conc " << Conc << " DO2 " << DO2 << " q " << q << endl;
          //if(abs(PO2_surf/PO2_inf -1.0) > 0.01){
          //  cout << " PO2_inf " << PO2_inf << " PO2_surf " << PO2_surf << " MWmix " << MWmix[c] << " f1 " << f1 << " f2 " << f2 << " f3 " << f3 << " icount " << icount << endl;
          //}
          //cout << " devol " << devol[c] << " devolgas " << devolGas[c] << " weight " << unscaled_weight << endl;
          //cout << "char_reaction_rate_ " << char_reaction_rate_ << " char_production_rate_ " << char_production_rate_ << 
          //     " particle_temp_rate_ " << particle_temp_rate_ <<endl;
          
        if(d_unweighted){
          char_rate[c] = (char_reaction_rate_ + char_production_rate_)/d_rh_scaling_constant;
          gas_char_rate[c] = -char_reaction_rate_*unscaled_weight;
          particle_temp_rate[c] = particle_temp_rate_;
        } else {
          char_rate[c] = (char_reaction_rate_*unscaled_weight + char_production_rate_)/(d_rh_scaling_constant*d_w_scaling_constant);
          gas_char_rate[c] = -char_reaction_rate_*unscaled_weight;
          particle_temp_rate[c] = particle_temp_rate_*unscaled_weight;
        }
 
        surface_rate[c] = WC*q;  // in kg/s/m^2
        PO2surf_[c] = PO2_surf;

      }
    }//end cell loop
  }//end patch loop
}



