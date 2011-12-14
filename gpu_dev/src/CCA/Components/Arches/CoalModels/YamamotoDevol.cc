#include <CCA/Components/Arches/CoalModels/YamamotoDevol.h>
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
  R   =  8.314;
  Av = 3.2159e16;
  Ev = 2.647e5;
  Yv = 0.58;
  c0 = 7.008;
  c1 = -79.38;
  c2 = 379.9;
  c3 = -853.0;
  c4 = 836.7;
  c5 = -301.1;

  part_temp_from_enth = false;
  compute_part_temp = false;
  compute_char_mass = false;
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
  // call parent's method first
  Devolatilization::problemSetup(params, qn);

  ProblemSpecP db = params; 
  compute_part_temp = false;
  part_temp_from_enth = false;
  compute_char_mass = false;

  string label_name;
  string role_name;
  string temp_label_name;
  
  string temp_ic_name;
  string temp_ic_name_full;

  const ProblemSpecP params_root = db->getRootNode();
  if (params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties")) {
    ProblemSpecP db_coal = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties");
    db_coal->require("initial_rawcoal_mass", rc_mass_init);
  } else {
    throw InvalidValue("ERROR: YamamotoDevol: problemSetup(): Missing <initial_rawcoal_mass> in <Coal_Properties> section in input file!",__FILE__,__LINE__);
  }


  // -----------------------------------------------------------------
  // Look for required internal coordinates
  ProblemSpecP db_icvars = params->findBlock("ICVars");
  if (db_icvars) {
    for (ProblemSpecP variable = db_icvars->findBlock("variable"); variable != 0; variable = variable->findNextBlock("variable") ) {

      variable->getAttribute("label",label_name);
      variable->getAttribute("role", role_name);

      temp_label_name = label_name;
      
      std::stringstream out;
      out << qn;
      string node = out.str();
      temp_label_name += "_qn";
      temp_label_name += node;

      // user specifies "role" of each internal coordinate
      // if it isn't an internal coordinate or a scalar, it's required explicitly
      // ( see comments in Arches::registerModels() for details )
      if ( role_name == "raw_coal_mass" ) {
        LabelToRoleMap[temp_label_name] = role_name;
      } else if( role_name == "particle_temperature" ) {  
        LabelToRoleMap[temp_label_name] = role_name;
        compute_part_temp = true;
      } else if( role_name == "particle_temperature_from_enthalpy" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        part_temp_from_enth = true;
      } else if( role_name == "char_mass" ) {    
        LabelToRoleMap[temp_label_name] = role_name;        
        compute_char_mass = true;                           
      } else {
        std::string errmsg;
        errmsg = "Invalid variable role for Yamamoto Devolatilization model: must be \"particle_temperature\" or \"raw_coal_mass\" or \"char_mass\", you specified \"" + role_name + "\".";
        throw InvalidValue(errmsg,__FILE__,__LINE__);
      }
    }
  }

  // fix the d_icLabels to point to the correct quadrature node (since there is 1 model per quad node)
  for ( vector<std::string>::iterator iString = d_icLabels.begin(); 
        iString != d_icLabels.end(); ++iString) {

    temp_ic_name        = (*iString);
    temp_ic_name_full   = temp_ic_name;

    std::stringstream out;
    out << qn;
    string node = out.str();
    temp_ic_name_full += "_qn";
    temp_ic_name_full += node;

    std::replace( d_icLabels.begin(), d_icLabels.end(), temp_ic_name, temp_ic_name_full);
  }


  // -----------------------------------------------------------------
  // Look for required scalars
 
/* 
  ProblemSpecP db_scalarvars = params->findBlock("scalarVars");
  if (db_scalarvars) {
    for( ProblemSpecP variable = db_scalarvars->findBlock("variable");
         variable != 0; variable = variable->findNextBlock("variable") ) {

      variable->getAttribute("label", label_name);
      variable->getAttribute("role",  role_name);

      temp_label_name = label_name;

      std::stringstream out; 
      out << qn;
      string node = out.str();
      temp_label_name += "_qn";
      temp_label_name += node;

      // user specifies "role" of each scalar
      // if it isn't an internal coordinate or a scalar, it's required explicitly
      // ( see comments in Arches::registerModels() for details )
      if( role_name == "particle_temperature_from_enthalpy" ) {  
        LabelToRoleMap[temp_label_name] = role_name;
        part_temp_from_enth = true;
      } else {
        std::string errmsg;
        errmsg = "Invalid variable role for Yamamoto Devolatilization model: must be \"particle_temperature\", you specified \"" + role_name + "\".";
        throw InvalidValue(errmsg,__FILE__,__LINE__);
      }
    }
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
*/


}

//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model 
//---------------------------------------------------------------------------
void 
YamamotoDevol::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "YamamotoDevol::computeModel";
  Task* tsk = scinew Task(taskname, this, &YamamotoDevol::computeModel);

  Ghost::GhostType gn = Ghost::None;

  d_timeSubStep = timeSubStep; 

  tsk->requires( Task::OldDW, d_fieldLabels->d_sharedState->get_delt_label(), Ghost::None, 0);

  if (d_timeSubStep == 0 && !d_labelSchedInit) {
    // Every model term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;

    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel); 
    tsk->computes(d_charLabel);
  } else {
    tsk->modifies(d_modelLabel);
    tsk->modifies(d_gasLabel);  
    tsk->modifies(d_charLabel);
  }

  //EqnFactory& eqn_factory = EqnFactory::self();
  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();

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
  tsk->requires(Task::OldDW, d_weight_label, gn, 0);

  // always require gas temperature
  tsk->requires(Task::OldDW, d_fieldLabels->d_tempINLabel, Ghost::AroundCells, 1);

  // For each required variable, determine what role it plays
  // - "gas_temperature" - require the "tempIN" label
  // - "particle_temperature" - look in DQMOMEqnFactory
  // - "raw_coal_mass" - look in DQMOMEqnFactory

  // for each required internal coordinate:
  for (vector<std::string>::iterator iter = d_icLabels.begin(); 
      iter != d_icLabels.end(); iter++) { 

    map<string, string>::iterator iMap = LabelToRoleMap.find(*iter);

    if ( iMap != LabelToRoleMap.end() ) {
      if ( iMap->second == "particle_temperature") {
        if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_particle_temperature_label = current_eqn.getTransportEqnLabel();
          d_pt_scaling_factor = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_particle_temperature_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: YamamotoDevol: Invalid variable given in <variable> tag for YamamotoDevol model";
          errmsg += "\nCould not find given particle temperature variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory or in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }


      } else if ( iMap->second == "particle_temperature_from_enthalpy") {
        std::string pt_temp_name = iMap->first;
        d_particle_temperature_label = VarLabel::find(pt_temp_name);
        d_pt_scaling_factor = 1.0;
        tsk->requires(Task::OldDW, d_particle_temperature_label, Ghost::None, 0);

      } else if ( iMap->second == "raw_coal_mass") {
        if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_raw_coal_mass_label = current_eqn.getTransportEqnLabel();
          d_rc_scaling_factor = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_raw_coal_mass_label, Ghost::None, 0);

        } else {
          std::string errmsg = "ARCHES: YamamotoDevol: Invalid variable given in <variable> tag for YamamotoDevol model";
          errmsg += "\nCould not find given raw coal mass variable \"";
          errmsg += *iter;
          errmsg += "\" in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }

      } else if ( iMap->second == "char_mass") {
        if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_char_mass_label = current_eqn.getTransportEqnLabel();
          d_rh_scaling_factor = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_char_mass_label, Ghost::None, 0);

        } else {
          std::string errmsg = "ARCHES: YamamotoDevol: Invalid variable given in <variable> tag for YamamotoDevol model";
          errmsg += "\nCould not find given raw coal mass variable \"";
          errmsg += *iter;
          errmsg += "\" in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      }

    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ARCHES: YamamotoDevol: You specified that the variable \"" + *iter + 
                           "\" was required, but you did not specify a role for it!\n";
      throw InvalidValue( errmsg, __FILE__, __LINE__);
    }
  }
  
  // for each required scalar variable:
/*
  for( vector<std::string>::iterator iter = d_scalarLabels.begin();
       iter != d_scalarLabels.end(); ++iter) {
    map<string, string>::iterator iMap = LabelToRoleMap.find(*iter);
    
    
    if( iMap != LabelToRoleMap.end() ) {
      if( iMap->second == "particle_temperature_from_enthalpy" ) {
        d_particle_temperature_label = VarLabel::find( iMap->first );

        if( eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& current_eqn = eqn_factory.retrieve_scalar_eqn(*iter);
          d_<insert role name here>_label = current_eqn.getTransportEqnLabel();
          tsk->requires(Task::OldDW, d_<insert role name here>_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: YamamotoDevol: Invalid variable given in <scalarVars> block for <variable> tag for YamamotoDevol model.";
          errmsg += "\nCould not find given <insert role name here> variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      }
    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ARCHES: YamamotoDevol: You specified that the variable \"" + *iter + 
                           "\" was required, but you did not specify a role for it!\n";
      throw InvalidValue( errmsg, __FILE__, __LINE__);
    }
    

  } //end for
*/

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
                                     DataWarehouse        * new_dw )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    delt_vartype DT;
    old_dw->get(DT, d_fieldLabels->d_sharedState->get_delt_label());
    double dt = DT;

    CCVariable<double> devol_rate;
    if( new_dw->exists( d_modelLabel, matlIndex, patch ) ) {
      new_dw->getModifiable( devol_rate, d_modelLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( devol_rate, d_modelLabel, matlIndex, patch );
      devol_rate.initialize(0.0);
    }

    CCVariable<double> gas_devol_rate; 
    if (new_dw->exists( d_gasLabel, matlIndex, patch )){
      new_dw->getModifiable( gas_devol_rate, d_gasLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( gas_devol_rate, d_gasLabel, matlIndex, patch ); 
      gas_devol_rate.initialize(0.0);
    } 

    CCVariable<double> char_rate;
    if (new_dw->exists( d_charLabel, matlIndex, patch )){
      new_dw->getModifiable( char_rate, d_charLabel, matlIndex, patch );
    } else {
      new_dw->allocateAndPut( char_rate, d_charLabel, matlIndex, patch );
      char_rate.initialize(0.0);
    }

    constCCVariable<double> temperature; // holds gas OR particle temperature...
    if (compute_part_temp || part_temp_from_enth) {
      old_dw->get( temperature, d_particle_temperature_label, matlIndex, patch, gn, 0 );
    } else {
      old_dw->get( temperature, d_fieldLabels->d_tempINLabel, matlIndex, patch, gac, 1 );
    }
 
    constCCVariable<double> wa_raw_coal_mass;
    old_dw->get( wa_raw_coal_mass, d_raw_coal_mass_label, matlIndex, patch, gn, 0 );

    constCCVariable<double> wa_char_mass;
    if(compute_char_mass){
      old_dw->get( wa_char_mass, d_char_mass_label, matlIndex, patch, gn, 0 );
    }

    constCCVariable<double> weight;
    old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 

      // weight - check if small
      bool weight_is_small = (weight[c] < d_w_small);

      double unscaled_weight;
      double unscaled_temperature;
      // raw coal mass - de-scaled, de-weighted
      double scaled_raw_coal_mass;
      double unscaled_raw_coal_mass;
     // char mass - de-scaled, de-weighted
      double scaled_char_mass;
      double unscaled_char_mass;

      // devol_rate: particle source
      double devol_rate_;

      // gase_devol_rate: gas source
      double gas_devol_rate_;

      // char_rate: particle source
      double char_rate_;

      if (weight_is_small  && !d_unweighted) {
        devol_rate_ = 0.0;
        gas_devol_rate_ = 0.0;
        char_rate_ = 0.0;
      } else {

        if(d_unweighted){
          unscaled_weight = weight[c]*d_w_scaling_factor;
          scaled_raw_coal_mass = wa_raw_coal_mass[c];
          unscaled_raw_coal_mass = scaled_raw_coal_mass*d_rc_scaling_factor;

          if(compute_char_mass){
            scaled_char_mass = wa_char_mass[c];
            unscaled_char_mass = scaled_char_mass*d_rh_scaling_factor;
          } else {
            scaled_char_mass = 0.0;
            unscaled_char_mass = 0.0;
          }

          if (compute_part_temp) {
            // particle temp
            unscaled_temperature = temperature[c]*d_pt_scaling_factor;
          } else {
            // gas temp
            unscaled_temperature = temperature[c];
          }
        } else {
          unscaled_weight = weight[c]*d_w_scaling_factor;
          if (compute_part_temp) {
            // particle temp
            unscaled_temperature = temperature[c]*d_pt_scaling_factor/weight[c];
          } else {
            // gas temp
            unscaled_temperature = temperature[c];
            unscaled_temperature = max(273.0, min(unscaled_temperature,3000.0));
          } 
          scaled_raw_coal_mass = wa_raw_coal_mass[c]/weight[c];
          unscaled_raw_coal_mass = scaled_raw_coal_mass*d_rc_scaling_factor;

          if(compute_char_mass){
            scaled_char_mass = wa_char_mass[c]/weight[c];
            unscaled_char_mass = scaled_char_mass*d_rh_scaling_factor;
          } else {
            scaled_char_mass = 0.0;
            unscaled_char_mass = 0.0;
          }

        }

        double unscaled_rawcoal_mass = rc_mass_init[d_quadNode];
        Xv = (unscaled_rawcoal_mass-unscaled_raw_coal_mass)/unscaled_rawcoal_mass;
        Xv = min(max(Xv,0.0),1.0);
        Fv = c5*pow(Xv,5.0) + c4*pow(Xv,4.0) + c3*pow(Xv,3.0) + c2*pow(Xv,2.0) + c1*Xv +c0;
        kv = exp(Fv)*Av*exp(-Ev/(R*unscaled_temperature));
 
        if(d_unweighted){
          rateMax = max((0.2*(unscaled_raw_coal_mass + min(0.0,unscaled_char_mass))/dt),0.0);
          testVal_part = -kv*(unscaled_raw_coal_mass + min(0.0,unscaled_char_mass))/d_rc_scaling_factor;
          testVal_gas = (Yv*kv)*(unscaled_raw_coal_mass+ min(0.0,unscaled_char_mass))*unscaled_weight;
          testVal_char = (1.0-Yv)*kv*(unscaled_raw_coal_mass + min(0.0,unscaled_char_mass));
          if( testVal_part < (-rateMax/d_rc_scaling_factor)) {
            testVal_part = -rateMax/(d_rc_scaling_factor);
            testVal_gas = Yv*rateMax;
            testVal_char = (1.0-Yv)*rateMax;
          }

        } else {
          rateMax = max((0.2*(unscaled_raw_coal_mass + min(0.0,unscaled_char_mass))*unscaled_weight/dt),0.0);
          testVal_part = -kv*(unscaled_raw_coal_mass + min(0.0,unscaled_char_mass))*unscaled_weight/(d_rc_scaling_factor*d_w_scaling_factor);
          testVal_gas = (Yv*kv)*(unscaled_raw_coal_mass+ min(0.0,unscaled_char_mass))*unscaled_weight;
          testVal_char = (1.0-Yv)*kv*(unscaled_raw_coal_mass + min(0.0,unscaled_char_mass));
          if( testVal_part < (-rateMax/(d_rc_scaling_factor*d_w_scaling_factor))) {
            testVal_part = -rateMax/(d_rc_scaling_factor*d_w_scaling_factor);
            testVal_gas = Yv*rateMax;
            testVal_char = (1.0-Yv)*rateMax;
          }
        }

        if( (testVal_part < -1e-16) && (unscaled_raw_coal_mass > 1e-16)) {
          devol_rate_ = testVal_part;
          gas_devol_rate_ = testVal_gas;
          char_rate_ = testVal_char;
        } else {
          devol_rate_ = 0.0;
          gas_devol_rate_ = 0.0;
          char_rate_ = 0.0;
        }
          
        //}
        //cout << "devol_rate_ " << devol_rate_ << " char_rate_ " << char_rate_ << " unscaled_char_mass " << unscaled_char_mass
        //     << " unscaled_raw_coal_mass " << unscaled_raw_coal_mass << endl;

      }

      devol_rate[c] = devol_rate_;
      gas_devol_rate[c] = gas_devol_rate_;
      char_rate[c] = char_rate_;

    }//end cell loop  
  }//end patch loop
}











