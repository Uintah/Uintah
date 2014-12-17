#include <CCA/Components/Arches/CoalModels/RichardsFletcherDevol.h>
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
RichardsFletcherDevolBuilder::RichardsFletcherDevolBuilder( const std::string         & modelName,
                                                            const vector<std::string> & reqICLabelNames,
                                                            const vector<std::string> & reqScalarLabelNames,
                                                            ArchesLabel         * fieldLabels,
                                                            SimulationStateP          & sharedState,
                                                            int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

RichardsFletcherDevolBuilder::~RichardsFletcherDevolBuilder(){}

ModelBase* RichardsFletcherDevolBuilder::build() {
  return scinew RichardsFletcherDevol( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

RichardsFletcherDevol::RichardsFletcherDevol( std::string modelName, 
                                              SimulationStateP& sharedState,
                                              ArchesLabel* fieldLabels,
                                              vector<std::string> icLabelNames, 
                                              vector<std::string> scalarLabelNames,
                                              int qn ) 
: Devolatilization(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  pi = 3.141592653589793;


  compute_part_temp = false;
  compute_char_mass = false;
  part_temp_from_enth = false;
}

RichardsFletcherDevol::~RichardsFletcherDevol()
{
}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
  void 
RichardsFletcherDevol::problemSetup(const ProblemSpecP& params, int qn)
{
  // call parent's method first
  Devolatilization::problemSetup(params, qn);

  ProblemSpecP db = params; 
  compute_part_temp = false;
  compute_char_mass = false;

  string label_name;
  string role_name;
  string temp_label_name;
  
  string temp_ic_name;
  string temp_ic_name_full;

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
        errmsg = "Invalid variable role for Richards Fletcher Devolatilization model: must be \"particle_temperature\" or \"raw_coal_mass\" or \"char_mass\", you specified \"" + role_name + "\".";
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
  const ProblemSpecP params_root = db->getRootNode();
  if (params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties")) {
    ProblemSpecP db_coal = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties");
    db_coal->require("RichardsFletcher_coefficients", RichardsFletcher_coefficients);
    // Values from Ubhayakar (1976):
    Av1=RichardsFletcher_coefficients[0];  // [=] 1/s; k1 pre-exponential factor
    Av2=RichardsFletcher_coefficients[1];  // [=] 1/s; k2 pre-exponential factor
    Ev1=RichardsFletcher_coefficients[2];  // [=] K; k1 activation energy
    Ev2=RichardsFletcher_coefficients[3];  // [=] K;  k2 activation energy
    // Y values from white book:
    Y1_=RichardsFletcher_coefficients[4];  // volatile fraction from proximate analysis
    Y2_=RichardsFletcher_coefficients[5];  // fraction devolatilized at higher temperatures
    c0_1=RichardsFletcher_coefficients[6]; 
    c1_1=RichardsFletcher_coefficients[7]; 
    c2_1=RichardsFletcher_coefficients[8]; 
    c3_1=RichardsFletcher_coefficients[9]; 
    c4_1=RichardsFletcher_coefficients[10]; 
    c5_1=RichardsFletcher_coefficients[11]; 
    c6_1=RichardsFletcher_coefficients[12]; 
    c0_2=RichardsFletcher_coefficients[13]; 
    c1_2=RichardsFletcher_coefficients[14]; 
    c2_2=RichardsFletcher_coefficients[15]; 
    c3_2=RichardsFletcher_coefficients[16]; 
    c4_2=RichardsFletcher_coefficients[17]; 
    c5_2=RichardsFletcher_coefficients[18]; 
    c5_2=RichardsFletcher_coefficients[19]; 
    c6_2=RichardsFletcher_coefficients[20]; 
    db_coal->require("particle_density", rhop);
    db_coal->require("particle_sizes", particle_sizes); // read the particle sizes [m]
    db_coal->require("as_received", as_received);
    total_rc=as_received[0]+as_received[1]+as_received[2]+as_received[3]+as_received[4]; // (C+H+O+N+S) dry ash free total
    total_dry=as_received[0]+as_received[1]+as_received[2]+as_received[3]+as_received[4]+as_received[5]+as_received[6]; // (C+H+O+N+S+char+ash)  moisture free total
    rc_mass_frac=total_rc/total_dry; // mass frac of rc (dry) 
    char_mass_frac=as_received[5]/total_dry; // mass frac of char (dry)
    ash_mass_frac=as_received[6]/total_dry; // mass frac of ash (dry)
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
    throw InvalidValue("ERROR: RichardsFletcherDevol: problemSetup(): Missing <Coal_Properties> section in input file!",__FILE__,__LINE__);
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
}

//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model 
//---------------------------------------------------------------------------
void 
RichardsFletcherDevol::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "RichardsFletcherDevol::computeModel";
  Task* tsk = scinew Task(taskname, this, &RichardsFletcherDevol::computeModel);

  Ghost::GhostType gn = Ghost::None;

  d_timeSubStep = timeSubStep; 

  tsk->requires( Task::OldDW, d_fieldLabels->d_sharedState->get_delt_label(), Ghost::None, 0);

  if (d_timeSubStep == 0 ) {
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
  d_gas_temperature_label = VarLabel::find( "temperature" ); 
  if ( d_gas_temperature_label == 0 ){ 
    throw InvalidValue("Error: Unable to find gas temperature label.",__FILE__,__LINE__);
  }
  tsk->requires(Task::OldDW, d_gas_temperature_label, Ghost::AroundCells, 1);

  // For each required variable, determine what role it plays
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
          std::string errmsg = "ARCHES: RichardsFletcherDevol: Invalid variable given in <variable> tag for RichardsFletcherDevol model";
          errmsg += "\nCould not find given particle temperature variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory or in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }


      } else if ( iMap->second == "particle_temperature_from_enthalpy") {
          d_particle_temperature_label = VarLabel::find(iMap->first);
          d_pt_scaling_factor = 1;
          tsk->requires(Task::OldDW, d_particle_temperature_label, Ghost::None, 0);

      } else if ( iMap->second == "raw_coal_mass") {
        if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_raw_coal_mass_label = current_eqn.getTransportEqnLabel();
          d_rc_scaling_factor = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_raw_coal_mass_label, Ghost::None, 0);

        } else {
          std::string errmsg = "ARCHES: RichardsFletcherDevol: Invalid variable given in <variable> tag for RichardsFletcherDevol model";
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
          std::string errmsg = "ARCHES: RichardsFletcherDevol: Invalid variable given in <variable> tag for RichardsFletcherDevol model";
          errmsg += "\nCould not find given raw coal mass variable \"";
          errmsg += *iter;
          errmsg += "\" in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      }

    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ARCHES: RichardsFletcherDevol: You specified that the variable \"" + *iter + 
                           "\" was required, but you did not specify a role for it!\n";
      throw InvalidValue( errmsg, __FILE__, __LINE__);
    }
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}

//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
RichardsFletcherDevol::computeModel( const ProcessorGroup * pc, 
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
    if (compute_part_temp) {
      old_dw->get( temperature, d_particle_temperature_label, matlIndex, patch, gn, 0 );
    }// else {
     // old_dw->get( temperature, d_gas_temperature_label, matlIndex, patch, gac, 1 );
    //}
    if (part_temp_from_enth) {
      old_dw->get( temperature, d_particle_temperature_label, matlIndex, patch, gn, 0 );
    }// else {
    constCCVariable<double> wa_raw_coal_mass;
    old_dw->get( wa_raw_coal_mass, d_raw_coal_mass_label, matlIndex, patch, gn, 0 );

    constCCVariable<double> wa_char_mass;
    if(compute_char_mass){
      old_dw->get( wa_char_mass, d_char_mass_label, matlIndex, patch, gn, 0 );
    }

    constCCVariable<double> weight;
    old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );

#if !defined(VERIFY_RICHARDS_MODEL)

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
            unscaled_temperature = temperature[c];
          } 
        } else {
          unscaled_weight = weight[c]*d_w_scaling_factor;
          if (compute_part_temp) {
            // particle temp
            unscaled_temperature = temperature[c]*d_pt_scaling_factor/weight[c];
          } else { 
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

#else
        bool weight_is_small = false;
        double unscaled_weight = 1e6;
        double unscaled_temperature = 2000;
        double unscaled_raw_coal_mass = 1e-8;
        double scaled_raw_coal_mass = unscaled_raw_coal_mass;
        double unscaled_char_mass = 1e-8;
        double scaled_char_mass = unscaled_char_mass;
        double devol_rate_;
        double gas_devol_rate_;
        double char_rate_;
#endif
        double unscaled_rawcoal_mass = rc_mass_init[d_quadNode];
        Xv = (unscaled_rawcoal_mass-unscaled_raw_coal_mass)/unscaled_rawcoal_mass;
        Xv = min(max(Xv,0.0),1.0);
        
        Fv1 = c6_1*pow(Xv,6.0) + c5_1*pow(Xv,5.0) + c4_1*pow(Xv,4.0) + c3_1*pow(Xv,3.0) + c2_1*pow(Xv,2.0) + c1_1*Xv +c0_1;
        k1 = exp(Fv1)*Av1*exp(-Ev1/(unscaled_temperature));
        
        Fv2 = c6_2*pow(Xv,6.0) + c5_2*pow(Xv,5.0) + c4_2*pow(Xv,4.0) + c3_2*pow(Xv,3.0) + c2_2*pow(Xv,2.0) + c1_2*Xv +c0_2;
        k2 = exp(Fv2)*Av2*exp(-Ev2/(unscaled_temperature));
 
        if(d_unweighted){ 
          rateMax = max((0.2*(unscaled_raw_coal_mass + min(0.0,unscaled_char_mass))/dt),0.0);
          testVal_part = -(k1+k2)*(unscaled_raw_coal_mass + min(0.0,unscaled_char_mass))/d_rc_scaling_factor;
          testVal_gas = (Y1_*k1 + Y2_*k2)*(unscaled_raw_coal_mass+ min(0.0,unscaled_char_mass))*unscaled_weight;
          testVal_char = ((1.0-Y1_)*k1 + (1.0-Y2_)*k2)*(unscaled_raw_coal_mass + min(0.0,unscaled_char_mass));
          if( testVal_part < (-rateMax/d_rc_scaling_factor)) {
            testVal_part = -rateMax/(d_rc_scaling_factor);
            testVal_gas = Y1_*rateMax;
            testVal_char = (1.0-Y1_)*rateMax;
          }
        } else {
          rateMax = max((0.2*(unscaled_raw_coal_mass + min(0.0,unscaled_char_mass))*unscaled_weight/dt),0.0);
          testVal_part = -(k1+k2)*(unscaled_raw_coal_mass + min(0.0,unscaled_char_mass))*unscaled_weight/(d_rc_scaling_factor*d_w_scaling_factor);
          testVal_gas = (Y1_*k1 + Y2_*k2)*(unscaled_raw_coal_mass+ min(0.0,unscaled_char_mass))*unscaled_weight;
          testVal_char = ((1.0-Y1_)*k1 + (1.0-Y2_)*k2)*(unscaled_raw_coal_mass + min(0.0,unscaled_char_mass))*unscaled_weight;
          if( testVal_part < (-rateMax/(d_rc_scaling_factor*d_w_scaling_factor))) {
            testVal_part = -rateMax/(d_rc_scaling_factor*d_w_scaling_factor);
            testVal_gas = Y1_*rateMax;
            testVal_char = (1.0-Y1_)*rateMax;
          }
        }

        if( (testVal_part < -1e-16) && ((unscaled_raw_coal_mass+min(0.0,unscaled_char_mass))> 1e-16)) {
          devol_rate_ = testVal_part;
          gas_devol_rate_ = testVal_gas;
          char_rate_ = testVal_char;
        } else {
          devol_rate_ = 0.0;
          gas_devol_rate_ = 0.0;
          char_rate_ = 0.0;
        }
      }      

#if defined(VERIFY_RICHARDS_MODEL)
      proc0cout << "****************************************************************" << endl;
      proc0cout << "Verification error, Richards-Fletcher Devolatilization model:   " << endl;
      proc0cout << endl;

      double error; 
      
      error = ( (-0.04053)-(devol_rate_) )/(-0.04053);
      if( fabs(error) < 0.01 ) {
        proc0cout << "Verification for particle model term successful:" << endl;
        proc0cout << "    Percent error is " << fabs(error)*100 << ", which is less than 1 percent." << endl;
      } else {
        proc0cout << "WARNING: VERIFICATION FOR PARTICLE MODEL TERM FAILED!!! " << endl;
        proc0cout << "    Verification value  = -0.04053" << endl;
        proc0cout << "    Calculated value    = " << devol_rate_ << endl;
        proc0cout << "    Percent error = " << fabs(error)*100 << ", which is greater than 1 percent." << endl;
      }

      proc0cout << endl;

      error = ( (40500.3) - (gas_devol_rate_) )/(40500.3);
      if( fabs(error) < 0.01 ) {
        proc0cout << "Verification for gas model term successful:" << endl;
        proc0cout << "    Percent error = " << fabs(error)*100 << ", which is less than 1 percent." << endl;
      } else {
        proc0cout << "WARNING: VERIFICATION FOR GAS MODEL TERM FAILED!!! " << endl;
        proc0cout << "    Verification value  = 40500.3" << endl;
        proc0cout << "    Calculated value    = " << gas_devol_rate_ << endl;
        proc0cout << "    Percent error = " << fabs(error)*100 << ", which is greater than 1 percent." << endl;
      }

      proc0cout << endl;
      proc0cout << "****************************************************************" << endl;

#else
      devol_rate[c] = devol_rate_;
      gas_devol_rate[c] = gas_devol_rate_;
      char_rate[c] = char_rate_;
    }//end cell loop
#endif
  
  }//end patch loop
}











