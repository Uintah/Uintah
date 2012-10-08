#include <CCA/Components/Arches/CoalModels/SimpleHeatTransfer.h>
#include <CCA/Components/Arches/CoalModels/PartVel.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/CoalModels/fortran/rqpart_fort.h>
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
SimpleHeatTransferBuilder::SimpleHeatTransferBuilder( const std::string         & modelName,
                                                      const vector<std::string> & reqICLabelNames,
                                                      const vector<std::string> & reqScalarLabelNames,
                                                      ArchesLabel         * fieldLabels,
                                                      SimulationStateP          & sharedState,
                                                      int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

SimpleHeatTransferBuilder::~SimpleHeatTransferBuilder(){}

ModelBase* SimpleHeatTransferBuilder::build() {
  return scinew SimpleHeatTransfer( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

SimpleHeatTransfer::SimpleHeatTransfer( std::string modelName, 
                                        SimulationStateP& sharedState,
                                        ArchesLabel* fieldLabels,
                                        vector<std::string> icLabelNames, 
                                        vector<std::string> scalarLabelNames,
                                        int qn ) 
: HeatTransfer(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  // Set constants
  Pr = 0.7;
  blow = 1.0;
  sigma = 5.67e-8;   // [=] J/s/m^2/K^4 : Stefan-Boltzmann constant (from white book)
  //rkg = 0.03;        // [=] J/s/m/K : thermal conductivity of gas

  pi = 3.14159265358979; 
}

SimpleHeatTransfer::~SimpleHeatTransfer()
{}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
SimpleHeatTransfer::problemSetup(const ProblemSpecP& params, int qn)
{
  HeatTransfer::problemSetup( params, qn );

  ProblemSpecP db = params; 
  
  // check for viscosity
  const ProblemSpecP params_root = db->getRootNode(); 
  if (params_root->findBlock("PhysicalConstants")) {
    ProblemSpecP db_phys = params_root->findBlock("PhysicalConstants");
    db_phys->require("viscosity", visc);
    if( visc == 0 ) {
      throw InvalidValue("ERROR: SimpleHeatTransfer: problemSetup(): Zero viscosity specified in <PhysicalConstants> section of input file.",__FILE__,__LINE__);
    }
  } else {
    throw InvalidValue("ERROR: SimpleHeatTransfer: problemSetup(): Missing <PhysicalConstants> section in input file!",__FILE__,__LINE__);
  }

  if (params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties")) {
    ProblemSpecP db_coal = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties");
    db_coal->require("C", yelem[0]);
    db_coal->require("H", yelem[1]);
    db_coal->require("N", yelem[2]);
    db_coal->require("O", yelem[3]);
    db_coal->require("S", yelem[4]);
    db_coal->require("initial_ash_mass", ash_mass_init);
    db_coal->require("initial_fixcarb_mass", fixcarb_mass_init);
  } else {
    throw InvalidValue("ERROR: SimpleHeatTransfer: problemSetup(): Missing <Coal_Properties> section in input file!",__FILE__,__LINE__);
  }

  // Assume no ash (for now)
  //d_ash = false;

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
      if( role_name == "gas_temperature" ) {
        // tempIN will be required explicitly
      } else if ( role_name == "particle_length" 
               || role_name == "raw_coal_mass"
               //|| role_name == "ash_mass"
               || role_name == "particle_temperature" ) {
        LabelToRoleMap[temp_label_name] = role_name;
      } else {
        std::string errmsg = "ERROR: SimpleHeatTransfer: problemSetup(): Invalid variable role for Simple Heat Transfer model!";
        throw InvalidValue(errmsg,__FILE__,__LINE__);
      }

      // set model clipping
      db->getWithDefault( "low_clip",  d_lowModelClip,  1.0e-6 );
      db->getWithDefault( "high_clip", d_highModelClip, 999999 );
    }
  }

  // Look for required scalars
  //   ( SimpleHeatTransfer model doesn't use any extra scalars (yet)
  //     but if it did, this "for" loop would have to be un-commented )
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
      /*
      if ( role_name == "raw_coal_mass") {
        LabelToRoleMap[temp_label_name] = role_name;
      } else if( role_name == "particle_temperature" ) {  
        LabelToRoleMap[temp_label_name] = role_name;
        compute_part_temp = true;
      } else {
        std::string errmsg;
        errmsg = "Invalid variable role for Simple Heat Transfer model: must be \"particle_temperature\" or \"raw_coal_mass\", you specified \"" + role_name + "\".";
        throw InvalidValue(errmsg,__FILE__,__LINE__);
      }
      */
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
SimpleHeatTransfer::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "SimpleHeatTransfer::initVars";
  Task* tsk = scinew Task(taskname, this, &SimpleHeatTransfer::initVars);

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
SimpleHeatTransfer::initVars( const ProcessorGroup * pc, 
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
SimpleHeatTransfer::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "SimpleHeatTransfer::computeModel";
  Task* tsk = scinew Task(taskname, this, &SimpleHeatTransfer::computeModel);

  d_timeSubStep = timeSubStep; 

  if (d_timeSubStep == 0 && !d_labelSchedInit) {
    // Every model term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;

    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel); 
    tsk->computes(d_abskpLabel);
  } else {
    tsk->modifies(d_modelLabel);
    tsk->modifies(d_gasLabel);  
    tsk->modifies(d_abskpLabel);
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
  tsk->requires(Task::OldDW, d_weight_label, Ghost::None, 0);
  
  // also require paticle velocity, gas velocity, and density
  ArchesLabel::PartVelMap::const_iterator iQuad = d_fieldLabels->partVel.find(d_quadNode);
  tsk->requires(Task::OldDW, iQuad->second, Ghost::None, 0);
  tsk->requires( Task::OldDW, d_fieldLabels->d_CCVelocityLabel, Ghost::None, 0);
  tsk->requires(Task::OldDW, d_fieldLabels->d_densityCPLabel, Ghost::None, 0);
  tsk->requires(Task::OldDW, d_fieldLabels->d_cpINLabel, Ghost::None, 0);
 
  if(_radiation){
    tsk->requires(Task::OldDW, d_fieldLabels->d_radiationSRCINLabel,  Ghost::None, 0);
    tsk->requires(Task::OldDW, d_fieldLabels->d_abskgINLabel,  Ghost::None, 0);   
    tsk->requires(Task::OldDW, d_fieldLabels->d_radiationVolqINLabel, Ghost::None, 0);
  }

  // always require the gas-phase temperature
  tsk->requires(Task::OldDW, d_fieldLabels->d_tempINLabel, Ghost::None, 0);

  // For each required variable, determine what role it plays
  // - "particle_temperature" - look in DQMOMEqnFactory
  // - "particle_length" - look in DQMOMEqnFactory
  // - "raw_coal_mass_fraction" - look in DQMOMEqnFactory

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
          std::string errmsg = "ARCHES: SimpleHeatTransfer: sched_computeModel(): Invalid variable given in <ICVars> block, for <variable> tag for SimpleHeatTransfer model.";
          errmsg += "\nCould not find given particle temperature variable \"";
          errmsg += *iter;
          errmsg += "\" in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }

      } else if( iMap->second == "particle_length" ) {
        if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_particle_length_label = current_eqn.getTransportEqnLabel();
          d_pl_scaling_constant = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_particle_length_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: SimpleHeatTransfer: sched_computeModel(): Invalid variable given in <ICVars> block, for <variable> tag for SimpleHeatTransfer model.";
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
          std::string errmsg = "ARCHES: SimpleHeatTransfer: sched_computeModel(): Invalid variable given in <ICVars> block, for <variable> tag for SimpleHeatTransfer model.";
          errmsg += "\nCould not find given coal mass  variable \"";
          errmsg += *iter;
          errmsg += "\" in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      } 
    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ARCHES: SimpleHeatTransfer: sched_computeModel(): You specified that the variable \"" + *iter + 
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
SimpleHeatTransfer::computeModel( const ProcessorGroup * pc, 
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

    CCVariable<double> heat_rate;
    if ( new_dw->exists( d_modelLabel, matlIndex, patch) ) {
      new_dw->getModifiable( heat_rate, d_modelLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( heat_rate, d_modelLabel, matlIndex, patch );
      heat_rate.initialize(0.0);
    }
    
    CCVariable<double> gas_heat_rate; 
    if( new_dw->exists( d_gasLabel, matlIndex, patch ) ) {
      new_dw->getModifiable( gas_heat_rate, d_gasLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( gas_heat_rate, d_gasLabel, matlIndex, patch );
      gas_heat_rate.initialize(0.0);
    }
    
    CCVariable<double> abskp; 
    if( new_dw->exists( d_abskpLabel, matlIndex, patch) ) {
      new_dw->getModifiable( abskp, d_abskpLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( abskp, d_abskpLabel, matlIndex, patch );
      abskp.initialize(0.0);
    }
    

    // get particle velocity used to calculate Reynolds number
    constCCVariable<Vector> partVel;  
    ArchesLabel::PartVelMap::const_iterator iQuad = d_fieldLabels->partVel.find(d_quadNode);
    old_dw->get( partVel, iQuad->second, matlIndex, patch, gn, 0);
    
    // gas velocity used to calculate Reynolds number
    constCCVariable<Vector> gasVel; 
    old_dw->get( gasVel, d_fieldLabels->d_CCVelocityLabel, matlIndex, patch, gn, 0 ); 
    
    constCCVariable<double> den;
    old_dw->get(den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0 ); 
    constCCVariable<double> cpg;
    old_dw->get(cpg, d_fieldLabels->d_cpINLabel, matlIndex, patch, gn, 0 );

    constCCVariable<double> radiationSRCIN;
    constCCVariable<double> abskgIN;
    constCCVariable<double> radiationVolqIN;
    CCVariable<double> enthNonLinSrc;

    if(_radiation){
      old_dw->get(radiationSRCIN, d_fieldLabels->d_radiationSRCINLabel, matlIndex, patch, gn, 0);
      old_dw->get(abskgIN, d_fieldLabels->d_abskgINLabel, matlIndex, patch, gn, 0);
      old_dw->get(radiationVolqIN, d_fieldLabels->d_radiationVolqINLabel, matlIndex, patch, gn, 0);
    }

    constCCVariable<double> temperature;
    constCCVariable<double> w_particle_temperature;
    constCCVariable<double> w_particle_length;
    constCCVariable<double> w_raw_coal_mass;
    constCCVariable<double> weight;

    old_dw->get( temperature, d_fieldLabels->d_tempINLabel, matlIndex, patch, gn, 0 );
    old_dw->get( w_particle_temperature, d_particle_temperature_label, matlIndex, patch, gn, 0 );
    old_dw->get( w_particle_length, d_particle_length_label, matlIndex, patch, gn, 0 );
    old_dw->get( w_raw_coal_mass, d_raw_coal_mass_label, matlIndex, patch, gn, 0 );
    old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );

#if !defined(VERIFY_SIMPLEHEATTRANSFER_MODEL)

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 
      IntVector cxp = *iter + IntVector(1,0,0);
      IntVector cxm = *iter - IntVector(1,0,0);
      IntVector cyp = *iter + IntVector(0,1,0);
      IntVector cym = *iter - IntVector(0,1,0);
      IntVector czp = *iter + IntVector(0,0,1);
      IntVector czm = *iter - IntVector(0,0,1);
      // define variables specific for non-verification runs:

      // velocities
      Vector gas_velocity = gasVel[c];
      Vector particle_velocity = partVel[c];

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

      // temperature - gas
      double gas_temperature = temperature[c];

      double unscaled_ash_mass = ash_mass_init[d_quadNode];
      double unscaled_fixcarb_mass = fixcarb_mass_init[d_quadNode];
      double density = den[c];
      // viscosity should be grabbed from data warehouse... right now it's constant

      double FSum = 0.0;

      double heat_rate_;
      double gas_heat_rate_;
      double abskp_;

      // intermediate calculation values
      double Re;
      double Nu;
      double Cpc;
      double Cph;
      double Cpa; 
      double mp_Cp;
      double rkg;
      double Q_convection;
      double Q_radiation;

      if (weight_is_small && !d_unweighted) {
        heat_rate_ = 0.0;
        gas_heat_rate_ = 0.0;
        abskp_ = 0.0;
      } else {

        if(d_unweighted){
          scaled_weight = weight[c];
          unscaled_weight = weight[c]*d_w_scaling_constant;
          unscaled_particle_temperature = w_particle_temperature[c]*d_pt_scaling_constant;
          unscaled_length = w_particle_length[c]*d_pl_scaling_constant;
          unscaled_raw_coal_mass = w_raw_coal_mass[c]*d_rc_scaling_constant;
        } else {
          scaled_weight = weight[c];
          unscaled_weight = weight[c]*d_w_scaling_constant;
          unscaled_particle_temperature = (w_particle_temperature[c]*d_pt_scaling_constant)/scaled_weight;
          unscaled_length = (w_particle_length[c]*d_pl_scaling_constant)/scaled_weight;
          unscaled_raw_coal_mass = (w_raw_coal_mass[c]*d_rc_scaling_constant)/scaled_weight;
        }

#else
        Vector gas_velocity       = Vector(3.0);
        Vector particle_velocity  = Vector(1.0);
        bool weight_is_small = false;
        double unscaled_weight = 1.0e6;
        double scaled_weight = unscaled_weight;
        double unscaled_particle_temperature = 2000.0;
        double d_pt_scaling_constant = 1.0;
        double gas_temperature = 2050;
        double unscaled_length = 1.0e-5;
        double unscaled_raw_coal_mass = 1.0e-8;
        double unscaled_ash_mass = 1.0e-9;
        double density = 1;
        visc = 1.0e-5;
        // redefine composition array
        yelem[0] = 0.75; // C
        yelem[1] = 0.05; // H
        yelem[2] = 0.00; // N
        yelem[3] = 0.20; // O
        yelem[4] = 0.00; // S
        double FSum = 0.0;
        double heat_rate_;
        double gas_heat_rate_;
        double abskp_;
#endif

        // Convection part: -----------------------

        // Reynolds number
        Re = abs(gas_velocity.length() - particle_velocity.length())*unscaled_length*density/visc;

        // Nusselt number
        Nu = 2.0 + 0.65*pow(Re,0.50)*pow(Pr,(1.0/3.0));

        // Heat capacity of raw coal
        Cpc = heatcp( unscaled_particle_temperature );

        // Heat capacity of char
        Cph = heatcph( unscaled_particle_temperature );
 
        // Heat capacity of ash
        Cpa = heatap( unscaled_particle_temperature );

        // Heat capacity
        mp_Cp = (Cpc*unscaled_raw_coal_mass + Cph*unscaled_fixcarb_mass + Cpa*unscaled_ash_mass);
   
        // Gas thermal conductivity
        rkg = props(gas_temperature, unscaled_particle_temperature); // [=] J/s/m/K

        // Q_convection (see Section 5.4 of LES_Coal document)
        Q_convection = Nu*pi*blow*rkg*unscaled_length*(gas_temperature - unscaled_particle_temperature);

        // Radiation part: -------------------------
        bool DO_NEW_ABSKP = false; 
        Q_radiation = 0.0;
        if ( _radiation  && DO_NEW_ABSKP){ 
          // New Glacier Code for ABSKP: 
          double qabs = 0.0; 
          double qsca = 0.0; 
          double init_ash_frac = 0.0; // THIS NEEDS TO BE FIXED!
          fort_rqpart( unscaled_length, unscaled_particle_temperature, unscaled_ash_mass, init_ash_frac, qabs, qsca ); 

          //what goes next?!
        } else if ( _radiation && !DO_NEW_ABSKP ) { 
          double Qabs = 0.8;
          double Apsc = (pi/4)*Qabs*pow(unscaled_length,2);
          double Eb = 4*sigma*pow(unscaled_particle_temperature,4);
          FSum = radiationVolqIN[c];    
          Q_radiation = Apsc*(FSum - Eb);
          abskp_ = pi/4*Qabs*unscaled_weight*pow(unscaled_length,2); 
        } else {
          abskp_ = 0.0;
        }
    
        if(d_unweighted){  
          heat_rate_ = (Q_convection + Q_radiation)/(mp_Cp*d_pt_scaling_constant);
        } else {
          heat_rate_ = (Q_convection + Q_radiation)*unscaled_weight/(mp_Cp*d_pt_scaling_constant*d_w_scaling_constant);
        }
        //cout << "Qconv " << Q_convection << " Qrad " << Q_radiation << endl;
        gas_heat_rate_ = -unscaled_weight*Q_convection;
 
      }


#if defined(VERIFY_SIMPLEHEATTRANSFER_MODEL)
      proc0cout << "****************************************************************" << endl;
      proc0cout << "Verification error, Simple Heat Trasnfer model: " << endl;
      proc0cout << endl;

      double error; 
      double verification_value;
      
      verification_value = 2979.4;
      error = ( (verification_value)-(Cpc) )/(verification_value);
      if( fabs(error) < 0.01 ) {
        proc0cout << "Verification for raw coal heat capacity successful:" << endl;
        proc0cout << "    Percent error = " << setw(4) << fabs(error)*100 << " \%, which is less than 1 percent." << endl;
      } else {
        proc0cout << "WARNING: VERIFICATION FOR RAW COAL HEAT CAPACITY FAILED!!! " << endl;
        proc0cout << "    Verification value  = " << verification_value << endl;
        proc0cout << "    Calculated value    = " << Cpc << endl;
        proc0cout << "    Percent error = " << setw(4) << setprecision(4) << fabs(error)*100 << " \%, which is greater than 1 percent." << endl;
      }

      proc0cout << endl;

      verification_value = 1765.00;
      error = ( (verification_value)-(Cpa) )/(verification_value);
      if( fabs(error) < 0.01 ) {
        proc0cout << "Verification for ash heat capacity successful:" << endl;
        proc0cout << "    Percent error = " << setw(4) << setprecision(4) << fabs(error)*100 << " \%, which is less than 1 percent." << endl;
      } else {
        proc0cout << "WARNING: VERIFICATION FOR ASH HEAT CAPACITY FAILED!!! " << endl;
        proc0cout << "    Verification value  = " << verification_value << endl;
        proc0cout << "    Calculated value    = " << Cpa << endl;
        proc0cout << "    Percent error = " << setw(4) << setprecision(4) << fabs(error)*100 << " \%, which is greater than 1 percent." << endl;
      }

      proc0cout << endl;
      
      verification_value = 4.6985e-4;
      error = ( (verification_value)-(Q_convection) )/(verification_value);
      if( fabs(error) < 0.01 ) {
        proc0cout << "Verification for convection heat transfer term successful:" << endl;
        proc0cout << "    Percent error = " << setw(4) << setprecision(4) << fabs(error)*100 << " \%, which is less than 1 percent." << endl;
      } else {
        proc0cout << "WARNING: VERIFICATION FOR CONVECTION HEAT TRANSFER TERM FAILED!!! " << endl;
        proc0cout << "    Verification value  = " << verification_value << endl;
        proc0cout << "    Calculated value    = " << Q_convection << endl;
        proc0cout << "    Percent error = " << setw(4) << setprecision(4) << fabs(error)*100 << " \%, which is greater than 1 percent." << endl;
      }

      proc0cout << endl;

      verification_value = 0.097312;
      error = ( (verification_value)-(rkg) )/(verification_value);
      if( fabs(error) < 0.01 ) {
        proc0cout << "Verification for gas thermal conductivity successful:" << endl;
        proc0cout << "    Percent error = " << setw(5) << setprecision(4) << fabs(error)*100 << " \%, which is less than 1 percent." << endl;
      } else {
        proc0cout << "WARNING: VERIFICATION FOR GAS THERMAL CONDUCTIVITY FAILED!!! " << endl;
        proc0cout << "    Verification value  = " << verification_value << endl;
        proc0cout << "    Calculated value    = " << rkg << endl;
        proc0cout << "    Percent error = " << setw(5) << setprecision(4) << fabs(error)*100 << " \%, which is greater than 1 percent." << endl;
      }

      proc0cout << endl;
      
      verification_value = 14.888;
      error = ( (verification_value)-(heat_rate_) )/(verification_value);
      if( fabs(error) < 0.01 ) {
        proc0cout << "Verification for particle heating rate term successful:" << endl;
        proc0cout << "    Percent error = " << setw(5) << setprecision(4) << fabs(error)*100 << " \%, which is less than 1 percent." << endl;
      } else {
        proc0cout << "WARNING: VERIFICATION FOR PARTICLE HEATING RATE TERM FAILED!!! " << endl;
        proc0cout << "    Verification value  = " << verification_value << endl;
        proc0cout << "    Calculated value    = " << heat_rate_ << endl;
        proc0cout << "    Percent error = " << setw(5) << setprecision(4) << fabs(error)*100 << " \%, which is greater than 1 percent." << endl;
      }

      proc0cout << endl;
      proc0cout << "****************************************************************" << endl;

      proc0cout << endl << endl;

#else
      heat_rate[c] = heat_rate_;
      gas_heat_rate[c] = gas_heat_rate_;
      abskp[c] = abskp_;
 
    }//end cell loop
#endif

  }//end patch loop
}



// ********************************************************
// Private methods:

double
SimpleHeatTransfer::g1( double z){
  double dum1 = (exp(z)-1)/z;
  double dum2 = pow(dum1,2);
  double sol = exp(z)/dum2;
  return sol;
}

double
SimpleHeatTransfer::heatcp(double Tp){
  if (Tp < 700) {
    // correlation is not valid
    return 1046.0;
  } else {
    double MW [5] = { 12., 1., 14., 16., 32.}; // Atomic weight of elements (C,H,N,O,S)
    double Rgas = 8314.3; // J/kg/K

    double MW_avg = 0.0; // Mean atomic weight of coal
    for(int i=0;i<5;i++){
      MW_avg += yelem[i]/MW[i];
    }
    MW_avg = 1/MW_avg;

    double z1 = 380.0/Tp;
    double z2 = 1800.0/Tp;
    double cp = (Rgas/MW_avg)*(g1(z1)+2.0*g1(z2));
    return cp; // J/kg/K
  }
}

double
SimpleHeatTransfer::heatcph(double Tp){
  if (Tp < 700) {
    // correlation is not valid
    return 1046.0;
  } else {
    double Rgas = 8314.3; // J/kg/K

    double z1 = 380.0/Tp;
    double z2 = 1800.0/Tp;
    double cp = (Rgas/12.0)*(g1(z1)+2.0*g1(z2));
    return cp; // J/kg/K
  }
}

double
SimpleHeatTransfer::heatap(double Tp){
  // c.f. PCGC2
  double cpa = 593.0 + 0.586*Tp;
  return cpa;  // J/kg/K
}


double
SimpleHeatTransfer::props(double Tg, double Tp){

  double tg0[10] = {300.,  400.,   500.,   600.,  700.,  800.,  900.,  1000., 1100., 1200. };
  double kg0[10] = {.0262, .03335, .03984, .0458, .0512, .0561, .0607, .0648, .0685, .07184};
  double T = (Tp+Tg)/2; // Film temperature

//   CALCULATE UG AND KG FROM INTERPOLATION OF TABLE VALUES FROM HOLMAN
//   FIND INTERVAL WHERE TEMPERATURE LIES. 

  double kg = 0.0;

  if( T > 1200.0 ) {
    kg = kg0[9] * pow( T/tg0[9], 0.58);

  } else if ( T < 300 ) {
    kg = kg0[0];
  
  } else {
    int J = -1;
    for ( int I=0; I < 9; I++ ) {
      if ( T > tg0[I] ) {
        J = J + 1;
      }
    }
    double FAC = ( tg0[J] - T ) / ( tg0[J] - tg0[J+1] );
    kg = ( -FAC*( kg0[J] - kg0[J+1] ) + kg0[J] );
  }

  return kg; // I believe this is in J/s/m/K, but not sure
}

