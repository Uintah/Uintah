#include <CCA/Components/Arches/CoalModels/SimpleHeatTransfer.h>
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


using namespace std;
using namespace Uintah; 

//---------------------------------------------------------------------------
// Builder:
SimpleHeatTransferBuilder::SimpleHeatTransferBuilder( const std::string         & modelName,
                                                            const vector<std::string> & reqICLabelNames,
                                                            const vector<std::string> & reqScalarLabelNames,
                                                            const ArchesLabel         * fieldLabels,
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
                                        const ArchesLabel* fieldLabels,
                                        vector<std::string> icLabelNames, 
                                        vector<std::string> scalarLabelNames,
                                        int qn ) 
: HeatTransfer(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  // Create a label for this model
  d_modelLabel = VarLabel::create( modelName, CCVariable<double>::getTypeDescription() );

  // Create the gas phase source term associated with this model
  std::string gasSourceName = modelName + "_gasSource";
  d_gasLabel = VarLabel::create( gasSourceName, CCVariable<double>::getTypeDescription() );
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
  } else
    throw InvalidValue("Missing <PhysicalConstants> section in input file!",__FILE__,__LINE__);

  if (params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties")) {
    ProblemSpecP db_coal = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties");
    db_coal->require("C", yelem[0]);
    db_coal->require("H", yelem[1]);
    db_coal->require("N", yelem[2]);
    db_coal->require("O", yelem[3]);
    db_coal->require("S", yelem[4]);
  } else
    throw InvalidValue("Missing <Coal_Properties> section in input file!",__FILE__,__LINE__);

  // Assume no ash (for now)
  d_ash = false;

  // Look for required internal coordinates
  ProblemSpecP db_icvars = params->findBlock("ICVars");
  for (ProblemSpecP variable = db_icvars->findBlock("variable"); variable != 0; variable = variable->findNextBlock("variable") ) {
  
    string label_name;
    string role_name;
    string temp_label_name;
    
    variable->getAttribute("label",label_name);
    variable->getAttribute("role",role_name);

    temp_label_name = label_name;
    
    string node;
    std::stringstream out;
    out << qn;
    node = out.str();
    temp_label_name += "_qn";
    temp_label_name += node;

    // user specifies "role" of each internal coordinate
    // if it isn't an internal coordinate or a scalar, it's required explicitly
    // ( see comments in Arches::registerModels() for details )
    if( role_name == "gas_temperature" ) {
      // tempIN will be required explicitly
    } else if ( role_name == "particle_length" 
             || role_name == "raw_coal_mass"
             || role_name == "ash_mass"
             || role_name == "particle_temperature" ) {
      LabelToRoleMap[temp_label_name] = role_name;
    } else {
      std::string errmsg = "Invalid variable role for Simple Heat Transfer model!";
      throw InvalidValue(errmsg,__FILE__,__LINE__);
    }

    // set model clipping
    db->getWithDefault( "low_clip",  d_lowModelClip,  1.0e-6 );
    db->getWithDefault( "high_clip", d_highModelClip, 999999 );
 
  }

  // Look for required scalars
  //   ( SimpleHeatTransfer model doesn't use any extra scalars (yet)
  //     but if it did, this "for" loop would have to be un-commented )
  /*
  ProblemSpecP db_scalarvars = params->findBlock("scalarVars");
  for( ProblemSpecP variable = db_scalarvars->findBlock("variable");
       variable != 0; variable = variable->findNextBlock("variable") ) {

    string label_name;
    string role_name;
    string temp_label_name;

    variable->getAttribute("label", label_name);
    variable->getAttribute("role",  role_name);

    temp_label_name = label_name;

    string node;
    std::stringstream out;
    out << qn;
    node = out.str();
    temp_label_name += "_qn";
    temp_label_name += node;

    // user specifies "role" of each scalar
    // if it isn't an internal coordinate or a scalar, it's required explicitly
    // ( see comments in Arches::registerModels() for details )
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

  }
  */



  // fix the d_icLabels to point to the correct quadrature node (since there is 1 model per quad node)
  for ( vector<std::string>::iterator iString = d_icLabels.begin(); 
        iString != d_icLabels.end(); ++iString) {
    
    string temp_ic_name;
    string temp_ic_name_full;

    temp_ic_name      = (*iString);
    temp_ic_name_full = temp_ic_name;

    string node;
    std::stringstream out;
    out << qn;
    node = out.str();
    temp_ic_name_full += "_qn";
    temp_ic_name_full += node;

    std::replace( d_icLabels.begin(), d_icLabels.end(), temp_ic_name, temp_ic_name_full);
  }

  // fix the d_scalarLabels to point to the correct quadrature node (since there is 1 model per quad node)
  // (Not needed for SimpleHeatTransfer model (yet)... If it is, uncomment the block below)
  /*
  for ( vector<std::string>::iterator iString = d_scalarLabels.begin(); 
        iString != d_scalarLabels.end(); ++iString) {

    string temp_ic_name;
    string temp_ic_name_full;

    temp_ic_name      = (*iString);
    temp_ic_name_full = temp_ic_name;

    string node;
    std::stringstream out;
    out << qn;
    node = out.str();
    temp_ic_name_full += "_qn";
    temp_ic_name_full += node;

    std::replace( d_scalarLabels.begin(), d_scalarLabels.end(), temp_ic_name, temp_ic_name_full);
  }
  */

  string node;
  std::stringstream out;
  out << qn; 
  node = out.str();

  // thermal conductivity (of particles, I think???)
  std::string abskpName = "abskp_qn";
  abskpName += node; 
  d_abskp = VarLabel::create(abskpName, CCVariable<double>::getTypeDescription());

}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void 
SimpleHeatTransfer::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "SimpleHeatTransfer::initVars";
  Task* tsk = scinew Task(taskname, this, &SimpleHeatTransfer::initVars);

  tsk->computes(d_abskp);

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
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> abskp; 
    new_dw->allocateAndPut( abskp, d_abskp, matlIndex, patch ); 
    abskp.initialize(0.0);

  }
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
    tsk->computes(d_abskp);
  } else {
    tsk->modifies(d_modelLabel);
    tsk->modifies(d_gasLabel);  
    tsk->modifies(d_abskp);
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
  d_w_small = weight_eqn.getSmallClip();
  d_w_scaling_factor = weight_eqn.getScalingConstant();
  tsk->requires(Task::OldDW, d_weight_label, Ghost::None, 0);
  
  // also require paticle velocity, gas velocity, and density
  ArchesLabel::PartVelMap::const_iterator iQuad = d_fieldLabels->partVel.find(d_quadNode);
  tsk->requires(Task::OldDW, iQuad->second, Ghost::None, 0);
  tsk->requires( Task::OldDW, d_fieldLabels->d_newCCVelocityLabel, Ghost::None, 0);
  tsk->requires(Task::OldDW, d_fieldLabels->d_densityCPLabel, Ghost::None, 0);
  tsk->requires(Task::OldDW, d_fieldLabels->d_cpINLabel, Ghost::None, 0);
 
  if(d_radiation){
    tsk->requires(Task::OldDW, d_fieldLabels->d_radiationSRCINLabel,  Ghost::None, 0);
    tsk->requires(Task::OldDW, d_fieldLabels->d_abskgINLabel,  Ghost::None, 0);   
  }

  // always require the gas-phase temperature
  tsk->requires(Task::OldDW, d_fieldLabels->d_tempINLabel, Ghost::AroundCells, 1);

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
          d_pt_scaling_factor = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_particle_temperature_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: SimpleHeatTransfer: Invalid variable given in <ICVars> block, for <variable> tag for SimpleHeatTransfer model.";
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
          d_pl_scaling_factor = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_particle_length_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: SimpleHeatTransfer: Invalid variable given in <ICVars> block, for <variable> tag for SimpleHeatTransfer model.";
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
          d_rc_scaling_factor = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_raw_coal_mass_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: SimpleHeatTransfer: Invalid variable given in <ICVars> block, for <variable> tag for SimpleHeatTransfer model.";
          errmsg += "\nCould not find given coal mass  variable \"";
          errmsg += *iter;
          errmsg += "\" in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      } else if ( iMap->second == "ash_mass") {
        if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_ash_mass_label = current_eqn.getTransportEqnLabel();
          d_ash_scaling_factor = current_eqn.getScalingConstant();
          d_ash = true;
          tsk->requires(Task::OldDW, d_ash_mass_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: SimpleHeatTransfer: Invalid variable given in <ICVars> block, for <variable> tag for SimpleHeatTransfer model.";
          errmsg += "\nCould not find given ash mass  variable \"";
          errmsg += *iter;
          errmsg += "\" in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }


      } 
    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ARCHES: SimpleHeatTransfer: You specified that the variable \"" + *iter + 
                           "\" was required, but you did not specify a role for it!\n";
      throw InvalidValue( errmsg, __FILE__, __LINE__);
    }
  }

  // for each required scalar variable:
  //  (but no scalar equation variables should be required for the SimpleHeatTransfer model, at least not for now...)
  /*
  for( vector<std::string>::iterator iter = d_scalarLabels.begin();
       iter != d_scalarLabels.end(); ++iter) {
    map<string, string>::iterator iMap = LabelToRoleMap.find(*iter);
    
    if( iMap != LabelToRoleMap.end() ) {
      if( iMap->second == <insert role name here> ) {
        if( eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& current_eqn = eqn_factory.retrieve_scalar_eqn(*iter);
          d_<insert role name here>_label = current_eqn.getTransportEqnLabel();
          tsk->requires(Task::OldDW, d_<insert role name here>_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: SimpleHeatTransfer: Invalid variable given in <scalarVars> block for <variable> tag for SimpleHeatTransfer model.";
          errmsg += "\nCould not find given <insert role name here> variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      }
    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ARCHES: SimpleHeatTransfer: You specified that the variable \"" + *iter + 
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
SimpleHeatTransfer::computeModel( const ProcessorGroup * pc, 
                                  const PatchSubset    * patches, 
                                  const MaterialSubset * matls, 
                                  DataWarehouse        * old_dw, 
                                  DataWarehouse        * new_dw )
{
  double pi = acos(-1.0);
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
    if( new_dw->exists( d_abskp, matlIndex, patch) ) {
      new_dw->getModifiable( abskp, d_abskp, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( abskp, d_abskp, matlIndex, patch );
      abskp.initialize(0.0);
    }
    

    // get particle velocity used to calculate Reynolds number
    constCCVariable<Vector> partVel;  
    ArchesLabel::PartVelMap::const_iterator iQuad = d_fieldLabels->partVel.find(d_quadNode);
    old_dw->get( partVel, iQuad->second, matlIndex, patch, gn, 0);
    
    // gas velocity used to calculate Reynolds number
    constCCVariable<Vector> gasVel; 
    old_dw->get( gasVel, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0 ); 
    
    constCCVariable<double> den;
    old_dw->get(den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0 ); 
    constCCVariable<double> cpg;
    old_dw->get(cpg, d_fieldLabels->d_cpINLabel, matlIndex, patch, gn, 0 );

    constCCVariable<double> radiationSRCIN;
    constCCVariable<double> abskgIN;
    CCVariable<double> enthNonLinSrc;

    if(d_radiation){
      old_dw->get(radiationSRCIN, d_fieldLabels->d_radiationSRCINLabel, matlIndex, patch, gn, 0);
      old_dw->get(abskgIN,        d_fieldLabels->d_abskgINLabel,        matlIndex, patch, gn, 0);
    }

    constCCVariable<double> temperature;
    constCCVariable<double> w_particle_temperature;
    constCCVariable<double> w_particle_length;
    constCCVariable<double> w_mass_raw_coal;
    constCCVariable<double> w_mass_ash;
    constCCVariable<double> weight;

    old_dw->get( temperature, d_fieldLabels->d_tempINLabel, matlIndex, patch, gn, 0 );
    old_dw->get( w_particle_temperature, d_particle_temperature_label, matlIndex, patch, gn, 0 );
    old_dw->get( w_particle_length, d_particle_length_label, matlIndex, patch, gn, 0 );
    old_dw->get( w_mass_raw_coal, d_raw_coal_mass_label, matlIndex, patch, gn, 0 );
    old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );
    if (d_ash) 
      old_dw->get( w_mass_ash, d_ash_mass_label, matlIndex, patch, gn, 0 );

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 

      Vector sphGas = Vector(0.,0.,0.);
      Vector cartGas = gasVel[c]; 
      Vector sphPart = Vector(0.,0.,0.);
      Vector cartPart = partVel[c]; 

      sphGas = cart2sph( cartGas ); 
      sphPart = cart2sph( cartPart ); 
	
      double length;
      double particle_temperature;
      double rawcoal_mass;
      double ash_mass;

      if (weight[c] < d_w_small ) { 
        heat_rate[c] = 0.0;
        gas_heat_rate[c] = 0.0;
      } else {
	      length = w_particle_length[c]*d_pl_scaling_factor/weight[c];
	      particle_temperature = w_particle_temperature[c]*d_pt_scaling_factor/weight[c];
        rawcoal_mass = w_mass_raw_coal[c]*d_rc_scaling_factor/weight[c];
        if(d_ash) {
          ash_mass = w_mass_ash[c]*d_ash_scaling_factor/weight[c];
        } else {
          ash_mass = 0.0;
        }

        double Pr = 0.7; // Prandtl number
        double blow = 1.0;
        double sigma = 5.67e-8; // [=] J/(s-m^2-K^4) : Stefan-Boltzmann constant from white book p. 354
        double rkg = 0.03; // [=] J/(s-m-K) : thermal conductivity of gas
        double Re  = abs(sphGas.z() - sphPart.z())*length*den[c]/visc;
        double Nu = 2.0 + 0.6*pow(Re,0.5)*pow(Pr,0.333); // Nusselt number

        // Calculate thermal conductivity
        rkg = props(temperature[c],particle_temperature);
        double cpc = heatcp(particle_temperature);
        double cpa = heatap(particle_temperature);
        double m_p = rawcoal_mass + ash_mass;
        double cp = (cpc*rawcoal_mass + cpa*ash_mass)/m_p;
        //cout << "heat capacities " << cpc << " " << cp << " " << rkg << " " <<  weight[c] << "ash " << ash_mass << endl;
        double Qconv = Nu*pi*blow*rkg*length*(temperature[c]-particle_temperature);

        // Radiative transfer
	      double Qrad = 0.0;
        if(d_radiation) {
	        if(abskgIN[c]<1e-6){
	          Qrad = 0;
	        } else {
	          double Qabs = 0.8;
	          double Apsc = (pi/4)*Qabs*pow(length/2,2);
	          double Eb = 4.0*sigma*pow(particle_temperature,4);
	          double Eg = 4.0*sigma*abskgIN[c]*pow(temperature[c],4);
	          Qrad = Apsc*((radiationSRCIN[c]+ Eg)/abskgIN[c] - Eb);
	          abskp[c] = pi/4*Qabs*weight[c]*pow(length,2);
          }
        }

        heat_rate[c] =(Qconv+Qrad)/(m_p*cp*d_pt_scaling_factor); 

        gas_heat_rate[c] = 0.0; // change this to get two-way coupling...
    	}
    }
  }
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
  double u [5] = { 12., 1., 14., 16., 32.}; // Atomic weight of elements (C,H,N,O,S)
  double rgas = 8314.3; // J/kg/K
  double a = 0.0; // Mean atomic weight
  for(int i=0;i<5;i++){
    a += yelem[i]/u[i];
  }
  a = 1/a;
  double f1 = 380./Tp;
  double f2 = 1800./Tp;
  double cp = rgas/a*(g1(f1)+2*g1(f2));
  return cp; // J/kg/K
}


double
SimpleHeatTransfer::heatap(double Tp){
  double cpa = (754.+.586*(Tp-273));
  return cpa;  // J/kg/K
}


double
SimpleHeatTransfer::props(double Tg, double Tp){

  double tg0[10] = {300.,400.,500.,600.,700.,800.,900.,1000.,1100.,1200.};
  double kg0[10] = {.0262,.03335,.03984,.0458,.0512,.0561,.0607,.0648,.0685,.07184};
  double T = (Tp+Tg)/2; // Film temperature
//   CALCULATE UG AND KG FROM INTERPOLATION OF TABLE VALUES FROM HOLMAN
//   FIND INTERVAL WHERE TEMPERATURE LIES. 
  double kg = 0.03; 
  if(T>1200.){
    kg = kg0[9]*pow(T/tg0[9],.58);
  } else if (T<300){
    kg = kg0[0];
  } else {
    int J = -1;
    for(int I=0;I<9;I++){
      if(T>tg0[I]) J=J+1;
    }
    double FAC = (tg0[J]-T)/(tg0[J]-tg0[J+1]);
    kg = (-FAC*(kg0[J]-kg0[J+1]) + kg0[J]);
  }
 return kg; 
}
