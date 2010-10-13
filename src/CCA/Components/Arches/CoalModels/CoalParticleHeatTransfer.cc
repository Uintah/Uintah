#include <CCA/Components/Arches/CoalModels/CoalParticleHeatTransfer.h>
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

//===========================================================================

using namespace std;
using namespace Uintah; 

//---------------------------------------------------------------------------
// Builder:
CoalParticleHeatTransferBuilder::CoalParticleHeatTransferBuilder( const std::string         & modelName,
                                                                  const vector<std::string> & reqICLabelNames,
                                                                  const vector<std::string> & reqScalarLabelNames,
                                                                  const ArchesLabel         * fieldLabels,
                                                                  SimulationStateP          & sharedState,
                                                                  int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

CoalParticleHeatTransferBuilder::~CoalParticleHeatTransferBuilder(){}

ModelBase* CoalParticleHeatTransferBuilder::build() {
  return scinew CoalParticleHeatTransfer( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

CoalParticleHeatTransfer::CoalParticleHeatTransfer( std::string modelName, 
                                        SimulationStateP& sharedState,
                                        const ArchesLabel* fieldLabels,
                                        vector<std::string> icLabelNames, 
                                        vector<std::string> scalarLabelNames,
                                        int qn ) 
: HeatTransfer(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  // Model/gas labels created in parent class

  // thermal conductivity of particles
  std::string abskpName = "abskp_qn";
  abskpName += qn; 
  d_abskp = VarLabel::create(abskpName, CCVariable<double>::getTypeDescription());

  // Set constants
  Pr = 0.7;
  blow = 1.0;
  sigma = 5.67e-8;   // [=] J/s/m^2/K^4 : Stefan-Boltzmann constant (from white book)

  pi = 3.14159265358979; 

  d_useLength = false;
  d_useRawCoal = false;
  d_useChar = false;
  d_useMoisture = false;
  d_useTp = false;
  d_useTgas = false;

}

CoalParticleHeatTransfer::~CoalParticleHeatTransfer()
{
  VarLabel::destroy(d_abskp);
}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
CoalParticleHeatTransfer::problemSetup(const ProblemSpecP& params)
{
  HeatTransfer::problemSetup( params );

  ProblemSpecP db = params; 
  const ProblemSpecP params_root = db->getRootNode(); 

  // check for viscosity
  if (params_root->findBlock("PhysicalConstants")) {
    ProblemSpecP db_phys = params_root->findBlock("PhysicalConstants");
    db_phys->require("viscosity", visc);
    if( visc == 0.0 ) {
      throw InvalidValue("ERROR: CoalParticleHeatTransfer: problemSetup(): Zero viscosity specified in <PhysicalConstants> section of input file.",__FILE__,__LINE__);
    }
  } else {
    throw InvalidValue("ERROR: CoalParticleHeatTransfer: problemSetup(): Missing <PhysicalConstants> section in input file, no viscosity value specified.",__FILE__,__LINE__);
  }

  if( params_root->findBlock("CFD") ) {
    if( params_root->findBlock("CFD")->findBlock("ARCHES") ) {
      if( params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties") ) {
        ProblemSpecP db_coal = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties");
        db_coal->require("C", yelem[0]);
        db_coal->require("H", yelem[1]);
        db_coal->require("N", yelem[2]);
        db_coal->require("O", yelem[3]);
        db_coal->require("S", yelem[4]);
        db_coal->require("initial_ash_mass", d_ash_mass);
        if( params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties")->findBlock("initial_fixcarb_mass") ) {
          d_use_fixcarb_mass = true;
          db_coal->require("initial_fixcarb_mass", d_fixcarb_mass );
        } else {
          d_use_fixcarb_mass = false;
        }
          

        // normalize amounts
        double ysum = yelem[0] + yelem[1] + yelem[2] + yelem[3] + yelem[4];
        yelem[0] = yelem[0]/ysum;
        yelem[1] = yelem[1]/ysum;
        yelem[2] = yelem[2]/ysum;
        yelem[3] = yelem[3]/ysum;
        yelem[4] = yelem[4]/ysum;
      } else {
        throw InvalidValue("ERROR: CoalParticleHeatTransfer: problemSetup(): Missing <Coal_Properties> section in input file. Please specify the elemental composition of the coal and the initial ash mass.",__FILE__,__LINE__);
      }
    }
  }

  // Check for radiation 
  b_radiation = false;
  if( params_root->findBlock("CFD") ) {
    if( params_root->findBlock("CFD")->findBlock("ARCHES") ) {
      if( params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ExplicitSolver") ) {
        if( params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ExplicitSolver")->findBlock("EnthalpySolver") ) {
          if( params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ExplicitSolver")->findBlock("EnthalpySolver")->findBlock("DORadiationModel") ) {
            b_radiation = true; // if gas phase radiation is turned on.  
          }
        }
      }
    }
  }
  
  //user can specifically turn off radiation heat transfer
  if (db->findBlock("noRadiation")) {
    b_radiation = false;
  }

  string label_name;
  string role_name;
  string temp_label_name;

  string temp_ic_name;
  string temp_ic_name_full;

  std::stringstream out;
  out << d_quadNode; 
  string node = out.str();

  // -----------------------------------------------------------------
  // Look for required internal coordinates
  ProblemSpecP db_icvars = params->findBlock("ICVars");
  if (db_icvars) {
    for (ProblemSpecP variable = db_icvars->findBlock("variable"); variable != 0; variable = variable->findNextBlock("variable") ) {
    
      variable->getAttribute("label",label_name);
      variable->getAttribute("role",role_name);

      temp_label_name = label_name;

      temp_label_name += "_qn";
      temp_label_name += node;

      // user specifies "role" of each internal coordinate
      if( role_name == "particle_length" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useLength = true;
      } else if( role_name == "raw_coal_mass" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useRawCoal = true;
      } else if( role_name == "char_mass" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useChar = true;
      } else if( role_name == "moisture_mass" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useMoisture = true;
      } else if( role_name == "particle_temperature" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useTp = true;
      } else {
        std::string errmsg = "ERROR: CoalParticleHeatTransfer: problemSetup(): Invalid variable role for Simple Heat Transfer model!";
        throw InvalidValue(errmsg,__FILE__,__LINE__);
      }
    }//end for dqmom variables
  }

  // fix the d_icLabels to point to the correct quadrature node (since there is 1 model per quad node)
  for ( vector<std::string>::iterator iString = d_icLabels.begin(); 
        iString != d_icLabels.end(); ++iString) {
    
    temp_ic_name      = (*iString);
    temp_ic_name_full = temp_ic_name;

    std::stringstream out;
    out << d_quadNode;
    string node = out.str();
    temp_ic_name_full += "_qn";
    temp_ic_name_full += node;

    std::replace( d_icLabels.begin(), d_icLabels.end(), temp_ic_name, temp_ic_name_full);
  }

  // -----------------------------------------------------------------
  // Look for required scalars
  ProblemSpecP db_scalarvars = params->findBlock("scalarVars");
  if (db_scalarvars) {
    for( ProblemSpecP variable = db_scalarvars->findBlock("variable");
         variable != 0; variable = variable->findNextBlock("variable") ) {

      variable->getAttribute("label", label_name);
      variable->getAttribute("role",  role_name);

      // user specifies "role" of each scalar
      // NOTE: only gas_temperature can be used
      // (otherwise, we have to differentiate which variables are weighted and which are not...)
      if( role_name == "gas_temperature" ) {
        LabelToRoleMap[label_name] = role_name;
        d_useTgas = true;
      } else {
        string errmsg = "ERROR: Arches: CoalParticleHeatTransfer: Invalid scalar variable role for Simple Heat Transfer model: must be \"particle_temperature\" or \"gas_temperature\", you specified \"" + role_name + "\".";
        throw ProblemSetupException(errmsg,__FILE__,__LINE__);
      }
    }//end for scalar variables
  }

  if(!d_useRawCoal) {
    string errmsg = "ERROR: Arches: CoalParticleHeatTransfer: No raw coal variable was specified. Quitting...";
    throw ProblemSetupException(errmsg,__FILE__,__LINE__);
  }

  if(!d_useTp) { 
    string errmsg = "ERROR: Arches: CoalParticleHeatTransfer: No particle temperature variable was specified. Quitting...";
    throw ProblemSetupException(errmsg,__FILE__,__LINE__);
  }

	if(!d_useLength) {
    string errmsg = "ERROR: Arches: CoalParticleHeatTransfer: No particle length variable was specified. Quitting...";
    throw ProblemSetupException(errmsg,__FILE__,__LINE__);
	}

  if(!d_useTgas) {
    d_gas_temperature_label = d_fieldLabels->d_tempINLabel;
  }


  ///////////////////////////////////////////


  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();
  EqnFactory& eqn_factory = EqnFactory::self();

  // assign labels for each required internal coordinate
  for( map<string,string>::iterator iter = LabelToRoleMap.begin();
       iter != LabelToRoleMap.end(); ++iter ) {

    EqnBase* current_eqn;
    if( dqmom_eqn_factory.find_scalar_eqn(iter->first) ) {
      current_eqn = &(dqmom_eqn_factory.retrieve_scalar_eqn(iter->first));
    } else if( eqn_factory.find_scalar_eqn(iter->first) ) {
      current_eqn = &(eqn_factory.retrieve_scalar_eqn(iter->first));
    } else {
      string errmsg = "ERROR: Arches: CoalParticleHeatTransfer: Invalid variable \"" + iter->first + "\" given for \""+iter->second+"\" role, could not find in EqnFactory or DQMOMEqnFactory!";
      throw ProblemSetupException(errmsg,__FILE__,__LINE__);
    }

    if( iter->second == "particle_length" ) {
      d_length_label = current_eqn->getTransportEqnLabel();
      d_length_scaling_constant = current_eqn->getScalingConstant();
    } else if( iter->second == "raw_coal_mass" ) {
      d_raw_coal_mass_label = current_eqn->getTransportEqnLabel();
      d_rc_scaling_constant = current_eqn->getScalingConstant();
    } else if( iter->second == "char_mass" ) {
      d_char_mass_label = current_eqn->getTransportEqnLabel();
      d_char_scaling_constant = current_eqn->getScalingConstant();
    } else if( iter->second == "moisture_mass" ) {
      d_moisture_mass_label = current_eqn->getTransportEqnLabel();
      d_moisture_scaling_constant = current_eqn->getScalingConstant();
    } else if( iter->second == "particle_temperature" ) {
      d_particle_temperature_label = current_eqn->getTransportEqnLabel();
      d_pt_scaling_constant = current_eqn->getScalingConstant();

      DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(current_eqn);
      dqmom_eqn->addModel( d_modelLabel );
    } else if( iter->second == "gas_temperature" ) {
      d_gas_temperature_label = current_eqn->getTransportEqnLabel();
    } else {
      string errmsg = "ERROR: Arches: CoalParticleHeatTransfer: Could not identify specified variable role \""+iter->second+"\".";
      throw ProblemSetupException(errmsg,__FILE__,__LINE__);
    }

  }

  // set model clipping
  //db->getWithDefault( "low_clip",  d_lowModelClip,  1.0e-6 );
  //db->getWithDefault( "high_clip", d_highModelClip, 999999 );

}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void 
CoalParticleHeatTransfer::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "CoalParticleHeatTransfer::initVars";
  Task* tsk = scinew Task(taskname, this, &CoalParticleHeatTransfer::initVars);

  tsk->computes( d_modelLabel );
  tsk->computes( d_gasLabel   );
  tsk->computes( d_abskp );

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}

//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model 
//---------------------------------------------------------------------------
void 
CoalParticleHeatTransfer::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "CoalParticleHeatTransfer::computeModel";
  Task* tsk = scinew Task(taskname, this, &CoalParticleHeatTransfer::computeModel, timeSubStep );

  Ghost::GhostType gn = Ghost::None;

  CoalModelFactory& coalFactory = CoalModelFactory::self();

  if (d_timeSubStep == 0 && !d_labelSchedInit) {
    // Every model term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;
  }

  if( timeSubStep == 0 ) {
  
    // calculated quantities
    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel); 
    tsk->computes(d_abskp);

    // density
    tsk->requires(Task::OldDW, d_fieldLabels->d_densityCPLabel, gn, 0);

    // velocity
    if( coalFactory.useParticleVelocityModel() ) {
      tsk->requires( Task::OldDW, coalFactory.getParticleVelocityLabel(d_quadNode), gn, 0 );
    }
    tsk->requires( Task::OldDW, d_fieldLabels->d_newCCVelocityLabel, gn, 0);

    // temperature
    tsk->requires( Task::OldDW, d_particle_temperature_label, gn, 0 );
    tsk->requires( Task::OldDW, d_gas_temperature_label, gn, 0 );

    // radiation
    if(b_radiation){
      tsk->requires(Task::OldDW, d_fieldLabels->d_radiationSRCINLabel,  gn, 0);
      tsk->requires(Task::OldDW, d_fieldLabels->d_abskgINLabel,  gn, 0);   
      tsk->requires(Task::OldDW, d_fieldLabels->d_radiationVolqINLabel, gn, 0);
    }

    // DQMOM internal coordinates
    tsk->requires(Task::OldDW, d_weight_label, gn, 0 );
    tsk->requires(Task::OldDW, d_raw_coal_mass_label, gn, 0);
    tsk->requires(Task::OldDW, d_length_label, gn, 0);
    tsk->requires(Task::OldDW, d_particle_temperature_label, gn, 0);
    if( d_useChar ) {
      tsk->requires(Task::OldDW, d_char_mass_label, gn, 0);
    }
    if( d_useMoisture ) {
      tsk->requires(Task::OldDW, d_moisture_mass_label, gn, 0);
    }

  } else {

    // calculated quantities
    tsk->modifies(d_modelLabel);
    tsk->modifies(d_gasLabel);  
    tsk->modifies(d_abskp);

    // density
    tsk->requires(Task::NewDW, d_fieldLabels->d_densityCPLabel, gn, 0);

    // velocity
    if( coalFactory.useParticleVelocityModel() ) {
      tsk->requires( Task::NewDW, coalFactory.getParticleVelocityLabel(d_quadNode), gn, 0 );
    }
    tsk->requires( Task::NewDW, d_fieldLabels->d_newCCVelocityLabel, gn, 0);

    // temperature
    tsk->requires( Task::NewDW, d_particle_temperature_label, gn, 0 );
    tsk->requires( Task::NewDW, d_gas_temperature_label, gn, 0 );

    // radiation 
    if(b_radiation){
      tsk->requires(Task::NewDW, d_fieldLabels->d_radiationSRCINLabel,  gn, 0);
      tsk->requires(Task::NewDW, d_fieldLabels->d_abskgINLabel,  gn, 0);   
      tsk->requires(Task::NewDW, d_fieldLabels->d_radiationVolqINLabel, gn, 0);
    }

    // DQMOM internal coordinates 
    tsk->requires(Task::NewDW, d_weight_label, gn, 0 );
    tsk->requires(Task::NewDW, d_raw_coal_mass_label, gn, 0);
    tsk->requires(Task::NewDW, d_length_label, gn, 0);
    tsk->requires(Task::NewDW, d_particle_temperature_label, gn, 0);
    if( d_useChar ) {
      tsk->requires(Task::NewDW, d_char_mass_label, gn, 0);
    }
    if( d_useMoisture ) {
      tsk->requires(Task::NewDW, d_moisture_mass_label, gn, 0);
    }

  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}

//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
CoalParticleHeatTransfer::computeModel( const ProcessorGroup * pc, 
                                        const PatchSubset    * patches, 
                                        const MaterialSubset * matls, 
                                        DataWarehouse        * old_dw, 
                                        DataWarehouse        * new_dw,
                                        int timeSubStep )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CoalModelFactory& coalFactory = CoalModelFactory::self();

    // calculated quantities
    CCVariable<double> heat_rate;
    CCVariable<double> gas_heat_rate; 
    CCVariable<double> abskp; 

    // density
    constCCVariable<double> gas_density;
    
    // velocity
    constCCVariable<Vector> particle_velocity;
    constCCVariable<Vector> gas_velocity;

    // temperature
    constCCVariable<double> wa_particle_temperature;
    constCCVariable<double> gas_temperature;

    // radiation
    constCCVariable<double> radiationSRCIN;
    constCCVariable<double> abskgIN;
    constCCVariable<double> radiationVolqIN;

    // DQMOM internal coordinates
    constCCVariable<double> weight;
    constCCVariable<double> wa_raw_coal_mass;
    constCCVariable<double> wa_particle_length;
    constCCVariable<double> wa_char_mass;
    constCCVariable<double> wa_moisture_mass;

    if( timeSubStep == 0 ) {

      // calculated quantities
      new_dw->allocateAndPut( heat_rate, d_modelLabel, matlIndex, patch );
      heat_rate.initialize(0.0);

      new_dw->allocateAndPut( gas_heat_rate, d_gasLabel, matlIndex, patch );
      gas_heat_rate.initialize(0.0);

      new_dw->allocateAndPut( abskp, d_abskp, matlIndex, patch );
      abskp.initialize(0.0);

      // density
      old_dw->get( gas_density, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0 ); 

      // velocity
      if( coalFactory.useParticleVelocityModel() ) {
        old_dw->get( particle_velocity, coalFactory.getParticleVelocityLabel( d_quadNode ), matlIndex, patch, gn, 0 );
      } else {
        old_dw->get( particle_velocity, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0 );
      }
      old_dw->get( gas_velocity, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0 );

      // temperature
      old_dw->get( wa_particle_temperature, d_particle_temperature_label, matlIndex, patch, gn, 0);
      old_dw->get( gas_temperature, d_gas_temperature_label, matlIndex, patch, gn, 0 );

      // radiation
      if(b_radiation) { 
        old_dw->get(radiationSRCIN, d_fieldLabels->d_radiationSRCINLabel, matlIndex, patch, gn, 0);
        old_dw->get(abskgIN, d_fieldLabels->d_abskgINLabel, matlIndex, patch, gn, 0);
        old_dw->get(radiationVolqIN, d_fieldLabels->d_radiationVolqINLabel, matlIndex, patch, gn, 0);
      }

      // DQMOM internal coordinates
      old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );
      old_dw->get( wa_raw_coal_mass, d_raw_coal_mass_label, matlIndex, patch, gn, 0 );
      if(d_useLength) {
        old_dw->get( wa_particle_length, d_length_label, matlIndex, patch, gn, 0 );
      }
      if(d_useChar) {
        old_dw->get( wa_char_mass, d_char_mass_label, matlIndex, patch, gn, 0 );
      }
      if(d_useMoisture) {
        old_dw->get( wa_moisture_mass, d_moisture_mass_label, matlIndex, patch, gn, 0 );
      }


    } else {

      // calculated quantities
      new_dw->getModifiable( heat_rate, d_modelLabel, matlIndex, patch ); 
      new_dw->getModifiable( gas_heat_rate, d_gasLabel, matlIndex, patch ); 
      new_dw->getModifiable( abskp, d_abskp, matlIndex, patch ); 

      // density
      new_dw->get( gas_density, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0 ); 

      // velocity
      if( coalFactory.useParticleVelocityModel() ) {
        new_dw->get( particle_velocity, coalFactory.getParticleVelocityLabel( d_quadNode ), matlIndex, patch, gn, 0 );
      } else {
        new_dw->get( particle_velocity, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0 );
      }
      new_dw->get( gas_velocity, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0 );

      // temperature
      new_dw->get( wa_particle_temperature, d_particle_temperature_label, matlIndex, patch, gn, 0);
      new_dw->get( gas_temperature, d_gas_temperature_label, matlIndex, patch, gn, 0 );

      // radiation
      if(b_radiation) {
        new_dw->get(radiationSRCIN, d_fieldLabels->d_radiationSRCINLabel, matlIndex, patch, gn, 0);
        new_dw->get(abskgIN, d_fieldLabels->d_abskgINLabel, matlIndex, patch, gn, 0);
        new_dw->get(radiationVolqIN, d_fieldLabels->d_radiationVolqINLabel, matlIndex, patch, gn, 0);
      }

      // DQMOM internal coordinates
      new_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );
      new_dw->get( wa_raw_coal_mass, d_raw_coal_mass_label, matlIndex, patch, gn, 0 );
      if(d_useLength) {
        new_dw->get( wa_particle_length, d_length_label, matlIndex, patch, gn, 0 );
      }
      if(d_useChar) {
        new_dw->get( wa_char_mass, d_char_mass_label, matlIndex, patch, gn, 0 );
      }
      if(d_useMoisture) {
        new_dw->get( wa_moisture_mass, d_moisture_mass_label, matlIndex, patch, gn, 0 );
      }

    }


    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 

      // weight - check if small
      bool weight_is_small = (weight[c] < d_w_small) || (weight[c] == 0 );

      double scaled_weight;
      double unscaled_weight;

      double scaled_particle_temperature;
      double unscaled_particle_temperature;

      double scaled_length;
      double unscaled_length;

      double scaled_raw_coal_mass;
      double unscaled_raw_coal_mass;

      double unscaled_ash_mass = d_ash_mass[d_quadNode];
      double unscaled_fixcarb_mass;
      if( d_use_fixcarb_mass ) {
        unscaled_fixcarb_mass = d_fixcarb_mass[d_quadNode];
      }

      double FSum = 0.0;

      double heat_rate_;
      double gas_heat_rate_;
      double abskp_;

      // intermediate calculation values
      double Re;
      double Nu;
      double Cpc;
      double Cpa; 
      double Cph;
      double mp_Cp;
      double rkg;
      double Q_convection;
      double Q_radiation;

      if (weight_is_small) {

        scaled_weight = 0.0;
        unscaled_weight = 0.0;

        scaled_particle_temperature = 0.0;
        unscaled_particle_temperature = 0.0;

        scaled_length = 0.0;
        unscaled_length = 0.0;

        scaled_raw_coal_mass = 0.0;
        unscaled_raw_coal_mass = 0.0;


        heat_rate_ = 0.0;
        gas_heat_rate_ = 0.0;
        abskp_ = 0.0;

      } else {

        scaled_weight = weight[c];
        unscaled_weight = weight[c]*d_w_scaling_constant;

        scaled_particle_temperature = wa_particle_temperature[c]/scaled_weight;
        unscaled_particle_temperature = scaled_particle_temperature*d_pt_scaling_constant;

        scaled_length = wa_particle_length[c]/scaled_weight;
        unscaled_length = scaled_length*d_length_scaling_constant;

        scaled_raw_coal_mass = wa_raw_coal_mass[c]/scaled_weight;
        unscaled_raw_coal_mass = scaled_raw_coal_mass*d_rc_scaling_constant;

        // ---------------------------------------------
        // Convection part: 

        // Reynolds number
        Re = abs(gas_velocity[c].length() - particle_velocity[c].length())*unscaled_length*gas_density[c]/visc;

        // Nusselt number
        //Nu = 2.0 + 0.65*pow(Re,0.50)*pow(Pr,(1.0/3.0));
        Nu = 2.0;

        // Heat capacity of raw coal
        Cpc = calc_Cp_rawcoal( unscaled_particle_temperature );

        // Heat capacity of ash
        Cpa = calc_Cp_ash( unscaled_particle_temperature );
        
        // Heat capacity of char
        Cph = calc_Cp_char( unscaled_particle_temperature );

        // Heat capacity
        mp_Cp = (Cpc*unscaled_raw_coal_mass + Cpa*unscaled_ash_mass);
        if( d_use_fixcarb_mass ) {
          mp_Cp += (Cph*unscaled_fixcarb_mass);
        }

        // Gas thermal conductivity
        rkg = props(gas_temperature[c], unscaled_particle_temperature); // [=] J/s/m/K

        // Q_convection (see Section 5.4 of LES_Coal document)
        Q_convection = Nu*pi*blow*rkg*unscaled_length*(gas_temperature[c] - unscaled_particle_temperature);


        // ---------------------------------------------
        // Radiation part: 

        Q_radiation = 0.0;

        if (b_radiation) {

          double Qabs = 0.8;
	        double Apsc = (pi/4)*Qabs*pow(unscaled_length,2);
	        double Eb = 4*sigma*pow(unscaled_particle_temperature,4);

          FSum = radiationVolqIN[c];    
	        Q_radiation = Apsc*(FSum - Eb);
	        abskp_ = pi/4*Qabs*unscaled_weight*pow(unscaled_length,2); 

        } else {

          abskp_ = 0.0;

        }
      
        heat_rate_ = (Q_convection + Q_radiation)/(mp_Cp*d_pt_scaling_constant);
        gas_heat_rate_ = -unscaled_weight*Q_convection;

      }

      heat_rate[c] = heat_rate_;
      gas_heat_rate[c] = gas_heat_rate_;
      abskp[c] = abskp_;

#ifdef DEBUG_MODELS
      if( c == IntVector(1,2,3) ) {
        cout << endl;
        cout << "Coal particle heat transfer: -------------" << endl;
        cout << "Heating rate for qn " << d_quadNode << " is ~ Q_convection + Q_radiation = " << Q_convection << " + " << Q_radiation << " = " << heat_rate_ << endl;
        cout << "Gas heating rate for qn " << d_quadNode << " is " << gas_heat_rate_ << endl;
        cout << endl;
      }

#endif
 
    }//end cell loop

  }//end patch loop
}



// ********************************************************
// Private methods:

double
CoalParticleHeatTransfer::g1( double z){
  double dum1 = (exp(z)-1)/z;
  double dum2 = pow(dum1,2);
  double sol = exp(z)/dum2;
  return sol;
}

//------------------------------------------------------
// Heat capacity of raw coal
//------------------------------------------------------
double
CoalParticleHeatTransfer::calc_Cp_rawcoal(double Tp){
  if (Tp < 273) {
    // correlation is not valid
    return 0.0;
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
CoalParticleHeatTransfer::calc_Cp_ash(double Tp){
  // c.f. PCGC2
  double cpa = 593.0 + 0.586*Tp;
  return cpa;  // J/kg/K
}


double
CoalParticleHeatTransfer::calc_Cp_char(double Tp) {
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
CoalParticleHeatTransfer::props(double Tg, double Tp){

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


//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
CoalParticleHeatTransfer::initVars( const ProcessorGroup * pc, 
                              const PatchSubset    * patches, 
                              const MaterialSubset * matls, 
                              DataWarehouse        * old_dw, 
                              DataWarehouse        * new_dw )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> model_value; 
    new_dw->allocateAndPut( model_value, d_modelLabel, matlIndex, patch ); 
    model_value.initialize(0.0);

    CCVariable<double> gas_value; 
    new_dw->allocateAndPut( gas_value, d_gasLabel, matlIndex, patch ); 
    gas_value.initialize(0.0);

    CCVariable<double> abskp; 
    new_dw->allocateAndPut( abskp, d_abskp, matlIndex, patch ); 
    abskp.initialize(0.0);

  }
}

