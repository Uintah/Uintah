#include <CCA/Components/Arches/CoalModels/InertParticleHeatTransfer.h>
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
InertParticleHeatTransferBuilder::InertParticleHeatTransferBuilder( const std::string         & modelName,
                                                                    const vector<std::string> & reqICLabelNames,
                                                                    const vector<std::string> & reqScalarLabelNames,
                                                                    const ArchesLabel         * fieldLabels,
                                                                    SimulationStateP          & sharedState,
                                                                    int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

InertParticleHeatTransferBuilder::~InertParticleHeatTransferBuilder(){}

ModelBase* InertParticleHeatTransferBuilder::build() {
  return scinew InertParticleHeatTransfer( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

InertParticleHeatTransfer::InertParticleHeatTransfer( std::string modelName, 
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
  d_Pr = 0.7;         ///< What is the justification for this value?
  d_blow = 1.0;       ///< Should be a function...
  d_sigma = 5.67e-8;  ///< [=] J/s/m^2/K^4 : Stefan-Boltzmann constant (from white book)

  pi = 3.14159265358979; 

  d_useLength = false;
  d_useMass = false;
  d_useTp = false;
  d_useTgas = false;
}

InertParticleHeatTransfer::~InertParticleHeatTransfer()
{
  VarLabel::destroy(d_abskp);
}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
InertParticleHeatTransfer::problemSetup(const ProblemSpecP& params)
{
  HeatTransfer::problemSetup( params );

  ProblemSpecP db = params; 
  const ProblemSpecP params_root = db->getRootNode(); 

  // check for viscosity
  if (params_root->findBlock("PhysicalConstants")) {
    ProblemSpecP db_phys = params_root->findBlock("PhysicalConstants");
    db_phys->require("viscosity", d_visc);
    if( d_visc == 0.0 ) {
      throw InvalidValue("ERROR: InertParticleHeatTransfer: problemSetup(): Zero viscosity specified in <PhysicalConstants> section of input file.",__FILE__,__LINE__);
    }
  } else {
    throw InvalidValue("ERROR: InertParticleHeatTransfer: problemSetup(): Missing <PhysicalConstants> section in input file, no viscosity value specified.",__FILE__,__LINE__);
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
      } else if( role_name == "particle_mass" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useMass = true;
      } else if( role_name == "particle_temperature" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useTp = true;
      } else {
        std::string errmsg = "ERROR: Arches: InertParticleHeatTransfer: Invalid variable role: must be \"particle_length\", \"particle_tempeature\", or \"particle_mass\", you specified \"" + role_name + "\".";
        throw ProblemSetupException(errmsg,__FILE__,__LINE__);
      }
    }//end for dqmom variables
  }

  // fix the d_icLabels to point to the correct quadrature node (since there is 1 model per quad node)
  for ( vector<std::string>::iterator iString = d_icLabels.begin(); 
        iString != d_icLabels.end(); ++iString) {
    
    temp_ic_name      = (*iString);
    temp_ic_name_full = temp_ic_name;

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
        string errmsg = "ERROR: Arches: InertParticleHeatTransfer: Invalid scalar variable role: must be \"gas_temperature\", you specified \"" + role_name + "\".";
        throw ProblemSetupException(errmsg,__FILE__,__LINE__);
      }
    }//end for scalar variables
  }

  if(!d_useMass) {
    string errmsg = "ERROR: Arches: InertParticleHeatTransfer: No particle mass variable was specified. Quitting...";
    throw ProblemSetupException(errmsg,__FILE__,__LINE__);
  }

  if(!d_useTp) { 
    string errmsg = "ERROR: Arches: InertParticleHeatTransfer: No particle temperature variable was specified. Quitting...";
    throw ProblemSetupException(errmsg,__FILE__,__LINE__);
  }

  if(!d_useLength) {
    string errmsg = "ERROR: Arches: InertParticleHeatTransfer: No particle length variable was specified. Quitting...";
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
      string errmsg = "ERROR: Arches: InertParticleHeatTransfer: Invalid variable \"" + iter->first + "\" given for \""+iter->second+"\" role, could not find in EqnFactory or DQMOMEqnFactory!";
      throw ProblemSetupException(errmsg,__FILE__,__LINE__);
    }

    if( iter->second == "particle_length" ) {
      d_length_label = current_eqn->getTransportEqnLabel();
      d_length_scaling_constant = current_eqn->getScalingConstant();
    } else if( iter->second == "particle_mass" ) {
      d_particle_mass_label = current_eqn->getTransportEqnLabel();
      d_mass_scaling_constant = current_eqn->getScalingConstant();
    } else if( iter->second == "particle_temperature" ) {
      d_particle_temperature_label = current_eqn->getTransportEqnLabel();
      d_pt_scaling_constant = current_eqn->getScalingConstant();
    } else if( iter->second == "gas_temperature" ) {
      d_gas_temperature_label = current_eqn->getTransportEqnLabel();
    } else {
      string errmsg = "ERROR: Arches: InertParticleHeatTransfer: Could not identify specified variable role \""+iter->second+"\".";
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
InertParticleHeatTransfer::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "InertParticleHeatTransfer::initVars";
  Task* tsk = scinew Task(taskname, this, &InertParticleHeatTransfer::initVars);

  tsk->computes( d_modelLabel );
  tsk->computes( d_gasLabel   );
  tsk->computes( d_abskp );

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}

//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model 
//---------------------------------------------------------------------------
void 
InertParticleHeatTransfer::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "InertParticleHeatTransfer::computeModel";
  Task* tsk = scinew Task(taskname, this, &InertParticleHeatTransfer::computeModel);

  Ghost::GhostType gn = Ghost::None;

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

  // required variables:
  // gas temp
  // particle temp
  // gas density
  // gas heat capacity
  // (particle heat capacity depends on internal coordinates)
  // radiation variables
  // particle internal coordinates and weights

  CoalModelFactory& coalFactory = CoalModelFactory::self();

  // gas density
  tsk->requires(Task::OldDW, d_fieldLabels->d_densityCPLabel, gn, 0);

  if( timeSubStep == 0 ) {

    // velocities - gas and particle velocities 
    if( coalFactory.useParticleVelocityModel() ) {
      tsk->requires( Task::OldDW, coalFactory.getParticleVelocityLabel(d_quadNode), gn, 0 );
    }
    tsk->requires( Task::OldDW, d_fieldLabels->d_newCCVelocityLabel, gn, 0);

    // gas temperature (particle temp is required below)
    tsk->requires( Task::OldDW, d_gas_temperature_label, gn, 0 );

    // radiation variables
    if(b_radiation){
      tsk->requires(Task::OldDW, d_fieldLabels->d_radiationSRCINLabel,  gn, 0);
      tsk->requires(Task::OldDW, d_fieldLabels->d_abskgINLabel,  gn, 0);   
      tsk->requires(Task::OldDW, d_fieldLabels->d_radiationVolqINLabel, gn, 0);
    }

    //// particle internal coordinates and weights
    tsk->requires(Task::OldDW, d_weight_label, gn, 0 );
    tsk->requires(Task::OldDW, d_length_label, gn, 0);
    tsk->requires(Task::OldDW, d_particle_mass_label, gn, 0);
    tsk->requires(Task::OldDW, d_particle_temperature_label, gn, 0);
    tsk->requires(Task::OldDW, d_gas_temperature_label, gn, 0);

  } else {

    // velocities - gas and particle velocities 
    if( coalFactory.useParticleVelocityModel() ) {
      tsk->requires( Task::NewDW, coalFactory.getParticleVelocityLabel(d_quadNode), gn, 0 );
    }
    tsk->requires( Task::NewDW, d_fieldLabels->d_newCCVelocityLabel, gn, 0);

    // gas temperature (particle temp is required below)
    tsk->requires( Task::NewDW, d_gas_temperature_label, gn, 0 );

    // radiation variables
    if(b_radiation){
      tsk->requires(Task::NewDW, d_fieldLabels->d_radiationSRCINLabel,  gn, 0);
      tsk->requires(Task::NewDW, d_fieldLabels->d_abskgINLabel,  gn, 0);   
      tsk->requires(Task::NewDW, d_fieldLabels->d_radiationVolqINLabel, gn, 0);
    }

    //// particle internal coordinates and weights
    tsk->requires(Task::NewDW, d_weight_label, gn, 0 );
    tsk->requires(Task::NewDW, d_length_label, gn, 0);
    tsk->requires(Task::NewDW, d_particle_mass_label, gn, 0);
    tsk->requires(Task::NewDW, d_particle_temperature_label, gn, 0);
    tsk->requires(Task::NewDW, d_gas_temperature_label, gn, 0);

  }


  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}

//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
InertParticleHeatTransfer::computeModel( const ProcessorGroup * pc, 
                                  const PatchSubset    * patches, 
                                  const MaterialSubset * matls, 
                                  DataWarehouse        * old_dw, 
                                  DataWarehouse        * new_dw )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

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
 
    CoalModelFactory& coalFactory = CoalModelFactory::self();

    //--------------------------------------------
    // Velocities - gas and particle

    constCCVariable<Vector> particle_velocity;
    if( coalFactory.useParticleVelocityModel() ) {
      if( new_dw->exists(coalFactory.getParticleVelocityLabel(d_quadNode), matlIndex, patch) ) {
        new_dw->get( particle_velocity, coalFactory.getParticleVelocityLabel( d_quadNode ), matlIndex, patch, gn, 0 );
      } else {
        old_dw->get( particle_velocity, coalFactory.getParticleVelocityLabel( d_quadNode ), matlIndex, patch, gn, 0 );
      }

    } else {
      if( new_dw->exists(d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch) ) {
        new_dw->get( particle_velocity, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0 );
      } else {
        old_dw->get( particle_velocity, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0 );
      }
    }
    
    constCCVariable<Vector> gas_velocity;
    if( new_dw->exists(d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch) ) {
      new_dw->get( gas_velocity, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0 );
    } else {
      old_dw->get( gas_velocity, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0 );
    }
    
    //------------------------------------------------
    // Densities - gas (no particle density needed)

    constCCVariable<double> gas_density;
    old_dw->get(gas_density, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0 ); 

    //-------------------------------------------------------
    // Temperatures (gas and particle)

    constCCVariable<double> wa_particle_temperature;
    if( new_dw->exists(d_particle_temperature_label, matlIndex, patch) ) {
      new_dw->get( wa_particle_temperature, d_particle_temperature_label, matlIndex, patch, gn, 0);
    } else {
      old_dw->get( wa_particle_temperature, d_particle_temperature_label, matlIndex, patch, gn, 0);
    }

    constCCVariable<double> gas_temperature;
    if( new_dw->exists(d_gas_temperature_label, matlIndex, patch) ) {
      new_dw->get( gas_temperature, d_gas_temperature_label, matlIndex, patch, gn, 0);
    } else {
      old_dw->get( gas_temperature, d_gas_temperature_label, matlIndex, patch, gn, 0);
    }

    //---------------------------------------------------------
    // Radiation variables

    constCCVariable<double> radiationSRCIN;
    constCCVariable<double> abskgIN;
    constCCVariable<double> radiationVolqIN;

    if(b_radiation){
      old_dw->get(radiationSRCIN, d_fieldLabels->d_radiationSRCINLabel, matlIndex, patch, gn, 0);
      old_dw->get(abskgIN, d_fieldLabels->d_abskgINLabel, matlIndex, patch, gn, 0);
      old_dw->get(radiationVolqIN, d_fieldLabels->d_radiationVolqINLabel, matlIndex, patch, gn, 0);
    }

    //----------------------------------------------------------
    // Internal coordinates and weights

    constCCVariable<double> weight;
    if( new_dw->exists(d_weight_label, matlIndex, patch) ) {
      new_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );
    } else {
      old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );
    }

    constCCVariable<double> wa_particle_length;
    if( new_dw->exists(d_length_label, matlIndex, patch) ) {
      new_dw->get( wa_particle_length, d_length_label, matlIndex, patch, gn, 0 );
    } else {
      old_dw->get( wa_particle_length, d_length_label, matlIndex, patch, gn, 0 );
    }

    constCCVariable<double> wa_particle_mass;
    if( new_dw->exists(d_particle_mass_label, matlIndex, patch) ) {
      new_dw->get( wa_particle_mass, d_particle_mass_label, matlIndex, patch, gn, 0 );
    } else {
      old_dw->get( wa_particle_mass, d_particle_mass_label, matlIndex, patch, gn, 0 );
    }


    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 

      // weight - check if small
      bool weight_is_small = (weight[c] < d_w_small);

      double scaled_weight;
      double unscaled_weight;

      double scaled_particle_temperature;
      double unscaled_particle_temperature;

      double scaled_particle_mass;
      double unscaled_particle_mass;

      double scaled_length;
      double unscaled_length;

      double FSum = 0.0;

      double heat_rate_;
      double gas_heat_rate_;
      double abskp_;

      // intermediate calculation values
      double Re;
      double Nu;
      double Cpc;
      double Cpa; 
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

        scaled_particle_mass = wa_particle_mass[c]/scaled_weight;
        unscaled_particle_mass = scaled_particle_mass*d_mass_scaling_constant;

        // ---------------------------------------------
        // Convection part: 

        // Reynolds number
        Re = abs(gas_velocity[c].length() - particle_velocity[c].length())*unscaled_length*gas_density[c]/d_visc;

        // Nusselt number
        Nu = 2.0 + 0.65*pow(Re,0.50)*pow(d_Pr,(1.0/3.0));

        // Heat capacity
        mp_Cp = calc_Cp();

        // Gas thermal conductivity
        rkg = props(gas_temperature[c], unscaled_particle_temperature); // [=] J/s/m/K

        // Q_convection (see Section 5.4 of LES_Coal document)
        Q_convection = Nu*pi*d_blow*rkg*unscaled_length*(gas_temperature[c] - unscaled_particle_temperature);


        // ---------------------------------------------
        // Radiation part: 

        Q_radiation = 0.0;

        if (b_radiation) {

          double Qabs = 0.8;
	        double Apsc = (pi/4)*Qabs*pow(unscaled_length,2);
	        double Eb = 4*d_sigma*pow(unscaled_particle_temperature,4);

          FSum = radiationVolqIN[c];    
	        Q_radiation = Apsc*(FSum - Eb);
	        abskp_ = pi/4*Qabs*unscaled_weight*pow(unscaled_length,2); 

        } else {

          abskp_ = 0.0;

        }
      
        heat_rate_ = (Q_convection + Q_radiation)/(mp_Cp*d_pt_scaling_constant);
        gas_heat_rate_ = 0.0;

      }

      heat_rate[c] = heat_rate_;
      gas_heat_rate[c] = gas_heat_rate_;
      abskp[c] = abskp_;
 
    }//end cell loop

  }//end patch loop
}



// ********************************************************
// Private methods:

double
InertParticleHeatTransfer::props(double Tg, double Tp){

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
InertParticleHeatTransfer::initVars( const ProcessorGroup * pc, 
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

