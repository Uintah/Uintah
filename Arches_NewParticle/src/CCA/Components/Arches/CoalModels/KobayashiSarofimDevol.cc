#include <CCA/Components/Arches/CoalModels/KobayashiSarofimDevol.h>
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
#include <iomanip>

//===========================================================================

using namespace std;
using namespace Uintah; 

//---------------------------------------------------------------------------
// Builder:
KobayashiSarofimDevolBuilder::KobayashiSarofimDevolBuilder( const std::string         & modelName,
                                                            const vector<std::string> & reqICLabelNames,
                                                            const vector<std::string> & reqScalarLabelNames,
                                                            const ArchesLabel         * fieldLabels,
                                                            SimulationStateP          & sharedState,
                                                            int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

KobayashiSarofimDevolBuilder::~KobayashiSarofimDevolBuilder(){}

ModelBase* KobayashiSarofimDevolBuilder::build() {
  return scinew KobayashiSarofimDevol( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

KobayashiSarofimDevol::KobayashiSarofimDevol( std::string modelName, 
                                              SimulationStateP& sharedState,
                                              const ArchesLabel* fieldLabels,
                                              vector<std::string> icLabelNames, 
                                              vector<std::string> scalarLabelNames,
                                              int qn ) 
: Devolatilization(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  // gas/model labels are created in parent class

  // Values from Ubhayakar (1976):
  A1  =  3.7e5;       // [=] 1/s; k1 pre-exponential factor
  A2  =  1.46e13;     // [=] 1/s; k2 pre-exponential factor
  E1  =  17600;      // [=] kcal/kmol;  k1 activation energy
  E2  =  60000;      // [=] kcal/kmol;  k2 activation energy

  /*
  // Values from Kobayashi (1976):
  A1  =  2.0e5;       // [=] 1/s; pre-exponential factor for k1
  A2  =  1.3e7;       // [=] 1/s; pre-exponential factor for k2
  E1  =  -25000;      // [=] kcal/kmol;  k1 activation energy
  E2  =  -40000;      // [=] kcal/kmol;  k2 activation energy
  */

  R_   =  1.987;       // [=] kcal/kmol; ideal gas constant

  // Y values from white book:
  Y1_ = 0.3; // volatile fraction from proximate analysis
  Y2_ = 1.0; // fraction devolatilized at higher temperatures

  /*
  // Y values from Ubhayakar (1976):
  Y1_ = 0.39; // volatile fraction from proximate analysis
  Y2_ = 0.80; // fraction devolatilized at higher temperatures
  */


  // Create a char model label (for char creation)
  d_charModelLabel    = VarLabel::create( modelName+"_char",    CCVariable<double>::getTypeDescription() );

  d_useRawCoal = false;
  d_useChar = false;
  d_useTparticle = false;
  d_useTgas = false;
}

KobayashiSarofimDevol::~KobayashiSarofimDevol()
{}

//-----------------------------------------------------------------------------
//Problem Setup
//-----------------------------------------------------------------------------
void 
KobayashiSarofimDevol::problemSetup(const ProblemSpecP& params)
{
  // call parent's method first
  Devolatilization::problemSetup(params);

  ProblemSpecP db = params; 

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
      out << d_quadNode;
      string node = out.str();
      temp_label_name += "_qn";
      temp_label_name += node;

      // user specifies "role" of each internal coordinate
      if( role_name == "raw_coal_mass" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useRawCoal = true;
      } else if( role_name == "char_mass" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useChar = true;
      } else if( role_name == "particle_temperature" ) {  
        LabelToRoleMap[temp_label_name] = role_name;
        d_useTparticle = true;
      } else {
        std::string errmsg;
        errmsg = "ERROR: Arches: KobayashiSarofimDevol: Invalid variable role for DQMOM equation: must be \"particle_temperature\" or \"raw_coal_mass\", you specified \"" + role_name + "\".";
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
      if ( role_name == "gas_temperature" ) {
        LabelToRoleMap[label_name] = role_name;
        d_useTgas = true;
      } else {
        std::string errmsg;
        errmsg = "ERROR: Arches: KobayashiSarofimDevol: Invalid variable role for scalar equation: must be \"gas_tempeature\", you specified \"" + role_name + "\".";
        throw InvalidValue(errmsg,__FILE__,__LINE__);
      }
    }
  }


  if(!d_useRawCoal) {
    string errmsg = "ERROR: Arches: KobayashiSarofimDevol: No raw coal variable was specified. Quitting...";
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

    if( iter->second == "raw_coal_mass" ) {
      d_raw_coal_mass_label = current_eqn->getTransportEqnLabel();
      d_rc_scaling_constant = current_eqn->getScalingConstant();
      
      DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(current_eqn);
      dqmom_eqn->addModel( d_modelLabel );

    } else if( iter->second == "char_mass" ) {
      d_char_mass_label = current_eqn->getTransportEqnLabel();
      d_char_scaling_constant = current_eqn->getScalingConstant();

      DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(current_eqn);
      dqmom_eqn->addModel( d_charModelLabel );

    } else if( iter->second == "particle_temperature" ) {
      d_particle_temperature_label = current_eqn->getTransportEqnLabel();
      d_pt_scaling_constant = current_eqn->getScalingConstant();
    } else if( iter->second == "gas_temperature" ) {
      d_gas_temperature_label = current_eqn->getTransportEqnLabel();
    } else {
      string errmsg = "ERROR: Arches: KobayashiSarofimDevol: Could not identify specified variable role \""+iter->second+"\".";
      throw ProblemSetupException(errmsg,__FILE__,__LINE__);
    }

  }


}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void 
KobayashiSarofimDevol::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  //Devolatilization::sched_initVars( level, sched );

  string taskname = "KobayashiSarofimDevol::initVars";
  Task* tsk = scinew Task(taskname, this, &KobayashiSarofimDevol::initVars);

  tsk->computes( d_modelLabel );
  tsk->computes( d_gasLabel   );
  tsk->computes( d_charModelLabel );

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
KobayashiSarofimDevol::initVars( const ProcessorGroup * pc, 
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

    CCVariable<double> char_model_value; 
    new_dw->allocateAndPut( char_model_value, d_charModelLabel, matlIndex, patch ); 
    char_model_value.initialize(0.0);
  }
}


//-----------------------------------------------------------------------------
//Schedule the calculation of the Model 
//-----------------------------------------------------------------------------
void 
KobayashiSarofimDevol::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "KobayashiSarofimDevol::computeModel";
  Task* tsk = scinew Task(taskname, this, &KobayashiSarofimDevol::computeModel, timeSubStep );

  Ghost::GhostType gn = Ghost::None;

  d_timeSubStep = timeSubStep; 

  //// require timestep label
  //tsk->requires(Task::OldDW, d_fieldLabels->d_sharedState->get_delt_label() );

  if (d_timeSubStep == 0 && !d_labelSchedInit) {
    // Every model term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;
  }

  if( d_timeSubStep == 0 ) {
    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel); 
    if(d_useChar) { 
      tsk->computes(d_charModelLabel);
    }

    tsk->requires(Task::OldDW, d_gas_temperature_label, gn, 0);
    tsk->requires(Task::OldDW, d_weight_label, gn, 0);
    tsk->requires(Task::OldDW, d_raw_coal_mass_label, gn, 0);

    if(d_useTparticle) {
      tsk->requires(Task::OldDW, d_particle_temperature_label, gn, 0);
    }

  } else {
    tsk->modifies(d_modelLabel);
    tsk->modifies(d_gasLabel);  
    if(d_useChar) {
      tsk->modifies(d_charModelLabel);
    }

    tsk->requires(Task::NewDW, d_gas_temperature_label, gn, 0);
    tsk->requires(Task::NewDW, d_weight_label, gn, 0);
    tsk->requires(Task::NewDW, d_raw_coal_mass_label, gn, 0);

    if(d_useTparticle) {
      tsk->requires(Task::NewDW, d_particle_temperature_label, gn, 0);
    }

  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}



//-----------------------------------------------------------------------------
//Actually compute the source term 
//-----------------------------------------------------------------------------
void
KobayashiSarofimDevol::computeModel( const ProcessorGroup * pc, 
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

    //delt_vartype delta_t;
    //old_dw->get( delta_t, d_fieldLabels->d_sharedState->get_delt_label() );
    //double dt = delta_t;

    CCVariable<double> devol_rate;
    CCVariable<double> gas_devol_rate; 
    CCVariable<double> char_production_rate;

    constCCVariable<double> weight;
    constCCVariable<double> wa_raw_coal_mass;
    constCCVariable<double> temperature; // holds gas OR particle temperature...

    if( timeSubStep == 0 ) {

      new_dw->allocateAndPut( devol_rate, d_modelLabel, matlIndex, patch );
      devol_rate.initialize(0.0);

      new_dw->allocateAndPut( gas_devol_rate, d_gasLabel, matlIndex, patch ); 
      gas_devol_rate.initialize(0.0);

      if(d_useChar) {
        new_dw->allocateAndPut( char_production_rate, d_charModelLabel, matlIndex, patch );
        char_production_rate.initialize(0.0);
      }

      old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );
      old_dw->get( wa_raw_coal_mass, d_raw_coal_mass_label, matlIndex, patch, gn, 0 );

      if(d_useTparticle) {
        old_dw->get( temperature, d_particle_temperature_label, matlIndex, patch, gn, 0 );
      } else {
        old_dw->get( temperature, d_gas_temperature_label, matlIndex, patch, gn, 0 );
      }

    } else {

      new_dw->getModifiable( devol_rate, d_modelLabel, matlIndex, patch ); 
      new_dw->getModifiable( gas_devol_rate, d_gasLabel, matlIndex, patch ); 

      if(d_useChar) {
        new_dw->getModifiable( char_production_rate, d_charModelLabel, matlIndex, patch );
      }

      new_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );
      new_dw->get( wa_raw_coal_mass, d_raw_coal_mass_label, matlIndex, patch, gn, 0 );
      
      if(d_useTparticle) {
        new_dw->get( temperature, d_particle_temperature_label, matlIndex, patch, gn, 0 );
      } else {
        new_dw->get( temperature, d_gas_temperature_label, matlIndex, patch, gn, 0 );
      }

    }

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 

      // weight - check if small
      bool weight_is_small = (weight[c] < d_w_small) || (weight[c] == 0.0);

      // devol_rate: raw caol internal coordinate source G
      double devol_rate_;

      // gas_devol_rate: gas source
      double gas_devol_rate_;

      // char_production_rate: char internal coordinate source G
      double char_production_rate_;


      double unscaled_temperature;
      double unscaled_raw_coal_mass;

      if (weight_is_small) {

        devol_rate_ = 0.0;
        gas_devol_rate_ = 0.0;
        char_production_rate_ = 0.0;

      } else {

        double unscaled_weight;
        double scaled_raw_coal_mass;


        unscaled_weight = weight[c]*d_w_scaling_constant;

        if (d_useTparticle) {
          // particle temp
          unscaled_temperature = temperature[c]*d_pt_scaling_constant/weight[c];
        } else {
          // particle temp = gas temp
          unscaled_temperature = temperature[c];
        }
        if( unscaled_temperature < TINY ) {
          unscaled_temperature = TINY;
        }

        scaled_raw_coal_mass = wa_raw_coal_mass[c]/weight[c];
        unscaled_raw_coal_mass = scaled_raw_coal_mass*d_rc_scaling_constant;

        k1 = A1*exp(-E1/(R_*unscaled_temperature)); // [=] 1/s
        k2 = A2*exp(-E2/(R_*unscaled_temperature)); // [=] 1/s

        devol_rate_ = -(k1+k2)*scaled_raw_coal_mass;
        if( devol_rate_ > TINY ) {
          devol_rate_ = 0.0;
        }

        gas_devol_rate_ = (Y1_*k1 + Y2_*k2)*unscaled_raw_coal_mass*unscaled_weight; // [=] kg/m^3
        if( gas_devol_rate_ < TINY ) {
          gas_devol_rate_ = 0.0;
        }

        if(d_useChar) {
          char_production_rate_ = ( (1-Y1_)*k1 + (1-Y2_)*k2)*unscaled_raw_coal_mass*(1/d_char_scaling_constant);
          if( char_production_rate_ < TINY ) {
            char_production_rate_ = 0.0;
          }
        }
 

        // FIXME - see ConstantDensityInert class for details
        /*
        //double big_rate = 1.0e10; // Limit model terms to 1e10

        if( fabs(testVal_part*dt) > scaled_raw_coal_mass ) {

          // too much devolatilization! set to maximum possible
          testVal_part = -scaled_raw_coal_mass/dt;
          devol_rate_ = testVal_part;
          // -----------------
          // total amt reacted = -(k1 + k2)*unscaled_raw_coal_mass*unscaled_weight
          // total amt into gas = (Y1 k1 + Y2 k2)*unscaled_raw_coal_mass*unscaled_weight
          //
          // when total amt reacted = unscaled_raw_coal_mass*unscaled_weight ,
          // there is no -(k1+k2) contained in it... 
          // so divide it out to get total amt into gas's (unscaled_raw_coal_mass*unscaled_weight)
          testVal_gas = (Y1_*k1 + Y2_*k2)*((unscaled_raw_coal_mass*unscaled_weight)/(k1+k2)); // [=] kg/m^3
          if( testVal_gas > TINY ) {
            gas_devol_rate_ = testVal_gas;
          } else {
            gas_devol_rate_ = 0.0;
          }


        } else if( fabs(testVal_part) > big_rate ) {
        
          // devolatilizing too fast! limit rate to maximum possible
          // (or, some arbitrarily large number)
          testVal_part = -big_rate;
          devol_rate_ = testVal_part;

          testVal_gas = (Y1_*k1 + Y2_*k2)*((big_rate*d_rc_scaling_constant)*(weight[c]*d_w_scaling_constant))/(-k1+k2);
          gas_devol_rate_ = testVal_gas;

        } else {
          // treat devolatilization like normal
          
          [see code]

        }
        */

      }

      devol_rate[c] = devol_rate_;
      gas_devol_rate[c] = gas_devol_rate_;
      if(d_useChar) {
        char_production_rate[c] = char_production_rate_;
      }

#ifdef DEBUG_MODELS
      if( c == IntVector(1,2,3) ) {
        cout << endl;
        cout << "Kobayashi Model: QN " << d_quadNode << endl;
        cout << "  For temperature = " << unscaled_temperature << " and raw coal mass = " << unscaled_raw_coal_mass << endl;
        cout << "  Coal devolatilization rate for qn " << d_quadNode << " = " << devol_rate_ << endl;
        cout << "  Gas production rate for qn " << d_quadNode << " = " << gas_devol_rate_ << endl;
        cout << endl;
      }
#endif

    }//end cell loop
  
  }//end patch loop
}


