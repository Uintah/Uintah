#include <CCA/Components/Arches/CoalModels/GlobalCharOxidation.h>
#include <CCA/Components/Arches/CoalModels/CharOxidation.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Directives.h>
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <CCA/Components/Arches/ChemMix/TabPropsInterface.h>

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
GlobalCharOxidationBuilder::GlobalCharOxidationBuilder( const std::string         & modelName,
                                                        const vector<std::string> & reqICLabelNames,
                                                        const vector<std::string> & reqScalarLabelNames,
                                                        const ArchesLabel         * fieldLabels,
                                                        SimulationStateP          & sharedState,
                                                        int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

GlobalCharOxidationBuilder::~GlobalCharOxidationBuilder(){}

ModelBase* GlobalCharOxidationBuilder::build() {
  return scinew GlobalCharOxidation( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

GlobalCharOxidation::GlobalCharOxidation( std::string modelName, 
                                          SimulationStateP& sharedState,
                                          const ArchesLabel* fieldLabels,
                                          vector<std::string> icLabelNames, 
                                          vector<std::string> scalarLabelNames,
                                          int qn ) 
: CharOxidation(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  d_modelLabel = VarLabel::create( modelName, CCVariable<double>::getTypeDescription() );

  /**
  This class creates a gas source term for each char oxidation reaction. The intention is to allow for
  later extension to the multiple solids progress variables (MSPV) formulation (see Brewster et al 1988).

  The current (and simplest) abstraction is to make (number of reactions = number of oxidizer species).

  However, to extend this to an arbitrary number of reactions, one would simply need to create
  numbered labels, and add a mechanism (i.e. a get() method in the header file) to allow the CharOxidationMixtureFraction
  source term to grab the VarLabels for the gas source terms.

  @seealso CharOxidationMixtureFraction
  */

  // ========================================================================
  // O2 reactions
  
  d_O2GasModelLabel = VarLabel::create( modelName+"_gasSource_O2",  CCVariable<double>::getTypeDescription() );
  GasModelLabels_.push_back(  d_O2GasModelLabel );
  nu.push_back( 2.0/2.0 );
  oxidizer_name_.push_back("O2");

  A_.push_back( 2.30 );   E_.push_back( 9.29e7 ); ///< Baxter 1987 (all ranks)
  //A_.push_back( 0.30 );   E_.push_back( 1.49e8 ); ///< Field et al 1967 (all ranks)
  //A_.push_back( 1.03 );   E_.push_back( 7.49e7 ); ///< Baxter 1987 (hv Bituminous A)
  //A_.push_back( 0.479);   E_.push_back( 5.25e7 ); ///< Baxter 1987 (hv Bituminous C)
  //A_.push_back( 10.4 );   E_.push_back( 9.31e7 ); ///< Baxter 1987 (hv Bituminous C)
  //A_.push_back( 2.25 );   E_.push_back( 8.52e7 ); ///< Goetz et al 1982 (hv Bituminous A)
  //A_.push_back( 2.02 );   E_.push_back( 7.18e7 ); ///< Goetz et al 1982 (hv Bituminous C)
  //A_.push_back( 4.96 );   E_.push_back( 8.36e7 ); ///< Goetz et al 1982 (hv Bituminous C)



  // ========================================================================
  // H2 reactions

  // d_H2GasModelLabel = VarLabel::create( modelName+"_gasSource_H2",  CCVariable<double>::getTypeDescription() );
  // GasModelLabels_.push_back(  d_H2GasModelLabel );
  // nu.push_back( 0.0 ); //< (not sure what value this should be)

  //A_.push_back(  );   E_.push_back(  );



  // ========================================================================
  // CO2 reactions
  
  d_CO2GasModelLabel = VarLabel::create( modelName+"_gasSource_CO2", CCVariable<double>::getTypeDescription() );
  GasModelLabels_.push_back( d_CO2GasModelLabel );
  nu.push_back( 2.0/2.0 );
  oxidizer_name_.push_back("CO2");

  A_.push_back( 3.419);   E_.push_back( 1.30e8 ); ///< Baxter 1987 (lignite)
  //A_.push_back( 1160 );   E_.push_back( 2.59e8 ); ///< Baxter 1987 (hv Bituminous A)
  //A_.push_back( 4890 );   E_.push_back( 2.60e8 ); ///< Baxter 1987 (hv Bituminous C)
  //A_.push_back( 6188 );   E_.push_back( 2.40e8 ); ///< Baxter 1987 (Subbituminous C)
  //A_.push_back( 45.0 );   E_.push_back( 1.65e8 ); ///< Goetz et al 1982 (lignite)
  //A_.push_back( 95.14);   E_.push_back( 1.25e8 ); ///< Goetz et al 1982 (hv Bituminous A)
  //A_.push_back( 88.5 );   E_.push_back( 2.36e8 ); ///< Goetz et al 1982 (hi Bituminous C)
  //A_.push_back( 70.95);   E_.push_back( 1.78e8 ); ///< Goetz et al 1982 (Subbituminous C)



  // ========================================================================
  // H2O reactions

  d_H2OGasModelLabel = VarLabel::create( modelName+"_gasSource_H2O", CCVariable<double>::getTypeDescription() );
  GasModelLabels_.push_back( d_H2OGasModelLabel );
  nu.push_back( 1.0/2.0 );
  oxidizer_name_.push_back("H2O");

  A_.push_back( 4.26e4 );   E_.push_back( 3.16e8 ); ///< Otto et al 1979 (lignite)
  //A_.push_back( 208    );   E_.push_back( 2.40e8 ); ///< Otto et al 1979 (lignite)


  // ==============


  // Schmidt numbers for each species, used to obtain diffusion coefficients
  // Assuming Schmidt number for all oxidizers are 0.4; this may be changed later 
  for( vector<const VarLabel*>::iterator i = GasModelLabels_.begin(); i != GasModelLabels_.end(); ++i ) {
    Sc_.push_back(0.4);
  }

  d_useTparticle = false;

}

GlobalCharOxidation::~GlobalCharOxidation()
{}

//-----------------------------------------------------------------------------
//Problem Setup
//-----------------------------------------------------------------------------
void 
GlobalCharOxidation::problemSetup(const ProblemSpecP& params)
{
  // call parent's method first
  CharOxidation::problemSetup(params);

  d_TabPropsInterface->addAdditionalDV(oxidizer_name_);
  for( vector<string>::iterator iOxidizer = oxidizer_name_.begin(); iOxidizer != oxidizer_name_.end(); ++iOxidizer ) {
    const VarLabel* temp_label = VarLabel::find( *iOxidizer );
    OxidizerLabels_.push_back( temp_label );
  }

  ProblemSpecP db = params; 

  std::stringstream out;
  out << d_quadNode;
  string node = out.str();

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
      
      temp_label_name += "_qn";
      temp_label_name += node;

      // user specifies "role" of each internal coordinate
      if( role_name == "char_mass" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useChar = true;

      } else if ( role_name == "particle_length" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useLength = true;

      } else if( role_name == "particle_temperature" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useTparticle = true;

      } else {
        std::string errmsg;
        errmsg = "ERROR: Arches: GlobalCharOxidation: Invalid variable role for DQMOM equation: must be \"char_mass\", \"particle_length\", or \"particle_temperature\", you specified \"" + role_name + "\".";
        throw InvalidValue(errmsg,__FILE__,__LINE__);
      }
    }
  }

  // fix the d_icLabels to point to the correct quadrature node (since there is 1 model per quad node)
  for ( vector<std::string>::iterator iString = d_icLabels.begin(); 
        iString != d_icLabels.end(); ++iString) {

    temp_ic_name        = (*iString);
    temp_ic_name_full   = temp_ic_name;

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
        errmsg = "ERROR: Arches: GlobalCharOxidation: Invalid variable role for scalar equation: must be \"gas_temperature\", you specified \"" + role_name + "\".";
        throw InvalidValue(errmsg,__FILE__,__LINE__);
      }
    }
  }

  if(!d_useChar) {
    string errmsg = "ERROR: Arches: GlobalCharOxidation: No char internal coordinate was specified.  Quitting...";
    throw ProblemSetupException(errmsg,__FILE__,__LINE__);
  }

  if(!d_useLength) { 
    string errmsg = "ERROR: Arches: GlobalCharOxidation: No length variable was specified.  Quitting...";
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
      string errmsg = "ERROR: Arches: GlobalCharOxidation: Invalid variable \"" + iter->first + "\" given for \""+iter->second+"\" role, could not find in EqnFactory or DQMOMEqnFactory!";
      throw ProblemSetupException(errmsg,__FILE__,__LINE__);
    }

    if( iter->second == "char_mass" ){
      d_char_mass_label = current_eqn->getTransportEqnLabel();
      d_char_scaling_constant = current_eqn->getScalingConstant();

      DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(current_eqn);
      dqmom_eqn->addModel( d_modelLabel );

    } else if( iter->second == "particle_length" ){
      d_length_label = current_eqn->getTransportEqnLabel();
      d_length_scaling_constant = current_eqn->getScalingConstant();

    } else if( iter->second == "particle_temperature" ) {
      d_particle_temperature_label = current_eqn->getTransportEqnLabel();
      d_pt_scaling_constant = current_eqn->getScalingConstant();

    } else if( iter->second == "gas_temperature" ) {
      d_gas_temperature_label = current_eqn->getTransportEqnLabel();

    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ERROR: Arches: GlobalCharOxidation: You specified that the variable \"" + iter->first + 
                           "\" was required, but you did not specify a valid role for it! (You specified \"" + iter->second + "\")\n";
      throw InvalidValue( errmsg, __FILE__, __LINE__);
    }
  
  }//end for ic/scalar labels
  
  // // set model clipping (not used)
  //db->getWithDefault( "low_clip", d_lowModelClip,   1.0e-6 );
  //db->getWithDefault( "high_clip", d_highModelClip, 999999 );  

}



//-----------------------------------------------------------------------------
//Schedule the calculation of the Model 
//-----------------------------------------------------------------------------
void 
GlobalCharOxidation::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "GlobalCharOxidation::computeModel";
  Task* tsk = scinew Task(taskname, this, &GlobalCharOxidation::computeModel, timeSubStep);
  
  Ghost::GhostType gn = Ghost::None;

  d_timeSubStep = timeSubStep; 

  // require timestep label
  tsk->requires(Task::OldDW, d_fieldLabels->d_sharedState->get_delt_label() );

  if (d_timeSubStep == 0 && !d_labelSchedInit) {
    // Every model term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;
  }

  if( timeSubStep == 0 ) {

    tsk->computes( d_modelLabel );

    for( vector<const VarLabel*>::iterator i = GasModelLabels_.begin(); i != GasModelLabels_.end(); ++i ) { 
      tsk->computes( *i );
    }

    for( vector<const VarLabel*>::iterator i = OxidizerLabels_.begin(); i != OxidizerLabels_.end(); ++i ) {
      tsk->requires(Task::OldDW, *i, gn, 0 );
    }

    tsk->requires(Task::OldDW, d_weight_label, gn, 0);
    tsk->requires(Task::OldDW, d_length_label, gn, 0);

    if(d_useTparticle) {
      tsk->requires(Task::OldDW, d_particle_temperature_label, gn, 0);
    } else {
      tsk->requires(Task::OldDW, d_gas_temperature_label, gn, 0);
    }

    tsk->requires(Task::OldDW, d_fieldLabels->d_viscosityCTSLabel, gn, 0);

  } else {

    tsk->modifies( d_modelLabel );

    for( vector<const VarLabel*>::iterator i = GasModelLabels_.begin(); i != GasModelLabels_.end(); ++i ) {
      tsk->modifies( *i );
    }
    
    for( vector<const VarLabel*>::iterator i = OxidizerLabels_.begin(); i != OxidizerLabels_.end(); ++i ) {
      tsk->requires(Task::NewDW, *i, gn, 0 );
    }

    tsk->requires(Task::NewDW, d_weight_label, gn, 0);
    tsk->requires(Task::NewDW, d_length_label, gn, 0);

    if(d_useTparticle) {
      tsk->requires(Task::NewDW, d_particle_temperature_label, gn, 0);
    } else {
      tsk->requires(Task::NewDW, d_gas_temperature_label, gn, 0);
    }

    tsk->requires(Task::NewDW, d_fieldLabels->d_viscosityCTSLabel, gn, 0);
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}



//-----------------------------------------------------------------------------
//Actually compute the source term 
//-----------------------------------------------------------------------------
void
GlobalCharOxidation::computeModel( const ProcessorGroup * pc, 
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

    constCCVariable<double> weight;
    constCCVariable<double> wa_length;
    constCCVariable<double> temperature; // holds gas OR particle temperature...
    constCCVariable<double> turbulent_viscosity;

    CCVariable<double> char_model;
    vector< CCVariable<double>* > gasModelCCVars;
    vector< constCCVariable<double>* > oxidizerCCVars;


    if( timeSubStep == 0 ) {

      new_dw->allocateAndPut( char_model,      d_modelLabel, matlIndex, patch );
      char_model.initialize(0.0);

      int zz = 0;
      for( vector<const VarLabel*>::iterator iLabel = GasModelLabels_.begin(); iLabel != GasModelLabels_.end(); ++iLabel, ++zz ) {
        gasModelCCVars.push_back( scinew CCVariable<double> );
        new_dw->allocateAndPut( *(gasModelCCVars[zz]), *iLabel, matlIndex, patch );
        (*gasModelCCVars[zz]).initialize(0.0);
      }
      zz=0;
      for( vector<const VarLabel*>::iterator iLabel = OxidizerLabels_.begin(); iLabel != OxidizerLabels_.end(); ++iLabel, ++zz ) {
        oxidizerCCVars.push_back( scinew constCCVariable<double> );
        old_dw->get( *(oxidizerCCVars[zz]), *iLabel, matlIndex, patch, gn, 0 );
      }

      old_dw->get( weight,       d_weight_label,    matlIndex, patch, gn, 0 );
      old_dw->get( wa_length,    d_length_label,    matlIndex, patch, gn, 0 );
      if(d_useTparticle) {
        old_dw->get( temperature, d_particle_temperature_label, matlIndex, patch, gn, 0 );
      } else {
        old_dw->get( temperature, d_gas_temperature_label, matlIndex, patch, gn, 0 );
      }
      old_dw->get( turbulent_viscosity, d_fieldLabels->d_viscosityCTSLabel, matlIndex, patch, gn, 0 );
      
    } else {

      new_dw->getModifiable( char_model,      d_modelLabel, matlIndex, patch );

      int zz = 0;
      for( vector<const VarLabel*>::iterator iLabel = GasModelLabels_.begin(); iLabel != GasModelLabels_.end(); ++iLabel, ++zz ) {
        gasModelCCVars.push_back( scinew CCVariable<double> );
        new_dw->getModifiable( *(gasModelCCVars[zz]), *iLabel, matlIndex, patch );
      }
      zz = 0;
      for( vector<const VarLabel*>::iterator iLabel = OxidizerLabels_.begin(); iLabel != OxidizerLabels_.end(); ++iLabel, ++zz ) {
        oxidizerCCVars.push_back( scinew constCCVariable<double> );
        new_dw->get( *(oxidizerCCVars[zz]), *iLabel, matlIndex, patch, gn, 0 );
      }

      new_dw->get( weight,       d_weight_label,    matlIndex, patch, gn, 0 );
      new_dw->get( wa_length,    d_length_label,    matlIndex, patch, gn, 0 );
      if(d_useTparticle) {
        new_dw->get( temperature, d_particle_temperature_label, matlIndex, patch, gn, 0 );
      } else {
        new_dw->get( temperature, d_gas_temperature_label, matlIndex, patch, gn, 0 );
      }
      new_dw->get( turbulent_viscosity, d_fieldLabels->d_viscosityCTSLabel, matlIndex, patch, gn, 0 );

    }

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 

      bool weight_is_small = (weight[c] < d_w_small) || (weight[c] == 0.0);

      double char_rxn_rate_;

      if(weight_is_small) {
        char_rxn_rate_ = 0.0;

        // set gas model terms (3 of them) equal to 0
        int z=0;
        for( vector< CCVariable<double>* >::iterator iGasModel = gasModelCCVars.begin(); iGasModel != gasModelCCVars.end(); ++iGasModel, ++z) {
          (**iGasModel)[c] = 0.0;
        }

      } else {

        double unscaled_weight;
        double unscaled_length;
        double unscaled_temperature;

        double A_p;   ///< particle area
        double xi_p;  ///< particle shape/area factor
        double MW_carbon = 12.0;

        vector<double> k_m( GasModelLabels_.size() ); ///< Mass transfer coefficient array for each of the 4 char reactions: O2, H2, CO2, H2O
        vector<double> k_r( GasModelLabels_.size() ); ///< Reaction rate constants array for each of the 4 char reactions: O2, H2, CO2, H2O
 
        int z;

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

        A_p = (4*pi_*pow(unscaled_length/2.0,2));

        xi_p = 1.0;

        // Mass transfer coefficients
        z=0;
        vector< constCCVariable<double>* >::iterator iOxidizer = oxidizerCCVars.begin();
        for( vector< CCVariable<double>* >::iterator iGasModel = gasModelCCVars.begin(); iGasModel != gasModelCCVars.end(); ++iGasModel, ++iOxidizer, ++z) {
          /// Mass transfer coefficients for char oxidation reactions are given by the expression:
          /// \f$ k_{m} = \frac{ 2 D_{om} }{ d_{p} } = \frac{ 2 \nu_{T,fluid} }
          k_m[z] = 2*(turbulent_viscosity[c]/Sc_[z])/unscaled_length;

          /// Rate constant for char oxidation reactions are given by the rate constant:
          /// \f$ k_{r} = A T^{n} exp \left( -\frac{E}{RT} \right) \left[ \mbox{Oxidizer Conc.} \right] \f$
          /// (Note: n=1 for all the char oxidation reactions... This is implicit in the code.)
          k_r[z] = A_[z] * unscaled_temperature * exp( - E_[z] / R_ / unscaled_temperature )*(**iOxidizer)[c];

          double rate = ( A_p*nu[z]*MW_carbon*xi_p*k_r[z]*k_m[z] )/( k_m[z] + k_r[z]*xi_p + TINY );
          (**iGasModel)[c] = rate;
          char_rxn_rate_ += (-rate/d_char_scaling_constant);
        }

      }

      char_model[c] = char_rxn_rate_;

    }//end cell loop

    for( vector< CCVariable<double>* >::iterator iC = gasModelCCVars.begin(); iC != gasModelCCVars.end(); ++iC ) {
      delete *iC;
    }
  
  }//end patch loop
}


