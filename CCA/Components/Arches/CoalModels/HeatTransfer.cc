#include <CCA/Components/Arches/CoalModels/HeatTransfer.h>
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

/* NOTE: abskp is thermal conductivity */
using namespace std;
using namespace Uintah; 

//---------------------------------------------------------------------------
// Builder:
HeatTransferBuilder::HeatTransferBuilder( const std::string         & modelName,
                                          const vector<std::string> & reqICLabelNames,
                                          const vector<std::string> & reqScalarLabelNames,
                                          const ArchesLabel         * fieldLabels,
                                          SimulationStateP          & sharedState,
                                          int qn ) :
  ModelBuilder( modelName, fieldLabels, reqICLabelNames, reqScalarLabelNames, sharedState, qn )
{
}

HeatTransferBuilder::~HeatTransferBuilder(){}

ModelBase* HeatTransferBuilder::build() {
  return scinew HeatTransfer( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

HeatTransfer::HeatTransfer( std::string modelName, 
                            SimulationStateP& sharedState,
                            const ArchesLabel* fieldLabels,
                            vector<std::string> icLabelNames, 
                            vector<std::string> scalarLabelNames,
                            int qn ) 
: ModelBase(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn),
  d_fieldLabels(fieldLabels)
{
  //d_radiation = false;
  d_quad_node = qn;

  // Create a label for this model
  d_modelLabel = VarLabel::create( modelName, CCVariable<double>::getTypeDescription() );

  // Create the gas phase source term associated with this model
  std::string gasSourceName = modelName + "_gasSource";
  d_gasLabel = VarLabel::create( gasSourceName, CCVariable<double>::getTypeDescription() );
}

HeatTransfer::~HeatTransfer()
{}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
  void 
HeatTransfer::problemSetup(const ProblemSpecP& params, int qn)
{
  ProblemSpecP db = params; 
  
  const ProblemSpecP params_root = db->getRootNode(); 

  // Check for radiation 
  d_radiation = false;
  if (params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ExplicitSolver")->findBlock("EnthalpySolver")->findBlock("DORadiationModel"))
    d_radiation = true; // if gas phase radiation is turned on.  

  //user can specifically turn off radiation heat transfer
  if (db->findBlock("noRadiation"))
    d_radiation = false; 

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
    if ( role_name == "particle_length" 
             || role_name == "raw_coal_mass"
             || role_name == "particle_temperature" ) {
      LabelToRoleMap[temp_label_name] = role_name;
    } else {
      std::string errmsg = "Invalid variable role for Heat Transfer model!";
      throw InvalidValue(errmsg,__FILE__,__LINE__);
    }

    // set model clipping
    db->getWithDefault( "low_clip",  d_lowModelClip,  1.0e-6 );
    db->getWithDefault( "high_clip", d_highModelClip, 999999 );
 
  }


  // Look for required scalars
  //   ( Kobayashi-Sarofim model doesn't use any extra scalars (yet)
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
      errmsg = "Invalid variable role for Kobayashi Sarofim Devolatilization model: must be \"particle_temperature\" or \"raw_coal_mass\", you specified \"" + role_name + "\".";
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
  // (Not needed for HeatTransfer model (yet)... If it is, uncomment the block below)
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

  std::string abskpName = "abskp_qn";
  abskpName += node; 
  d_abskp = VarLabel::create(abskpName, CCVariable<double>::getTypeDescription());

  std::string smoothTName = "smoothTfield_qn";
  smoothTName += node; 
  d_smoothTfield = VarLabel::create(smoothTName, CCVariable<double>::getTypeDescription());

}



//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
HeatTransfer::sched_dummyInit( const LevelP& level, SchedulerP& sched ) 
{
  string taskname = "HeatTransfer::dummyInit"; 

  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &HeatTransfer::dummyInit);

  tsk->computes(d_modelLabel);
  tsk->computes(d_gasLabel); 

  tsk->requires( Task::OldDW, d_modelLabel, gn, 0);
  tsk->requires( Task::OldDW, d_gasLabel,   gn, 0);

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}

//-------------------------------------------------------------------------
// Method: Actually do the dummy initialization
//-------------------------------------------------------------------------
/** @details
This is called from ExplicitSolver::noSolve(), which skips the first timestep
 so that the initial conditions are correct.

This method was originally in ModelBase, but it requires creating CCVariables
 for the model and gas source terms, and the CCVariable type (double, Vector, &c.)
 is model-dependent.  Putting the method here eliminates if statements in 
 ModelBase and keeps the ModelBase class as generic as possible.
 */
void
HeatTransfer::dummyInit( const ProcessorGroup* pc,
                         const PatchSubset* patches, 
                         const MaterialSubset* matls, 
                         DataWarehouse* old_dw, 
                         DataWarehouse* new_dw )
{
  for( int p=0; p < patches->size(); ++p ) {

    Ghost::GhostType  gn = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> model;
    CCVariable<double> gasHeatRate;
    
    constCCVariable<double> oldModel;
    constCCVariable<double> oldGasHeatRate;

    new_dw->allocateAndPut( model,       d_modelLabel, matlIndex, patch );
    new_dw->allocateAndPut( gasHeatRate, d_gasLabel,   matlIndex, patch ); 

    old_dw->get( oldModel,       d_modelLabel, matlIndex, patch, gn, 0 );
    old_dw->get( oldGasHeatRate, d_gasLabel,   matlIndex, patch, gn, 0 );
    
    model.copyData(oldModel);
    gasHeatRate.copyData(oldGasHeatRate);

  }
}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of some variables 
//---------------------------------------------------------------------------
void 
HeatTransfer::sched_initVars( const LevelP& level, SchedulerP& sched )
{

  std::string taskname = "HeatTransfer::initVars";
  Task* tsk = scinew Task(taskname, this, &HeatTransfer::initVars);

  tsk->computes(d_abskp);
  tsk->computes(d_smoothTfield);

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}

//-------------------------------------------------------------------------
// Method: Initialize variables
//-------------------------------------------------------------------------
void
HeatTransfer::initVars( const ProcessorGroup * pc, 
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
    abskp.initialize(0.);

    CCVariable<double> smoothTfield; 
    new_dw->allocateAndPut( smoothTfield, d_smoothTfield, matlIndex, patch ); 
    smoothTfield.initialize(0.);

  }
}

//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model 
//---------------------------------------------------------------------------
void 
HeatTransfer::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "HeatTransfer::computeModel";
  Task* tsk = scinew Task(taskname, this, &HeatTransfer::computeModel);

  d_timeSubStep = timeSubStep; 

  if (d_timeSubStep == 0 && !d_labelSchedInit) {
    // Every model term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;

    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel); 
    tsk->computes(d_abskp);
    tsk->computes(d_smoothTfield);
  } else {
    tsk->modifies(d_modelLabel);
    tsk->modifies(d_gasLabel);  
    tsk->modifies(d_abskp);
    tsk->modifies(d_smoothTfield);
  }

  //EqnFactory& eqn_factory = EqnFactory::self();
  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();

  // construct the weight label corresponding to this quad node
  std::string temp_weight_name = "w_qn";
  std::string node;
  std::stringstream out;
  out << d_quad_node;
  node = out.str();
  temp_weight_name += node;
  EqnBase& t_weight_eqn = dqmom_eqn_factory.retrieve_scalar_eqn( temp_weight_name );
  DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(t_weight_eqn);
  d_weight_label = weight_eqn.getTransportEqnLabel();
  d_w_small = weight_eqn.getSmallClip();
  d_w_scaling_factor = weight_eqn.getScalingConstant();
  tsk->requires(Task::OldDW, d_weight_label, Ghost::None, 0);
  
  // also require paticle velocity, gas velocity, gas temperature, and density
  ArchesLabel::PartVelMap::const_iterator iQuad = d_fieldLabels->partVel.find(d_quad_node);
  tsk->requires(Task::OldDW, iQuad->second, Ghost::None, 0);
  tsk->requires(Task::OldDW, d_fieldLabels->d_newCCVelocityLabel, Ghost::None, 0);
  tsk->requires(Task::OldDW, d_fieldLabels->d_densityCPLabel, Ghost::None, 0);
  tsk->requires(Task::OldDW, d_fieldLabels->d_tempINLabel, Ghost::AroundCells, 1);
 
  if(d_radiation){
    tsk->requires(Task::OldDW, d_fieldLabels->d_radiationSRCINLabel,  Ghost::None, 0);
    tsk->requires(Task::OldDW, d_fieldLabels->d_abskgINLabel,  Ghost::None, 0);   
  }


  // For each required variable, determine what role it plays
  // - "gas_temperature" - require the "tempIN" label
  // - "particle_temperature" - look in DQMOMEqnFactory
  // - "particle_length" - look in DQMOMEqnFactory
  // - "raw_coal_mass" - look in DQMOMEqnFactory

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
          std::string errmsg = "ARCHES: HeatTransfer: Invalid variable given in <ICVars> block, for <variable> tag for HeatTransfer model.";
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
          std::string errmsg = "ARCHES: HeatTransfer: Invalid variable given in <ICVars> block, for <variable> tag for HeatTransfer model.";
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
          std::string errmsg = "ARCHES: HeatTransfer: Invalid variable given in <ICVars> block, for <variable> tag for HeatTransfer model.";
          errmsg += "\nCould not find given raw coal mass variable \"";
          errmsg += *iter;
          errmsg += "\" in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      } 
    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ARCHES: HeatTransfer: You specified that the variable \"" + *iter + 
                           "\" was required, but you did not specify a role for it!\n";
      throw InvalidValue( errmsg, __FILE__, __LINE__);
    }
  }

  // for each required scalar variable:
  //  (but no scalar equation variables should be required for the HeatTransfer model, at least not for now...)
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
          std::string errmsg = "ARCHES: HeatTransfer: Invalid variable given in <scalarVars> block for <variable> tag for HeatTransfer model.";
          errmsg += "\nCould not find given <insert role name here> variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      }
    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ARCHES: HeatTransfer: You specified that the variable \"" + *iter + 
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
HeatTransfer::computeModel( const ProcessorGroup * pc, 
    const PatchSubset    * patches, 
    const MaterialSubset * matls, 
    DataWarehouse        * old_dw, 
    DataWarehouse        * new_dw )
{
  double pi = acos(-1.0);
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> heat_rate;
    if( new_dw->exists( d_modelLabel, matlIndex, patch) ) {
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
    
    CCVariable<double> smoothTfield;
    if( new_dw->exists( d_smoothTfield, matlIndex, patch) ) {
      new_dw->getModifiable( smoothTfield, d_smoothTfield, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( smoothTfield, d_smoothTfield, matlIndex, patch );  
      smoothTfield.initialize(0.0);
    }
   

    // get particle velocity used to calculate Reynolds number
    constCCVariable<Vector> partVel;  
    ArchesLabel::PartVelMap::const_iterator iQuad = d_fieldLabels->partVel.find(d_quad_node);
    old_dw->get( partVel, iQuad->second, matlIndex, patch, gn, 0);
    
    // gas velocity used to calculate Reynolds number
    constCCVariable<Vector> gasVel; 
    old_dw->get( gasVel, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0 ); 
    
    constCCVariable<double> den;
    old_dw->get(den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0 ); 
 
    constCCVariable<double> radiationSRCIN;
    constCCVariable<double> abskgIN;
    //CCVariable<double> enthNonLinSrc;

    if(d_radiation){
      old_dw->get(radiationSRCIN, d_fieldLabels->d_radiationSRCINLabel, matlIndex, patch, gn, 0);
      old_dw->get(abskgIN,        d_fieldLabels->d_abskgINLabel,        matlIndex, patch, gn, 0);
    }

    constCCVariable<double> gas_temperature;
    old_dw->get( gas_temperature, d_fieldLabels->d_tempINLabel, matlIndex, patch, gac, 1 );

    constCCVariable<double> w_particle_temperature;
    old_dw->get( w_particle_temperature, d_particle_temperature_label, matlIndex, patch, gn, 0 );

    constCCVariable<double> w_particle_length;
    old_dw->get( w_particle_length, d_particle_length_label, matlIndex, patch, gn, 0 );

    //constCCVariable<double> w_mass_raw_coal;
    //old_dw->get( w_mass_raw_coal, d_raw_coal_mass_label, matlIndex, patch, gn, 0 );

    constCCVariable<double> weight;
    old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );

    for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
      IntVector c = *iter; 

      Vector sphGas = Vector(0.,0.,0.);
      Vector cartGas = gasVel[c]; 
      Vector sphPart = Vector(0.,0.,0.);
      Vector cartPart = partVel[c]; 

      sphGas = cart2sph( cartGas ); 
      sphPart = cart2sph( cartPart ); 
	
	    double length;
	    double particle_temperature;

// ****** ARCHES KLUDGE ******  inserted 10/03/2009

      //if (weight[c] < d_w_small ) {  // if you use this one, temperature blows up everywhere
                                       // (the smaller d_w_small, the bigger the blow-up)
      if( weight[c] < 1e-4 ) {

//  ****** END KLUDGE ******
        heat_rate[c] = 0.0;
        length = 0.0;
        particle_temperature = 0.0;
        smoothTfield[c] = gas_temperature[c];
      } else {
	      length = w_particle_length[c]*d_pl_scaling_factor/weight[c];
	      particle_temperature = w_particle_temperature[c]*d_pt_scaling_factor/weight[c];
        smoothTfield[c] = particle_temperature;

        double Pr = 0.7; // Prandtl number
        double blow = 1.0;
        double sigma = 5.67e-8; // [=] J/(s-m^2-K^4) : Stefan-Boltzmann constant from white book p. 354

        double rkg; // [=] W/(m-K) : thermal conductivity of gas 
                    // (values are from Yos 1963, "Transport properties of nitrogen, hydrogen, oxygen, and air to 30,000 K", p.49 and p.68)
        if( gas_temperature[c] < 1500 ) {
          rkg = 0.0690;
        } else if( gas_temperature[c] >= 1500 && gas_temperature[c] < 2500 ) {
          rkg = 0.121;
        } else if( gas_temperature[c] >= 2500 ) {
          rkg = 0.383;
        } else {
          rkg = 0.0;
        }
        double visc = 2.0e-5; // [=] m^2/s : viscosity of gas

        double Re  = abs(sphGas.z() - sphPart.z())*length*den[c]/visc;

        double Nu = 2.0 + 0.6*pow(Re,0.5)*pow(Pr,0.333); // Nusselt number
// ****** FIXME ******
        double rhop = 1000.0; // [=] kg/m^3 : Density of particle
// ****** END FIXME ******
        
        //double cp = 3000.0; // [=] J/(kg-K) : heat capacity
        double cp = 1500.0; // [=] J/(kg-K) : heat capacity (new value of 1500 recommended by Julien as more realistic)
        double m_p = rhop*4.0/3.0*pi*pow(length/2.0,3.0); // [=] kg : mass of particle
        double Qconv = Nu*pi*blow*rkg*length*(gas_temperature[c]-particle_temperature);

	      // Radiative transfer
	      double Qrad = 0.0;
	
	      if(d_radiation) {
	        if(abskgIN[c]<1e-6){

	          Qrad = 0;

	        } else {

	          double Qabs = 0.8;
	          double Apsc = (pi/4)*Qabs*pow(length/2,2);
	          double Eb = 4.0*sigma*pow(particle_temperature,4);
	          double Eg = 4.0*sigma*abskgIN[c]*pow(gas_temperature[c],4);

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

