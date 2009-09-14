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

//===========================================================================

using namespace std;
using namespace Uintah; 

//---------------------------------------------------------------------------
// Builder:
HeatTransferBuilder::HeatTransferBuilder( const std::string         & modelName,
    const vector<std::string> & reqLabelNames,
    const ArchesLabel              * fieldLabels,
    SimulationStateP          & sharedState,
    int qn ) :
  ModelBuilder( modelName, fieldLabels, reqLabelNames, sharedState, qn )
{}

HeatTransferBuilder::~HeatTransferBuilder(){}

ModelBase*
HeatTransferBuilder::build(){
  return scinew HeatTransfer( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

HeatTransfer::HeatTransfer( std::string srcName, SimulationStateP& sharedState,
    const ArchesLabel* fieldLabels,
    vector<std::string> icLabelNames, int qn ) 
: ModelBase(srcName, sharedState, fieldLabels, icLabelNames, qn),d_fieldLabels(fieldLabels)
{
  //d_radiation = true;
  d_quad_node = qn;
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
  ProblemSpecP db_icvars = params->findBlock("ICVars");
  
  const ProblemSpecP params_root = db->getRootNode(); 

  // Check for radiation 
  bool d_radiation = false;
  if (params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ExplicitSolver")->findBlock("EnthalpySolver")->findBlock("DORadiationModel"))
    d_radiation = true; // if gas phase radiation is turned on.  

  //user can specifically turn off radiation heat transfer
  if (db->findBlock("noRadiation"))
    d_radiation = false; 


  for (ProblemSpecP variable = db_icvars->findBlock("variable"); variable != 0; variable = variable->findNextBlock("variable") ) {
    string label_name;
    string role_name;

    variable->getAttribute("label",label_name);
    variable->getAttribute("role",role_name);

    std::string temp_label_name = label_name;
    std::string node;
    std::stringstream out;
    out << qn;
    node = out.str();
    temp_label_name += "_qn";
    temp_label_name += node;

    // This way restricts what "roles" the user can specify (less flexible)
    if (role_name == "temperature"
    || role_name == "particle_length" || role_name == "raw_coal_mass_fraction") {
      LabelToRoleMap[temp_label_name] = role_name;
    } else if(role_name == "particle_temperature" ){  
       LabelToRoleMap[temp_label_name] = role_name;
    } else {
      std::string errmsg;
      errmsg = "Invalid variable role for Heat Transfer model";
      throw InvalidValue(errmsg,__FILE__,__LINE__);
    }

    //This way does not restrict what "roles" the user can specify (more flexible)
    //LabelToRoleMap[label_name] = role_name;

    db->getWithDefault( "low_clip", d_lowClip, 1.e-6 );
    db->getWithDefault( "high_clip", d_highClip, 1 );  
 
  }

  // now fix the d_icLabels to point to the correct quadrature node (since there is 1 model per quad node)
  for ( vector<std::string>::iterator iString = d_icLabels.begin(); iString != d_icLabels.end(); ++iString) {
    std::string temp_ic_name        = (*iString);
    std::string temp_ic_name_full   = temp_ic_name;

    std::string node;
    std::stringstream out;
    out << qn;
    node = out.str();
    temp_ic_name_full += "_qn";
    temp_ic_name_full += node;

    std::replace( d_icLabels.begin(), d_icLabels.end(), temp_ic_name, temp_ic_name_full);
  }

  std::string node; 
  std::stringstream out; 
  out << qn; 
  node = out.str();
  std::string gasHeatName = "gasHeatRate_qn";
  gasHeatName += node; 
  d_gasHeatRate = VarLabel::create(gasHeatName, CCVariable<double>::getTypeDescription());
  
  std::string abskpName = "abskp_qn";
  abskpName += node; 
  d_abskp = VarLabel::create(abskpName, CCVariable<double>::getTypeDescription());


}
//---------------------------------------------------------------------------
// Method: Schedule the initialization of some variables 
//---------------------------------------------------------------------------
void 
HeatTransfer::sched_initVars( const LevelP& level, SchedulerP& sched )
{

  std::string taskname = "HeatTransfer::initVars";
  Task* tsk = scinew Task(taskname, this, &HeatTransfer::initVars);

  tsk->computes(d_gasHeatRate); 

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}
void
HeatTransfer::initVars( const ProcessorGroup * pc, 
    const PatchSubset    * patches, 
    const MaterialSubset * matls, 
    DataWarehouse        * old_dw, 
    DataWarehouse        * new_dw )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    //Ghost::GhostType  gac = Ghost::AroundCells;
    //Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int matlIndex = 0;

    CCVariable<double> gasHeatRate; 
    new_dw->allocateAndPut( gasHeatRate, d_gasHeatRate, matlIndex, patch ); 
    gasHeatRate.initialize(0.);
    
    CCVariable<double> abskp; 
    new_dw->allocateAndPut( abskp, d_abskp, matlIndex, patch ); 
    abskp.initialize(0.);

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
    tsk->computes(d_gasHeatRate);
    tsk->computes(d_abskp);
  } else {
    tsk->modifies(d_modelLabel); 
    tsk->modifies(d_gasHeatRate);
    tsk->modifies(d_abskp);
  }

  EqnFactory& eqn_factory = EqnFactory::self();
  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();

  // construct the weight label corresponding to this quad node
  std::string temp_weight_name = "w_qn";
  std::string node;
  std::stringstream out;
  out << d_quad_node;
  node = out.str();
  temp_weight_name += node;
  EqnBase& weight_eqn = dqmom_eqn_factory.retrieve_scalar_eqn( temp_weight_name );
  d_weight_label = weight_eqn.getTransportEqnLabel();
  tsk->requires(Task::OldDW, d_weight_label, Ghost::None, 0);
  

    ArchesLabel::PartVelMap::const_iterator iQuad = d_fieldLabels->partVel.find(d_quad_node);
    tsk->requires(Task::OldDW, iQuad->second, Ghost::None, 0);
    
    tsk->requires( Task::OldDW, d_fieldLabels->d_newCCVelocityLabel, Ghost::None, 0);
    tsk->requires(Task::OldDW, d_fieldLabels->d_densityCPLabel, Ghost::None, 0);
    
    if(d_radiation){
    tsk->requires(Task::OldDW, d_fieldLabels->d_radiationSRCINLabel,  Ghost::None, 0);
    tsk->requires(Task::OldDW, d_fieldLabels->d_abskgINLabel,  Ghost::None, 0);   
    }

  // For each required variable, determine if it plays the role of temperature or mass fraction;
  //  if it plays the role of mass fraction, then look for it in equation factories
  for (vector<std::string>::iterator iter = d_icLabels.begin(); 
      iter != d_icLabels.end(); iter++) { 

    map<string, string>::iterator iMap = LabelToRoleMap.find(*iter);

   if ( iMap != LabelToRoleMap.end() ) {
      if ( iMap->second == "gas_temperature") {
        // automatically use Arches' temperature label if role="temperature"
        tsk->requires(Task::OldDW, d_fieldLabels->d_tempINLabel, Ghost::AroundCells, 1);

        // Only require() variables found in equation factories (right now we're not tracking temperature this way)
      } 
      else if ( iMap->second == "particle_temperature") {
        // if it's a normal scalar
        if ( eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& current_eqn = eqn_factory.retrieve_scalar_eqn(*iter);
          d_particle_temperature_label = current_eqn.getTransportEqnLabel();
          tsk->requires(Task::OldDW, d_particle_temperature_label, Ghost::None, 0);
          // if it's a dqmom scalar
        } else if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_particle_temperature_label = current_eqn.getTransportEqnLabel();
          d_pt_scaling_factor = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_particle_temperature_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: HeatTransfer: Invalid variable given in <variable> tag for HeatTransfer model";
          errmsg += "\nCould not find given particle temperature variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory or in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      } //else... we don't need that variable!!!
      else if ( iMap->second == "particle_length") {
        // if it's a normal scalar
        if ( eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& current_eqn = eqn_factory.retrieve_scalar_eqn(*iter);
          d_particle_length_label = current_eqn.getTransportEqnLabel();
          tsk->requires(Task::OldDW, d_particle_length_label, Ghost::None, 0);
          // if it's a dqmom scalar
        } else if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_particle_length_label = current_eqn.getTransportEqnLabel();
          d_pl_scaling_factor = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_particle_length_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: HeatTransfer: Invalid variable given in <variable> tag for HeatTransfer model";
          errmsg += "\nCould not find given particle length variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory or in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      } //else... we don't need that variable!!!
      else if ( iMap->second == "raw_coal_mass_fraction") {
        // if it's a normal scalar
        if ( eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& current_eqn = eqn_factory.retrieve_scalar_eqn(*iter);
          d_raw_coal_mass_fraction_label = current_eqn.getTransportEqnLabel();
          tsk->requires(Task::OldDW, d_raw_coal_mass_fraction_label, Ghost::None, 0);
          // if it's a dqmom scalar
        } else if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_raw_coal_mass_fraction_label = current_eqn.getTransportEqnLabel();
          d_rc_scaling_factor = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_raw_coal_mass_fraction_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: HeatTransfer: Invalid variable given in <variable> tag for HeatTransfer model";
          errmsg += "\nCould not find given coal mass fraction variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory or in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      } //else... we don't need that variable!!!
    } 
    else {
      // can't find it in the labels-to-roles map!
      std::string errmsg = "ARCHES: HeatTransfer: Could not find role for given variable \"" + *iter + "\".";
      throw InvalidValue(errmsg,__FILE__,__LINE__);
    }
  }

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
  //cout << "computemodel start" << endl;
  double pi = 3.14159265;
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int matlIndex = 0;

    CCVariable<double> heat_rate;
    CCVariable<double> gas_heat_rate; 
    CCVariable<double> abskp; 
    if (new_dw->exists( d_modelLabel, matlIndex, patch )){
      new_dw->getModifiable( heat_rate, d_modelLabel, matlIndex, patch ); 
      new_dw->getModifiable( gas_heat_rate, d_gasHeatRate, matlIndex, patch ); 
      new_dw->getModifiable( abskp, d_abskp, matlIndex, patch ); 
      heat_rate.initialize(0.0);
      gas_heat_rate.initialize(0.0);
      abskp.initialize(0.0);
    } else {
      new_dw->allocateAndPut( heat_rate, d_modelLabel, matlIndex, patch );
      new_dw->allocateAndPut( gas_heat_rate, d_gasHeatRate, matlIndex, patch );
      new_dw->allocateAndPut( abskp, d_abskp, matlIndex, patch );  
      heat_rate.initialize(0.0);
      gas_heat_rate.initialize(0.0);
      abskp.initialize(0.0);
    }
    
    constCCVariable<Vector> partVel;  
    ArchesLabel::PartVelMap::const_iterator iQuad = d_fieldLabels->partVel.find(d_quad_node);
    old_dw->get( partVel, iQuad->second, matlIndex, patch, gn, 0);
    
    constCCVariable<Vector> gasVel; 
    old_dw->get( gasVel, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0 ); 
    constCCVariable<double> den;
    old_dw->get(den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0 ); 
    
    constCCVariable<double> radiationSRCIN;
    constCCVariable<double> abskgIN;
    CCVariable<double> enthNonLinSrc;
    if(d_radiation){
    old_dw->get(radiationSRCIN,   d_fieldLabels->d_radiationSRCINLabel,  matlIndex, patch,gn, 0);
    old_dw->get(abskgIN,   d_fieldLabels->d_abskgINLabel,  matlIndex, patch,gn, 0);
    }

    constCCVariable<double> temperature;
    old_dw->get( temperature, d_fieldLabels->d_dummyTLabel, matlIndex, patch, gac, 1 );
    constCCVariable<double> w_particle_temperature;
    old_dw->get( w_particle_temperature, d_particle_temperature_label, matlIndex, patch, gn, 0 );
    constCCVariable<double> w_particle_length;
    old_dw->get( w_particle_length, d_particle_length_label, matlIndex, patch, gn, 0 );
    constCCVariable<double> w_omegac;
    old_dw->get( w_omegac, d_raw_coal_mass_fraction_label, matlIndex, patch, gn, 0 );
    constCCVariable<double> weight;
    old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );

    double small = 1e-16; // for now... 

    for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
      IntVector c = *iter; 

        Vector sphGas = Vector(0.,0.,0.);
        Vector cartGas = gasVel[c]; 
        Vector sphPart = Vector(0.,0.,0.);
        Vector cartPart = partVel[c]; 

        sphGas = cart2sph( cartGas ); 
        sphPart = cart2sph( cartPart ); 
	
        if (weight[c] < 1e-4 ) {
		      heat_rate[c] = 0.0;
	      } else {
	
	      double length;
	      double particle_temperature;

        if ( weight[c] < small ) {
          length = 0.0;
          particle_temperature = 0.0;
        } else {
	        length = w_particle_length[c]*d_pl_scaling_factor/weight[c];
	        particle_temperature = w_particle_temperature[c]*d_pt_scaling_factor/weight[c];
        }

	      double Pr = 0.7;
	      double blow = 1.0;
	      double sigma = 5.67e-8;

	      double rkg = 0.03;
	      double visc = 2.0e-5;
	      double Re  = abs(sphGas.z() - sphPart.z())*length*den[c]/visc;
	      double Nu = 2.0 + 0.6*pow(Re,0.5)*pow(Pr,0.333);
	      double rhop = 1000.0;
	      double cp = 3000.0;
	      double alpha = rhop*(4/3*pi*pow(length/2,3));
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

        heat_rate[c] =(Qconv+Qrad)/(alpha*cp*d_pt_scaling_factor); 

        gas_heat_rate[c] = 0.0;
    	}
    }
  }
}

