#include <CCA/Components/Arches/CoalModels/YDragModel.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/CoalModels/PartVel.h>
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

YDragModelBuilder::YDragModelBuilder( const std::string         & modelName, 
                                            const vector<std::string> & reqICLabelNames,
                                            const vector<std::string> & reqScalarLabelNames,
                                            ArchesLabel         * fieldLabels,
                                            SimulationStateP          & sharedState,
                                            int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{}

YDragModelBuilder::~YDragModelBuilder(){}

ModelBase* YDragModelBuilder::build(){
  return scinew YDragModel( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}

// End Builder
//---------------------------------------------------------------------------

YDragModel::YDragModel( std::string           modelName, 
                              SimulationStateP    & sharedState,
                              ArchesLabel   * fieldLabels,
                              vector<std::string>   icLabelNames, 
                              vector<std::string>   scalarLabelNames,
                              int qn ) 
: ModelBase(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  // Create a label for this model
  d_modelLabel = VarLabel::create( modelName, CCVariable<double>::getTypeDescription() );

  // Create the gas phase source term associated with this model
  std::string gasSourceName = modelName + "_gasSource";
  d_gasLabel = VarLabel::create( gasSourceName, CCVariable<double>::getTypeDescription() );
}

YDragModel::~YDragModel()
{}



//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
YDragModel::problemSetup(const ProblemSpecP& params, int qn)
{
  pi = 3.141592653589793;

  ProblemSpecP db = params;

  string label_name;
  string role_name;
  string temp_label_name;
  string node;
  std::stringstream out;

  // check for gravity
  const ProblemSpecP params_root = db->getRootNode();
  if (params_root->findBlock("PhysicalConstants")) {
    ProblemSpecP db_phys = params_root->findBlock("PhysicalConstants");
    db_phys->require("gravity", gravity);
    db_phys->require("viscosity", kvisc);
  } else {
    throw InvalidValue("Missing <PhysicalConstants> section in input file!",__FILE__,__LINE__);
  }

  if (params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties")) {
    ProblemSpecP db_coal = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties");
    db_coal->require("particle_density", rhop);
  } else {
    throw InvalidValue("ERROR: YDragmodel: problemSetup(): Missing <Coal_Properties> section in input file!",__FILE__,__LINE__);
  }

  // Look for required internal coordinates
  ProblemSpecP db_icvars = params->findBlock("ICVars");
  for (ProblemSpecP variable = db_icvars->findBlock("variable"); variable != 0; variable = variable->findNextBlock("variable") ) {

    variable->getAttribute("label",label_name);
    variable->getAttribute("role", role_name);

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
    if (role_name == "particle_length") {
      LabelToRoleMap[temp_label_name] = role_name;
    } else if (role_name == "particle_yvel") {
      LabelToRoleMap[temp_label_name] = role_name;
    } else {
      std::string errmsg;
      errmsg = "Invalid variable role for YDrag model: must be \"particle_length\" or \"particle_yvel\", you specified \"" + role_name + "\".";
      throw InvalidValue(errmsg,__FILE__,__LINE__);
    }

    // set model clipping (not used yet...)
    db->getWithDefault( "low_clip", d_lowModelClip,   1.0e-6 );
    db->getWithDefault( "high_clip", d_highModelClip, 999999 );

  }

  for ( vector<std::string>::iterator iString = d_icLabels.begin();
        iString != d_icLabels.end(); ++iString) {
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

}

void
YDragModel::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "YDragModel::dummyInit";

  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &YDragModel::dummyInit);

  tsk->requires( Task::OldDW, d_modelLabel, gn, 0);
  tsk->requires( Task::OldDW, d_gasLabel,   gn, 0);

  tsk->computes(d_modelLabel);
  tsk->computes(d_gasLabel);

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

@see ExplicitSolver::noSolve()
 */
void
YDragModel::dummyInit( const ProcessorGroup* pc,
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

    CCVariable<double> ModelTerm;
    CCVariable<double> GasModelTerm;
    
    constCCVariable<double> oldModelTerm;
    constCCVariable<double> oldGasModelTerm;

    new_dw->allocateAndPut( ModelTerm,    d_modelLabel, matlIndex, patch );
    new_dw->allocateAndPut( GasModelTerm, d_gasLabel,   matlIndex, patch ); 

    old_dw->get( oldModelTerm,    d_modelLabel, matlIndex, patch, gn, 0 );
    old_dw->get( oldGasModelTerm, d_gasLabel,   matlIndex, patch, gn, 0 );
    
    ModelTerm.copyData(oldModelTerm);
    GasModelTerm.copyData(oldGasModelTerm);
  }
}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void 
YDragModel::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "YDragModel::initVars";
  Task* tsk = scinew Task(taskname, this, &YDragModel::initVars);

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
YDragModel::initVars( const ProcessorGroup * pc, 
                            const PatchSubset    * patches, 
                            const MaterialSubset * matls, 
                            DataWarehouse        * old_dw, 
                            DataWarehouse        * new_dw )
{
  // This method left intentionally blank...
  // It has the form:
  /*
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> something; 
    new_dw->allocateAndPut( something, d_something_label, matlIndex, patch ); 
    something.initialize(0.0)

  }
  */
}



//---------------------------------------------------------------------------
// Method: Schedule the calculation of the model 
//---------------------------------------------------------------------------
void 
YDragModel::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "YDragModel::computeModel";
  Task* tsk = scinew Task(taskname, this, &YDragModel::computeModel);

  Ghost::GhostType  gn  = Ghost::None;

  d_timeSubStep = timeSubStep; 

  if (d_timeSubStep == 0 ){ // && !d_labelSchedInit) {
    // Every model term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;

    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel);
  } else {
    tsk->modifies(d_modelLabel); 
    tsk->modifies(d_gasLabel); 
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
  tsk->requires( Task::OldDW, d_weight_label, gn, 0);

  tsk->requires( Task::OldDW, d_fieldLabels->d_CCVelocityLabel, gn, 0 );
  tsk->requires(Task::OldDW, d_fieldLabels->d_densityCPLabel, Ghost::None, 0);

  // require particle velocity
  ArchesLabel::PartVelMap::const_iterator i = d_fieldLabels->partVel.find(d_quadNode);
  tsk->requires( Task::NewDW, i->second, gn, 0 );

  for (vector<std::string>::iterator iter = d_icLabels.begin();
      iter != d_icLabels.end(); iter++) {

    map<string, string>::iterator iMap = LabelToRoleMap.find(*iter);

    if ( iMap != LabelToRoleMap.end() ) {
      if ( iMap->second == "particle_length" ) {
        if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_particle_length_label = current_eqn.getTransportEqnLabel();
          d_pl_scaling_factor = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_particle_length_label, gn, 0);
        } else {
          std::string errmsg = "ARCHES: YDragModel: Invalid variable given in <variable> tag for Drag model";
          errmsg += "\nCould not find given particle length variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory or in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      } else if ( iMap->second == "particle_yvel" ) {
        if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_yvel_scaling_factor = current_eqn.getScalingConstant();
        } else {
          std::string errmsg = "ARCHES: YDragModel: Invalid variable given in <variable> tag for Drag model";
          errmsg += "\nCould not find given particle x-velocity variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory or in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      } //else... we don't need that variable!


    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ARCHES: YDragModel: You specified that the variable \"" + *iter +
                           "\" was required, but you did not specify a role for it!\n";
      throw InvalidValue(errmsg,__FILE__,__LINE__);
    }
  }


  for (vector<std::string>::iterator iter = d_icLabels.begin(); 
       iter != d_icLabels.end(); iter++) { 
    // require any internal coordinates needed to compute the model
  }

  for( vector<string>::iterator iter = d_scalarLabels.begin();
       iter != d_scalarLabels.end(); ++iter ) {
    // require any scalars needed to compute the model
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
YDragModel::computeModel( const ProcessorGroup* pc, 
                   const PatchSubset* patches, 
                   const MaterialSubset* matls, 
                   DataWarehouse* old_dw, 
                   DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    //Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> model; 
    CCVariable<double> gas_source;

    if (new_dw->exists( d_modelLabel, matlIndex, patch )){
      new_dw->getModifiable( model, d_modelLabel, matlIndex, patch );
    } else {
      new_dw->allocateAndPut( model, d_modelLabel, matlIndex, patch );
      model.initialize(0.0);
    }

    if (new_dw->exists( d_gasLabel, matlIndex, patch )){
      new_dw->getModifiable( gas_source, d_gasLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( gas_source, d_gasLabel, matlIndex, patch );
      gas_source.initialize(0.0);
    }

    constCCVariable<Vector> gasVel;
    old_dw->get( gasVel, d_fieldLabels->d_CCVelocityLabel, matlIndex, patch, gn, 0 );

    constCCVariable<double> den;
    old_dw->get(den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0 );

    constCCVariable<Vector> partVel;
    ArchesLabel::PartVelMap::const_iterator iter = d_fieldLabels->partVel.find(d_quadNode);
    new_dw->get(partVel, iter->second, matlIndex, patch, gn, 0);

    constCCVariable<double> weight;
    old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );

    constCCVariable<double> w_particle_length;
    old_dw->get( w_particle_length, d_particle_length_label, matlIndex, patch, gn, 0 );

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;

      if( weight[c] < d_w_small && !d_unweighted) {
        model[c] = 0.0;
        gas_source[c] = 0.0;
      } else {
      
        double length;
        if(d_unweighted) {
          length = w_particle_length[c]*d_pl_scaling_factor;
        } else {
          length = w_particle_length[c]/weight[c]*d_pl_scaling_factor;
        }

        Vector sphGas = Vector(0.,0.,0.);
        Vector cartGas = gasVel[c];

        Vector sphPart = Vector(0.,0.,0.);
        Vector cartPart = partVel[c];

        sphGas = cart2sph( cartGas );
        sphPart = cart2sph( cartPart );

        double diff = sphGas.z() - sphPart.z();
        double Re  = std::abs(diff)*length / (kvisc/den[c]);
        double phi;

        if(Re < 1.0) {
          phi = 1.0;
        } else if(Re>1000.0) {
          phi = 0.0183*Re;
        } else {
          phi = 1. + .15*pow(Re, 0.687);
        }

        double t_p = rhop/(18.0*kvisc)*pow(length,2.0);

        if(d_unweighted){        
          model[c] = (phi/t_p*(cartGas.y()-cartPart.y())+gravity.y())/(d_yvel_scaling_factor);
        } else {
          model[c] = weight[c]*(phi/t_p*(cartGas.y()-cartPart.y())+gravity.y())/(d_yvel_scaling_factor);
        }

        gas_source[c] = -weight[c]*d_w_scaling_factor*rhop/6.0*pi*phi/t_p*(cartGas.y()-cartPart.y())*pow(length,3.0);

        /*
        // Debugging
        cout << "quad_node " << d_quad_node << endl;
        cout << "drag source " << drag_part[c] << endl;
        if (cartPart.y() > 1.0) {
          cout << "quad_node " << d_quad_node  << " cartgasy " << cartGas.y() << " " << "catrparty " << cartPart.y() << endl;
          cout << "length " << length << " Re " << Re <<  endl;
          cout << "w_scaling " << d_w_scaling_factor << endl;
          cout << "phi " << phi << endl;
          cout << "t_p " << t_p << endl;
          cout << "pi " << pi << endl;
          cout << "diff " << diff << endl;
        }
        */
       }
    }
  }
}
