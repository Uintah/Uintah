#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/CoalModels/DragModel.h>
#include <CCA/Components/Arches/CoalModels/PartVel.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <Core/Exceptions/InvalidValue.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

//---------------------------------------------------------------------------
// Builder:
DragModelBuilder::DragModelBuilder( const std::string         & modelName, 
                                    const vector<std::string> & reqICLabelNames,
                                    const vector<std::string> & reqScalarLabelNames,
                                    ArchesLabel         * fieldLabels,
                                    SimulationStateP          & sharedState,
                                    int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{}

DragModelBuilder::~DragModelBuilder(){}

ModelBase* DragModelBuilder::build(){
  return scinew DragModel( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

DragModel::DragModel( std::string modelName, 
                      SimulationStateP& sharedState,
                      ArchesLabel* fieldLabels,
                      vector<std::string> icLabelNames, 
                      vector<std::string> scalarLabelNames,
                      int qn ) 
: ModelBase(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  pi = 3.141592653589793;
  d_quadNode = qn;
  
  // Create a label for this model
  d_modelLabel = VarLabel::create( modelName, CCVariable<Vector>::getTypeDescription() );

  // Create the gas phase source term associated with this model
  std::string gasSourceName = modelName + "_gasSource";
  d_gasLabel = VarLabel::create( gasSourceName, CCVariable<Vector>::getTypeDescription() );
}

DragModel::~DragModel()
{}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
DragModel::problemSetup(const ProblemSpecP& params, int qn)
{

  ProblemSpecP db = params; 

  string label_name;
  string role_name;
  string temp_label_name;
  string node;
  std::stringstream out;

  // Look for required internal coordinates
  ProblemSpecP db_icvars = params->findBlock("ICVars");
  for (ProblemSpecP variable = db_icvars->findBlock("variable"); variable != 0; variable = variable->findNextBlock("variable") ) {

    variable->getAttribute("label",label_name);
    variable->getAttribute("role", role_name);

    temp_label_name = label_name;
    
    out << qn;
    node = out.str();
    temp_label_name += "_qn";
    temp_label_name += node;

    // user specifies "role" of each internal coordinate
    // if it isn't an internal coordinate or a scalar, it's required explicitly
    // ( see comments in Arches::registerModels() for details )
    if (role_name == "particle_length") {
      LabelToRoleMap[temp_label_name] = role_name;
    } else {
      std::string errmsg;
      errmsg = "Invalid variable role for Drag model: must be \"particle_length\", you specified \"" + role_name + "\".";
      throw InvalidValue(errmsg,__FILE__,__LINE__);
    }

    // set model clipping (not used yet...)
    db->getWithDefault( "low_clip", d_lowModelClip,   1.0e-6 );
    db->getWithDefault( "high_clip", d_highModelClip, 999999 );  
 
  }

  // Look for required scalars
  //   ( Drag model doesn't use any extra scalars (yet)
  //     but if it did, this "for" loop would have to be un-commented )
  /*
  ProblemSpecP db_scalarvars = params->findBlock("scalarVars");
  for( ProblemSpecP variable = db_scalarvars->findBlock("variable");
       variable != 0; variable = variable->findNextBlock("variable") ) {

    variable->getAttribute("label", label_name);
    variable->getAttribute("role",  role_name);

    temp_label_name = label_name;
    out << qn;
    node = out.str();
    temp_label_name += "_qn";
    temp_label_name += node;

    // user specifies "role" of each scalar
    // if it isn't an internal coordinate or a scalar, it's required explicitly
    // ( see comments in Arches::registerModels() for details )
    if ( role_name == "particle_length" ) {
      LabelToRoleMap[temp_label_name] = role_name;
    } else {
      std::string errmsg;
      errmsg = "Invalid variable role for Drag model: must be \"particle_length\", you specified \"" + role_name + "\".";
      throw InvalidValue(errmsg,__FILE__,__LINE__);
    }

  }
  */

  string temp_ic_name;
  string temp_ic_name_full;

  // fix the d_icLabels to point to the correct quadrature node (since there is 1 model per quad node)
  for ( vector<std::string>::iterator iString = d_icLabels.begin(); 
        iString != d_icLabels.end(); ++iString) {
    temp_ic_name      = (*iString);
    temp_ic_name_full = temp_ic_name;

    out << qn;
    node = out.str();
    temp_ic_name_full += "_qn";
    temp_ic_name_full += node;

    std::replace( d_icLabels.begin(), d_icLabels.end(), temp_ic_name, temp_ic_name_full);
  }

  // fix the d_scalarLabels to point to the correct quadrature node (since there is 1 model per quad node)
  // (Not needed for DragModel (yet)... If it is, uncomment the block below)
  /*
  for ( vector<std::string>::iterator iString = d_scalarLabels.begin(); 
        iString != d_scalarLabels.end(); ++iString) {
    temp_ic_name      = (*iString);
    temp_ic_name_full = temp_ic_name;

    out << qn;
    node = out.str();
    temp_ic_name_full += "_qn";
    temp_ic_name_full += node;

    std::replace( d_scalarLabels.begin(), d_scalarLabels.end(), temp_ic_name, temp_ic_name_full);
  }
  */

}

//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
DragModel::sched_dummyInit( const LevelP& level, SchedulerP& sched ) 
{
  string taskname = "DragModel::dummyInit"; 

  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &DragModel::dummyInit);

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
DragModel::dummyInit( const ProcessorGroup* pc,
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
    CCVariable<double> gasSource;
    
    constCCVariable<double> oldModel;
    constCCVariable<double> oldGasSource;

    new_dw->allocateAndPut( model,       d_modelLabel, matlIndex, patch );
    new_dw->allocateAndPut( gasSource,   d_gasLabel,   matlIndex, patch ); 

    old_dw->get( oldModel,       d_modelLabel, matlIndex, patch, gn, 0 );
    old_dw->get( oldGasSource,   d_gasLabel,   matlIndex, patch, gn, 0 );

    model.copyData(oldModel);
    gasSource.copyData(oldGasSource);

  }
}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of some variables 
//---------------------------------------------------------------------------
void 
DragModel::sched_initVars( const LevelP& level, SchedulerP& sched )
{

  std::string taskname = "DragModel::initVars";
  Task* tsk = scinew Task(taskname, this, &DragModel::initVars);

  // d_modelLabel and d_gasLabel are "required" in the ModelBase class...

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}

//-------------------------------------------------------------------------
// Method: Initialize variables
//-------------------------------------------------------------------------
void
DragModel::initVars( const ProcessorGroup * pc, 
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

    CCVariable<double> model;
    CCVariable<double> gasSource;
    
    constCCVariable<double> oldModel;
    constCCVariable<double> oldGasSource;

    new_dw->allocateAndPut( model,     d_modelLabel, matlIndex, patch );
    new_dw->allocateAndPut( gasSource, d_gasLabel,   matlIndex, patch ); 

    old_dw->get( oldModel,     d_modelLabel, matlIndex, patch, gn, 0 );
    old_dw->get( oldGasSource, d_gasLabel,   matlIndex, patch, gn, 0 );
    
    model.copyData(oldModel);
    gasSource.copyData(oldGasSource);

  }
}


//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model 
//---------------------------------------------------------------------------
void 
DragModel::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "DragModel::computeModel";
  Task* tsk = scinew Task(taskname, this, &DragModel::computeModel);

  Ghost::GhostType  gn  = Ghost::None;

  d_timeSubStep = timeSubStep; 

  if (d_timeSubStep == 0 && !d_labelSchedInit) {
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
  tsk->requires( Task::OldDW, d_weight_label, Ghost::None, 0);

  // require gas velocity
  tsk->requires( Task::OldDW, d_fieldLabels->d_CCVelocityLabel, gn, 0 );

  // require particle velocity
  ArchesLabel::PartVelMap::const_iterator i = d_fieldLabels->partVel.find(d_quadNode);
  tsk->requires( Task::OldDW, i->second, gn, 0 );

  // For each required variable, determine what role it plays
  // - "gas_velocity" - require the "NewCCVelocity" label (that's already done)
  // - "particle_velocity" - require the "partVel" label from PartVel map (that's alrady done)
  //                         (this could potentially be an internal coordinate... 
  //                          so look for it in internal coordinate map if it's in the
  //                          <ICVar> block!!!)
  // - "particle_length" - look in DQMOMEqnFactory

  // for each required internal coordinate:
  for (vector<std::string>::iterator iter = d_icLabels.begin(); 
      iter != d_icLabels.end(); iter++) { 

    map<string, string>::iterator iMap = LabelToRoleMap.find(*iter);

    if ( iMap != LabelToRoleMap.end() ) {
      if ( iMap->second == "gas_velocity") {
        // automatically using Arches' velocity label, so do nothing
      } else if ( iMap->second == "particle_velocity" ) {
        // particle_velocity is specified as an internal coordinate (found in <ICVar> block)
        // so look for it in the DQMOMEqn map
        if( dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_particle_velocity_label = current_eqn.getTransportEqnLabel();
          d_pv_scaling_factor = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_particle_length_label, gn, 0);
        } else {
          std::string errmsg = "ARCHES: DragModel: Invalid variable given in <variable> tag for Drag model";
          errmsg += "\nCould not find given particle velocity variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory or in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }

      } else if ( iMap->second == "particle_length" ) {
        if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_particle_length_label = current_eqn.getTransportEqnLabel();
          d_pl_scaling_factor = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_particle_length_label, gn, 0);
        } else {
          std::string errmsg = "ARCHES: DragModel: Invalid variable given in <variable> tag for Drag model";
          errmsg += "\nCould not find given particle length variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory or in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      } //else... we don't need that variable!
           
     
    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ARCHES: DragModel: You specified that the variable \"" + *iter + 
                           "\" was required, but you did not specify a role for it!\n";
      throw InvalidValue(errmsg,__FILE__,__LINE__);
    }
  }


  // for each required scalar variable:
  //  (but no scalar equation variables should be required for the Drag model, at least not for now...)
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
          std::string errmsg = "ARCHES: DragModel: Invalid variable given in <scalarVars> block for <variable> tag for DragModel model.";
          errmsg += "\nCould not find given <insert role name here> variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      }
    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ARCHES: DragModel: You specified that the variable \"" + *iter + 
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
DragModel::computeModel( const ProcessorGroup* pc, 
                         const PatchSubset* patches, 
                         const MaterialSubset* matls, 
                         DataWarehouse* old_dw, 
                         DataWarehouse* new_dw )
{
  for (int p=0; p < patches->size(); p++){

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    //Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    CCVariable<Vector> drag_part;
    if (new_dw->exists( d_modelLabel, matlIndex, patch )){
      new_dw->getModifiable( drag_part, d_modelLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( drag_part, d_modelLabel, matlIndex, patch );
      drag_part.initialize(Vector(0.,0.,0.));
    }

    CCVariable<Vector> drag_gas; 
    if (new_dw->exists( d_gasLabel, matlIndex, patch )){
      new_dw->getModifiable( drag_gas, d_gasLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( drag_gas, d_gasLabel, matlIndex, patch );
      drag_gas.initialize(Vector(0.,0.,0.));
    }

    constCCVariable<Vector> gasVel; 
    old_dw->get( gasVel, d_fieldLabels->d_CCVelocityLabel, matlIndex, patch, gn, 0 );

    constCCVariable<Vector> partVel; 
    ArchesLabel::PartVelMap::const_iterator iter = d_fieldLabels->partVel.find(d_quadNode);
    old_dw->get(partVel, iter->second, matlIndex, patch, gn, 0);

    constCCVariable<double> w_particle_length; 
    old_dw->get( w_particle_length, d_particle_length_label, matlIndex, patch, gn, 0 );

    constCCVariable<double> weight;
    old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );
    

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 

      if( weight[c] < d_w_small ) {
        drag_part[c] = Vector(0.,0.,0.);
        drag_gas[c] = Vector(0.,0.,0.);
      } else {
        double length = w_particle_length[c]/weight[c]*d_pl_scaling_factor;   
        Vector sphGas = Vector(0.,0.,0.);
        Vector cartGas = gasVel[c]; 
        
        Vector sphPart = Vector(0.,0.,0.);
        Vector cartPart = partVel[c]; 

        sphGas = PartVel::cart2sph( cartGas ); 
        sphPart = PartVel::cart2sph( cartPart ); 
        double kvisc = 2.0e-5; 
        double rhop = 1000.0;
        double diff = sphGas.z() - sphPart.z(); 
        double Re  = abs(diff)*length / kvisc;
        double phi;
        if(Re < 1) {
          phi = 1;
        } else {
          phi = 1. + .15*pow(Re, 0.687);
        }
        double t_p = rhop/(18*kvisc)*pow(length,2); 
        double part_src_mag = phi/t_p*diff;
        sphPart = Vector(sphPart.x(), sphPart.y(), part_src_mag);
        drag_part[c] = PartVel::sph2cart(sphPart);
        double gas_src_mag = -weight[c]*d_w_scaling_factor*rhop*4/3*pi*phi/t_p*diff*pow(length,3);
        sphGas = Vector(sphGas.x(), sphGas.y(), gas_src_mag);
        drag_gas[c] = PartVel::sph2cart(sphGas);

        /*
        cout << "quad_node " << d_quadNode << endl;
        cout << "drag source " << drag_gas[c] << endl;
        cout << "partvel " << partVel[c] << endl;
        cout << "length " << length << endl;
        cout << "w_scaling " << d_w_scaling_factor << endl;
        cout << "phi " << phi << endl;
        cout << "t_p " << t_p << endl;
        cout << "pi " << pi << endl;
        cout << "diff " << diff << endl;
        */
       }  
    }
  }
}
