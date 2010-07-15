#include <CCA/Components/Arches/CoalModels/DragModel.h>
#include <CCA/Components/Arches/CoalModels/ParticleVelocity.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Directives.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>

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
DragModelBuilder::DragModelBuilder( const std::string         & modelName, 
                                    const vector<std::string> & reqICLabelNames,
                                    const vector<std::string> & reqScalarLabelNames,
                                    const ArchesLabel         * fieldLabels,
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
                      const ArchesLabel* fieldLabels,
                      vector<std::string> icLabelNames, 
                      vector<std::string> scalarLabelNames,
                      int qn ) 
: ParticleVelocity(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  d_uvel_model_label = VarLabel::create( modelName + "_uvel", CCVariable<double>::getTypeDescription() );
  d_vvel_model_label = VarLabel::create( modelName + "_vvel", CCVariable<double>::getTypeDescription() );
  d_wvel_model_label = VarLabel::create( modelName + "_wvel", CCVariable<double>::getTypeDescription() );

  // particle velocity label is created in parent class
  pi = 3.141592653589793;
}

DragModel::~DragModel()
{}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
DragModel::problemSetup(const ProblemSpecP& params)
{
  // call parent's method first
  ParticleVelocity::problemSetup(params);

  ProblemSpecP db = params; 

  string label_name;
  string role_name;
  string temp_label_name;

  string temp_ic_name;
  string temp_ic_name_full;
  
  std::stringstream out;
  out << d_quadNode; 
  string node = out.str();

  // Look for required internal coordinates
  ProblemSpecP db_icvars = params->findBlock("ICVars");
  if(db_icvars) {
    for (ProblemSpecP variable = db_icvars->findBlock("variable"); variable != 0; variable = variable->findNextBlock("variable") ) {

      variable->getAttribute("label",label_name);
      variable->getAttribute("role", role_name);

      temp_label_name = label_name;
      
      temp_label_name += "_qn";
      temp_label_name += node;

      // user specifies "role" of each internal coordinate
      if (role_name == "particle_length") {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useLength = true;
      } else if ( role_name == "u_velocity" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useUVelocity = true;
      } else if ( role_name == "v_velocity" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useVVelocity = true;
      } else if ( role_name == "w_velocity" ) {
        LabelToRoleMap[temp_label_name] = role_name;
        d_useWVelocity = true;
      } else {
        std::string errmsg;
        errmsg = "ERROR: Arches: DragModel: Invalid variable role:";
        errmsg += "must be \"length\", \"u_velocity\", \"v_velocity\", or \"w_velocity\", you specified \"" + role_name + "\".";
        throw ProblemSetupException(errmsg,__FILE__,__LINE__);
      }
    }
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
  // (Not used by DragModel)
  /*
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
  */


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
      string errmsg = "ERROR: Arches: DragModel: Invalid variable \"" + iter->first + 
                      "\" given for \""+iter->second+"\" role, could not find in EqnFactory or DQMOMEqnFactory!";
      throw ProblemSetupException(errmsg,__FILE__,__LINE__);
    }

    if( iter->second == "particle_length" ) {
      d_length_label = current_eqn->getTransportEqnLabel();
      d_length_scaling_factor = current_eqn->getScalingConstant();

    } else if( iter->second == "u_velocity" ) {
      d_uvel_label = current_eqn->getTransportEqnLabel();
      d_uvel_scaling_factor = current_eqn->getScalingConstant();

      DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(current_eqn);
      dqmom_eqn->addModel(d_uvel_model_label);

    } else if( iter->second == "v_velocity" ) {
      d_vvel_label = current_eqn->getTransportEqnLabel();
      d_vvel_scaling_factor = current_eqn->getScalingConstant();

      DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(current_eqn);
      dqmom_eqn->addModel(d_vvel_model_label);

    } else if( iter->second == "w_velocity" ) {
      d_wvel_label = current_eqn->getTransportEqnLabel();
      d_wvel_scaling_factor = current_eqn->getScalingConstant();

      DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(current_eqn);
      dqmom_eqn->addModel(d_wvel_model_label);

    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ERROR: Arches: DragModel: You specified that the variable \"" + iter->first + 
                           "\" was required, but you did not specify a valid role for it! (You specified \"" + iter->second + "\"\n";
      throw ProblemSetupException( errmsg, __FILE__, __LINE__);
    }

  }


  if(!d_useLength) {
    string errmsg = "ERROR: Arches: DragModel: No particle length variable was specified. Quitting...";
    throw ProblemSetupException(errmsg,__FILE__,__LINE__);
  }

  if(!d_useUVelocity || !d_useVVelocity || !d_useWVelocity ) {
    string errmsg = "ERROR: Arches: DragModel: Not all particle velocity variables were specified. Quitting...";
    throw ProblemSetupException(errmsg,__FILE__,__LINE__);
  }

}


//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model 
//---------------------------------------------------------------------------
void 
DragModel::sched_computeModel( const LevelP& level, 
                               SchedulerP& sched, 
                               int timeSubStep )
{
  // calculate model source term for dqmom velocity internal coodinate
  std::string taskname = "DragModel::computeModel";
  Task* tsk = scinew Task(taskname, this, &DragModel::computeModel, timeSubStep);

  // move this assignment to a private member
  CoalModelFactory& coalFactory = CoalModelFactory::self();
  const VarLabel* particle_density_label = coalFactory.getParticleDensityLabel(d_quadNode);

  Ghost::GhostType  gn  = Ghost::None;

  if( timeSubStep == 0 && !d_labelSchedInit) {
    // Every model term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;
  }

  if( timeSubStep == 0 ) {

    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel);

    tsk->computes(d_uvel_model_label);  
    tsk->computes(d_vvel_model_label); 
    tsk->computes(d_wvel_model_label); 
    
    tsk->requires( Task::OldDW, d_fieldLabels->d_densityCPLabel,   gn, 0);   // gas density
    tsk->requires( Task::OldDW, d_fieldLabels->d_newCCVelocityLabel, gn, 0 ); // gas velocity
    tsk->requires( Task::OldDW, d_velocity_label, gn, 0 ); // particle velocity
    // particle density is always calculated FIRST, so get it from the new DW
    tsk->requires( Task::NewDW, particle_density_label, gn, 0); // particle density 

    tsk->requires( Task::OldDW, d_weight_label, gn, 0);
    tsk->requires( Task::OldDW, d_length_label, gn, 0);
    //tsk->requires(Task::OldDW, d_uvel_label, gn, 0);  // <-- These are available via d_velocity_label
    //tsk->requires(Task::OldDW, d_vvel_label, gn, 0);  //     (that quantity is calculated from the internal coordinate values
    //tsk->requires(Task::OldDW, d_wvel_label, gn, 0);  //      in DragModel::computeParticleVelocity() )

  } else {

    tsk->modifies(d_modelLabel); 
    tsk->modifies(d_gasLabel);

    tsk->modifies(d_uvel_model_label);  
    tsk->modifies(d_vvel_model_label); 
    tsk->modifies(d_wvel_model_label); 

    tsk->requires( Task::NewDW, d_fieldLabels->d_densityTempLabel,   gn, 0);   // gas density
    tsk->requires( Task::NewDW, d_fieldLabels->d_newCCVelocityLabel, gn, 0); // gas velocity
    tsk->requires( Task::NewDW, d_velocity_label,       gn, 0); // particle velocity
    tsk->requires( Task::NewDW, particle_density_label, gn, 0); // particle density 

    tsk->requires( Task::NewDW, d_weight_label, gn, 0);
    tsk->requires( Task::NewDW, d_length_label, gn, 0);
    //tsk->requires(Task::NewDW, d_uvel_label, gn, 0);  // <-- These are available via d_velocity_label
    //tsk->requires(Task::NewDW, d_vvel_label, gn, 0);  //     (that quantity is calculated from the internal coordinate values
    //tsk->requires(Task::NewDW, d_wvel_label, gn, 0);  //      in DragModel::computeParticleVelocity() )

  }

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
                         DataWarehouse* new_dw,
                         int timeSubStep )
{
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    // move this assignment to a private member
    CoalModelFactory& coalFactory = CoalModelFactory::self();
    const VarLabel* particle_density_label = coalFactory.getParticleDensityLabel(d_quadNode);

    constCCVariable<double> gas_density;
    constCCVariable<double> particle_density;

    constCCVariable<Vector> gas_velocity;
    constCCVariable<Vector> particle_velocity;

    constCCVariable<double> w_particle_length; 
    constCCVariable<double> weight;

    CCVariable<Vector> drag_particle;
    CCVariable<Vector> drag_gas; 
    CCVariable<double> uvel_model;
    CCVariable<double> vvel_model;
    CCVariable<double> wvel_model;

    if( timeSubStep == 0 ) {

      old_dw->get( gas_density, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0);

      // particle density is always calculated FIRST, so get from new DW
      new_dw->get( particle_density, particle_density_label, matlIndex, patch, gn, 0);

      old_dw->get( gas_velocity, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0);
      old_dw->get( particle_velocity, d_velocity_label, matlIndex, patch, gn, 0);

      old_dw->get( w_particle_length, d_length_label, matlIndex, patch, gn, 0 );
      old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );

      new_dw->allocateAndPut( drag_particle, d_modelLabel, matlIndex, patch );
      drag_particle.initialize(Vector(0.,0.,0.));

      new_dw->allocateAndPut( drag_gas, d_gasLabel, matlIndex, patch );
      drag_gas.initialize(Vector(0.,0.,0.));

      new_dw->allocateAndPut( uvel_model, d_uvel_model_label, matlIndex, patch );
      uvel_model.initialize(0.0);

      new_dw->allocateAndPut( vvel_model, d_vvel_model_label, matlIndex, patch );
      vvel_model.initialize(0.0);

      new_dw->allocateAndPut( wvel_model, d_wvel_model_label, matlIndex, patch );
      wvel_model.initialize(0.0);

    } else { 

      new_dw->get( gas_density, d_fieldLabels->d_densityTempLabel, matlIndex, patch, gn, 0);
      new_dw->get( particle_density, particle_density_label, matlIndex, patch, gn, 0);

      new_dw->get( gas_velocity, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0);
      new_dw->get( particle_velocity, d_velocity_label, matlIndex, patch, gn, 0);

      new_dw->get( w_particle_length, d_length_label, matlIndex, patch, gn, 0 );
      new_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );

      new_dw->getModifiable( drag_particle, d_modelLabel, matlIndex, patch ); 
      new_dw->getModifiable( drag_gas, d_gasLabel, matlIndex, patch ); 

      new_dw->getModifiable( uvel_model, d_uvel_model_label, matlIndex, patch );
      new_dw->getModifiable( vvel_model, d_vvel_model_label, matlIndex, patch );
      new_dw->getModifiable( wvel_model, d_wvel_model_label, matlIndex, patch );

    }


    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 

      if( weight[c] < d_w_small ) {
        drag_particle[c] = Vector(0.,0.,0.);
        drag_gas[c] = Vector(0.,0.,0.);
      } else {
        double length = w_particle_length[c]/weight[c]*d_length_scaling_factor;   

        Vector gasVel = gas_velocity[c];
        Vector partVel = particle_velocity[c];

        double rhop = particle_density[c];

        double x_diff = gasVel.x() - partVel.x();
        double y_diff = gasVel.y() - partVel.y();
        double z_diff = gasVel.z() - partVel.z();
        double diff   = fabs(gasVel.length() - partVel.length());
        double Re     = (diff*length)/d_visc;

        double phi;
        if(Re < 1) {
          phi = 1;
        } else if(Re>1000) {
          phi = 0.0183*Re;
        } else {
          phi = 1. + .15*pow(Re, 0.687);
        }

        double t_p;

        if( length > TINY ) {

          t_p = (rhop/(18*d_visc))*pow(length,2); 

          uvel_model[c]    = (phi/t_p)*(x_diff + d_gravity.x());
          vvel_model[c]    = (phi/t_p)*(y_diff + d_gravity.y());
          wvel_model[c]    = (phi/t_p)*(z_diff + d_gravity.z());
          drag_particle[c] = Vector( uvel_model[c], vvel_model[c], wvel_model[c] );

          ////cmr
          //if( c == IntVector(1,2,3) && d_quadNode == 0 ) {
          //  cout << "tau_p = (rhop/(18*d_visc))*pow(length,2) = (" << rhop << "/(18*" << d_visc << "))*" << length*length << endl;
          //  cout << "X-Vel Drag = (phi/t_p)*(x_diff + d_gravity.x()) = (" << phi << "/" << t_p << ")*(" << x_diff << ")" << endl;
          //  cout << "Y-Vel Drag = (phi/t_p)*(y_diff + d_gravity.y()) = (" << phi << "/" << t_p << ")*(" << y_diff << ")" << endl;
          //  cout << "Z-Vel Drag = (phi/t_p)*(z_diff + d_gravity.z()) = (" << phi << "/" << t_p << ")*(" << z_diff << ")" << endl;
          //}
  
          double prefix = -(weight[c]*d_w_scaling_factor)*(rhop/6)*pi*(phi/t_p)*pow(length,3);
          double gasSrc_x = prefix*x_diff;
          double gasSrc_y = prefix*y_diff;
          double gasSrc_z = prefix*z_diff;
          drag_gas[c] = Vector( gasSrc_x, gasSrc_y, gasSrc_z );

        } else {

          uvel_model[c]    = 0.0;
          vvel_model[c]    = 0.0;
          wvel_model[c]    = 0.0;
          drag_particle[c] = 0.0;
          drag_gas[c]      = 0.0;

        }

      }//end if small weight  
    }// end cells
  }//end patches
}

void
DragModel::sched_computeParticleVelocity( const LevelP& level,
                                          SchedulerP&   sched,
                                          const int timeSubStep )
{
  string taskname = "DragModel::computeParticleVelocity";
  Task* tsk = scinew Task(taskname, this, &DragModel::computeParticleVelocity, timeSubStep );

  Ghost::GhostType gn = Ghost::None;

  // setting particle velocity (Vector)
  if( timeSubStep == 0 ) {

    tsk->computes( d_velocity_label );

    tsk->requires( Task::OldDW, d_fieldLabels->d_newCCVelocityLabel, gn, 0);

    tsk->requires( Task::OldDW, d_weight_label, gn, 0);
    tsk->requires( Task::OldDW, d_uvel_label, gn, 0);
    tsk->requires( Task::OldDW, d_vvel_label, gn, 0);
    tsk->requires( Task::OldDW, d_wvel_label, gn, 0);

  } else {

    tsk->modifies( d_velocity_label );

    tsk->requires( Task::NewDW, d_fieldLabels->d_newCCVelocityLabel, gn, 0);

    tsk->requires( Task::NewDW, d_weight_label, gn, 0);
    tsk->requires( Task::NewDW, d_uvel_label, gn, 0);
    tsk->requires( Task::NewDW, d_vvel_label, gn, 0);
    tsk->requires( Task::NewDW, d_wvel_label, gn, 0);

  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}

void 
DragModel::computeParticleVelocity( const ProcessorGroup* pc,
                                    const PatchSubset*    patches,
                                    const MaterialSubset* matls,
                                    DataWarehouse*        old_dw,
                                    DataWarehouse*        new_dw,
                                    int timeSubStep )
{
  for( int p=0; p<patches->size(); ++p ) {

    Ghost::GhostType gn = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    constCCVariable<Vector> gas_velocity;

    constCCVariable<double> weight;
    constCCVariable<double> wtd_particle_uvel;
    constCCVariable<double> wtd_particle_vvel;
    constCCVariable<double> wtd_particle_wvel;

    CCVariable<Vector> particle_velocity;

    if( timeSubStep == 0 ) {

      old_dw->get( gas_velocity, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0);

      old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );
      old_dw->get( wtd_particle_uvel, d_uvel_label, matlIndex, patch, gn, 0 );
      old_dw->get( wtd_particle_vvel, d_vvel_label, matlIndex, patch, gn, 0 );
      old_dw->get( wtd_particle_wvel, d_wvel_label, matlIndex, patch, gn, 0 );

      new_dw->allocateAndPut( particle_velocity, d_velocity_label, matlIndex, patch );
      particle_velocity.initialize( Vector(0.0, 0.0, 0.0) );

    } else {

      new_dw->get( gas_velocity, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0);

      new_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );
      new_dw->get( wtd_particle_uvel, d_uvel_label, matlIndex, patch, gn, 0 );
      new_dw->get( wtd_particle_vvel, d_vvel_label, matlIndex, patch, gn, 0 );
      new_dw->get( wtd_particle_wvel, d_wvel_label, matlIndex, patch, gn, 0 );

      new_dw->getModifiable( particle_velocity, d_velocity_label, matlIndex, patch );

    }



    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 

      double U = (wtd_particle_uvel[c]/weight[c])*d_uvel_scaling_factor;
      double V = (wtd_particle_vvel[c]/weight[c])*d_vvel_scaling_factor;
      double W = (wtd_particle_wvel[c]/weight[c])*d_wvel_scaling_factor;
      particle_velocity[c] = Vector(U,V,W);

    }

    //cmr
    // TODO - add this to the class description in the header file

    // NOTE: When tracking particle velocity as an internal coordinate,
    //       it is broken up into three separate scalars.
    //       HOWEVER, we still have to treat it as a Vector in the code,
    //       so when the boundary conditions for particle velocity are set,
    //       they are set as a Vector
    //
    //       Velocity boundary conditions and velocity internal coordinate
    //       boundary values are set via "vel_qn" vectors at each boundary,
    //       NOT through the DQMOM scalars!!!
    //
    // e.g., if the input file contains
    // 
    //      <BCType id = "0" label = "vel_qn0" var = "Dirichlet">
    //        <value>[5.0,1.0,3.0]</value>
    //      </BCType>
    //
    // Then "vel_qn0" will be set to Vector(5.0, 1.0, 3.0),
    // and "uvel_internal_coordinate" will be set to 5.0
    //     "vvel_internal_coordinate" will be set to 1.0
    //     "wvel_internal_coordinate" will be set to 3.0
    // (no boundary condition is explicitly set for the internal coordinates)
    //
    // This keeps from having to specify the boundary condition twice
    // (and potential mistakes from using two different values)

    // Now apply boundary conditions
    if (d_gasBC) {
      // assume particle velocity = gas velocity at boundary
      // DON'T DO THIS, IT'S WRONG!
      d_boundaryCond->setVectorValueBC( 0, patch, particle_velocity, gas_velocity, d_velocity_label->getName() );
    } else {
      // Particle velocity at boundary is set by user
      d_boundaryCond->setVectorValueBC( 0, patch, particle_velocity, d_velocity_label->getName() );
    }

  }//end patch loop
}

