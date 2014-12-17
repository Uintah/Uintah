#include <CCA/Components/Arches/CoalModels/XDragModel.h>
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

XDragModelBuilder::XDragModelBuilder( const std::string         & modelName, 
                                            const vector<std::string> & reqICLabelNames,
                                            const vector<std::string> & reqScalarLabelNames,
                                            ArchesLabel         * fieldLabels,
                                            SimulationStateP          & sharedState,
                                            int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{}

XDragModelBuilder::~XDragModelBuilder(){}

ModelBase* XDragModelBuilder::build(){
  return scinew XDragModel( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}

// End Builder
//---------------------------------------------------------------------------

XDragModel::XDragModel( std::string           modelName, 
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

XDragModel::~XDragModel()
{}



//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
XDragModel::problemSetup(const ProblemSpecP& params, int qn)
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
    db_coal->require("particle_sizes", particle_sizes); // read the particle sizes [m]
    db_coal->require("as_received", as_received);
    total_rc=as_received[0]+as_received[1]+as_received[2]+as_received[3]+as_received[4]; // (C+H+O+N+S) dry ash free total
    total_dry=as_received[0]+as_received[1]+as_received[2]+as_received[3]+as_received[4]+as_received[5]+as_received[6]; // (C+H+O+N+S+char+ash)  moisture free total
    rc_mass_frac=total_rc/total_dry; // mass frac of rc (dry) 
    char_mass_frac=as_received[5]/total_dry; // mass frac of char (dry)
    ash_mass_frac=as_received[6]/total_dry; // mass frac of ash (dry)
    int p_size=particle_sizes.size();
    for (int n=0; n<p_size; n=n+1)
      {
        vol_dry.push_back((pi/6)*pow(particle_sizes[n],3)); // m^3/particle
        mass_dry.push_back(vol_dry[n]*rhop); // kg/particle
        ash_mass_init.push_back(mass_dry[n]*ash_mass_frac); // kg_ash/particle (initial)  
        char_mass_init.push_back(mass_dry[n]*char_mass_frac); // kg_char/particle (initial)
        rc_mass_init.push_back(mass_dry[n]*rc_mass_frac); // kg_ash/particle (initial)
      }
  } else {
    throw InvalidValue("ERROR: XDragmodel: problemSetup(): Missing <Coal_Properties> section in input file!",__FILE__,__LINE__);
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
    } else if (role_name == "particle_xvel") {
      LabelToRoleMap[temp_label_name] = role_name;
    } else if (role_name == "raw_coal_mass") {
      LabelToRoleMap[temp_label_name] = role_name;
    } else if (role_name == "char_mass") {
      LabelToRoleMap[temp_label_name] = role_name;
    } else {
      std::string errmsg;
      errmsg = "Invalid variable role for XDrag model: must be \"particle_length\" or \"particle_xvel\" or \"raw_coal_mass\" or \"char_mass\", you specified \"" + role_name + "\".";
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

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void 
XDragModel::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "XDragModel::initVars";
  Task* tsk = scinew Task(taskname, this, &XDragModel::initVars);

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
XDragModel::initVars( const ProcessorGroup * pc, 
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
XDragModel::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "XDragModel::computeModel";
  Task* tsk = scinew Task(taskname, this, &XDragModel::computeModel);

  Ghost::GhostType  gn  = Ghost::None;

  d_timeSubStep = timeSubStep; 

  if (d_timeSubStep == 0 ) { //&& !d_labelSchedInit) {
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
          std::string errmsg = "ARCHES: XDragModel: Invalid variable given in <variable> tag for Drag model";
          errmsg += "\nCould not find given particle length variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory or in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      } else if ( iMap->second == "raw_coal_mass" ) {
        if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_raw_coal_mass_label = current_eqn.getTransportEqnLabel();
          d_rcmass_scaling_factor = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_raw_coal_mass_label, gn, 0);
        } else {
          std::string errmsg = "ARCHES: XDragModel: Invalid variable given in <variable> tag for Drag model";
          errmsg += "\nCould not find given raw coal mass variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory or in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      } else if ( iMap->second == "char_mass" ) {
        if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_char_mass_label = current_eqn.getTransportEqnLabel();
          d_charmass_scaling_factor = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_char_mass_label, gn, 0);
        } else {
          std::string errmsg = "ARCHES: XDragModel: Invalid variable given in <variable> tag for Drag model";
          errmsg += "\nCould not find given raw coal mass variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory or in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      } else if ( iMap->second == "particle_xvel" ) {
        if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_xvel_scaling_factor = current_eqn.getScalingConstant();
        } else {
          std::string errmsg = "ARCHES: XDragModel: Invalid variable given in <variable> tag for Drag model";
          errmsg += "\nCould not find given particle x-velocity variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory or in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      } //else... we don't need that variable!


    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ARCHES: XDragModel: You specified that the variable \"" + *iter +
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
XDragModel::computeModel( const ProcessorGroup* pc, 
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

    constCCVariable<double> w_raw_coal_mass;
    old_dw->get( w_raw_coal_mass, d_raw_coal_mass_label, matlIndex, patch, gn, 0 );

    constCCVariable<double> w_char_mass;
    old_dw->get( w_char_mass, d_char_mass_label, matlIndex, patch, gn, 0 );

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
        double rc_mass;
        if(d_unweighted) {
          rc_mass = w_raw_coal_mass[c]*d_rcmass_scaling_factor;
        } else {
          rc_mass = w_raw_coal_mass[c]/weight[c]*d_rcmass_scaling_factor;
        }
        double char_mass;
        if(d_unweighted) {
          char_mass = w_char_mass[c]*d_charmass_scaling_factor;
        } else {
          char_mass = w_char_mass[c]/weight[c]*d_charmass_scaling_factor;
        }

        // KLUDGE: implicit clipping
        //length = max(min(length,1e-3),1e-6);

        Vector cartGas = gasVel[c];
        Vector cartPart = partVel[c];
        double gasVelMag = velMag( cartGas );
        double partVelMag = velMag( cartPart );
 
        double diff = gasVelMag - partVelMag;
        double Re  = std::abs(diff)*length/(kvisc/den[c]);  
        double phi;
        
        if(Re < 994.0) {
          phi = 1.0 + 0.15*pow(Re, 0.687);
        } else {
          phi = 0.0183*Re;
        }

        double rho_factor = (char_mass+rc_mass+ash_mass_init[d_quadNode])/(rc_mass_init[d_quadNode]+ash_mass_init[d_quadNode]);

        if(!(rho_factor>=0 && rho_factor<=1.0)){
          if(!(rho_factor>=0 && rho_factor<=1.2))
          //  cout <<"X_rho_factor =" <<rho_factor <<", c"<<c<<"charmass="<<w_char_mass[c]<<" ,"<<char_mass<<"rcmass="<<w_raw_coal_mass[c]<<" ,"<<rc_mass<<" ,"<<rc_mass_init[d_quadNode]<<" weights="<<weight[c]<<", lenght="<<length<<endl;
          rho_factor =1;
        }

        double t_p = rhop*rho_factor/(18.0*kvisc)*pow(length,2.0);

        if(d_unweighted){
          model[c] = (phi/t_p*(cartGas.x()-cartPart.x())+gravity.x())/(d_xvel_scaling_factor);
        } else {
          model[c]= weight[c]*(phi/t_p*(cartGas.x()-cartPart.x())+gravity.x())/d_xvel_scaling_factor;
        }

        gas_source[c] = -weight[c]*d_w_scaling_factor*rhop*rho_factor/6.0*pi*phi/t_p*(cartGas.x()-cartPart.x())*pow(length,3.0);

        /*
        // Debugging
        cout << "quad_node " << d_quad_node << endl;
        cout << "drag source " << drag_part[c] << endl;
        if (cartPart.x() > 1.0) {
          cout << "quad_node " << d_quad_node  << " cartgasx " << cartGas.x() << " " << "catrpartx " << cartPart.x() << endl;
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
