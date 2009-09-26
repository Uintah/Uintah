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

                                            const vector<std::string> & icLabelNames, 
                                            const ArchesLabel         * fieldLabels,
                                            SimulationStateP          & sharedState,
                                            int qn ) :
  ModelBuilder( modelName, fieldLabels, icLabelNames, sharedState, qn )
{}

DragModelBuilder::~DragModelBuilder(){}

ModelBase*
DragModelBuilder::build(){
  return scinew DragModel( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, 
                              d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

DragModel::DragModel( std::string srcName, SimulationStateP& sharedState,
                            const ArchesLabel* fieldLabels,
                            vector<std::string> icLabelNames, int qn ) 
: ModelBase(srcName, sharedState, fieldLabels, icLabelNames, qn)
{
  d_quad_node = qn;
}

DragModel::~DragModel()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
DragModel::problemSetup(const ProblemSpecP& inputdb, int qn)
{

  ProblemSpecP db = inputdb; 

  ProblemSpecP db_icvars = db->findBlock("ICVars");
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
    if (role_name == "particle_length") {
      LabelToRoleMap[temp_label_name] = role_name;
    } else {
      std::string errmsg;
      errmsg = "Invalid variable role for Drag model: must be \"particle_length\", you specified \"" + role_name + "\".";
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

}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of some variables 
//---------------------------------------------------------------------------
void 
DragModel::sched_initVars( const LevelP& level, SchedulerP& sched )
{

  std::string taskname = "DragModel::initVars";
  Task* tsk = scinew Task(taskname, this, &DragModel::initVars);

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}
void
DragModel::initVars( const ProcessorGroup * pc, 
    const PatchSubset    * patches, 
    const MaterialSubset * matls, 
    DataWarehouse        * old_dw, 
    DataWarehouse        * new_dw )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop
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

  EqnFactory& eqn_factory = EqnFactory::self();
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
  d_w_scaling_factor = weight_eqn.getScalingConstant();
  tsk->requires(Task::OldDW, d_weight_label, Ghost::None, 0);
  tsk->requires( Task::OldDW, d_fieldLabels->d_newCCVelocityLabel, gn, 0 );

   ArchesLabel::PartVelMap::const_iterator i = d_fieldLabels->partVel.find(d_quad_node);
   tsk->requires( Task::OldDW, i->second, gn, 0 );

  // For each required variable, determine if it plays the role of temperature or mass fraction;
  //  if it plays the role of mass fraction, then look for it in equation factories
  for (vector<std::string>::iterator iter = d_icLabels.begin(); 
      iter != d_icLabels.end(); iter++) { 

    map<string, string>::iterator iMap = LabelToRoleMap.find(*iter);
    
    

    if ( iMap != LabelToRoleMap.end() ) {
      if ( iMap->second == "gas_velocity") {
        // automatically use Arches' velocity label if role="gas_velocity"
        //tsk->requires( Task::OldDW, d_fieldLabels->d_newCCVelocityLabel, gn, 0 );

        // Only require() variables found in equation factories (right now we're not tracking temperature this way)
      } 
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
          std::string errmsg = "ARCHES: DragModel: Invalid variable given in <variable> tag for Drag model";
          errmsg += "\nCould not find given particle length variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory or in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      } //else... we don't need that variable!!! 
           
     
    } else {
      // can't find it in the labels-to-roles map!
      std::string errmsg = "ARCHES: KobayashiSarofimDevol: Could not find role for given variable \"" + *iter + "\".";
      throw InvalidValue(errmsg,__FILE__,__LINE__);
    }
  }


  for (vector<std::string>::iterator iter = d_icLabels.begin(); 
       iter != d_icLabels.end(); iter++) { 
    // HERE I WOULD REQUIRE ANY VARIABLES NEEDED TO COMPUTE THE MODEL
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
                   DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    //Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    
    int archIndex = 0;
    double pi = acos(-1);
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    double time = d_sharedState->getElapsedTime();

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
    old_dw->get( gasVel, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0 );

    ArchesLabel::PartVelMap::const_iterator iter = d_fieldLabels->partVel.find(d_quad_node);
    constCCVariable<Vector> partVel; 
    old_dw->get(partVel, iter->second, matlIndex, patch, gn, 0);

    constCCVariable<double> w_particle_length; 
    old_dw->get( w_particle_length, d_particle_length_label, matlIndex, patch, gn, 0 );
    constCCVariable<double> weight;
    old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );
    

    for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
      IntVector c = *iter; 

      if (weight[c] < 1e-4) {
        drag_part[c] = Vector(0.,0.,0.);
        drag_gas[c] = Vector(0.,0.,0.);
      } else {
       // d_pl_scaling_factor = current_eqn.getScalingConstant();
        double length = w_particle_length[c]/weight[c]*d_pl_scaling_factor;   
        Vector sphGas = Vector(0.,0.,0.);
        Vector cartGas = gasVel[c]; 
        
        Vector sphPart = Vector(0.,0.,0.);
        Vector cartPart = partVel[c]; 

        sphGas = cart2sph( cartGas ); 
        sphPart = cart2sph( cartPart ); 
        double kvisc = 2.0e-5; 
        double rhop = 3000.0;
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
        drag_part[c] = sph2cart(sphPart);
        double gas_src_mag = -weight[c]*d_w_scaling_factor*rhop*4/3*pi*phi/t_p*diff*pow(length,3);
        sphGas = Vector(sphGas.x(), sphGas.y(), gas_src_mag);
        drag_gas[c] = sph2cart(sphGas);

      /*cout << "quad_node " << d_quad_node << endl;
      cout << "drag source " << drag_gas[c] << endl;
      cout << "partvel " << partVel[c] << endl;
      cout << "length " << length << endl;
      cout << "w_scaling " << d_w_scaling_factor << endl;
      cout << "phi " << phi << endl;
      cout << "t_p " << t_p << endl;
      cout << "pi " << pi << endl;
      cout << "diff " << diff << endl;*/
       }  
    }
  }
}
