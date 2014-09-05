#include <CCA/Components/Arches/CoalModels/BadHawkDevol.h>
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
BadHawkDevolBuilder::BadHawkDevolBuilder( const std::string         & modelName,
                                                  const vector<std::string> & reqICLabelNames,
                                                  const vector<std::string> & reqScalarLabelNames,
                                                  ArchesLabel         * fieldLabels,
                                                  SimulationStateP          & sharedState,
                                                  int qn ) :
  ModelBuilder( modelName, fieldLabels, reqICLabelNames, reqScalarLabelNames, sharedState, qn )
{
}

BadHawkDevolBuilder::~BadHawkDevolBuilder(){}

ModelBase* BadHawkDevolBuilder::build() {
  return scinew BadHawkDevol( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

BadHawkDevol::BadHawkDevol( std::string modelName, 
                                    SimulationStateP& sharedState,
                                    ArchesLabel* fieldLabels,
                                    vector<std::string> icLabelNames, 
                                    vector<std::string> scalarLabelNames,
                                    int qn ) 
: ModelBase(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn), 
  d_fieldLabels(fieldLabels)
{
  d_quad_node = qn;
  
  compute_part_temp = false;

  // Create a label for this model
  d_modelLabel = VarLabel::create( modelName, CCVariable<double>::getTypeDescription() );

  // Create the gas phase source term associated with this model
  std::string gasSourceName = modelName + "_gasSource";
  d_gasLabel = VarLabel::create( gasSourceName, CCVariable<double>::getTypeDescription() );
}

BadHawkDevol::~BadHawkDevol()
{}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
  void 
BadHawkDevol::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params; 
  compute_part_temp = false;

  // Look for required internal coordinates
  ProblemSpecP db_icvars = params->findBlock("ICVars");
  for (ProblemSpecP variable = db_icvars->findBlock("variable"); variable != 0; variable = variable->findNextBlock("variable") ) {

    string label_name;
    string role_name;
    string temp_label_name;

    variable->getAttribute("label",label_name);
    variable->getAttribute("role", role_name);

    temp_label_name = label_name;
    
    string node;
    std::stringstream out;
    out << d_quadNode;
    node = out.str();
    temp_label_name += "_qn";
    temp_label_name += node;

    // user specifies "role" of each internal coordinate
    // if it isn't an internal coordinate or a scalar, it's required explicitly
    // ( see comments in Arches::registerModels() for details )
    if ( role_name == "raw_coal_mass" ) {
      LabelToRoleMap[temp_label_name] = role_name;
    } else if( role_name == "particle_temperature" ) {  
      LabelToRoleMap[temp_label_name] = role_name;
      compute_part_temp = true;
    } else {
      std::string errmsg;
      errmsg = "Invalid variable role for Badzioch Hawksley Devolatilization model: must be \"particle_temperature\" or \"raw_coal_mass\", you specified \"" + role_name + "\".";
      throw InvalidValue(errmsg,__FILE__,__LINE__);
    }

    // set model clipping (not used yet...)
    db->getWithDefault( "low_clip",  d_lowModelClip,  1.0e-6 );
    db->getWithDefault( "high_clip", d_highModelClip, 999999 );

  }

  // Look for required scalars
  //   ( Badzioch Hawksley model doesn't use any extra scalars (yet)
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
    out << d_quadNode;
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
      errmsg = "Invalid variable role for Badzioch Hawksley Devolatilization model: must be \"particle_temperature\" or \"raw_coal_mass\", you specified \"" + role_name + "\".";
      throw InvalidValue(errmsg,__FILE__,__LINE__);
    }

  }
  */


  // fix the d_icLabels to point to the correct quadrature node (since there is 1 model per quad node)
  for ( vector<std::string>::iterator iString = d_icLabels.begin(); 
        iString != d_icLabels.end(); ++iString) {
    std::string temp_ic_name        = (*iString);
    std::string temp_ic_name_full   = temp_ic_name;

    std::string node;
    std::stringstream out;
    out << d_quadNode;
    node = out.str();
    temp_ic_name_full += "_qn";
    temp_ic_name_full += node;

    std::replace( d_icLabels.begin(), d_icLabels.end(), temp_ic_name, temp_ic_name_full);
  }

  // fix the d_scalarLabels to point to the correct quadrature node (since there is 1 model per quad node)
  // (Not needed for BadHawkDevol model (yet)... If it is, uncomment the block below)
  /*
  for ( vector<std::string>::iterator iString = d_scalarLabels.begin(); 
        iString != d_scalarLabels.end(); ++iString) {
    temp_ic_name      = (*iString);
    temp_ic_name_full = temp_ic_name;

    out << d_quadNode;
    node = out.str();
    temp_ic_name_full += "_qn";
    temp_ic_name_full += node;

    std::replace( d_scalarLabels.begin(), d_scalarLabels.end(), temp_ic_name, temp_ic_name_full);
  }
  */
  
}



//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model 
//---------------------------------------------------------------------------
void 
BadHawkDevol::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "BadHawkDevol::computeModel";
  Task* tsk = scinew Task(taskname, this, &BadHawkDevol::computeModel);

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}

//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
BadHawkDevol::computeModel( const ProcessorGroup * pc, 
    const PatchSubset    * patches, 
    const MaterialSubset * matls, 
    DataWarehouse        * old_dw, 
    DataWarehouse        * new_dw )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

  }
}

