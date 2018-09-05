#include <CCA/Components/Arches/CoalModels/CharOxidation.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

CharOxidation::CharOxidation( std::string modelName, 
                              MaterialManagerP& materialManager,
                              ArchesLabel* fieldLabels,
                              vector<std::string> icLabelNames,
                              vector<std::string> scalarLabelNames,
                              int qn )
: ModelBase(modelName, materialManager, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  d_quadNode = qn;

  // Create a label for this model
  d_modelLabel = VarLabel::create( modelName, CCVariable<double>::getTypeDescription() );

  // Create the gas phase source term associated with this model
  std::string gasSourceName = modelName + "_gasSource";
  d_gasLabel = VarLabel::create( gasSourceName, CCVariable<double>::getTypeDescription() );

  // Create the particle temperature source term associated with this model
  std::string particletempSourceName = modelName + "_particletempSource";
  d_particletempLabel = VarLabel::create( particletempSourceName, CCVariable<double>::getTypeDescription() );
  _extra_local_labels.push_back(d_particletempLabel);
  
  // Create the particle size source term associated with this model
  std::string particleSizeSourceName = modelName + "_particleSizeSource";
  d_particleSizeLabel = VarLabel::create( particleSizeSourceName, CCVariable<double>::getTypeDescription() );
  _extra_local_labels.push_back(d_particleSizeLabel);


  // Create the char oxidation surface rate term associated with this model
  std::string surfacerateName = modelName + "_surfacerate";
  d_surfacerateLabel = VarLabel::create( surfacerateName, CCVariable<double>::getTypeDescription() );
  _extra_local_labels.push_back(d_surfacerateLabel);

  // Create the char oxidation PO2 surf term associated with this model
  std::string PO2surfName = modelName + "_PO2surf";
  d_PO2surfLabel = VarLabel::create( PO2surfName, CCVariable<double>::getTypeDescription() );
  _extra_local_labels.push_back(d_PO2surfLabel);

}

CharOxidation::~CharOxidation()
{
  for (vector<const VarLabel*>::iterator iter = _extra_local_labels.begin();
       iter != _extra_local_labels.end(); iter++) {
    VarLabel::destroy( *iter );
  }
}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
  void 
CharOxidation::problemSetup(const ProblemSpecP& params, int qn)
{
  // This method is called by all devolatilizaiton classes' problemSetup()

  ProblemSpecP db = params; 

  // set model clipping (not used yet...)
  db->getWithDefault( "low_clip",  d_lowModelClip,  1.0e-6 );
  db->getWithDefault( "high_clip", d_highModelClip, 999999 );

  // grab weight scaling factor and small value
  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();

  std::string temp_weight_name = "w_qn";
  std::string node;
  std::stringstream out;
  out << d_quadNode;
  node = out.str();
  temp_weight_name += node;
  EqnBase& t_weight_eqn = dqmom_eqn_factory.retrieve_scalar_eqn( temp_weight_name );
  DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(t_weight_eqn);

  d_w_small = weight_eqn.getSmallClipPlusTol();
  d_w_scaling_constant = weight_eqn.getScalingConstant(d_quadNode);
}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
/** @details
This method intentionally does nothing.

@see CharOxidation::initVars()
*/
void 
CharOxidation::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "CharOxidation::initVars";
  Task* tsk = scinew Task(taskname, this, &CharOxidation::initVars);

  sched->addTask(tsk, level->eachPatch(), d_materialManager->allMaterials( "Arches" )); 
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
/** @details
This method is left intentionally blank.  This way, if the method is
 called for a heat transfer model, and that heat transfer model
 doesn't require the initialization of any variables, the child 
 class will not need to re-define this (empty) method.

If additional variables are needed, and initVars needs to do stuff,
 the model can redefine it.
*/
void
CharOxidation::initVars( const ProcessorGroup * pc, 
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
    int matlIndex = d_fieldLabels->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex(); 

  }
  */
  
}
