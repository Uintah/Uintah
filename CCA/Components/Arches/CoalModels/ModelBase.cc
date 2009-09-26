
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/ModelFactory.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <Core/Grid/Variables/CCVariable.h>

using namespace std;
using namespace Uintah; 


ModelBase::ModelBase( std::string modelName, SimulationStateP& sharedState,
                      const ArchesLabel* fieldLabels,
                      vector<std::string> icLabelNames, int qn ) : 
d_modelName(modelName), d_sharedState( sharedState ), d_fieldLabels(fieldLabels), d_icLabels(icLabelNames), d_quadNode(qn)
{
  //Create a label for this source term. 
  if(modelName == "dragforce") {
    d_modelLabel = VarLabel::create(modelName, CCVariable<Vector>::getTypeDescription());
    std::string varname = modelName + "_gassSource"; 
    d_gasLabel = VarLabel::create( varname, CCVariable<Vector>::getTypeDescription()); 

  } else {  
    d_modelLabel = VarLabel::create(modelName, CCVariable<double>::getTypeDescription()); 
    std::string varname = modelName + "_gassSource"; 
    d_gasLabel = VarLabel::create( varname, CCVariable<double>::getTypeDescription()); 
  }

 
  d_labelSchedInit  = false; 
}

ModelBase::~ModelBase()
{
  VarLabel::destroy(d_modelLabel); 
  VarLabel::destroy(d_gasLabel); 
}

//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
ModelBase::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "ModelBase::dummyInit"; 

  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &ModelBase::dummyInit);

  tsk->requires( Task::OldDW, d_modelLabel, gn, 0);
  tsk->requires( Task::OldDW, d_gasLabel,   gn, 0);

  tsk->computes(d_modelLabel);
  tsk->computes(d_gasLabel); 

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());

}
void 
ModelBase::dummyInit( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw )		      
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    Ghost::GhostType  gn = Ghost::None;
    
     if(d_modelName == "dragforce") {
      CCVariable<Vector> model;
      CCVariable<Vector> gas_src; 
      constCCVariable<Vector> old_model;
      constCCVariable<Vector> old_gas_src;
      new_dw->allocateAndPut( model, d_modelLabel, matlIndex, patch ); 
      new_dw->allocateAndPut( gas_src, d_gasLabel, matlIndex, patch ); 
      old_dw->get(old_model, d_modelLabel, matlIndex, patch, gn, 0);
      old_dw->get(old_gas_src, d_gasLabel, matlIndex, patch, gn, 0);
      model.copyData(old_model);
      gas_src.copyData(old_gas_src);
     } else {
      CCVariable<double> model;
      CCVariable<double> gas_src; 
      constCCVariable<double> old_model;
      constCCVariable<double> old_gas_src;
      new_dw->allocateAndPut( model, d_modelLabel, matlIndex, patch ); 
      new_dw->allocateAndPut( gas_src, d_gasLabel, matlIndex, patch ); 
      old_dw->get(old_model, d_modelLabel, matlIndex, patch, gn, 0);
      old_dw->get(old_gas_src, d_gasLabel, matlIndex, patch, gn, 0);
      model.copyData(old_model);
      gas_src.copyData(old_gas_src);
     }     

  }
}

