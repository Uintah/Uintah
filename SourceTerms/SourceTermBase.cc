
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <Core/Grid/Variables/CCVariable.h>

using namespace std;
using namespace Uintah; 

SourceTermBase::SourceTermBase( std::string srcName, SimulationStateP& sharedState,
                        vector<std::string> reqLabelNames ) : 
d_srcName(srcName), d_sharedState( sharedState ), d_requiredLabels(reqLabelNames)
{
  //Create a label for this source term. 
  //d_srcLabel = VarLabel::create(srcName, CCVariable<double>::getTypeDescription()); 
  if ( srcName == "coal_gas_momentum") {
    d_srcLabel = VarLabel::create(srcName, CCVariable<Vector>::getTypeDescription()); 
  }
  else {
    d_srcLabel = VarLabel::create(srcName, CCVariable<double>::getTypeDescription());
  }
  d_labelSchedInit  = false; 
}

SourceTermBase::~SourceTermBase()
{
  VarLabel::destroy(d_srcLabel); 
}

//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
SourceTermBase::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "SourceTermBase::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &SourceTermBase::dummyInit);

  Ghost::GhostType  gn = Ghost::None;

  tsk->computes(d_srcLabel);
  tsk->requires(Task::OldDW, d_srcLabel, gn, 0); 

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());

}
void 
SourceTermBase::dummyInit( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    Ghost::GhostType  gn = Ghost::None;

    CCVariable<double> src;
    constCCVariable<double> old_src;

    new_dw->allocateAndPut( src, d_srcLabel, matlIndex, patch ); 
    old_dw->get( old_src, d_srcLabel, matlIndex, patch, gn, 0); 

    src.copyData(old_src);

  }
}


