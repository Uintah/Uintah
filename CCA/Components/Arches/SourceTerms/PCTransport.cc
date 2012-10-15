#include <CCA/Components/Arches/SourceTerms/PCTransport.h>

using namespace std;
using namespace Uintah; 

PCTransport::PCTransport( std::string src_name, SimulationStateP& shared_state,
                      vector<std::string> req_label_names ) 
: SourceTermBase( src_name, shared_state, req_label_names )
{
  _label_sched_init = false; 

  //Source Label
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() ); 

  //Declare the source type: 
  _source_type = CC_SRC; // or FX_SRC, or FY_SRC, or FZ_SRC, or CCVECTOR_SRC

}

PCTransport::~PCTransport()
{
  
  // source label is destroyed in the base class 

  for (vector<std::string>::iterator iter = _required_labels.begin(); 
       iter != _required_labels.end(); iter++) { 

    VarLabel::destroy( *iter ); 

  }

  // clean up any other stuff here
  
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
PCTransport::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  // add input file interface here 

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
PCTransport::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "PCTransport::eval";
  Task* tsk = scinew Task(taskname, this, &PCTransport::computeSource, timeSubStep);

  if (timeSubStep == 0 && !_label_sched_init) {
    // Every source term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    _label_sched_init = true;

    tsk->computes(_src_label);

  } else {

    tsk->modifies(_src_label); 

  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
PCTransport::computeSource( const ProcessorGroup* pc, 
                   const PatchSubset* patches, 
                   const MaterialSubset* matls, 
                   DataWarehouse* old_dw, 
                   DataWarehouse* new_dw, 
                   int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

    // add actual calculation of the source here. 

  }
}

//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
PCTransport::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  //not needed
}
void 
PCTransport::dummyInit( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw )
{
  //not needed
}

