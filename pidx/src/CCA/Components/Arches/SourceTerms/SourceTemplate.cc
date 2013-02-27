#include <CCA/Components/Arches/SourceTerms/CLASSNAME.h>

// Instructions: 
//  1) Make sure you add doxygen comments!!
//  2) If this is not a CCVariable, then either replace with the appropriate 
//     type or use the templated template.  
//  2) Do a find and replace on CLASSNAME to change the your class name 
//  3) Add implementaion details of your source term. 
//  4) Here is a brief checklist: 
//     a) Any extra grid variables for this source term need to be 
//        given VarLabels in the constructor
//     b) Any extra grid variable VarLabels need to be destroyed
//        in the local destructor
//     c) Add your input file details in problemSetup
//     d) Add actual calculation of property in computeSource
//     e) Make sure that you initialize any new variables that require OldDW 
//        values.
//     f) Make sure that _before_table_lookup is set propertly for this model.
//        See _before_table_lookup variable. 
//   5) Please clean up unused code from this template in your final version
//   6) Please add comments to this list as you see fit to help the next person

using namespace std;
using namespace Uintah; 

CLASSNAME::CLASSNAME( std::string src_name, SimulationStateP& shared_state,
                      vector<std::string> req_label_names ) 
: SourceTermBase( src_name, shared_state, req_label_names )
{
  _label_sched_init = false; 

  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() ); 

  // Add any other local variables here. 
  _extra_local_labels.resize(1); 
  _another_label = VarLabel::create( "string", CCVariable<double>::getTypeDescription() ); 
  _extra_local_labels[0] = _another_label;

  //Declare the source type: 
  _source_type = CC_SRC; // or FX_SRC, or FY_SRC, or FZ_SRC, or CCVECTOR_SRC

}

CLASSNAME::~CLASSNAME()
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
CLASSNAME::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  // add input file interface here 

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
CLASSNAME::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "CLASSNAME::eval";
  Task* tsk = scinew Task(taskname, this, &CLASSNAME::computeSource, timeSubStep);

  if (timeSubStep == 0 && !_label_sched_init) {
    // Every source term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    _label_sched_init = true;

    tsk->computes(_src_label);

  } else {

    tsk->modifies(_src_label); 

  }

  for (vector<std::string>::iterator iter = _required_labels.begin(); 
       iter != _required_labels.end(); iter++) { 
    // HERE I WOULD REQUIRE ANY VARIABLES NEEDED TO COMPUTE THE SOURCe
    //tsk->requires( Task::OldDW, .... ); 
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
CLASSNAME::computeSource( const ProcessorGroup* pc, 
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
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
CLASSNAME::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "CLASSNAME::initialize"; 

  Task* tsk = scinew Task(taskname, this, &CLASSNAME::initialize);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); 
       iter != _extra_local_labels.end(); iter++){

    tsk->computes(*iter); 

  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void 
CLASSNAME::initialize( const ProcessorGroup* pc, 
                       const PatchSubset* patches, 
                       const MaterialSubset* matls, 
                       DataWarehouse* old_dw, 
                       DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> src;

    new_dw->allocateAndPut( src, _src_label, matlIndex, patch ); 

    src.initialize(0.0); 

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); 
         iter != _extra_local_labels.end(); iter++){
      CCVariable<double> tempVar; 
      new_dw->allocateAndPut(tempVar, *iter, matlIndex, patch ); 
      tempVar.initialize(0.0); 
    }
  }
}

