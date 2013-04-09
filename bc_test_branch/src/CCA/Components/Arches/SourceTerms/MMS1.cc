#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/MMS1.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

MMS1::MMS1( std::string src_name, SimulationStateP& shared_state,
                            vector<std::string> req_label_names, std::string type ) 
: SourceTermBase(src_name, shared_state, req_label_names, type)
{
  _label_sched_init = false; 
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() ); 
}

MMS1::~MMS1()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
MMS1::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  _source_grid_type = CC_SRC; 

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
MMS1::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "MMS1::eval";
  Task* tsk = scinew Task(taskname, this, &MMS1::computeSource, timeSubStep);

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
MMS1::computeSource( const ProcessorGroup* pc, 
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
    Vector Dx = patch->dCell(); 

    CCVariable<double> mms1Src; 
    if ( new_dw->exists(_src_label, matlIndex, patch ) ){
      new_dw->getModifiable( mms1Src, _src_label, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( mms1Src, _src_label, matlIndex, patch );
      mms1Src.initialize(0.0);
    } 

    for (vector<std::string>::iterator iter = _required_labels.begin(); 
         iter != _required_labels.end(); iter++) { 
      //CCVariable<double> temp; 
      //old_dw->get( *iter.... ); 
    }


    double pi = acos(-1.0); 

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      
      IntVector c = *iter; 
      double x = c[0]*Dx.x() + Dx.x()/2.; 
      double y = c[1]*Dx.y() + Dx.y()/2.;
      //double z = c[2]*Dx.z() + Dx.z()/2.;

      mms1Src[c] = 2.*pi*cos(2.*pi*x)*cos(2.*pi*y) - 2.*pi*sin(2.*pi*x)*sin(2.*pi*y); 
    }
  }
}
//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
MMS1::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "MMS1::initialize"; 

  Task* tsk = scinew Task(taskname, this, &MMS1::initialize);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter); 
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void 
MMS1::initialize( const ProcessorGroup* pc, 
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

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
      CCVariable<double> tempVar; 
      new_dw->allocateAndPut(tempVar, *iter, matlIndex, patch ); 
    }
  }
}
