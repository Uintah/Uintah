#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/ShunnMoinMMSCont.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

ShunnMoinMMSCont::ShunnMoinMMSCont( std::string src_name, MaterialManagerP& materialManager,
                            vector<std::string> req_label_names, std::string type ) 
: SourceTermBase(src_name, materialManager, req_label_names, type)
{
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() ); 
}

ShunnMoinMMSCont::~ShunnMoinMMSCont()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
ShunnMoinMMSCont::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  _source_grid_type = CC_SRC; 

  ProblemSpecP db_mom = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("ExplicitSolver")->findBlock("MomentumSolver"); 
  string init_type;
  db_mom->findBlock("initialization")->getAttribute("type",init_type); 
  if ( init_type != "shunn_moin"){ 
    throw InvalidValue("Error: Trying to initialize the Shunn/Moin MMS for the mixture fraction and not matching same IC in momentum", __FILE__,__LINE__); 
  }
  db_mom->findBlock("initialization")->require("k",_k);
  db_mom->findBlock("initialization")->require("w",_w); 
  db_mom->findBlock("initialization")->require("plane",_plane); 
  db_mom->findBlock("initialization")->getWithDefault("uf",_uf,0.0);
  db_mom->findBlock("initialization")->getWithDefault("vf",_vf,0.0);

  ProblemSpecP db_prop = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("Properties")->findBlock("ColdFlow"); 
  
  db_prop->findBlock("stream_0")->getAttribute("density",_r0);
  db_prop->findBlock("stream_1")->getAttribute("density",_r1);

  if ( _plane == "x-y" ){ 
    _i1 = 0; 
    _i2 = 1; 
  } else if ( _plane == "y-z" ){ 
    _i1 = 1; 
    _i2 = 2; 
  } else if ( _plane == "z-x" ){ 
    _i1 = 2; 
    _i2 = 1; 
  } else { 
    throw InvalidValue("Error: Plane value not recognized.",__FILE__,__LINE__);
  }

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
ShunnMoinMMSCont::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "ShunnMoinMMSCont::eval";
  Task* tsk = scinew Task(taskname, this, &ShunnMoinMMSCont::computeSource, timeSubStep);

  if (timeSubStep == 0) { 

    tsk->computes(_src_label);

  } else {

    tsk->modifies(_src_label); 

  }

  tsk->requires(Task::OldDW, _simulationTimeLabel);
  
  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" )); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
ShunnMoinMMSCont::computeSource( const ProcessorGroup* pc, 
                   const PatchSubset* patches, 
                   const MaterialSubset* matls, 
                   DataWarehouse* old_dw, 
                   DataWarehouse* new_dw, 
                   int timeSubStep )
{
//  double simTime = _materialManager->getElapsedSimTime();
  simTime_vartype simTime;
  old_dw->get( simTime, _simulationTimeLabel );

  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex(); 

    CCVariable<double> src; 
    double time = simTime;
    double pi = acos(-1.0);
    double t = _w * pi * time; 

    if ( timeSubStep ==0 ){ 
      new_dw->allocateAndPut( src, _src_label, matlIndex, patch );
      src.initialize(0.0);
    } else {
      new_dw->getModifiable( src, _src_label, matlIndex, patch ); 
    } 

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 
      Point PP = patch->getCellPosition(c); 
      const Vector P = PP.toVector(); 

      double x = pi * _k *(P[_i1] - _uf * time);
      double y = pi * _k *(P[_i2] - _vf * time); 
     
      src[c] += 0.5 * pi * _k * (_r1 - _r0) * cos(t) * ( 
                _uf * cos(x) * sin(y)  + _vf * sin(x) * cos(y));

    }
  }
}

//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
ShunnMoinMMSCont::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "ShunnMoinMMSCont::initialize"; 

  Task* tsk = scinew Task(taskname, this, &ShunnMoinMMSCont::initialize);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter); 
  }

  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));

}
void 
ShunnMoinMMSCont::initialize( const ProcessorGroup* pc, 
                          const PatchSubset* patches, 
                          const MaterialSubset* matls, 
                          DataWarehouse* old_dw, 
                          DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex(); 


    CCVariable<double> src;

    new_dw->allocateAndPut( src, _src_label, matlIndex, patch ); 

    src.initialize(0.0); 

  }
}

