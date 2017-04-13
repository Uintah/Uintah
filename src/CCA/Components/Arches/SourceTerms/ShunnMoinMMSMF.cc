#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/ShunnMoinMMSMF.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

ShunnMoinMMSMF::ShunnMoinMMSMF( std::string src_name, SimulationStateP& shared_state,
                            vector<std::string> req_label_names, std::string type ) 
: SourceTermBase(src_name, shared_state, req_label_names, type)
{
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() ); 
}

ShunnMoinMMSMF::~ShunnMoinMMSMF()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
ShunnMoinMMSMF::problemSetup(const ProblemSpecP& inputdb)
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
  db_mom->findBlock("initialization")->getWithDefault("A",_A,0.); 
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
ShunnMoinMMSMF::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "ShunnMoinMMSMF::eval";
  Task* tsk = scinew Task(taskname, this, &ShunnMoinMMSMF::computeSource, timeSubStep);

  Task::WhichDW which_dw; 

  if (timeSubStep == 0) { 

    tsk->computes(_src_label);
    which_dw = Task::OldDW; 

  } else {

    tsk->modifies(_src_label); 
    which_dw = Task::NewDW; 

  }

  tsk->requires(which_dw, VarLabel::find("density"), Ghost::None, 0); 

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
ShunnMoinMMSMF::computeSource( const ProcessorGroup* pc, 
                   const PatchSubset* patches, 
                   const MaterialSubset* matls, 
                   DataWarehouse* old_dw, 
                   DataWarehouse* new_dw, 
                   int timeSubStep )
{
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> src; 
    constCCVariable<double> density; 

    double time = _shared_state->getElapsedSimTime(); 
    double pi = acos(-1.0); 

    if ( timeSubStep ==0 ){  
      new_dw->allocateAndPut( src, _src_label, matlIndex, patch );
      src.initialize(0.0);
    } else {
      new_dw->getModifiable( src, _src_label, matlIndex, patch ); 
    } 

    double k2 = _k*_k;

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 

      Point PP = patch->getCellPosition(c); 
      const Vector P = PP.toVector(); 

      double x = P[_i1];
      double y = P[_i2]; 

      double s0 = cos(pi*_w*time); 
      double s1 = sin(pi*_w*time); 
      double s6 = sin(pi*_k*(x-_uf*time)); 
      double s7 = sin(pi*_k*(y-_vf*time)); 
      double s8 = cos(pi*_k*(x-_uf*time)); 
      double s9 = cos(pi*_k*(y-_vf*time));
      double s10= _r0 + _r1; 
      double s11= _r1 - _r0; 

      src[c] = -0.25*pi*_r1/pow(s10 - s0*s11*s6*s7,3)*
            ( 16*_A*k2*pi*_r0*s0*s0*s9*s9*s11*s6*s6
             - 16*_A*k2*pi*_r0*s0*s10*s6*s7
             + 16*_A*k2*pi*_r0*s0*s0*s8*s8*s11*s7*s7
             + 16*_A*k2*pi*_r0*s0*s0*s11*s6*s6*s7*s7
             + 2*_k*s0*s8*s10*s10*s10*s7*_uf
             - 6*_k*_r0*_r0*s0*s0*s8*s11*s6*s7*s7*_uf
             - 12*_k*_r0*_r1*s0*s0*s8*s11*s6*s7*s7*_uf
             - 6*_k*_r1*_r1*s0*s0*s8*s11*s6*s7*s7*_uf
             + 6*_k*s0*s0*s0*s8*s10*s11*s11*s6*s6*s7*s7*s7*_uf
             - 2*_k*s0*s0*s0*s0*s8*s11*s11*s11*s6*s6*s6*s7*s7*s7*s7*_uf
             + 2*_k*s0*s9*s10*s10*s10*s6*_vf
             - 6*_k*_r0*_r0*s0*s0*s9*s11*s6*s6*s7*_vf
             - 12*_k*_r0*_r1*s0*s0*s9*s11*s6*s6*s7*_vf
             - 6*_k*_r1*_r1*s0*s0*s9*s11*s6*s6*s7*_vf
             + 6*_k*s0*s0*s0*s9*s10*s11*s11*s6*s6*s6*s7*s7*_vf
             - 2*_k*s0*s0*s0*s0*s9*s11*s11*s11*s6*s6*s6*s6*s7*s7*s7*_vf
             - 2*_r0*s0*s1*s9*s9*s10*s11*s6*s6*_w
             + 4*_r0*s1*s10*s10*s6*s7*_w
             + 2*_r0*s0*s0*s1*s9*s9*s11*s11*s6*s6*s6*s7*_w
             - 5*_r0*s0*s1*s10*s11*s7*s7*_w
             + 3*_r0*s0*s1*s8*s8*s10*s11*s7*s7*_w
             - 3*_r0*s0*s1*s10*s11*s6*s6*s7*s7*_w
             + 2*_r0*s0*s0*s1*s8*s8*s11*s11*s6*s7*s7*s7*_w
             + 4*_r0*_r0*s0*s0*s1*s11*s6*s6*s6*s7*s7*s7*_w
             - 4*_r0*_r1*s0*s0*s1*s11*s6*s6*s6*s7*s7*s7*_w );

    }
  }
}

//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
ShunnMoinMMSMF::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "ShunnMoinMMSMF::initialize"; 

  Task* tsk = scinew Task(taskname, this, &ShunnMoinMMSMF::initialize);

  tsk->computes(_src_label);

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void 
ShunnMoinMMSMF::initialize( const ProcessorGroup* pc, 
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

  }
}

