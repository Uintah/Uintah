#include <CCA/Components/Arches/SourceTerms/RMCRT.h>
#include <CCA/Components/Arches/Radiation/DORadiationModel.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/Directives.h>

using namespace std;
using namespace Uintah; 
static DebugStream dbg("RMCRT", false);
//______________________________________________________________________
//
RMCRT_Radiation::RMCRT_Radiation( std::string src_name, 
                                  ArchesLabel* labels, 
                                  MPMArchesLabel* MAlab,
                                  BoundaryCondition* bc, 
                                  vector<std::string> req_label_names, 
                                  const ProcessorGroup* my_world, 
                                  std::string type ) 
: SourceTermBase( src_name, 
                  labels->d_sharedState, 
                  req_label_names, type ), 
  _labels( labels ),
  _MAlab(MAlab), 
  _bc(bc), 
  _my_world(my_world)
{

  // NOTE: This boundary condition here is bogus.  Passing it for 
  // now until the boundary condition reference can be stripped out of 
  // the radiation model. 
  
  _label_sched_init = false; 
  
  const TypeDescription* CC_double = CCVariable<double>::getTypeDescription();
  _src_label      = VarLabel::create( src_name,  CC_double ); 
  _sigmaT4Label   = VarLabel::create("sigmaT4",  CC_double );
  _colorLabel     = VarLabel::create( "color",   CC_double );
  _abskgLabel     = VarLabel::create( "abskg",   CC_double );
  _absorpLabel    = VarLabel::create( "absorp",  CC_double );
  
  //Declare the source type: 
  _source_grid_type = CC_SRC; // or FX_SRC, or FY_SRC, or FZ_SRC, or CCVECTOR_SRC

  _prop_calculator       = 0;
  _using_prop_calculator = 0; 
  _RMCRT                 = 0;
  _sharedState           = labels->d_sharedState;
  
  _gac = Ghost::AroundCells;
  _gn  = Ghost::None; 
  
  // HACK
  _initColor = -9;
  _initAbskg = -9;
  
  _CoarseLevelRMCRTMethod = true;
  _multiLevelRMCRTMethod  = false;
  
  //__________________________________
  //  define the materialSet
  int archIndex = 0;
  _matl = _labels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
  
  _matlSet = scinew MaterialSet();
  vector<int> m;
  m.push_back(_matl);
  _matlSet->addAll(m);
  _matlSet->addReference();
}
//______________________________________________________________________
//
RMCRT_Radiation::~RMCRT_Radiation()
{
  // source label is destroyed in the base class 
  VarLabel::destroy( _sigmaT4Label ); 
  VarLabel::destroy( _colorLabel );
  VarLabel::destroy( _abskgLabel );
  VarLabel::destroy( _absorpLabel );

  delete _prop_calculator; 
  delete _RMCRT; 

  if( _matlSet && _matlSet->removeReference()) {
    delete _matlSet;
  }
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
RMCRT_Radiation::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 
  db->getWithDefault( "calc_frequency",       _radiation_calc_freq, 3 ); 
  db->getWithDefault( "calc_on_all_RKsteps",  _all_rk,              false );  
//  db->getWithDefault( "T_label",              _T_label_name,        "temperature" ); 
//  db->getWithDefault( "abskp_label",          _abskp_label_name,    "new_abskp" ); 


  //__________________________________
  //  Bulletproofing:
  if(_all_rk){
    throw ProblemSetupException("ERROR:  RMCRT_radiation only works if calc_on_all_RKstes = false", __FILE__, __LINE__);
  }

  if (db->findBlock("RMCRT")){
    ProblemSpecP rmcrt_ps = db->findBlock("RMCRT");
    
    _RMCRT = scinew Ray(); 
    _RMCRT->registerVarLabels(_matl, 
                              _abskgLabel,
                              _absorpLabel,
                              _colorLabel,
                              _src_label,
                              _flux_label);

    _RMCRT->problemSetup( db, rmcrt_ps );
   
    //  HACK                           
    rmcrt_ps->require("Temperature",  _initColor);
    rmcrt_ps->require("abskg",        _initAbskg);
    
    //__________________________________
    //
    _prop_calculator = scinew RadPropertyCalculator(); 
    _using_prop_calculator = _prop_calculator->problemSetup( rmcrt_ps );
    
  }                               

  //__________________________________
  //  Read in the AMR section
  ProblemSpecP prob_spec = db->getRootNode();
  
  ProblemSpecP amr_ps = prob_spec->findBlock("AMR");
  if (amr_ps){
    ProblemSpecP rmcrt_ps = amr_ps->findBlock("RMCRT");

    if(!rmcrt_ps){
      string warn;
      warn ="\n INPUT FILE ERROR:\n <RMCRT>  block not found inside of <AMR> block \n";
      throw ProblemSetupException(warn, __FILE__, __LINE__);
    }

    rmcrt_ps->require("CoarseLevelRMCRTMethod", _CoarseLevelRMCRTMethod);
    rmcrt_ps->require("multiLevelRMCRTMethod",  _multiLevelRMCRTMethod);

    //__________________________________
    //  bulletproofing
    if(!_sharedState->isLockstepAMR()){
      ostringstream msg;
      msg << "\n ERROR: You must add \n"
          << " <useLockStep> true </useLockStep> \n"
          << " inside of the <AMR> section. \n"; 
      throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
    }
  } 
}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//
//  See: CCA/Components/Models/Radiation/RMCRT/Ray.cc
//       for the actual tasks that are scheduled.
//---------------------------------------------------------------------------
void 
RMCRT_Radiation::sched_computeSource( const LevelP& level, 
                                      SchedulerP& sched, 
                                      int timeSubStep )
{

  //__________________________________
  //
  if(level->getIndex() > 0){  // only schedule once
    return;
  }

  int timestep = _sharedState->getCurrentTopLevelTimeStep();
  if ( timestep%_radiation_calc_freq != 0 ) {  // is it the right timestep
    return;
  } 

  if ( timeSubStep != 0 ) {                   // only works on on RK step 0
    return;
  }
  
     
 
  dbg << " ---------------timeSubStep: " << timeSubStep << endl;
  printSchedule(level,dbg,"sched_computeSource: main task");
  
  GridP grid = level->getGrid();
  int maxLevels = level->getGrid()->numLevels();
  bool modifies_divQ =false;
  bool modifies_VRFlux =false;
  Task::WhichDW temp_dw   = Task::NewDW;
  
  if (timeSubStep == 0 && !_label_sched_init) {
    modifies_divQ  = false;
    modifies_VRFlux  = false;
  } else {
    modifies_divQ  = true;
    modifies_VRFlux  = true;
    temp_dw        = Task::NewDW;
  }

  


  
  //______________________________________________________________________
  //   2 - L E V E L   A P P R O A C H
  //  If the RMCRT is performed on only the coarse level
  // and the results are interpolated to the fine level
  if(_CoarseLevelRMCRTMethod){
    const LevelP& fineLevel = grid->getLevel(maxLevels-1);
    const PatchSet* finestPatches = fineLevel->eachPatch();
   
    // compute Radiative properties and sigmaT4 on the finest level
    sched_radProperties( fineLevel, sched, timeSubStep );
    
    _RMCRT->sched_sigmaT4( fineLevel,  sched, temp_dw );
    
    
    for (int l = 0; l <= maxLevels-1; l++) {
      const LevelP& level = grid->getLevel(l);
      
      _RMCRT->sched_CoarsenAll (level, sched);
      
      if(level->hasFinerLevel() || maxLevels == 1){
        Task::WhichDW abskg_dw   = Task::NewDW;
        Task::WhichDW sigmaT4_dw = Task::NewDW;
        _RMCRT->sched_rayTrace(level, sched, abskg_dw, sigmaT4_dw, modifies_divQ, modifies_VRFlux);
      }
    }

    // push divQ  to the coarser levels 
    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      const PatchSet* patches = level->eachPatch();
      _RMCRT->sched_Refine_Q (sched,  patches, _matlSet);
    }
  }
  
  // HACK until we figure out what to do with temperature
  std::string taskname = "RMCRT_Radiation::computeSource";
  Task* tsk = scinew Task(taskname, this, &RMCRT_Radiation::computeSource, timeSubStep);
  printSchedule(level,dbg,"sched_computeSource HACK");
  tsk->computes(_colorLabel);
  sched->addTask(tsk, level->eachPatch(), _matlSet);

  
}


//---------------------------------------------------------------------------
// Method: Wrapper to RadPropertyCalculator 
//---------------------------------------------------------------------------
void
RMCRT_Radiation::sched_radProperties( const LevelP& level, 
                                      SchedulerP& sched, 
                                      const int time_sub_step )
{
  Task* tsk = scinew Task( "RMCRT_Radiation::radProperties", this, 
                           &RMCRT_Radiation::radProperties, time_sub_step ); 
                           
  printSchedule(level,dbg, "RMCRT_Radiation::sched_radProperties");

  if ( time_sub_step == 0 ) { 
    tsk->computes( _abskgLabel ); 
  } else {  
    tsk->modifies( _abskgLabel );  
  }

  sched->addTask( tsk, level->eachPatch(), _matlSet ); 

}

//______________________________________________________________________
//
void
RMCRT_Radiation::radProperties( const ProcessorGroup* ,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* ,
                                DataWarehouse* new_dw, 
                                const int time_sub_step )
{
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing RMCRT_Radiation::radProperties");

    CCVariable<double> abskg; 

    if ( time_sub_step == 0 ) { 
      new_dw->allocateAndPut( abskg, _abskgLabel,  _matl, patch ); 
    } else { 
      new_dw->getModifiable( abskg,  _abskgLabel,  _matl, patch );
    }
    _prop_calculator->compute( patch, abskg );
    
    _RMCRT->setBC(abskg, _abskgLabel->getName(), patch, _matl);
  }
  

}

//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
RMCRT_Radiation::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "RMCRT_Radiation::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &RMCRT_Radiation::dummyInit);
  printSchedule(level,dbg,taskname);

  tsk->computes( _src_label  );
  tsk->computes( _colorLabel );
  tsk->computes( _abskgLabel );
  tsk->computes( _sigmaT4Label );

  sched->addTask(tsk, level->eachPatch(), _matlSet);
}
//______________________________________________________________________
//
void 
RMCRT_Radiation::dummyInit( const ProcessorGroup*, 
                      const PatchSubset* patches, 
                      const MaterialSubset*, 
                      DataWarehouse* , 
                      DataWarehouse* new_dw )
{
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing RMCRT_Radiation::dummyInit");

    CCVariable<double> divQ;
    CCVariable<double> color;
    CCVariable<double> abskg;
    CCVariable<double> sigmaT4;
    
    new_dw->allocateAndPut( color,    _colorLabel,    _matl, patch );
    new_dw->allocateAndPut( abskg,    _abskgLabel,    _matl, patch );
    new_dw->allocateAndPut( divQ,     _src_label,     _matl, patch );
    new_dw->allocateAndPut( sigmaT4,  _sigmaT4Label,  _matl, patch );
     
    divQ.initialize( 0.);
    sigmaT4.initialize( 0. ); 
    abskg.initialize( _initAbskg );
    color.initialize( _initColor );
    
     // set boundary conditions 
    _RMCRT->setBC(color,  _colorLabel->getName(), patch, _matl);
    _RMCRT->setBC(abskg,  _abskgLabel->getName(), patch, _matl);
  }
}

//______________________________________________________________________
// STUB  
//______________________________________________________________________
void
RMCRT_Radiation::computeSource( const ProcessorGroup* , 
                            const PatchSubset* patches, 
                            const MaterialSubset* matls, 
                            DataWarehouse* old_dw, 
                            DataWarehouse* new_dw, 
                            int timeSubStep ){
  // see sched_computeSource & CCA/Components/Models/Radiation/RMCRT/Ray.cc
  // for the actual tasks
  
  
  // HACK: until we start using the correct temperature.
  new_dw->transferFrom(old_dw, _colorLabel, patches, matls);

}
