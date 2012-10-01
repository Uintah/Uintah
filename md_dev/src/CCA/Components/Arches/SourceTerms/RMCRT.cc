#include <CCA/Components/Arches/SourceTerms/RMCRT.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>

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
  _abskgLabel     = VarLabel::create( "abskg",   CC_double );
  _absorpLabel    = VarLabel::create( "absorp",  CC_double );
  _cellTypeLabel  = _labels->d_cellTypeLabel; 
  
  //Declare the source type: 
  _source_grid_type = CC_SRC; // or FX_SRC, or FY_SRC, or FZ_SRC, or CCVECTOR_SRC

  _prop_calculator       = 0;
  _using_prop_calculator = 0; 
  _RMCRT                 = 0;
  _sharedState           = labels->d_sharedState;
  
  _gac = Ghost::AroundCells;
  _gn  = Ghost::None; 
  _whichAlgo = coarseLevel;
  
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

  _ps = inputdb; 
  _ps->getWithDefault( "calc_frequency",       _radiation_calc_freq, 3 ); 
  _ps->getWithDefault( "calc_on_all_RKsteps",  _all_rk,              false );  

  //__________________________________
  //  Bulletproofing:
  if(_all_rk){
    throw ProblemSetupException("ERROR:  RMCRT_radiation only works if calc_on_all_RKstes = false", __FILE__, __LINE__);
  }
  
  
  ProblemSpecP rmcrt_ps = _ps->findBlock("RMCRT");
  if (!rmcrt_ps){
    throw ProblemSetupException("ERROR:  RMCRT_radiation, the xml tag <RMCRT> was not found", __FILE__, __LINE__);
  }  

  //__________________________________
  //  Read in the algorithm
  ProblemSpecP alg_ps = rmcrt_ps->findBlock("algorithm");
  if (alg_ps){

    string type="NULL";
    alg_ps->getAttribute("type", type);

    if (type == "dataOnion" ) {
      _whichAlgo = dataOnion;

      //__________________________________
      //  bulletproofing
      if(!_sharedState->isLockstepAMR()){
        ostringstream msg;
        msg << "\n ERROR: You must add \n"
            << " <useLockStep> true </useLockStep> \n"
            << " inside of the <AMR> section. \n"; 
        throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
      }
    } else if ( type == "RMCRT_coarseLevel" ) {
      _whichAlgo = coarseLevel;
    }
  }

  //__________________________________
  //
  _prop_calculator = scinew RadPropertyCalculator(); 
  _using_prop_calculator = _prop_calculator->problemSetup( rmcrt_ps ); 
}

//______________________________________________________________________
//  Additional call made in problem setup
//______________________________________________________________________
void 
RMCRT_Radiation::extraSetup()
{ 
  _tempLabel = _labels->getVarlabelByRole("temperature");
  proc0cout << "RMCRT: temperature label name: " << _tempLabel->getName() << endl;

  _RMCRT = scinew Ray(); 
  _RMCRT->registerVarLabels(_matl, 
                            _abskgLabel,
                            _absorpLabel,
                            _tempLabel,
                            _cellTypeLabel, 
                            _src_label);

  ProblemSpecP rmcrt_ps = _ps->findBlock("RMCRT");
  _RMCRT->problemSetup( _ps, rmcrt_ps );
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
  bool modifies_divQ     =false;
  bool includeExtraCells = false;  // domain for sigmaT4 computation
  
  Task::WhichDW temp_dw   = Task::OldDW;
  
  if (timeSubStep == 0 && !_label_sched_init) {
    modifies_divQ  = false;
  } else {
    modifies_divQ  = true;
  }
  
  //______________________________________________________________________
  //   2 - L E V E L   A P P R O A C H
  //  If the RMCRT is performed on only the coarse level
  // and the results are interpolated to the fine level
  if( _whichAlgo == coarseLevel ){
    const LevelP& fineLevel = grid->getLevel(maxLevels-1);
    //const PatchSet* finestPatches = fineLevel->eachPatch(); //commented because it is not use.
   
    // compute Radiative properties and sigmaT4 on the finest level
    sched_radProperties( fineLevel, sched, timeSubStep );
    
    _RMCRT->sched_sigmaT4( fineLevel,  sched, temp_dw, includeExtraCells );
    
    _RMCRT->sched_setBoundaryConditions( fineLevel, sched, temp_dw );
    
    for (int l = 0; l <= maxLevels-1; l++) {
      const LevelP& level = grid->getLevel(l);
      const bool modifies_abskg   = false;
      const bool modifies_sigmaT4 = false;
      _RMCRT->sched_CoarsenAll (level, sched, modifies_abskg, modifies_sigmaT4);
      
      if(level->hasFinerLevel() || maxLevels == 1){
        Task::WhichDW abskg_dw    = Task::NewDW;
        Task::WhichDW sigmaT4_dw  = Task::NewDW;
        Task::WhichDW celltype_dw = Task::NewDW;
        _RMCRT->sched_rayTrace(level, sched, abskg_dw, sigmaT4_dw, celltype_dw, modifies_divQ);
      }
    }

    // push divQ  to the coarser levels 
    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      const PatchSet* patches = level->eachPatch();
      _RMCRT->sched_Refine_Q (sched,  patches, _matlSet);
    }
  }
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
    CCVariable<double> temp;
    if ( time_sub_step == 0 ) { 
      new_dw->allocateAndPut( abskg, _abskgLabel, _matl, patch ); 
    } else { 
      new_dw->getModifiable( abskg,  _abskgLabel,  _matl, patch );
    }
    
    // compute absorption coefficient via RadPropertyCalulator
    _prop_calculator->compute( patch, abskg );
    
    // abskg boundary conditions are set in setBoundaryCondition()
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
  tsk->computes( _tempLabel );
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
    CCVariable<double> temp;
    CCVariable<double> abskg;
    CCVariable<double> sigmaT4;
    
    new_dw->allocateAndPut( temp,     _tempLabel,    _matl, patch );
    new_dw->allocateAndPut( abskg,    _abskgLabel,    _matl, patch );
    new_dw->allocateAndPut( divQ,     _src_label,     _matl, patch );
    new_dw->allocateAndPut( sigmaT4,  _sigmaT4Label,  _matl, patch );
     
    divQ.initialize( 0.);
    sigmaT4.initialize( 0. ); 
    abskg.initialize( 0. );
    
     // set boundary conditions 
    _RMCRT->setBC(temp,   _tempLabel->getName(),  patch, _matl);
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
  throw InternalError("Stub Task: RMCRT_Radiation::computeSource you should never land here ", __FILE__, __LINE__);
}
