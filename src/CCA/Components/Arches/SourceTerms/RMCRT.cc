#include <CCA/Components/Arches/SourceTerms/RMCRT.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarLabel.h>


using namespace std;
using namespace Uintah; 
static DebugStream dbg("RMCRT", false);

/*______________________________________________________________________
          TO DO:
          
  - fix coarsen operator      
  
  - Don't like how _matlSet is being defined.
  
  - Initialize cellType on the non-arches levels.  Right now
    it's hard wired to 0
    
  
    

______________________________________________________________________*/

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
  
  const TypeDescription* CC_double = CCVariable<double>::getTypeDescription();
  _src_label      = VarLabel::create( src_name,  CC_double ); 
  _sigmaT4Label   = VarLabel::create("sigmaT4",  CC_double );
  _extra_local_labels.push_back(_sigmaT4Label); 
  _cellTypeLabel  = _labels->d_cellTypeLabel; 
  
   _RMCRT = scinew Ray( TypeDescription::double_type );          // HARDWIRED: double;
  
  //Declare the source type: 
  _source_grid_type = CC_SRC; // or FX_SRC, or FY_SRC, or FZ_SRC, or CCVECTOR_SRC
  _archesLevelIndex = -9;                         
  _sharedState      = labels->d_sharedState;      
  
  _gac = Ghost::AroundCells;
  _gn  = Ghost::None; 
  _whichAlgo = singleLevel;
  
  //__________________________________
  //  define the materialSet
  int archIndex = 0;                // HARDWIRED
  _matl = _sharedState->getArchesMaterial(archIndex)->getDWIndex();
  
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

  delete _RMCRT; 

  if( _matlSet && _matlSet->removeReference()) {
    delete _matlSet;
  }
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
RMCRT_Radiation::problemSetup( const ProblemSpecP& inputdb )
{

  _ps = inputdb; 
  _ps->getWithDefault( "calc_frequency",       _radiation_calc_freq, 3 ); 
  _ps->getWithDefault( "calc_on_all_RKsteps",  _all_rk,              false );  
  _T_label_name = "radiation_temperature"; 
  
  if ( _ps->findBlock("abskg")){ 
    _ps->findBlock("abskg")->getAttribute("label", _abskg_label_name); 
  } else { 
    throw ProblemSetupException("Error: RMCRT - The absorption coefficient is not defined.",__FILE__,__LINE__);
  }

  //__________________________________
  //  Bulletproofing:
  if(_all_rk){
    throw ProblemSetupException("ERROR:  RMCRT_radiation only works if calc_on_all_RKstes = false", __FILE__, __LINE__);
  }
  
  ProblemSpecP rmcrt_ps = _ps->findBlock("RMCRT");
  if (!rmcrt_ps){
    throw ProblemSetupException("ERROR:  RMCRT_radiation, the xml tag <RMCRT> was not found", __FILE__, __LINE__);
  }  


  _RMCRT->setBC_onOff( false );

  //__________________________________
  //  Read in the RMCRT algorithm that will be used
  ProblemSpecP alg_ps = rmcrt_ps->findBlock("algorithm");
  if (alg_ps){

    string type="NULL";
    alg_ps->getAttribute("type", type);

    if (type == "dataOnion" ) {                   // DATA ONION
    
      _whichAlgo = dataOnion;
      _RMCRT->setBC_onOff( true );

      //__________________________________
      //  bulletproofing
      if(!_sharedState->isLockstepAMR()){
        ostringstream msg;
        msg << "\n ERROR: You must add \n"
            << " <useLockStep> true </useLockStep> \n"
            << " inside of the <AMR> section. \n"; 
        throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
      }
    } else if ( type == "RMCRT_coarseLevel" ) {   // 2 LEVEL
      
      _whichAlgo = coarseLevel;
      _RMCRT->setBC_onOff( true );
      
    } else if ( type == "singleLevel" ) {         // 1 LEVEL
      _whichAlgo = singleLevel;

    }
  }
}

//______________________________________________________________________
//  We need this additiional call to problemSetup
//  so the reaction models can create the  VarLabel
//______________________________________________________________________
void 
RMCRT_Radiation::extraSetup( GridP& grid )
{ 

  // determing the temperature label
  //_tempLabel = _labels->getVarlabelByRole(ArchesLabel::TEMPERATURE);
  _tempLabel = VarLabel::find(_T_label_name); 
  proc0cout << "RMCRT: temperature label name: " << _tempLabel->getName() << endl;

  if ( _tempLabel == 0 ){ 
    throw ProblemSetupException("Error: No temperature label found.",__FILE__,__LINE__); 
  } 

  _abskg_label = VarLabel::find(_abskg_label_name); 
  if ( _abskg_label == 0){
    throw InvalidValue("Error: For RMCRT Radiation source term -- Could not find the abskg label.", __FILE__, __LINE__);
  }
  
  // create RMCRT and register the labels
  _RMCRT->registerVarLabels(_matl, 
                            _abskg_label,
                            _tempLabel,
                            _cellTypeLabel, 
                            _src_label);

  // read in RMCRT problem spec
  ProblemSpecP rmcrt_ps = _ps->findBlock("RMCRT");
  
  _RMCRT->problemSetup( _ps, rmcrt_ps, grid, _sharedState);
  
  _RMCRT->BC_bulletproofing( rmcrt_ps );
  
  //__________________________________
  //  Bulletproofing: 
  // dx must get smaller as level-index increases
  // Arches is always computed on the finest level
  int maxLevels = grid->numLevels();
  _archesLevelIndex = maxLevels - 1;
  
  if( maxLevels > 1) {
    Vector dx_prev = grid->getLevel(0)->dCell();
    
    for (int L = 1; L < maxLevels; L++) {
      Vector dx = grid->getLevel(L)->dCell();
      
      Vector ratio = dx/dx_prev;
      if( ratio.x() > 1 || ratio.y() > 1 || ratio.z() > 1){
        ostringstream warn;
        warn << "RMCRT: ERROR Level-"<< L << " cell spacing is not smaller than Level-"<< L-1 << ratio;
        throw ProblemSetupException(warn.str(),__FILE__,__LINE__);
      }
      dx_prev = dx;
    }
  }
}


//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term (divQ)
//
//  See: CCA/Components/Models/Radiation/RMCRT/Ray.cc
//       for the actual tasks that are scheduled.
//---------------------------------------------------------------------------
void 
RMCRT_Radiation::sched_computeSource( const LevelP& level, 
                                      SchedulerP& sched, 
                                      int timeSubStep )
{ 
  GridP grid = level->getGrid();

  // only sched on RK step 0 and on arches level
  if ( timeSubStep != 0  || level->getIndex() != _archesLevelIndex) {  
    return;
  } 

  int maxLevels = grid->numLevels();

  dbg << " ---------------timeSubStep: " << timeSubStep << endl;
  printSchedule(level,dbg,"RMCRT_Radiation::sched_computeSource");


  // common flags
  bool modifies_divQ     = false;
  bool includeExtraCells = false;  // domain for sigmaT4 computation

  if (timeSubStep == 0) {
    modifies_divQ  = false;
  } else {
    modifies_divQ  = true;
  }
  
  //______________________________________________________________________
  //   D A T A   O N I O N   A P P R O A C H
  if( _whichAlgo == dataOnion ){
    const LevelP& fineLevel = grid->getLevel(_archesLevelIndex);
    Task::WhichDW temp_dw  = Task::OldDW;
    Task::WhichDW abskg_dw = Task::NewDW;
    
    // modify Radiative properties on the finest level
    // convert abskg:dbl -> abskg:flt if needed
    _RMCRT->sched_DoubleToFloat( fineLevel,sched, abskg_dw, _radiation_calc_freq );
    
     // compute sigmaT4 on the finest level
    _RMCRT->sched_sigmaT4( fineLevel,  sched, temp_dw, _radiation_calc_freq, includeExtraCells );
 
    _RMCRT->sched_setBoundaryConditions( fineLevel, sched, temp_dw, _radiation_calc_freq );
        
    // coarsen data to the coarser levels.  
    // do it in reverse order
    Task::WhichDW notUsed = Task::OldDW;
    const bool backoutTemp = true;
    
    for (int l = maxLevels - 2; l >= 0; l--) {
      const LevelP& level = grid->getLevel(l);
      const bool modifies_abskg   = false;
      const bool modifies_sigmaT4 = false;
      
      _RMCRT->sched_CoarsenAll( level, sched, modifies_abskg, modifies_sigmaT4, _radiation_calc_freq );
      _RMCRT->sched_setBoundaryConditions( level, sched, notUsed, _radiation_calc_freq, backoutTemp );
    }
    
    //__________________________________
    //  compute the extents of the rmcrt region of interest
    //  on the finest level
    _RMCRT->sched_ROI_Extents( fineLevel, sched );

    Task::WhichDW sigmaT4_dw   = Task::NewDW;
    bool modifies_divQ       = false;
    _RMCRT->sched_rayTrace_dataOnion(fineLevel, sched, abskg_dw, sigmaT4_dw, modifies_divQ, _radiation_calc_freq);
  }
  
  //______________________________________________________________________
  //   2 - L E V E L   A P P R O A C H
  //  RMCRT is performed on the coarse level
  //  and the results are interpolated to the fine (arches) level
  if( _whichAlgo == coarseLevel ){
    const LevelP& fineLevel = grid->getLevel(_archesLevelIndex);
    Task::WhichDW temp_dw  = Task::OldDW;
    Task::WhichDW abskg_dw = Task::NewDW;

    // convert abskg:dbl -> abskg:flt if needed
    _RMCRT->sched_DoubleToFloat( fineLevel,sched, abskg_dw, _radiation_calc_freq );
    
    // compute sigmaT4 on the finest level
    _RMCRT->sched_sigmaT4( fineLevel,  sched, temp_dw, _radiation_calc_freq, includeExtraCells );
    
    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);;
      const bool modifies_abskg   = false;
      const bool modifies_sigmaT4 = false;
      const bool backoutTemp      = true;
      
      _RMCRT->sched_CoarsenAll (level, sched, modifies_abskg, modifies_sigmaT4, _radiation_calc_freq);
      
      if(level->hasFinerLevel() || maxLevels == 1){               // FIX ME:  Why maxLevels == 1?
        Task::WhichDW sigmaT4_dw  = Task::NewDW;
        Task::WhichDW celltype_dw = Task::NewDW;
        
        _RMCRT->sched_setBoundaryConditions( level, sched, temp_dw, _radiation_calc_freq, backoutTemp);
        
        _RMCRT->sched_rayTrace(level, sched, abskg_dw, sigmaT4_dw, celltype_dw, modifies_divQ, _radiation_calc_freq );
      }
    }

    // push divQ  to the coarser levels 
    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      const PatchSet* patches = level->eachPatch();
      _RMCRT->sched_Refine_Q (sched,  patches, _matlSet, _radiation_calc_freq);
    }
  }
  
  //______________________________________________________________________
  //   1 - L E V E L   A P P R O A C H
  //  RMCRT is performed on the same level as CFD
  if( _whichAlgo == singleLevel ){
    const LevelP& level = grid->getLevel(_archesLevelIndex);
    Task::WhichDW temp_dw  = Task::OldDW;
    Task::WhichDW abskg_dw = Task::NewDW;
    includeExtraCells = true;
    
    // convert abskg:dbl -> abskg:flt if needed
    _RMCRT->sched_DoubleToFloat( level,sched, abskg_dw, _radiation_calc_freq );
          
    // compute sigmaT4 on the CFD level
    _RMCRT->sched_sigmaT4( level,  sched, temp_dw, _radiation_calc_freq, includeExtraCells );
                                                                           
    Task::WhichDW sigmaT4_dw  = Task::NewDW;                                                                       
    Task::WhichDW celltype_dw = Task::NewDW;                                                                       

    _RMCRT->sched_rayTrace(level, sched, abskg_dw, sigmaT4_dw, celltype_dw, modifies_divQ, _radiation_calc_freq ); 
  }
}

//---------------------------------------------------------------------------
// Method: Schedule initialization
// This will only be called on the Archeslevel
//---------------------------------------------------------------------------
void
RMCRT_Radiation::sched_initialize( const LevelP& level, 
                                   SchedulerP& sched )
{
  GridP grid = level->getGrid();
  int maxLevels = grid->numLevels();

  //__________________________________
  //  Additional bulletproofing, this belongs in problem setup
  if (_whichAlgo == dataOnion && maxLevels == 1){
    throw ProblemSetupException("ERROR:  RMCRT_radiation, there must be more than 1 level if you're using the Data Onion algorithm", __FILE__, __LINE__);
  }  
  
  //__________________________________
  //  schedule the tasks
  for (int L=0; L< maxLevels; ++L){
    string taskname = "RMCRT_Radiation::sched_initialize"; 
    Task* tsk = scinew Task(taskname, this, &RMCRT_Radiation::initialize);

    LevelP level = grid->getLevel(L);
    printSchedule(level,dbg,taskname);
               
    if( L == _archesLevelIndex ) {
      tsk->computes(_src_label);
    } else {
      tsk->computes( _cellTypeLabel );
    } 
    
    //__________________________________
    //  all levels
    tsk->computes( _sigmaT4Label );
    
    sched->addTask(tsk, level->eachPatch(), _matlSet);
  }
}
//______________________________________________________________________
//
void 
RMCRT_Radiation::initialize( const ProcessorGroup*,
                             const PatchSubset* patches, 
                             const MaterialSubset*, 
                             DataWarehouse* , 
                             DataWarehouse* new_dw )
{
  const int L_indx = getLevel(patches)->getIndex();
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing RMCRT_Radiation::initialize");
    
    if( L_indx == _archesLevelIndex ){    // arches level
      CCVariable<double> src;
      new_dw->allocateAndPut( src, _src_label, _matl, patch ); 
      src.initialize(0.0); 
    }else{                                // other levels
      CCVariable<int> cellType;        
      new_dw->allocateAndPut( cellType,    _cellTypeLabel,    _matl, patch );
      cellType.initialize( 0 );           // FIX ME  do we still need this?
    }
    
    //__________________________________
    // all levels
    CCVariable<double> sigmaT4;
    new_dw->allocateAndPut(sigmaT4, _sigmaT4Label, _matl, patch );
    
    sigmaT4.initialize( 0.0 );           // FIX ME  do we still need this?
    
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
