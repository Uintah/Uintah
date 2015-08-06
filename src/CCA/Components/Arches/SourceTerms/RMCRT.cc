#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
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
 
 - Allow the user to select between double or float RMCRT, see _FLT_DBL
 
 ______________________________________________________________________*/

RMCRT_Radiation::RMCRT_Radiation( std::string src_name, 
                                  ArchesLabel* labels, 
                                  MPMArchesLabel* MAlab,
                                  vector<std::string> req_label_names, 
                                  const ProcessorGroup* my_world, 
                                  std::string type ) 
: SourceTermBase( src_name, 
                  labels->d_sharedState, 
                  req_label_names, type ), 
  _labels( labels ),
  _MAlab(MAlab), 
  _my_world(my_world)
{  

  _src_label = VarLabel::create( src_name,  CCVariable<double>::getTypeDescription() ); 
  
  _FLT_DBL = TypeDescription::double_type;        // HARDWIRED: double;
  
   _RMCRT = scinew Ray( _FLT_DBL );          
  
  //Declare the source type: 
  _source_grid_type = CC_SRC; // or FX_SRC, or FY_SRC, or FZ_SRC, or CCVECTOR_SRC
  _archesLevelIndex = -9;                         
  _sharedState      = labels->d_sharedState;      
  
  _gac = Ghost::AroundCells;
  _gn  = Ghost::None; 
  _whichAlgo = singleLevel;
  
  //__________________________________
  //  define the material index
  int archIndex = 0;                // HARDWIRED
  _matl = _sharedState->getArchesMaterial(archIndex)->getDWIndex();
  
  
  const TypeDescription* CC_double = CCVariable<double>::getTypeDescription();
  _radFluxE_Label = VarLabel::create("radiationFluxE",  CC_double);
  _radFluxW_Label = VarLabel::create("radiationFluxW",  CC_double);
  _radFluxN_Label = VarLabel::create("radiationFluxN",  CC_double);
  _radFluxS_Label = VarLabel::create("radiationFluxS",  CC_double);
  _radFluxT_Label = VarLabel::create("radiationFluxT",  CC_double);
  _radFluxB_Label = VarLabel::create("radiationFluxB",  CC_double);
  
}
//______________________________________________________________________
//
RMCRT_Radiation::~RMCRT_Radiation()
{
  // source label is destroyed in the base class
  delete _RMCRT; 
  
  VarLabel::destroy(_radFluxE_Label);
  VarLabel::destroy(_radFluxW_Label);
  VarLabel::destroy(_radFluxN_Label);
  VarLabel::destroy(_radFluxS_Label);
  VarLabel::destroy(_radFluxT_Label);
  VarLabel::destroy(_radFluxB_Label);
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
RMCRT_Radiation::problemSetup( const ProblemSpecP& inputdb )
{

  _ps = inputdb; 
  _ps->getWithDefault( "calc_frequency",       _radiation_calc_freq, 3 ); 
  _ps->getWithDefault( "calc_on_all_RKsteps",  _all_rk, false );  
  
  _T_label_name = "radiation_temperature";                        // HARDWIRED
  
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
RMCRT_Radiation::extraSetup( GridP& grid, BoundaryCondition* bc, Properties* prop )
{ 

  _boundaryCondition = bc; 

  // determing the temperature label
  _tempLabel = VarLabel::find(_T_label_name); 
  proc0cout << "RMCRT: temperature label name: " << _tempLabel->getName() << endl;

  if ( _tempLabel == 0 ){ 
    throw ProblemSetupException("Error: No temperature label found.",__FILE__,__LINE__); 
  } 

  _abskgLabel = VarLabel::find(_abskg_label_name); 
  if ( _abskgLabel == 0 ){
    throw InvalidValue("Error: For RMCRT Radiation source term -- Could not find the abskg label.", __FILE__, __LINE__);
  }
  
  // create RMCRT and register the labels
  _RMCRT->registerVarLabels(_matl, 
                            _abskgLabel,
                            _tempLabel,
                            _labels->d_cellTypeLabel, 
                            _src_label);

  // read in RMCRT problem spec
  ProblemSpecP rmcrt_ps = _ps->findBlock("RMCRT");
  
  _RMCRT->problemSetup( _ps, rmcrt_ps, grid, _sharedState);
  
//  _RMCRT->BC_bulletproofing( rmcrt_ps );
  
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
  
  
  //__________________________________
  //  carryForward cellType on NON arches level
  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    if( level->getIndex() != _archesLevelIndex ){  
      _RMCRT->sched_CarryForward_Var ( level,  sched, _labels->d_cellTypeLabel );
    }
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
 
    sched_setBoundaryConditions( fineLevel, sched, temp_dw, _radiation_calc_freq );
        
    // coarsen data to the coarser levels.  
    // do it in reverse order
    Task::WhichDW notUsed = Task::OldDW;
    const bool backoutTemp = true;
    
    for (int l = maxLevels - 2; l >= 0; l--) {
      const LevelP& level = grid->getLevel(l);
      const bool modifies_abskg   = false;
      const bool modifies_sigmaT4 = false;
      
      _RMCRT->sched_CoarsenAll( level, sched, modifies_abskg, modifies_sigmaT4, _radiation_calc_freq );
      sched_setBoundaryConditions( level, sched, notUsed, _radiation_calc_freq, backoutTemp );
    }
    
    //__________________________________
    //  compute the extents of the rmcrt region of interest
    //  on the finest level
    _RMCRT->sched_ROI_Extents( fineLevel, sched );

    Task::WhichDW sigmaT4_dw  = Task::NewDW;
    Task::WhichDW celltype_dw = Task::NewDW;
    bool modifies_divQ  = false;
    _RMCRT->sched_rayTrace_dataOnion(fineLevel, sched, abskg_dw, sigmaT4_dw, celltype_dw, modifies_divQ, _radiation_calc_freq);
    
    // convert boundaryFlux<Stencil7> -> 6 doubles
    sched_stencilToDBLs( fineLevel, sched );
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
      
      if( level->hasFinerLevel() ){
        Task::WhichDW sigmaT4_dw  = Task::NewDW;
        Task::WhichDW celltype_dw = Task::NewDW;
        
        sched_setBoundaryConditions( level, sched, temp_dw, _radiation_calc_freq, backoutTemp);
        
        _RMCRT->sched_rayTrace(level, sched, abskg_dw, sigmaT4_dw, celltype_dw, modifies_divQ, _radiation_calc_freq );
      }
    }

    // push divQ  to the coarser levels 
    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      const PatchSet* patches = level->eachPatch();
      _RMCRT->sched_Refine_Q (sched,  patches, _sharedState->allArchesMaterials() , _radiation_calc_freq);
    }
    
    // convert boundaryFlux<Stencil7> -> 6 doubles
    sched_stencilToDBLs( fineLevel, sched );
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
    
    // convert boundaryFlux<Stencil7> -> 6 doubles
    sched_stencilToDBLs( level, sched );
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
    
  for (int l = 0; l < maxLevels; l++) {
    const LevelP& myLevel = grid->getLevel(l);

    int L_ID= myLevel->getIndex();
    ostringstream taskname;
    taskname << "RMCRT_Radiation::sched_initialize_L-" << L_ID;

    Task* tsk = scinew Task( taskname.str(), this, &RMCRT_Radiation::initialize );
    printSchedule( level, dbg, taskname.str() );

    //  only schedule src on arches level
    if( L_ID == _archesLevelIndex ){
      tsk->computes(_src_label);
      tsk->computes(VarLabel::find("radiationVolq"));
    } else {
      tsk->computes( _abskgLabel );
    }
    sched->addTask( tsk, myLevel->eachPatch(), _sharedState->allArchesMaterials() );
  }
  
  //__________________________________
  //  initialize cellType on NON arches level
  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    if( level->getIndex() != _archesLevelIndex ){
      _boundaryCondition->sched_cellTypeInit( sched, level, _sharedState->allArchesMaterials() );
    }
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
  const Level* level = getLevel(patches);

  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing RMCRT_Radiation::initialize");

    if( level->getIndex() == _archesLevelIndex ){
      CCVariable<double> src;
      new_dw->allocateAndPut( src, _src_label, _matl, patch ); 
      src.initialize(0.0); 

      CCVariable<double> radVolq;
      new_dw->allocateAndPut( radVolq,VarLabel::find("radiationVolq"), _matl, patch ); 
      radVolq.initialize(0.0);  // needed for coal
    } else {
      CCVariable<double> abskg;
      new_dw->allocateAndPut( abskg, _abskgLabel, _matl, patch );
      abskg.initialize(0.0);
    }
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


//______________________________________________________________________
//   Set the the boundary conditions for sigmaT4 & abskg.
//______________________________________________________________________
void
RMCRT_Radiation::sched_setBoundaryConditions( const LevelP& level,
                                              SchedulerP& sched,
                                              Task::WhichDW temp_dw,
                                              const int radCalc_freq,
                                              const bool backoutTemp )
{

  std::string taskname = "RMCRT_radiation::setBoundaryConditions";

  Task* tsk = NULL;
  if( _FLT_DBL == TypeDescription::double_type ){

    tsk= scinew Task( taskname, this, &RMCRT_Radiation::setBoundaryConditions< double >,
                      temp_dw, radCalc_freq, backoutTemp );
  } else {
    tsk= scinew Task( taskname, this, &RMCRT_Radiation::setBoundaryConditions< float >,
                      temp_dw, radCalc_freq, backoutTemp );
  }

  printSchedule(level, dbg, "RMCRT_radiation::sched_setBoundaryConditions");

  if(!backoutTemp){
    tsk->requires( temp_dw, _tempLabel, Ghost::None,0 );
  }

  tsk->modifies( _RMCRT->d_sigmaT4Label );
  tsk->modifies( _abskgLabel );

  sched->addTask( tsk, level->eachPatch(), _sharedState->allArchesMaterials() );
}
//______________________________________________________________________

template<class T>
void RMCRT_Radiation::setBoundaryConditions( const ProcessorGroup* pc,
                                             const PatchSubset* patches,
                                             const MaterialSubset*,
                                             DataWarehouse*,
                                             DataWarehouse* new_dw,
                                             Task::WhichDW temp_dw,
                                             const int radCalc_freq,
                                             const bool backoutTemp )
{

  // Only run if it's time
  if ( _RMCRT->doCarryForward( radCalc_freq ) ) {
    return;
  }

  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);

    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    if( bf.size() > 0){

      printTask(patches,patch,dbg,"Doing RMCRT_Radiation::setBoundaryConditions");

      double sigma_over_pi = (_RMCRT->d_sigma)/M_PI;

      CCVariable<double> temp;
      CCVariable< T > abskg;
      CCVariable< T > sigmaT4OverPi;

      new_dw->allocateTemporary(temp,  patch);
      new_dw->getModifiable( abskg,         _abskgLabel,             _matl, patch );
      new_dw->getModifiable( sigmaT4OverPi, _RMCRT->d_sigmaT4Label,  _matl, patch );
      //__________________________________
      // loop over boundary faces and backout the temperature
      // one cell from the boundary.  Note that the temperature
      // is not available on all levels but sigmaT4 is.
      if (backoutTemp){
        for( vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
          Patch::FaceType face = *itr;

          Patch::FaceIteratorType IFC = Patch::InteriorFaceCells;

          for(CellIterator iter=patch->getFaceIterator(face, IFC); !iter.done();iter++) {
            const IntVector& c = *iter;
            double T4 =  sigmaT4OverPi[c]/sigma_over_pi;
            temp[c]   =  pow( T4, 1./4.);
          }
        }
      } else {
        //__________________________________
        // get a copy of the temperature and set the BC
        // on the copy and do not put it back in the DW.
        DataWarehouse* t_dw = new_dw->getOtherDataWarehouse( temp_dw );
        constCCVariable<double> varTmp;
        t_dw->get(varTmp, _tempLabel,   _matl, patch, Ghost::None, 0);
        temp.copyData(varTmp);
      }


      //__________________________________
      // set the boundary conditions
//      setBC< T, double >  (abskg,    d_abskgBC_tag,               patch, d_matl);
//      setBC<double,double>(temp,     d_compTempLabel->getName(),  patch, d_matl);

      string comp_abskg = _abskgLabel->getName();
      string comp_Temp =  _tempLabel->getName();

      BoundaryCondition_new* new_BC = _boundaryCondition->getNewBoundaryCondition();
      new_BC->setExtraCellScalarValueBC< T >(      pc, patch, abskg, comp_abskg );
      new_BC->setExtraCellScalarValueBC< double >( pc, patch, temp,  comp_Temp );

      //__________________________________
      // loop over boundary faces and compute sigma T^4
      for( vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
        Patch::FaceType face = *itr;

        Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;

        for(CellIterator iter=patch->getFaceIterator(face, PEC); !iter.done();iter++) {
          const IntVector& c = *iter;
          double T_sqrd = temp[c] * temp[c];
          sigmaT4OverPi[c] = sigma_over_pi * T_sqrd * T_sqrd;
        }
      }
    } // has a boundaryFace
  }
}
//______________________________________________________________________
// Explicit template instantiations:

template 
void RMCRT_Radiation::setBoundaryConditions< double >( const ProcessorGroup*,
                                                       const PatchSubset* ,
                                                       const MaterialSubset*,
                                                       DataWarehouse*,
                                                       DataWarehouse* ,
                                                       Task::WhichDW ,
                                                       const int ,
                                                       const bool );

template 
void RMCRT_Radiation::setBoundaryConditions< float >( const ProcessorGroup*,
                                                      const PatchSubset* ,
                                                      const MaterialSubset*,
                                                      DataWarehouse*,
                                                      DataWarehouse* ,
                                                      Task::WhichDW ,
                                                      const int ,
                                                      const bool );
                                                      
//______________________________________________________________________
//
//______________________________________________________________________
void
RMCRT_Radiation::sched_stencilToDBLs( const LevelP& level, 
                                      SchedulerP& sched )
{

  if( level->getID() != _archesLevelIndex){
    throw InternalError("RMCRT_Radiation::sched_stencilToDBLs.  You cannot schedule this task on a non-arches level", __FILE__, __LINE__);
  }

  if(_RMCRT->d_solveBoundaryFlux) {
    Task* tsk = scinew Task( "RMCRT_Radiation::stencilToDBLs", this, &RMCRT_Radiation::stencilToDBLs );
    printSchedule( level, dbg, "RMCRT_Radiation::sched_stencilToDBLs" );

    //  only schedule task on arches level
    tsk->requires(Task::NewDW, VarLabel::find("RMCRTboundFlux"), _gn, 0);

    tsk->computes( _radFluxE_Label );
    tsk->computes( _radFluxW_Label );
    tsk->computes( _radFluxN_Label );
    tsk->computes( _radFluxS_Label );
    tsk->computes( _radFluxT_Label );
    tsk->computes( _radFluxB_Label );
    sched->addTask( tsk, level->eachPatch(), _sharedState->allArchesMaterials() );
  }
}

//______________________________________________________________________
//
void 
RMCRT_Radiation::stencilToDBLs( const ProcessorGroup*,
                             const PatchSubset* patches, 
                             const MaterialSubset*, 
                             DataWarehouse* , 
                             DataWarehouse* new_dw )
{
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing RMCRT_Radiation::stencilToDBLs");

    constCCVariable<Stencil7>  boundaryFlux;
    new_dw->get( boundaryFlux,     VarLabel::find("RMCRTboundFlux"), _matl, patch, _gn, 0 ); 

    CCVariable<double> East, West;
    CCVariable<double> North, South;
    CCVariable<double> Top, Bot;
    new_dw->allocateAndPut( East,  _radFluxE_Label, _matl, patch );
    new_dw->allocateAndPut( West,  _radFluxW_Label, _matl, patch );
    new_dw->allocateAndPut( North, _radFluxN_Label, _matl, patch );
    new_dw->allocateAndPut( South, _radFluxS_Label, _matl, patch );
    new_dw->allocateAndPut( Top,   _radFluxT_Label, _matl, patch );
    new_dw->allocateAndPut( Bot,   _radFluxB_Label, _matl, patch );

    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      const Stencil7& me = boundaryFlux[c];
      East[c]  = me.e;
      West[c]  = me.w;
      North[c] = me.n;         // THIS MAPPING MUST BE VERIFIED
      South[c] = me.s;
      Top[c]   = me.t;
      Bot[c]   = me.b;
    }
  }
}
