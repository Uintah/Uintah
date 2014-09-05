#include <CCA/Components/Arches/SourceTerms/RMCRT.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>


using namespace std;
using namespace Uintah; 
static DebugStream dbg("RMCRT", false);

/*______________________________________________________________________
          TO DO:
          
  - fix coarsen operator      
  - pull in _archesLevelIndex from arches, don't compute it locally
  
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
  _label_sched_init = false; 
  
  const TypeDescription* CC_double = CCVariable<double>::getTypeDescription();
  _src_label      = VarLabel::create( src_name,  CC_double ); 
  _sigmaT4Label   = VarLabel::create("sigmaT4",  CC_double );
  _abskgLabel     = VarLabel::create( "abskg",   CC_double );
  _absorpLabel    = VarLabel::create( "absorp",  CC_double );
  _abskpLabel     = VarLabel::create( "abskp",   CC_double ); 
  _cellTypeLabel  = _labels->d_cellTypeLabel; 
  
  //Declare the source type: 
  _source_grid_type = CC_SRC; // or FX_SRC, or FY_SRC, or FZ_SRC, or CCVECTOR_SRC

//  _archesLevelIndex      = -9;
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
  //for particles: 
  _ps->getWithDefault( "abskp_label", _abskp_label_name, "abskp" ); 
  _ps->getWithDefault( "psize_label", _size_label_name, "length");
  _ps->getWithDefault( "ptemperature_label", _pT_label_name, "temperature"); 

  //get the number of quadrature nodes and store it locally 
  _nQn_part = 0;
  if ( _ps->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM") ){
    _ps->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM")->require( "number_quad_nodes", _nQn_part ); 
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
  _tempLabel = _labels->getVarlabelByRole(ArchesLabel::TEMPERATURE);
  proc0cout << "RMCRT: temperature label name: " << _tempLabel->getName() << endl;

  if ( _tempLabel == 0 ){ 
    throw ProblemSetupException("Error: No temperature label found.",__FILE__,__LINE__); 
  } 
  
#ifdef HAVE_CUDA
  _RMCRT = scinew Ray(_sharedState->getUnifiedScheduler());
#else
  _RMCRT = scinew Ray();
#endif

  _RMCRT->registerVarLabels(_matl, 
                            _abskgLabel,
                            _absorpLabel,
                            _tempLabel,
                            _cellTypeLabel, 
                            _src_label);

  ProblemSpecP rmcrt_ps = _ps->findBlock("RMCRT");
  _RMCRT->problemSetup( _ps, rmcrt_ps, _sharedState);
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
  // HACK This should be pulled in from arches, not computed here.
  GridP grid = level->getGrid();
  int archesLevelIndex = grid->numLevels()-1; // this is the finest level

  // only sched on RK step 0 and on arches level
  if ( timeSubStep != 0  || level->getIndex() != archesLevelIndex) {  
    return;
  } 

  int maxLevels = grid->numLevels();
  
  //__________________________________ 
  // move data on non-arches level to the new_dw for simplicity
  // do this on all timesteps
  for (int L = 0; L < maxLevels; L++) {
    if( L != archesLevelIndex ){
      const LevelP& level = grid->getLevel(L);
      _RMCRT->sched_CarryForward ( level, sched, _cellTypeLabel );
    }
  }

  dbg << " ---------------timeSubStep: " << timeSubStep << endl;
  printSchedule(level,dbg,"RMCRT_Radiation::sched_computeSource");


  // common flags
  bool modifies_divQ     = false;
  const bool includeExtraCells = false;  // domain for sigmaT4 computation

  if (timeSubStep == 0 && !_label_sched_init) {
    modifies_divQ  = false;
  } else {
    modifies_divQ  = true;
  }
  
  //______________________________________________________________________
  //   D A T A   O N I O N   A P P R O A C H
  if( _whichAlgo == dataOnion ){
    const LevelP& fineLevel = grid->getLevel(archesLevelIndex);
    Task::WhichDW temp_dw = Task::OldDW;
    
    // modify Radiative properties on the finest level
    // compute Radiative properties and sigmaT4 on the finest level
    sched_radProperties( fineLevel, sched, timeSubStep );
    
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
    
    Task::WhichDW abskg_dw     = Task::NewDW;
    Task::WhichDW sigmaT4_dw   = Task::NewDW;
    bool modifies_divQ       = false;
    _RMCRT->sched_rayTrace_dataOnion(fineLevel, sched, abskg_dw, sigmaT4_dw, modifies_divQ, _radiation_calc_freq);
  }
  
  //______________________________________________________________________
  //   2 - L E V E L   A P P R O A C H
  //  RMCRT is performed on the coarse level
  // and the results are interpolated to the fine (arches) level
  if( _whichAlgo == coarseLevel ){
    const LevelP& fineLevel = grid->getLevel(archesLevelIndex);
    Task::WhichDW temp_dw = Task::OldDW;
   
    // compute Radiative properties and sigmaT4 on the finest level
    sched_radProperties( fineLevel, sched, timeSubStep );
    
    _RMCRT->sched_sigmaT4( fineLevel,  sched, temp_dw, _radiation_calc_freq, includeExtraCells );
    
    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      const bool modifies_abskg   = false;
      const bool modifies_sigmaT4 = false;
      const bool backoutTemp      = true;
      
      _RMCRT->sched_CoarsenAll (level, sched, modifies_abskg, modifies_sigmaT4, _radiation_calc_freq);
      
      if(level->hasFinerLevel() || maxLevels == 1){
        Task::WhichDW abskg_dw    = Task::NewDW;
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
  std::vector<std::string> part_sp = _prop_calculator->get_participating_sp(); 

  if ( time_sub_step == 0 ) { 
    tsk->computes( _abskgLabel );
    tsk->computes( _abskpLabel ); 
    tsk->requires( Task::OldDW, _tempLabel, Ghost::None, 0 ); 
    //gas
    for ( std::vector<std::string>::iterator iter = part_sp.begin(); iter != part_sp.end(); iter++){

      const VarLabel* label = VarLabel::find(*iter);
      _species_varlabels.push_back(label); 

      if ( label != 0 ){ 
        tsk->requires( Task::OldDW, label, Ghost::None, 0 ); 
      } else { 
        throw ProblemSetupException("Error: Could not match species with varlabel: "+*iter,__FILE__, __LINE__);
      }
    }
    //particles
    for ( int i = 0; i < _nQn_part; i++ ){ 

      //--size--
      std::string label_name = _size_label_name + "_qn"; 
      std::stringstream out; 
      out << i; 
      label_name += out.str(); 

      const VarLabel* sizelabel = VarLabel::find( label_name ); 
      _size_varlabels.push_back( sizelabel ); 

      if ( sizelabel != 0 ){ 
        tsk->requires( Task::OldDW, sizelabel, Ghost::None, 0 ); 
      } else { 
        throw ProblemSetupException("Error: Could not find particle size quadrature node: " + label_name, __FILE__, __LINE__);
      }

      //--temperature--
      label_name = _pT_label_name + "_qn"; 
      label_name += out.str(); 

      const VarLabel* tlabel = VarLabel::find( label_name ); 
      _T_varlabels.push_back( tlabel ); 

      if ( tlabel != 0 ){ 
        tsk->requires( Task::OldDW, tlabel, Ghost::None, 0 ); 
      } else { 
        throw ProblemSetupException("Error: Could not find particle temperature quadrature node: " + label_name , __FILE__, __LINE__);
      }

      //--weight--
      label_name = "w_qn"+out.str(); 
      const VarLabel* wlabel = VarLabel::find( label_name ); 
      _w_varlabels.push_back( wlabel ); 

      if ( wlabel != 0 ){ 
        tsk->requires( Task::OldDW, wlabel, Ghost::None, 0 ); 
      } else { 
        throw ProblemSetupException("Error: Could not find particle weight quadrature node: w_qn"+out.str() , __FILE__, __LINE__);
      }
    } 
  } else {  
    tsk->modifies( _abskgLabel );
    tsk->modifies( _abskpLabel ); 
    tsk->requires( Task::NewDW, _tempLabel, Ghost::None, 0 ); 

    for ( std::vector<const VarLabel*>::iterator iter = _species_varlabels.begin();  iter != _species_varlabels.end(); iter++ ){ 
      tsk->requires( Task::NewDW, *iter, Ghost::None, 0 ); 
    } 
    for ( int i = 0; i < _nQn_part; i++ ){ 

      //--size--
      tsk->requires( Task::NewDW, _size_varlabels[i], Ghost::None, 0 ); 

      //--temperature--
      tsk->requires( Task::NewDW, _T_varlabels[i], Ghost::None, 0 ); 

      //--weight--
      tsk->requires( Task::NewDW, _w_varlabels[i], Ghost::None, 0 ); 

    } 
  }
  
  sched->addTask( tsk, level->eachPatch(), _matlSet ); 
}

//______________________________________________________________________
//
void
RMCRT_Radiation::radProperties( const ProcessorGroup* ,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw, 
                                const int time_sub_step )
{
  for (int p=0; p < patches->size(); p++){

    std::vector<constCCVariable<double> > species; 
    constCCVariable<double> gas_temperature; 

    const Patch* patch = patches->get(p);

    printTask(patches,patch,dbg,"Doing RMCRT_Radiation::radProperties");

    DataWarehouse* which_dw; 
    CCVariable<double> abskg; 
    CCVariable<double> abskp; 
    if ( time_sub_step == 0 ) { 
      new_dw->allocateAndPut( abskg, _abskgLabel, _matl, patch ); 
      new_dw->allocateAndPut( abskp, _abskpLabel, _matl, patch ); 
      which_dw = old_dw; 
    } else { 
      new_dw->getModifiable( abskg,  _abskgLabel,  _matl, patch );
      new_dw->getModifiable( abskp,  _abskpLabel,  _matl, patch );
      which_dw = new_dw; 
    }

    which_dw->get( gas_temperature, _tempLabel, _matl, patch, Ghost::None, 0 ); 

    typedef std::vector<constCCVariable<double> > CCCV; 
    typedef std::vector<const VarLabel*> CCCVL; 

    CCCV weights; 
    CCCV size;
    CCCV pT; 

    double weights_scaling_constant;
    double size_scaling_constant;
    DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();
    string tlabelname;

    //--size--
    for ( CCCVL::iterator iter = _size_varlabels.begin(); iter != _size_varlabels.end(); iter++ ){ 
      constCCVariable<double> var; 
      which_dw->get( var, *iter, _matl, patch, Ghost::None, 0 ); 
      size.push_back( var ); 
      //to get size scaling constant
      if(iter == _size_varlabels.begin()){
        tlabelname = (*iter)->getName();
        size_scaling_constant = dqmom_eqn_factory.retrieve_scalar_eqn(tlabelname).getScalingConstant();
      } 

    } 

    //--temperature--
    for ( CCCVL::iterator iter = _T_varlabels.begin(); iter != _T_varlabels.end(); iter++ ){ 
      constCCVariable<double> var; 
      which_dw->get( var, *iter, _matl, patch, Ghost::None, 0 ); 
      pT.push_back( var ); 
    } 

    //--weight--
    for ( CCCVL::iterator iter = _w_varlabels.begin(); iter != _w_varlabels.end(); iter++ ){ 
      constCCVariable<double> var; 
      which_dw->get( var, *iter, _matl, patch, Ghost::None, 0 ); 
      weights.push_back( var ); 
      //to get weight scaling constant
      if(iter == _w_varlabels.begin()){
        tlabelname = (*iter)->getName();
        weights_scaling_constant = dqmom_eqn_factory.retrieve_scalar_eqn(tlabelname).getScalingConstant();
      } 

    } 

    //--participating species--
    for ( std::vector<const VarLabel*>::iterator iter = _species_varlabels.begin();  iter != _species_varlabels.end(); iter++ ){ 
      constCCVariable<double> var; 
      which_dw->get( var, *iter, _matl, patch, Ghost::None, 0 ); 
      species.push_back( var ); 
    }

    // compute absorption (gas and particle) coefficient(s) via RadPropertyCalulator
    if ( _prop_calculator->does_scattering() ){
      _prop_calculator->compute( patch, species, size_scaling_constant, size, pT, weights_scaling_constant, weights, _nQn_part, gas_temperature, abskg, abskp ); 
    } else { 
      _prop_calculator->compute( patch, species, gas_temperature, abskg );
    } 
    
    // abskg boundary conditions are set in setBoundaryCondition()
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
  // HACK archesLevelIndex should be pulled in from arches, not computed here.
  GridP grid = level->getGrid();
  int maxLevels = grid->numLevels();
  int archesLevelIndex = maxLevels-1; // this is the index of the finest level  

  //__________________________________
  //  Additional bulletproofing, this belongs in problem setup
  if (_whichAlgo == dataOnion && maxLevels == 1){
    throw ProblemSetupException("ERROR:  RMCRT_radiation, there must be more than 1 level if you're using the Data Onion algorithm", __FILE__, __LINE__);
  }  
  
  //__________________________________
  //  schedule the tasks
  for (int L=0; L< maxLevels; ++L){
  
    if( L != archesLevelIndex ){
    
      string taskname = "RMCRT_Radiation::sched_initialize"; 
      Task* tsk = scinew Task(taskname, this, &RMCRT_Radiation::initialize);

      LevelP level = grid->getLevel(L);
      printSchedule(level,dbg,taskname);
      
      tsk->computes( _cellTypeLabel );
      sched->addTask(tsk, level->eachPatch(), _matlSet);
    }

  // THIS IS THE RIGHT WAY TO INITIALIZE cellType
  // The problem is _bc is not defined at this point in Arches::problemSetup
  #if 0 
    //__________________________________
    // cellType initialization
    const PatchSet* patches = level->eachPatch();
    if ( _bc->isUsingNewBC() ) {
      _bc->sched_cellTypeInit__NEW( sched, patches, _matlSet );
    } else {
      _bc->sched_cellTypeInit(sched, patches, _matlSet);
    }
   #endif
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
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing RMCRT_Radiation::initialize");


    CCVariable<int> cellType;        // HACK UNTIL WE KNOW WHAT TO DO
    new_dw->allocateAndPut( cellType,    _cellTypeLabel,    _matl, patch );
    cellType.initialize( 0 ); 
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
