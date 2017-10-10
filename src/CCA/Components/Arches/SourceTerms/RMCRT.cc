/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/Arches/SourceTerms/RMCRT.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Util/DOUT.hpp>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>


using namespace Uintah;

Dout dbg("RMCRT", false);

/*______________________________________________________________________
          TO DO:
 ______________________________________________________________________*/

RMCRT_Radiation::RMCRT_Radiation( std::string src_name,
                                  ArchesLabel* labels,
                                  MPMArchesLabel* MAlab,
                                  std::vector<std::string> req_label_names,
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

  //Declare the source type:
  _source_grid_type = CC_SRC; // or FX_SRC, or FY_SRC, or FZ_SRC, or CCVECTOR_SRC
  _sharedState      = labels->d_sharedState;

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
  _ps->getWithDefault( "calc_on_all_RKsteps",  _all_rk, false );

  _T_label_name = "radiation_temperature";                        // HARDWIRED

  if ( _ps->findBlock("abskt")){
    _ps->findBlock("abskt")->getAttribute("label", _abskt_label_name);
  } else {
    throw ProblemSetupException("Error: RMCRT - The total absorption coefficient is not defined.",__FILE__,__LINE__);
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

  // Are we using floats for all-to-all variables
  std::map<std::string, std::string> type;
  rmcrt_ps->getAttributes(type);

  std::string isFloat = type["type"];

  if( isFloat == "float" ){
    _FLT_DBL = TypeDescription::float_type;
  }

  _RMCRT = scinew Ray( _FLT_DBL );


  _RMCRT->setBC_onOff( false );

  //__________________________________
  //  Read in the RMCRT algorithm that will be used
  ProblemSpecP alg_ps = rmcrt_ps->findBlock("algorithm");
  if (alg_ps){

    std::string type="nullptr";
    alg_ps->getAttribute("type", type);

    if (type == "dataOnion" ) {                   // DATA ONION

      _whichAlgo = dataOnion;
      _RMCRT->setBC_onOff( true );

      //__________________________________
      //  bulletproofing
      if(!_sharedState->isLockstepAMR()){
        std::ostringstream msg;
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
#if 0
 //Todd and Derek need to confirm if this should be used May 19th 2007
  std::string baseNameAbskp;
  std::string modelName;
  std::string baseNameTemperature;
  _radiateAtGasTemp=true; // this flag is arbitrary for no particles
  ProblemSpecP db_propV2 = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("PropertyModelsV2");
  if  (db_propV2){
    for ( ProblemSpecP db_model = db_propV2->findBlock("model"); db_model != nullptr;
        db_model = db_model->findNextBlock("model")){
      db_model->getAttribute("type", modelName);

      if (modelName=="partRadProperties"){
        bool doing_dqmom = ParticleTools::check_for_particle_method(db,ParticleTools::DQMOM);
        bool doing_cqmom = ParticleTools::check_for_particle_method(db,ParticleTools::CQMOM);

        if ( doing_dqmom ){
          _nQn_part = ParticleTools::get_num_env( db, ParticleTools::DQMOM );
        } else if ( doing_cqmom ){
          _nQn_part = ParticleTools::get_num_env( db, ParticleTools::CQMOM );
        } else {
          throw ProblemSetupException("Error: This method only working for DQMOM/CQMOM.",__FILE__,__LINE__);
        }

        db_model->getWithDefault( "part_temp_label", baseNameTemperature, "heat_pT" );
        db_model->getWithDefault( "radiateAtGasTemp", _radiateAtGasTemp, true );
        db_model->getAttribute("label",baseNameAbskp);
        //  db_model->findBlock("calculator")->findBlock("abskg")->getAttribute("label",_abskg_label_name);
        break;
      }
    }
  } else{
    _nQn_part =0; // No property model found, so particles do not interact radiatively
  }


    for (int qn=0; qn < _nQn_part; qn++){
      std::stringstream absorp;
      std::stringstream temper;
      absorp <<baseNameAbskp <<"_"<< qn;
      temper <<baseNameTemperature <<"_"<< qn;
      _absk_name_vector.push_back( absorp.str());
      _temperature_name_vector.push_back( temper.str());
    }

// get gas-only absorption coefficient

  std::string modelName2;
  ProblemSpecP db_prop2 = _ps->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("PropertyModels");
  if  (db_prop2){
    for ( ProblemSpecP db_model = db_prop2->findBlock("model"); db_model != 0;
        db_model = db_model->findNextBlock("model")){
      db_model->getAttribute("type", modelName2);
      if (modelName2=="radiation_properties"){
        db_model->getAttribute("label",_abskg_label_name);
        db_model->findBlock("calculator")->findBlock("abskg")->getAttribute("label",_abskg_label_name);
      }
    }
  }else{
    proc0cout << " **WARNING**: Couldn't find property model (old interface), using user specified absorption coefficient.    \n";
      _ps->findBlock("abskg")->getAttribute("label", _abskg_label_name);
     _abskg_label_name=_abskg_label_name;
  }
#endif
}

//______________________________________________________________________
//  We need this additional call to problemSetup
//  so the reaction models can create the  VarLabel
//______________________________________________________________________
void
RMCRT_Radiation::extraSetup( GridP& grid, 
                             BoundaryCondition* bc, 
                             TableLookup* table_lookup )
{

  _boundaryCondition = bc;

  // determining the temperature label
  _tempLabel = VarLabel::find(_T_label_name);
  proc0cout << "RMCRT: temperature label name: " << _tempLabel->getName() << std::endl;

  if (_tempLabel == nullptr) {
    throw ProblemSetupException("Error: No temperature label found.", __FILE__, __LINE__);
  }

  _absktLabel = VarLabel::find(_abskt_label_name);
  if ( _absktLabel == nullptr ){
    throw InvalidValue("Error: For RMCRT Radiation source term -- Could not find the abskt label.", __FILE__, __LINE__);
  }

  // create RMCRT and register the labels
  _RMCRT->registerVarLabels(_matl,
                            _absktLabel,
                            _tempLabel,
                            _labels->d_cellTypeLabel,
                            _src_label);

  // read in RMCRT problem spec
  ProblemSpecP rmcrt_ps = _ps->findBlock("RMCRT");

  _RMCRT->problemSetup(_ps, rmcrt_ps, grid, _sharedState);

//  _RMCRT->BC_bulletproofing( rmcrt_ps );

  //__________________________________
  //  Bulletproofing:
  // dx must get smaller as level-index increases
  // Arches is always computed on the finest level
  int maxLevels = grid->numLevels();
  _archesLevelIndex = maxLevels - 1;

  if (maxLevels > 1) {
    Vector dx_prev = grid->getLevel(0)->dCell();

    for (int L = 1; L < maxLevels; L++) {
      Vector dx = grid->getLevel(L)->dCell();

      Vector ratio = dx / dx_prev;
      if (ratio.x() > 1 || ratio.y() > 1 || ratio.z() > 1) {
        std::ostringstream warn;
        warn << "RMCRT: ERROR Level-" << L << " cell spacing is not smaller than Level-" << L - 1 << ratio;
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
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
#if 0
//----------------gas-only stuff-----------//
  _abskgLabel = VarLabel::find(_abskg_label_name);
  if (_abskgLabel == nullptr ){
    throw InvalidValue("Error: For DO Radiation source term -- Could not find the abskg label.", __FILE__, __LINE__);
  }
//-----------------------------------------//
 // particle stuff
    _absk_label_vector.push_back(_abskgLabel);
    _temperature_label_vector.push_back(VarLabel::find(_T_label_name));

  //_tempLabel = VarLabel::find(_T_label_name);
  //_abskgLabel = VarLabel::find(_abskg_label_name);

  for (int qn=0; qn < _nQn_part; qn++){
    _absk_label_vector.push_back(VarLabel::find(_absk_name_vector[qn]));
    if (_absk_label_vector[qn]==0){
      throw ProblemSetupException("Error: particle absorption coefficient node not found."+_absk_name_vector[qn], __FILE__, __LINE__);
    }

    _temperature_label_vector.push_back(VarLabel::find(_temperature_name_vector[qn]));

    if (_temperature_label_vector[qn]==0) {
      throw ProblemSetupException("Error: particle temperature node not found! "+_temperature_name_vector[qn], __FILE__, __LINE__);
    }
  }
//----------------end particle stuff-----------//
#endif

  GridP grid = level->getGrid();

  // only sched on RK step 0 and on arches level
  if (timeSubStep != 0 || level->getIndex() != _archesLevelIndex) {
    return;
  }

  int maxLevels = grid->numLevels();

  DOUT( dbg ," ---------------timeSubStep: " << timeSubStep );
  printSchedule(level, dbg, "RMCRT_Radiation::sched_computeSource");

  // common flags
  bool modifies_divQ     = false;
  bool includeExtraCells = true;  // domain for sigmaT4 computation

  if (timeSubStep == 0) {
    modifies_divQ = false;
  }
  else {
    modifies_divQ = true;
  }

  //__________________________________
  //  carryForward cellType on NON arches level
  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    if (level->getIndex() != _archesLevelIndex) {
      _RMCRT->sched_CarryForward_Var(level, sched, _labels->d_cellTypeLabel);
    }
  }

  Task::WhichDW notUsed = Task::None;
  //______________________________________________________________________
  //   D A T A   O N I O N   A P P R O A C H
  if (_whichAlgo == dataOnion) {

    Task::WhichDW temp_dw       = Task::OldDW;
    Task::WhichDW sigmaT4_dw    = Task::NewDW;
    Task::WhichDW celltype_dw   = Task::NewDW;
    const bool backoutTemp      = true;
    const bool modifies_abskg   = false;
    const bool modifies_sigmaT4 = false;

    const LevelP& fineLevel = grid->getLevel(_archesLevelIndex);

    // define per level which abskg dw
    _RMCRT->set_abskg_dw_perLevel( fineLevel, Task::OldDW );

    // modify Radiative properties on the finest level
    // convert abskg:dbl -> abskg:flt if needed
    _RMCRT->sched_DoubleToFloat( fineLevel, sched, notUsed );

    // compute sigmaT4 on the finest level
    _RMCRT->sched_sigmaT4( fineLevel, sched, temp_dw, includeExtraCells );

    // carry forward if it's time
    _RMCRT->sched_CarryForward_FineLevelLabels ( fineLevel, sched );

    // coarse levels
    for (int l = 0; l < maxLevels-1; ++l) {
      const LevelP& level = grid->getLevel(l);
      _RMCRT->sched_CarryForward_Var ( level,  sched, _RMCRT->d_abskgLabel,   RMCRT_Radiation::TG_CARRY_FORWARD );
      _RMCRT->sched_CarryForward_Var ( level,  sched, _RMCRT->d_sigmaT4Label, RMCRT_Radiation::TG_CARRY_FORWARD );
    }

    // coarsen data to the coarser levels.
    // do it in reverse order
    for (int l = maxLevels - 2; l >= 0; l--) {
      const LevelP& level = grid->getLevel(l);

      _RMCRT->sched_CoarsenAll( level, sched, modifies_abskg, modifies_sigmaT4 );

      if( _RMCRT->d_coarsenExtraCells == false ) {
        sched_setBoundaryConditions( level, sched, notUsed, backoutTemp );
      }
    }

    //__________________________________
    //  compute the extents of the RMCRT region of interest on the finest level
    _RMCRT->sched_ROI_Extents( fineLevel, sched );

    _RMCRT->sched_rayTrace_dataOnion( fineLevel, sched, notUsed, sigmaT4_dw, celltype_dw, modifies_divQ );

    // convert boundaryFlux<Stencil7> -> 6 doubles
    sched_stencilToDBLs( fineLevel, sched );
  }

  //______________________________________________________________________
  //   2 - L E V E L   A P P R O A C H
  //  RMCRT is performed on the coarse level
  //  and the results are interpolated to the fine (arches) level
  if (_whichAlgo == coarseLevel) {

    Task::WhichDW temp_dw       = Task::OldDW;
    Task::WhichDW sigmaT4_dw    = Task::NewDW;
    Task::WhichDW celltype_dw   = Task::NewDW;
    const bool modifies_abskg   = false;
    const bool modifies_sigmaT4 = false;
    const bool backoutTemp      = true;

    // carry forward if it's time
    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);

      _RMCRT->sched_CarryForward_FineLevelLabels ( level, sched );

      // coarse levels
      if( level->hasFinerLevel() ){
        _RMCRT->sched_CarryForward_Var ( level, sched, _RMCRT->d_abskgLabel, RMCRT_Radiation::TG_CARRY_FORWARD );
      }
    }

    const LevelP& fineLevel = grid->getLevel(_archesLevelIndex);

    _RMCRT->set_abskg_dw_perLevel ( fineLevel, Task::OldDW );

    // convert abskg:dbl -> abskg:flt if needed
    _RMCRT->sched_DoubleToFloat( fineLevel, sched, notUsed );

    // compute sigmaT4 on the finest level
    _RMCRT->sched_sigmaT4( fineLevel, sched, temp_dw, includeExtraCells );

    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);

      _RMCRT->sched_CoarsenAll( level, sched, modifies_abskg, modifies_sigmaT4 );

      if (level->hasFinerLevel()) {
        if( _RMCRT->d_coarsenExtraCells == false ) {
          sched_setBoundaryConditions( level, sched, temp_dw, backoutTemp );
        }

        _RMCRT->sched_rayTrace( level, sched, notUsed, sigmaT4_dw, celltype_dw, modifies_divQ );
      }
    }

    // push divQ  to the coarser levels
    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      const PatchSet* patches = level->eachPatch();
      _RMCRT->sched_Refine_Q( sched, patches, _sharedState->allArchesMaterials() );
    }

    // convert boundaryFlux<Stencil7> -> 6 doubles
    sched_stencilToDBLs( fineLevel, sched );
  }

  //______________________________________________________________________
  //   1 - L E V E L   A P P R O A C H
  //  RMCRT is performed on the same level as CFD
  if (_whichAlgo == singleLevel) {

    Task::WhichDW temp_dw     = Task::OldDW;
    Task::WhichDW sigmaT4_dw  = Task::NewDW;
    Task::WhichDW celltype_dw = Task::NewDW;

    const LevelP& level = grid->getLevel(_archesLevelIndex);

    _RMCRT->set_abskg_dw_perLevel( level, Task::OldDW );

    // carry forward if it's time
    _RMCRT->sched_CarryForward_FineLevelLabels( level, sched );

    // convert abskg:dbl -> abskg:flt if needed
    _RMCRT->sched_DoubleToFloat( level, sched, notUsed );

    // compute sigmaT4 on the CFD level
    _RMCRT->sched_sigmaT4( level, sched, temp_dw, includeExtraCells );

    _RMCRT->sched_rayTrace( level, sched, notUsed, sigmaT4_dw, celltype_dw, modifies_divQ );

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

    int L_index= myLevel->getIndex();
    std::ostringstream taskname;
    taskname << "RMCRT_Radiation::sched_initialize_L-" << L_index;

    Task* tsk = scinew Task( taskname.str(), this, &RMCRT_Radiation::initialize );
    printSchedule( level, dbg, taskname.str() );

    // all levels
    tsk->computes(VarLabel::find("radiationVolq"));
    tsk->computes(VarLabel::find("RMCRTboundFlux"));
    tsk->computes(_RMCRT->d_sigmaT4Label);

    // only cfd level
    if ( L_index == _archesLevelIndex) {
      tsk->computes( _src_label );
    }

    // coarse levels
    if ( L_index != _archesLevelIndex) {
      tsk->computes( _RMCRT->d_abskgLabel );  // abskt or abskgRMCRT
      
      // divQ computed on all levels
      if (_whichAlgo == coarseLevel) {
        tsk->computes(_src_label);
      }
    }
    sched->addTask( tsk, myLevel->eachPatch(), _sharedState->allArchesMaterials() );
  }

  //__________________________________
  //  initialize cellType on NON arches level
  for (int l = maxLevels - 1; l >= 0; l--) {
    const LevelP& level = grid->getLevel(l);
    if( level->getIndex() != _archesLevelIndex ){
      // Set the BC on the coarse level
      _boundaryCondition->sched_cellTypeInit( sched, level, _sharedState->allArchesMaterials() );

      // Coarsen the interior cells
       _RMCRT->sched_computeCellType ( level, sched, Ray::modifiesVar);
    }
  }

  sched_fluxInit( level, sched );
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
  const int L_index  = level->getIndex();

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    printTask(patches, patch, dbg, "Doing RMCRT_Radiation::initialize");

    CCVariable<double> abskg;
    CCVariable<double> radVolq;
    CCVariable<double> sigmaT4;
    CCVariable<double> src;
    CCVariable<Stencil7> RMCRTboundFlux ;

    //__________________________________
    // all levels
    new_dw->allocateAndPut( radVolq, VarLabel::find("radiationVolq"), _matl, patch );
    radVolq.initialize(0.0);  // needed for coal

    new_dw->allocateAndPut( sigmaT4, _RMCRT->d_sigmaT4Label, _matl, patch );
    sigmaT4.initialize(0.0);

    new_dw->allocateAndPut( RMCRTboundFlux, VarLabel::find("RMCRTboundFlux"),_matl, patch );
    Uintah::BlockRange range( patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range,[&](int i, int j, int k){
      RMCRTboundFlux(i,j,k).initialize(0.0);
    });

    //__________________________________
    //  CFD level
    if ( L_index == _archesLevelIndex) {
      new_dw->allocateAndPut( src, _src_label, _matl, patch );
      src.initialize(0.0);
    }
    
    //__________________________________
    //  Coarse levels
    if ( L_index != _archesLevelIndex) {
      new_dw->allocateAndPut( abskg, _RMCRT->d_abskgLabel, _matl, patch );  // could be abskt or abskgRMCRT
      abskg.initialize(0.0);

      // divQ computed on all levels
      if (_whichAlgo == coarseLevel) {
        new_dw->allocateAndPut( src, _src_label, _matl, patch );
        src.initialize(0.0);
      }
    }
  }
}

//______________________________________________________________________
// Method: Schedule initialization
// This will only be called on the Archeslevel
//______________________________________________________________________
void
RMCRT_Radiation::sched_restartInitialize( const LevelP& level,
                                           SchedulerP& sched )
{
  GridP grid = level->getGrid();

  DataWarehouse* new_dw = sched->getLastDW();

  const LevelP& archesLevel = grid->getLevel(_archesLevelIndex);

  // Find the first patch, on the arches level, that this mpi rank owns.
  const Uintah::PatchSet* const ps = sched->getLoadBalancer()->getPerProcessorPatchSet(archesLevel);
  const PatchSubset* myPatches = ps->getSubset(_my_world->myrank());
  const Patch* firstPatch = myPatches->get(0);

  if (level == archesLevel) {
    printSchedule(level, dbg, "RMCRT_Radiation::sched_restartInitialize");
    //__________________________________
    //  As the name implies this is a hack
    Task* t1 = scinew Task("RMCRT_Radiation::restartInitializeHack", this, &RMCRT_Radiation::restartInitializeHack);

    //  Only schedule if radFlux*_Label are in the checkpoint uda
    if (new_dw->exists(_radFluxE_Label, _matl, firstPatch)) {
      printSchedule(level, dbg, "RMCRT_Radiation::sched_restartInitializeHack");

      t1->computes(_radFluxE_Label);
      t1->computes(_radFluxW_Label);
      t1->computes(_radFluxN_Label);   // Before you can require something from the new_dw
      t1->computes(_radFluxS_Label);   // there must be a compute() for that variable.
      t1->computes(_radFluxT_Label);
      t1->computes(_radFluxB_Label);
    }
    t1->computes(_tempLabel);          // needed by sched_sigmaT4
    sched->addTask(t1, archesLevel->eachPatch(), _sharedState->allArchesMaterials());

    //__________________________________
    //  convert flux from 6 doubles -> CCVarible
    if (new_dw->exists(_radFluxE_Label, _matl, firstPatch)) {
      sched_DBLsToStencil(archesLevel, sched);
    }

    //__________________________________
    // compute sigmaT4 if it doesn't already exist
    // on the arches level
    if (!new_dw->exists(_RMCRT->d_sigmaT4Label, _matl, firstPatch)) {
      bool includeExtraCells = true;
      _RMCRT->sched_sigmaT4(archesLevel, sched, Task::NewDW, includeExtraCells);
    }
  }
}

//______________________________________________________________________
// HACK
//______________________________________________________________________
void
RMCRT_Radiation::restartInitializeHack( const ProcessorGroup* , const PatchSubset*,
                                        const MaterialSubset*, DataWarehouse*, DataWarehouse*)
{
  // This task is used to "fake" out the taskgraph createDetailedDependency() logic
  // Before you can require something from the new_dw there must be a compute() for that
  // variable.
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
  // see sched_computeSource & CCA/Components/Models/Radiation/RMCRT/Ray.cc  for the actual tasks
  throw InternalError("Stub Task: RMCRT_Radiation::computeSource you should never land here ", __FILE__, __LINE__);
}


//______________________________________________________________________
//   Set the the boundary conditions for sigmaT4 & abskg.
//______________________________________________________________________
void
RMCRT_Radiation::sched_setBoundaryConditions( const LevelP& level,
                                              SchedulerP& sched,
                                              Task::WhichDW temp_dw,
                                              const bool backoutTemp /* = false */ )
{

  std::string taskname = "RMCRT_radiation::setBoundaryConditions";
  Task* tsk = nullptr;

  if ( _FLT_DBL == TypeDescription::double_type ) {

    tsk= scinew Task( taskname, this, &RMCRT_Radiation::setBoundaryConditions< double >, temp_dw, backoutTemp );
  } else {
    tsk= scinew Task( taskname, this, &RMCRT_Radiation::setBoundaryConditions< float >, temp_dw, backoutTemp );
  }

  printSchedule(level, dbg, "RMCRT_radiation::sched_setBoundaryConditions");

  if (!backoutTemp) {
    tsk->requires( temp_dw, _tempLabel, Ghost::None, 0 );
  }

  tsk->modifies( _RMCRT->d_sigmaT4Label );
  tsk->modifies( _RMCRT->d_abskgLabel );         // this label changes name if using floats

  sched->addTask( tsk, level->eachPatch(), _sharedState->allArchesMaterials(), RMCRT_Radiation::TG_RMCRT );
}
//______________________________________________________________________

template<class T>
void RMCRT_Radiation::setBoundaryConditions( const ProcessorGroup* pc,
                                             const PatchSubset* patches,
                                             const MaterialSubset*,
                                             DataWarehouse*,
                                             DataWarehouse* new_dw,
                                             Task::WhichDW temp_dw,
                                             const bool backoutTemp )
{

  for (int p=0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

    std::vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    if( bf.size() > 0){

      printTask(patches,patch,dbg,"Doing RMCRT_Radiation::setBoundaryConditions");

      double sigma_over_pi = (_RMCRT->d_sigma)/M_PI;

      CCVariable<double> temp;
      CCVariable< T > abskg;
      CCVariable< T > sigmaT4OverPi;

      new_dw->allocateTemporary(temp,  patch);
      new_dw->getModifiable( abskg,         _RMCRT->d_abskgLabel,    _matl, patch );
      new_dw->getModifiable( sigmaT4OverPi, _RMCRT->d_sigmaT4Label,  _matl, patch );
      //__________________________________
      // loop over boundary faces and backout the temperature
      // one cell from the boundary.  Note that the temperature
      // is not available on all levels but sigmaT4 is.
      if (backoutTemp){
        for( std::vector<Patch::FaceType>::const_iterator itr = bf.cbegin(); itr != bf.cend(); ++itr ){
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

      std::string comp_abskt = _RMCRT->d_abskgLabel->getName();
      std::string comp_Temp  = _tempLabel->getName();

      BoundaryCondition_new* new_BC = _boundaryCondition->getNewBoundaryCondition();
      new_BC->setExtraCellScalarValueBC< T >(      pc, patch, abskg, comp_abskt );
      new_BC->setExtraCellScalarValueBC< double >( pc, patch, temp,  comp_Temp );

      //__________________________________
      // loop over boundary faces and compute sigma T^4
      for( std::vector<Patch::FaceType>::const_iterator itr = bf.cbegin(); itr != bf.cend(); ++itr ){
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
                                                       const bool );

template
void RMCRT_Radiation::setBoundaryConditions< float >( const ProcessorGroup*,
                                                      const PatchSubset* ,
                                                      const MaterialSubset*,
                                                      DataWarehouse*,
                                                      DataWarehouse* ,
                                                      Task::WhichDW ,
                                                      const bool );

//______________________________________________________________________
//    Conversion doubls -> stencil tasks
//______________________________________________________________________
void
RMCRT_Radiation::sched_stencilToDBLs( const LevelP& level,
                                      SchedulerP& sched )
{

  if( level->getIndex() != _archesLevelIndex){
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
RMCRT_Radiation::sched_fluxInit( const LevelP& level,
                                      SchedulerP& sched )
{
  if( level->getIndex() != _archesLevelIndex){
    throw InternalError("RMCRT_Radiation::sched_stencilToDBLs.  You cannot schedule this task on a non-arches level", __FILE__, __LINE__);
  }

  if(_RMCRT->d_solveBoundaryFlux) {
    Task* tsk = scinew Task( "RMCRT_Radiation::fluxInit", this, &RMCRT_Radiation::fluxInit );

    printSchedule( level, dbg, "RMCRT_Radiation::sched_stencilToDBLs" );

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
RMCRT_Radiation::fluxInit( const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset*,
                             DataWarehouse* ,
                             DataWarehouse* new_dw )
{
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing RMCRT_Radiation::stencilToDBLs");

    CCVariable<double> East, West;
    CCVariable<double> North, South;
    CCVariable<double> Top, Bot;
    new_dw->allocateAndPut( East,  _radFluxE_Label, _matl, patch );
    new_dw->allocateAndPut( West,  _radFluxW_Label, _matl, patch );
    new_dw->allocateAndPut( North, _radFluxN_Label, _matl, patch );
    new_dw->allocateAndPut( South, _radFluxS_Label, _matl, patch );
    new_dw->allocateAndPut( Top,   _radFluxT_Label, _matl, patch );
    new_dw->allocateAndPut( Bot,   _radFluxB_Label, _matl, patch );

      East.initialize(0);
      West.initialize(0);
      North.initialize(0);          // THIS MAPPING MUST BE VERIFIED
      South.initialize(0);
      Top.initialize(0);
      Bot.initialize(0);
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
  for (int p = 0; p < patches->size(); ++p) {

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
//______________________________________________________________________
//
void
RMCRT_Radiation::sched_DBLsToStencil( const LevelP& level,
                                      SchedulerP& sched )
{

  if( level->getIndex() != _archesLevelIndex) {
    throw InternalError("RMCRT_Radiation::sched_stencilToDBLs.  You cannot schedule this task on a non-arches level", __FILE__, __LINE__);
  }

  if (_RMCRT->d_solveBoundaryFlux) {
    Task* tsk = scinew Task( "RMCRT_Radiation::DBLsToStencil", this, &RMCRT_Radiation::DBLsToStencil );
    printSchedule( level, dbg, "RMCRT_Radiation::sched_DBLsToStencil" );

    //  only schedule task on arches level
    tsk->requires(Task::NewDW, _radFluxE_Label, _gn, 0);
    tsk->requires(Task::NewDW, _radFluxW_Label, _gn, 0);
    tsk->requires(Task::NewDW, _radFluxN_Label, _gn, 0);
    tsk->requires(Task::NewDW, _radFluxS_Label, _gn, 0);
    tsk->requires(Task::NewDW, _radFluxT_Label, _gn, 0);
    tsk->requires(Task::NewDW, _radFluxB_Label, _gn, 0);

    tsk->computes( _RMCRT->d_boundFluxLabel );

    sched->addTask( tsk, level->eachPatch(), _sharedState->allArchesMaterials() );
  }
}

//______________________________________________________________________
//
void
RMCRT_Radiation::DBLsToStencil( const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse* ,
                                DataWarehouse* new_dw )
{
  for (int p=0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing RMCRT_Radiation::DBLsToStencil");

    CCVariable<Stencil7>  boundaryFlux;
    new_dw->allocateAndPut( boundaryFlux, _RMCRT->d_boundFluxLabel, _matl, patch );

    constCCVariable<double> East, West;
    constCCVariable<double> North, South;
    constCCVariable<double> Top, Bot;

    new_dw->get( East,  _radFluxE_Label, _matl, patch, _gn, 0 );
    new_dw->get( West,  _radFluxW_Label, _matl, patch, _gn, 0 );
    new_dw->get( North, _radFluxN_Label, _matl, patch, _gn, 0 );
    new_dw->get( South, _radFluxS_Label, _matl, patch, _gn, 0 );
    new_dw->get( Top,   _radFluxT_Label, _matl, patch, _gn, 0 );
    new_dw->get( Bot,   _radFluxB_Label, _matl, patch, _gn, 0 );

    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      Stencil7& me = boundaryFlux[c];
      me.e = East[c];
      me.w = West[c];
      me.n = North[c];         // THIS MAPPING MUST BE VERIFIED
      me.s = South[c];
      me.t = Top[c];
      me.b = Bot[c];
    }
  }
}
