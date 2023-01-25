/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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
#include <CCA/Components/Models/Radiation/RMCRT/Radiometer.h>

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Util/DOUT.hpp>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>


using namespace Uintah;

Dout dbg("RMCRT", "Arches", "RMCRT debug info", false);

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
                  labels->d_materialManager,
                  req_label_names, type ),
  m_labels( labels ),
  m_MAlab(MAlab),
  m_my_world(my_world)
{

  _src_label = VarLabel::create( src_name,  CCVariable<double>::getTypeDescription() );

  //Declare the source type:
  _source_grid_type = CC_SRC; // or FX_SRC, or FY_SRC, or FZ_SRC, or CCVECTOR_SRC

  m_materialManager  = labels->d_materialManager;

  m_partGas_temp_names.push_back("radiation_temperature");      // HARDWIRED!!!

  //__________________________________
  //  define the material index
  int archIndex = 0;                // HARDWIRED
  m_matl = m_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

  const TypeDescription* CC_double = CCVariable<double>::getTypeDescription();
  m_radFluxE_Label = VarLabel::create("radiationFluxE",  CC_double);
  m_radFluxW_Label = VarLabel::create("radiationFluxW",  CC_double);
  m_radFluxN_Label = VarLabel::create("radiationFluxN",  CC_double);
  m_radFluxS_Label = VarLabel::create("radiationFluxS",  CC_double);
  m_radFluxT_Label = VarLabel::create("radiationFluxT",  CC_double);
  m_radFluxB_Label = VarLabel::create("radiationFluxB",  CC_double);
}

//______________________________________________________________________
//
RMCRT_Radiation::~RMCRT_Radiation()
{
  // source label is destroyed in the base class
  delete m_RMCRT;

  VarLabel::destroy( m_radFluxE_Label );
  VarLabel::destroy( m_radFluxW_Label );
  VarLabel::destroy( m_radFluxN_Label );
  VarLabel::destroy( m_radFluxS_Label );
  VarLabel::destroy( m_radFluxT_Label );
  VarLabel::destroy( m_radFluxB_Label );
  VarLabel::destroy( m_sumAbsk_Label );

  if( m_matlSet ) {
    m_matlSet->removeReference();
    delete m_matlSet;
  }

}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
RMCRT_Radiation::problemSetup( const ProblemSpecP& inputdb )
{

  DOUT( dbg, Uintah::Parallel::getMPIRank() << "Doing RMCRT_Radiation::problemSetup");

  m_ps = inputdb;
  m_ps->getWithDefault( "calc_on_all_RKsteps",  m_all_rk, false );

  // gas absorption coefficient
  ProblemSpecP ac_ps = m_ps->findBlock("absorptionCoeffs");
  if ( ac_ps == nullptr ){
    throw ProblemSetupException("Error: RMCRT - <absorptionCoeffs> is not defined.",__FILE__,__LINE__);
  }
  std::string gas_absk_name;
  ac_ps ->require( "gas_absk", gas_absk_name );

  m_partGas_absk_names.push_back( gas_absk_name );


  //__________________________________
  //  Bulletproofing:
  if( m_all_rk){
    throw ProblemSetupException("ERROR:  RMCRT_radiation only works if calc_on_all_RKstes = false", __FILE__, __LINE__);
  }

  ProblemSpecP rmcrt_ps = m_ps->findBlock("RMCRT");
  if (!rmcrt_ps){
    throw ProblemSetupException("ERROR:  RMCRT_radiation, the xml tag <RMCRT> was not found", __FILE__, __LINE__);
  }

  // Are we using floats for all-to-all variables
  std::map<std::string, std::string> type;
  rmcrt_ps->getAttributes(type);

  std::string isFloat = type["type"];

  if( isFloat == "float" ){
    m_FLT_DBL = TypeDescription::float_type;
  }

  m_RMCRT = scinew Ray( m_FLT_DBL );

  m_RMCRT->setBC_onOff( false );

  //__________________________________
  //  Read in the RMCRT algorithm that will be used
  ProblemSpecP alg_ps = rmcrt_ps->findBlock("algorithm");
  if (alg_ps){

    std::string type="nullptr";
    alg_ps->getAttribute("type", type);

    if (type == "dataOnion" ) {                   // DATA ONION

      m_whichAlgo = dataOnion;
      m_RMCRT->setBC_onOff( true );

    }
    else if ( type == "dataOnionSlim" ) {       // DATA ONION SLIM

      m_whichAlgo = dataOnionSlim;
      m_RMCRT->setBC_onOff( true );

    } 
    else if ( type == "RMCRT_coarseLevel" ) {   // 2 LEVEL

      m_whichAlgo = coarseLevel;
      m_RMCRT->setBC_onOff( true );

    }
    else if ( type == "singleLevel" ) {         // 1 LEVEL
      m_whichAlgo = singleLevel;

    }
    else if ( type == "radiometerOnly" ) {      // Only when radiometer is used
      m_whichAlgo = radiometerOnly;
      _stage     = 2;                           // needed to avoid Arches bulletproofing
    }
  }

  //__________________________________
  //  Particle contributions
  ProblemSpecP icpr_ps = ac_ps->findBlock("includeParticleRad");

  if( ac_ps->findBlock("includeParticleRad") ){
    m_do_partRadiation = true;

#if 0
    // Only read in the particle absk
    std::string pAbskName;
    icpr_ps->require( "particle_absk", pAbskName );

    m_partGas_absk_names.push_back( pAbskName );
#endif
    //-----------------------------------------
    //This code is for when we use abskp_0 abskp_1 abskp_N
#if 1
    std::string pTempName;
    std::string pAbskName;
    icpr_ps->require( "particle_absk",   pAbskName );
    icpr_ps->require( "part_temp_label", pTempName );

    // find the number of particle labels
    bool doing_dqmom = ArchesCore::check_for_particle_method( m_ps, ArchesCore::DQMOM_METHOD );
    bool doing_cqmom = ArchesCore::check_for_particle_method( m_ps, ArchesCore::CQMOM_METHOD );

    if ( doing_dqmom ){
      m_nQn_part = ArchesCore::get_num_env( m_ps, ArchesCore::DQMOM_METHOD );
    } else if ( doing_cqmom ){
      m_nQn_part = ArchesCore::get_num_env( m_ps, ArchesCore::CQMOM_METHOD );
    } else {
      throw ProblemSetupException("RMCRT: This method only works for DQMOM/CQMOM.",__FILE__,__LINE__);
    }

    for (int qn=0; qn < m_nQn_part; qn++){
      std::stringstream absk;
      std::stringstream temp;

      temp << pTempName <<"_"<< qn;
      absk << pAbskName <<"_"<< qn;
      m_partGas_temp_names.push_back( temp.str() );
      m_partGas_absk_names.push_back( absk.str() );
    }
#endif

  }
}

//______________________________________________________________________
//  We need this additional call to problemSetup
//  so the reaction models can create the needed VarLabels
//______________________________________________________________________
void
RMCRT_Radiation::extraSetup( GridP& grid,
                             BoundaryCondition* bc,
                             TableLookup* table_lookup )
{

  m_boundaryCondition = bc;

  DOUT( dbg, Uintah::Parallel::getMPIRank() << "Doing RMCRT_Radiation::extraSetup");

  //__________________________________
  //  create sumAbskLabel
  const TypeDescription* td = CCVariable<double>::getTypeDescription();
  if( m_FLT_DBL == TypeDescription::float_type ){
    td = CCVariable<float>::getTypeDescription();
  }
  m_sumAbsk_Label = VarLabel::create("RMCRT_sumAbsk", td);

  //__________________________________
  // gas radiaton
    for (size_t i=0; i < m_partGas_absk_names.size(); i++){
      const VarLabel * Temp = VarLabel::find( m_partGas_temp_names[i], "ERROR RMCRT_Radiation::extraSetup: ");
      const VarLabel * absk = VarLabel::find( m_partGas_absk_names[i], "ERROR RMCRT_Radiation::extraSetup: ");

      m_partGas_temp_Labels.push_back( Temp );
      m_partGas_absk_Labels.push_back( absk );
    }

    m_gasTemp_Label  = m_partGas_temp_Labels[0];
    m_nPartGasLabels = m_partGas_absk_Labels.size();

  proc0cout << "\n __________________________________ RMCRT SETTINGS\n"
             <<"  - Temperature label:          " << m_partGas_temp_names[0] << "\n"
             <<"  - gas absorption Coeff label: " <<  m_partGas_absk_names[0] << "\n"
             <<"  - The boundary condition for the absorption coeff used in the RMCRT intensity calculation is 1.0.\n";


  //-----------------------------------------
  // Gas radiation
  if( ! m_do_partRadiation ){
    proc0cout << "  - sigmaT4 = (sigma/M_PI) * " << m_partGas_temp_names[0] << "^4\n\n";
  }
  //__________________________________
  //  Particle radiation
  else {

    // output to screen the sigmaT4 equation
    proc0cout << "  - Including the particle radiation contributions \n";
    proc0cout << "      sumT    = ";

    for (int i=0; i < m_nPartGasLabels; i++){
      proc0cout <<  "(" << m_partGas_absk_names[i] <<" * "<< m_partGas_temp_names[i]<<"^4 ) "
                << (i<m_nPartGasLabels-1 ? " + " : "\n");
    }

    proc0cout << "      sumAbsk = (";
    for (int i=0; i < m_nPartGasLabels; i++){
      proc0cout << m_partGas_absk_names[i] << (i<m_nPartGasLabels-1 ? " + " : ")\n");
    }

    proc0cout << "      sigmaT4 = (sigma/M_PI) * sumT/sumAbsk\n\n";
  }

  proc0cout << "  - Absorption coefficient used in intensity calculation: (";

  for (int i=0; i < m_nPartGasLabels; i++){
    proc0cout << m_partGas_absk_names[i] << (i<m_nPartGasLabels-1 ? " + " : ")\n");
  }

  //__________________________________
  // create RMCRT and register the labels
  m_RMCRT->registerVariables(m_matl,
                             m_sumAbsk_Label,
                             m_gasTemp_Label,
                             m_labels->d_cellTypeLabel,
                             _src_label,
                             m_whichAlgo);

  //__________________________________
  // read in RMCRT problem spec
  ProblemSpecP rmcrt_ps = m_ps->findBlock("RMCRT");

  m_RMCRT->problemSetup(m_ps, rmcrt_ps, grid);

  m_RMCRT->BC_bulletproofing( rmcrt_ps, true, false );

  //__________________________________
  //  Bulletproofing:
  // dx must get smaller as level-index increases
  // Arches is always computed on the finest level
  int maxLevels = grid->numLevels();
  m_archesLevelIndex = maxLevels - 1;

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
//  Schedule the calculation of the source term (divQ) and radiometer_VR
//
//  See: CCA/Components/Models/Radiation/RMCRT/Ray.cc
//       for the actual tasks that are scheduled.
//---------------------------------------------------------------------------
void
RMCRT_Radiation::sched_computeSource( const LevelP& level,
                                      SchedulerP  & sched,
                                      int timeSubStep )
{

  GridP grid = level->getGrid();

  // only sched on RK step 0 and on arches level
  if (timeSubStep != 0 || level->getIndex() != m_archesLevelIndex) {
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
    if (level->getIndex() != m_archesLevelIndex) {
      m_RMCRT->sched_CarryForward_Var(level, sched, m_labels->d_cellTypeLabel);
    }
  }

  typedef std::vector<const VarLabel*> VarLabelVec;

  VarLabelVec fineLevelVarLabels, coarseLevelVarLabels;

  fineLevelVarLabels.push_back(m_RMCRT->d_divQLabel);
  fineLevelVarLabels.push_back(m_RMCRT->d_boundFluxLabel);
  fineLevelVarLabels.push_back(m_RMCRT->d_radiationVolqLabel);            // ToDo: only carry forward saved vars
  fineLevelVarLabels.push_back(m_RMCRT->d_abskgLabel);
  fineLevelVarLabels.push_back(m_RMCRT->d_sigmaT4Label);

  coarseLevelVarLabels.push_back(m_RMCRT->d_abskgLabel);
  coarseLevelVarLabels.push_back(m_RMCRT->d_sigmaT4Label);

  Task::WhichDW notUsed = Task::None;
  //______________________________________________________________________
  //   D A T A   O N I O N   A P P R O A C H
  if (m_whichAlgo == dataOnion || m_whichAlgo == dataOnionSlim) {

    Task::WhichDW temp_dw       = Task::OldDW;
    Task::WhichDW celltype_dw   = Task::NewDW;
    Task::WhichDW sigmaT4_dw    = Task::NewDW;
    const bool backoutTemp      = true;
    const bool modifies_abskg   = false;
    const bool modifies_sigmaT4 = false;

    const LevelP& fineLevel = grid->getLevel(m_archesLevelIndex);

    // define per level which abskg dw
    m_RMCRT->set_abskg_dw_perLevel( fineLevel, Task::NewDW );

    // compute sigmaT4, sumAbsk on the finest level
    sched_sigmaT4( fineLevel, sched );

    sched_sumAbsk( fineLevel, sched );

    // carry forward if it's time
    m_RMCRT->sched_carryForward_VarLabels( fineLevel, sched, fineLevelVarLabels );

    // coarse levels
    for (int l = 0; l < maxLevels-1; ++l) {
      const LevelP& level = grid->getLevel(l);
      m_RMCRT->sched_carryForward_VarLabels( level, sched, coarseLevelVarLabels );
    }

    // coarsen data to the coarser levels.
    // do it in reverse order
    for (int l = maxLevels - 2; l >= 0; l--) {
      const LevelP& level = grid->getLevel(l);

      m_RMCRT->sched_CoarsenAll( level, sched, modifies_abskg, modifies_sigmaT4 );

      if( m_RMCRT->d_coarsenExtraCells == false ) {
        sched_setBoundaryConditions( level, sched, notUsed, backoutTemp );
      }
    }

    if (m_whichAlgo == dataOnionSlim) {
      //Combine vars for every level
      for (int l = maxLevels - 1; l >= 0; l--) {
        const LevelP& level = grid->getLevel(l);
        m_RMCRT->sched_combineAbskgSigmaT4CellType(level, sched, temp_dw, includeExtraCells);
      }
    }

    //__________________________________
    //  compute the extents of the RMCRT region of interest on the finest level
    m_RMCRT->sched_ROI_Extents( fineLevel, sched );

    m_RMCRT->sched_rayTrace_dataOnion( fineLevel, sched, notUsed, sigmaT4_dw, celltype_dw, modifies_divQ );

    // convert boundaryFlux<Stencil7> -> 6 doubles
    sched_stencilToDBLs( fineLevel, sched );
  }

  //______________________________________________________________________
  //   2 - L E V E L   A P P R O A C H
  //  RMCRT is performed on the coarse level
  //  and the results are interpolated to the fine (arches) level
  if ( m_whichAlgo == coarseLevel ) {

    Task::WhichDW temp_dw       = Task::OldDW;
    Task::WhichDW sigmaT4_dw    = Task::NewDW;
    Task::WhichDW celltype_dw   = Task::NewDW;
    const bool modifies_abskg   = false;
    const bool modifies_sigmaT4 = false;
    const bool backoutTemp      = true;

    // carry forward if it's time
    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      m_RMCRT->sched_carryForward_VarLabels( level, sched, fineLevelVarLabels );
    }

    const LevelP& fineLevel = grid->getLevel( m_archesLevelIndex );

    m_RMCRT->set_abskg_dw_perLevel ( fineLevel, Task::NewDW );

    // compute sigmaT4, sumAbsk on the finest level
    sched_sigmaT4( fineLevel, sched );

    sched_sumAbsk( fineLevel, sched );


    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);

      m_RMCRT->sched_CoarsenAll( level, sched, modifies_abskg, modifies_sigmaT4 );

      if (level->hasFinerLevel()) {
        if( m_RMCRT->d_coarsenExtraCells == false ) {
          sched_setBoundaryConditions( level, sched, temp_dw, backoutTemp );
        }

        m_RMCRT->sched_rayTrace( level, sched, notUsed, sigmaT4_dw, celltype_dw, modifies_divQ );
      }
    }

    // push divQ  to the fine levels
    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      const PatchSet* patches = level->eachPatch();
      m_RMCRT->sched_Refine_Q( sched, patches, m_matlSet );
    }

    // convert boundaryFlux<Stencil7> -> 6 doubles
    sched_stencilToDBLs( fineLevel, sched );
  }

  //______________________________________________________________________
  //   1 - L E V E L   A P P R O A C H
  //  RMCRT is performed on the same level as CFD
  if ( m_whichAlgo == singleLevel ) {
    Task::WhichDW sigmaT4_dw  = Task::NewDW;
    Task::WhichDW celltype_dw = Task::NewDW;

    const LevelP& level = grid->getLevel( m_archesLevelIndex );

    m_RMCRT->set_abskg_dw_perLevel( level, Task::NewDW );

    m_RMCRT->sched_carryForward_VarLabels( level, sched, fineLevelVarLabels );

    // compute sigmaT4 on the CFD level
    sched_sigmaT4( level, sched );

    sched_sumAbsk( level, sched );

    m_RMCRT->sched_rayTrace( level, sched, notUsed, sigmaT4_dw, celltype_dw, modifies_divQ );

    // convert boundaryFlux<Stencil7> -> 6 doubles
    sched_stencilToDBLs( level, sched );
  }

  //______________________________________________________________________
  //   R A D I O M E T E R
  //  No other calculations
  if ( m_whichAlgo == radiometerOnly ) {
    Radiometer* radiometer    = m_RMCRT->getRadiometer();
    Task::WhichDW sigmaT4_dw  = Task::NewDW;
    Task::WhichDW celltype_dw = Task::NewDW;

    const LevelP& level = grid->getLevel( m_archesLevelIndex );

    m_RMCRT->set_abskg_dw_perLevel ( level, Task::NewDW );

    VarLabelVec varLabels = { m_RMCRT->d_abskgLabel,
                              m_RMCRT->d_sigmaT4Label,
                              radiometer->d_VRFluxLabel,
                              radiometer->d_VRIntensityLabel};

    m_RMCRT->sched_carryForward_VarLabels( level, sched, varLabels );

    sched_sigmaT4( level, sched );

    sched_sumAbsk( level, sched );

    radiometer->sched_radiometer( level, sched, notUsed, sigmaT4_dw, celltype_dw );

  }
}
//______________________________________________________________________
//    Schedule task that initializes the boundary fluxes and divQ
//    This will only be called on the Archeslevel
//______________________________________________________________________
void
RMCRT_Radiation::sched_initialize( const LevelP& level,
                                   SchedulerP& sched )
{
  GridP grid = level->getGrid();
  int maxLevels = grid->numLevels();

  //__________________________________
  //  Must do after problemSetup
  m_matlSet = m_materialManager->allMaterials( "Arches" );

  //__________________________________
  //  Additional bulletproofing, this belongs in problem setup
  if ( m_whichAlgo == dataOnion && maxLevels == 1){
    throw ProblemSetupException("ERROR:  RMCRT_radiation, there must be more than 1 level if you're using the Data Onion algorithm", __FILE__, __LINE__);
  }

  if ( m_whichAlgo == radiometerOnly && maxLevels != 1){
    throw ProblemSetupException("ERROR:  RMCRT_radiation.  The virtual radiometer only works on 1 level", __FILE__, __LINE__);
  }


  //__________________________________
  // Initialize on all levels
  for (int l = 0; l < maxLevels; l++) {
    const LevelP& myLevel = grid->getLevel(l);
    m_RMCRT->sched_initialize_VarLabel( myLevel, sched, m_RMCRT->d_sigmaT4Label  );
    m_RMCRT->sched_initialize_VarLabel( myLevel, sched, m_RMCRT->d_abskgLabel  );
  }

  //__________________________________
  //  Radiometer only
  if( m_whichAlgo == radiometerOnly ){
    Radiometer* radiometer = m_RMCRT->getRadiometer();
    radiometer->sched_initialize_VRFlux( level, sched );
    return;
  }

  //__________________________________
  //   Other RMCRT algorithms
  //__________________________________
  for (int l = 0; l < maxLevels; l++) {
    const LevelP& myLevel = grid->getLevel(l);

    int L_index= myLevel->getIndex();
    std::ostringstream taskName;
    std::ostringstream schedName;
    taskName  << "RMCRT_Radiation::initialize_L-" << L_index;
    schedName << "RMCRT_Radiation::sched_initialize_L-" << L_index;

    Task* tsk = scinew Task( schedName.str(), this, &RMCRT_Radiation::initialize );
    printSchedule( level, dbg, taskName.str() );

    // all levels
    tsk->computes(VarLabel::find("radiationVolq"));
    tsk->computes(VarLabel::find("RMCRTboundFlux"));

    // only cfd level
    if ( L_index == m_archesLevelIndex) {
      tsk->computes( _src_label );
    }

    // coarse levels
    if ( L_index != m_archesLevelIndex) {
      // divQ computed on all levels
      if ( m_whichAlgo == coarseLevel ) {
        tsk->computes( _src_label );
      }
    }
    sched->addTask( tsk, myLevel->eachPatch(), m_matlSet );
  }

  //__________________________________
  //  initialize cellType on NON arches level
  for (int l = maxLevels - 1; l >= 0; l--) {
    const LevelP& level = grid->getLevel(l);

    if( level->getIndex() != m_archesLevelIndex ){
      // Set the BC on the coarse level
      m_boundaryCondition->sched_cellTypeInit( sched, level, m_matlSet );

      // Coarsen the interior cells
       m_RMCRT->sched_computeCellType ( level, sched, Ray::modifiesVar);
    }
  }

  sched_fluxInit( level, sched );
}

//______________________________________________________________________
//    Task that initializes the boundary fluxes and divQ
//______________________________________________________________________
void
RMCRT_Radiation::initialize( const ProcessorGroup *,
                             const PatchSubset    * patches,
                             const MaterialSubset *,
                                   DataWarehouse  * ,
                                   DataWarehouse  * new_dw )
{
  const Level* level = getLevel(patches);
  const int L_index  = level->getIndex();

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    printTask(patches, patch, dbg, "Doing RMCRT_Radiation::initialize");

    CCVariable<double> radVolq;
    CCVariable<double> src;
    CCVariable<Stencil7> RMCRTboundFlux;

    //__________________________________
    // all levels
    new_dw->allocateAndPut( radVolq, VarLabel::find("radiationVolq"), m_matl, patch );
    radVolq.initialize( 0.0 );  // needed for coal

    new_dw->allocateAndPut( RMCRTboundFlux, VarLabel::find("RMCRTboundFlux"), m_matl, patch );
    for ( CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++ ) {
      IntVector c = *iter;
      RMCRTboundFlux[c].initialize(0.0);
    }

    //__________________________________
    //  CFD level
    if ( L_index == m_archesLevelIndex) {
      new_dw->allocateAndPut( src, _src_label, m_matl, patch );
      src.initialize(0.0);
    }

    //__________________________________
    //  Coarse levels
    if ( L_index != m_archesLevelIndex) {

      if( m_RMCRT->RMCRTCommon::d_FLT_DBL == TypeDescription::double_type ) {
        CCVariable<double> abskgDouble;
        new_dw->allocateAndPut( abskgDouble, m_RMCRT->d_abskgLabel, m_matl, patch );
        abskgDouble.initialize( 0.0 );
      }
      else {
        CCVariable<float> abskgFloat;
        new_dw->allocateAndPut( abskgFloat, m_RMCRT->d_abskgLabel, m_matl, patch );
        abskgFloat.initialize( 0.0 );
      }

      // divQ computed on all levels
      if ( m_whichAlgo == coarseLevel ) {
        new_dw->allocateAndPut( src, _src_label, m_matl, patch );
        src.initialize(0.0);
      }
    }
  }
}

//______________________________________________________________________
// Schedule restart initialization
// This is only called on the Archeslevel
//______________________________________________________________________
void
RMCRT_Radiation::sched_restartInitialize( const LevelP& level,
                                           SchedulerP& sched )
{
  //__________________________________
  //  Must do after problemSetup
  m_matlSet = m_materialManager->allMaterials( "Arches" );

  GridP grid = level->getGrid();

  DataWarehouse* new_dw = sched->getLastDW();

  const LevelP& archesLevel = grid->getLevel( m_archesLevelIndex );

  if (level != archesLevel) {
    return;
  }

  printSchedule(level, dbg, "RMCRT_Radiation::sched_restartInitialize");

  // Find the first patch, on the arches level, that this mpi rank owns.
  const Uintah::PatchSet* const ps = sched->getLoadBalancer()->getPerProcessorPatchSet(archesLevel);
  const PatchSubset* myPatches = ps->getSubset( m_my_world->myRank() );
  const Patch* firstPatch = myPatches->get(0);

  //  Only schedule if radFlux*_Label are in the checkpoint uda
  if ( ( m_whichAlgo != radiometerOnly ) && new_dw->exists( m_radFluxE_Label, m_matl, firstPatch) ) {
    printSchedule(level, dbg, "RMCRT_Radiation::sched_restartInitializeHack");

    Task* t1 = scinew Task("RMCRT_Radiation::restartInitializeHack", this,
                           &RMCRT_Radiation::restartInitializeHack);
    t1->computes( m_radFluxE_Label );
    t1->computes( m_radFluxW_Label );
    t1->computes( m_radFluxN_Label );   // Before you can require something from the new_dw
    t1->computes( m_radFluxS_Label );   // there must be a compute() for that variable.
    t1->computes( m_radFluxT_Label );
    t1->computes( m_radFluxB_Label );

    sched->addTask( t1, archesLevel->eachPatch(), m_matlSet );

    //__________________________________
    //  convert rad flux from 6 doubles -> CCVarible
    sched_DBLsToStencil(archesLevel, sched);
  }

  //__________________________________
  //  Radiometer only
  Radiometer* radiometer = m_RMCRT->getRadiometer();
  if( m_whichAlgo == radiometerOnly && !new_dw->exists( radiometer->d_VRFluxLabel, m_matl, firstPatch) ){
    radiometer->sched_initialize_VRFlux( level, sched );
  }

  //__________________________________
  //  If any of the absk or temperature variables are missing
  //  from the checkpoint then initialize them

  const double initAbsk = 1.0;             // initialization values  HARDWIRED!!!
  const double initTemp = 300;

  if ( myPatches->size() > 0 ){

    for (int i=0 ; i< m_nPartGasLabels; i++){

      const VarLabel * abskLabel  = m_partGas_absk_Labels[i];
      if( !new_dw->exists( abskLabel,  m_matl, firstPatch ) ){
        m_missingCkPt_Labels[ abskLabel ] = initAbsk;
      }

      const VarLabel * tempLabel  = m_partGas_temp_Labels[i];
      if(  !new_dw->exists( tempLabel,  m_matl, firstPatch ) ){
        m_missingCkPt_Labels[ tempLabel ] = initTemp;
      }
    }
  }


  if( m_missingCkPt_Labels.size() > 0 ){

    std::string taskname = "RMCRT_Radiation::sched_restartInitialize";
    printSchedule(level, dbg, taskname);

    Task* t2 = scinew Task( taskname, this, &RMCRT_Radiation::restartInitialize);

    for ( auto  iter = m_missingCkPt_Labels.begin(); iter != m_missingCkPt_Labels.end(); iter++){
      t2->computes( iter->first );
    }

    sched->addTask( t2, archesLevel->eachPatch(), m_matlSet );
  }

  //__________________________________
  // if sumAbsk or sigmaT4 is missing from checkpoint compute them

  if( !new_dw->exists( m_sumAbsk_Label,         m_matl, firstPatch ) ||
      !new_dw->exists( m_RMCRT->d_sigmaT4Label, m_matl, firstPatch ) ){

    // Before you can require something from the new_dw
    // there must be a compute() for that variable.
    std::string taskname = "RMCRT_Radiation::sched_restartInitializeHack2";
    printSchedule(level, dbg, taskname);

    Task* t3 = scinew Task( taskname, this, &RMCRT_Radiation::restartInitializeHack2);

    // Some variables may have been computed in RMCRT_Radiation::sched_restartInitialize
    // Filter out those that have been computed.
    std::set<const VarLabel*, VarLabel::Compare> computedVars;
    computedVars = sched->getComputedVars();   // find all the computed vars

    for (int i=0 ; i< m_nPartGasLabels; i++){

      const VarLabel * abskLabel  = m_partGas_absk_Labels[i];
      if( computedVars.find( abskLabel ) == computedVars.end() ) {
        t3->computes( abskLabel );
      }

      const VarLabel * tempLabel  = m_partGas_temp_Labels[i];
      if( computedVars.find( tempLabel ) == computedVars.end() ) {
        t3->computes( tempLabel );
      }
    }

    sched->addTask( t3, archesLevel->eachPatch(), m_matlSet );


    sched_sumAbsk( level, sched );

    sched_sigmaT4( level, sched );
  }
}

//______________________________________________________________________
//    Task to initialize variables that were not found in the checkpoints
//______________________________________________________________________
void
RMCRT_Radiation::restartInitialize( const ProcessorGroup  * pg,
                                    const PatchSubset     * patches,
                                    const MaterialSubset  * matls,
                                    DataWarehouse         * old_dw,
                                    DataWarehouse         * new_dw )
{
  static bool doCout=( pg->myRank() == 0 );

  DOUT( doCout, "__________________________________\n"
             << "  RMCRT_Radiation::restartIntialize \n"
             << "    These variables were not found in the checkpoints\n"
             << "    and will be initialized\n");


  printTask(patches,  dbg, "Doing RMCRT_Radiation::restartIntialize");


  for ( auto  iter = m_missingCkPt_Labels.begin(); iter != m_missingCkPt_Labels.end(); iter++){
    const VarLabel* QLabel = iter->first;
    const double initValue = iter->second;
    DOUT( doCout, "    Label:  " << QLabel-> getName() << ":" <<  initValue );

    for (int p=0; p < patches->size(); p++){
      const Patch* patch = patches->get(p);
      CCVariable<double> Q;
      new_dw->allocateAndPut( Q, QLabel, m_matl, patch);
      Q.initialize( initValue );
    }
  }
  doCout=false;
}

//______________________________________________________________________
//    Schedule Task to compute intensity over all wave lengths (sigma * Temperature^4/pi)
//______________________________________________________________________
void
RMCRT_Radiation::sched_sigmaT4( const LevelP & level,
                                SchedulerP   & sched )
{
  std::string taskname = "RMCRT_Radiation::sigmaT4";

  Task* tsk = nullptr;
  std::string type = "null";

  Task::WhichDW oldNew_dw = Task::OldDW;
  if ( sched->isRestartInitTimestep() ){
    oldNew_dw = Task::NewDW;
  }

  if ( m_FLT_DBL == TypeDescription::double_type ) {
    type = "double";
    tsk = scinew Task( taskname, this, &RMCRT_Radiation::sigmaT4<double>, oldNew_dw );
  }
  else {
    type = "float";
    tsk = scinew Task( taskname, this, &RMCRT_Radiation::sigmaT4<float>, oldNew_dw );
  }

  printSchedule(level, dbg, "RMCRT_Radiation::sched_sigmaT4 (" +type+")");

  tsk->requires( oldNew_dw, m_labels->d_volFractionLabel, m_gn, 0 );

  for (int i=0 ; i< m_nPartGasLabels; i++){
    tsk->requires( oldNew_dw, m_partGas_absk_Labels[i], m_gn, 0 );
    tsk->requires( oldNew_dw, m_partGas_temp_Labels[i], m_gn, 0 );
  }

  tsk->computes( m_RMCRT->d_sigmaT4Label );
  sched->addTask( tsk, level->eachPatch(), m_matlSet, RMCRT_Radiation::TG_RMCRT );
}
//______________________________________________________________________
//    Task to compute intensity over all wave lengths (sigma * Temperature^4/pi)
//______________________________________________________________________
template< class T>
void
RMCRT_Radiation::sigmaT4( const ProcessorGroup  *,
                          const PatchSubset     * patches,
                          const MaterialSubset  * matls,
                          DataWarehouse         * old_dw,
                          DataWarehouse         * new_dw,
                          Task::WhichDW           which_dw )
{
  DataWarehouse* oldNew_dw = new_dw->getOtherDataWarehouse(which_dw);

  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, dbg, "Doing RMCRT_Radiation::sigmaT4");

    double sigma_over_pi = (m_RMCRT->d_sigma)/M_PI;

    constCCVariable<double> gasVolFrac;
    oldNew_dw->get( gasVolFrac, m_labels->d_volFractionLabel, m_matl, patch, m_gn, 0);

    // sigma T^4/pi
    CCVariable< T > sigmaT4;
    new_dw->allocateAndPut(sigmaT4, m_RMCRT->d_sigmaT4Label,m_matl, patch);

    // gas or  particle temperature & absk
    std::vector<constCCVariable<double> > partGas_absk( m_nPartGasLabels );
    std::vector<constCCVariable<double> > partGas_temp( m_nPartGasLabels );

    for (int i=0;  i< m_nPartGasLabels; i++){
      oldNew_dw->get( partGas_absk[i],  m_partGas_absk_Labels[i], m_matl, patch, m_gn, 0);
      oldNew_dw->get( partGas_temp[i],  m_partGas_temp_Labels[i], m_matl, patch, m_gn, 0);
    }

    constCCVariable<double> radTemp = partGas_temp[0];               // radiation_temperature

    //__________________________________
    //  sigmaT4: Gas radiation Only

    if( !m_do_partRadiation ){
      for (auto iter = patch->getExtraCellIterator();!iter.done();iter++){
        const IntVector& c = *iter;

        double T_sqrd = radTemp[c] * radTemp[c];
        sigmaT4[c] = sigma_over_pi * T_sqrd * T_sqrd;
      }
    }

    //__________________________________
    //  sigmaT4: Gas and particle radiation
    if( m_do_partRadiation ){

      for (auto iter = patch->getExtraCellIterator();!iter.done();iter++){
        const IntVector& c = *iter;

        if ( gasVolFrac[c] > 1e-16 ){       // interior cells
          double sumT    = 0.0;
          double sumAbsk = 0.0;
                                           // summations
          for (int i=0; i< m_nPartGasLabels; i++){
            double T_sqrd  = partGas_temp[i][c] * partGas_temp[i][c];
            sumT     += T_sqrd * T_sqrd * partGas_absk[i][c];
            sumAbsk  += partGas_absk[i][c];
          }

          sigmaT4[c] = 0.0;

          // weighted average
          if ( sumAbsk > 1e-16 ){
            sigmaT4[c] = sigma_over_pi * sumT/sumAbsk;
          }
        }
        else {                          // walls or intrustions
          double T_sqrd = radTemp[c] * radTemp[c];
          sigmaT4[c]    = sigma_over_pi * T_sqrd * T_sqrd;

        }  // intrusion or wall
      }  // loop
    }  // particle
  }  // patch
}

//______________________________________________________________________
//    Schedule task to compute the absoprtion coefficient
//______________________________________________________________________
void
RMCRT_Radiation::sched_sumAbsk( const LevelP & level,
                                SchedulerP   & sched )
{
  std::string taskname = "RMCRT_Radiation::sumAbsk";

  Task* tsk = nullptr;
  std::string type = "null";

  Task::WhichDW oldNew_dw = Task::OldDW;
  if ( sched->isRestartInitTimestep() ){
    oldNew_dw = Task::NewDW;
  }

  if ( m_FLT_DBL == TypeDescription::double_type ) {
    type = "double";
    tsk = scinew Task( taskname, this, &RMCRT_Radiation::sumAbsk<double>, oldNew_dw );
  }
  else {
    type = "float";
    tsk = scinew Task( taskname, this, &RMCRT_Radiation::sumAbsk<float>, oldNew_dw);
  }

  printSchedule(level, dbg, "RMCRT_Radiation::sched_sumAbsk (" +type+")");

  tsk->requires( oldNew_dw, m_labels->d_volFractionLabel, m_gn, 0 );      // New or old dw???

  for (int i=0 ; i< m_nPartGasLabels; i++){
    tsk->requires( oldNew_dw, m_partGas_absk_Labels[i], m_gn, 0 );
  }

  tsk->computes( m_sumAbsk_Label );
  sched->addTask( tsk, level->eachPatch(), m_matlSet, RMCRT_Radiation::TG_RMCRT );
}
//______________________________________________________________________
//    Task to compute the absoprtion coefficient
//______________________________________________________________________
template< class T>
void
RMCRT_Radiation::sumAbsk( const ProcessorGroup  *,
                          const PatchSubset     * patches,
                          const MaterialSubset  * matls,
                          DataWarehouse         * old_dw,
                          DataWarehouse         * new_dw,
                          Task::WhichDW           which_dw )
{
  DataWarehouse* oldNew_dw = new_dw->getOtherDataWarehouse(which_dw);

  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, dbg, "Doing RMCRT_Radiation::sumAbsk");

    constCCVariable<double> gasVolFrac;
    oldNew_dw->get( gasVolFrac, m_labels->d_volFractionLabel, m_matl, patch, m_gn, 0);

    // gas and particle temperature & absk
    std::vector<constCCVariable<double> > partGas_absk( m_nPartGasLabels );

    for (int i=0;  i< m_nPartGasLabels; i++){
      oldNew_dw->get( partGas_absk[i],  m_partGas_absk_Labels[i], m_matl, patch, m_gn, 0);
    }

    CCVariable<double> sumAbsk_tmp;
    new_dw->allocateTemporary( sumAbsk_tmp, patch, m_gn, 0);
    sumAbsk_tmp.initialize(0.0);

    //__________________________________
    //  Domain interior
    for (int i=0; i< m_nPartGasLabels; i++){

      for ( auto iter = patch->getCellIterator();!iter.done();iter++){
        const IntVector& c = *iter;

        if (gasVolFrac[c] > 1e-16){
          sumAbsk_tmp[c] += partGas_absk[i][c];     // gas
        }
        else{
          sumAbsk_tmp[c] = 1.0;                    // walls  HARDWIRED
        }
      }
    }

    //__________________________________
    //  Boundary Conditions                       // HARDWIRED!!
    std::vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    for( auto itr = bf.cbegin(); itr != bf.cend(); ++itr ){
      Patch::FaceType face = *itr;

      Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;

      for( auto iter=patch->getFaceIterator(face, PEC); !iter.done();iter++) {
        const IntVector& c = *iter;
        sumAbsk_tmp[c] = 1.0;
      }
    }

    //__________________________________
    //  convert to double or float
    CCVariable< T > sumAbsk;
    new_dw->allocateAndPut( sumAbsk, m_sumAbsk_Label, m_matl, patch);

    for ( auto iter = patch->getExtraCellIterator();!iter.done();iter++){
      const IntVector& c = *iter;
      sumAbsk[c] = (T) sumAbsk_tmp[c];
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
  // see sched_computeSource & CCA/Components/Models/Radiation/RMCRT/Ray.cc  for the actual tasks
  throw InternalError("Stub Task: RMCRT_Radiation::computeSource you should never land here ", __FILE__, __LINE__);
}


//______________________________________________________________________
//    Schedule task to set boundary conditions for sigmaT4 & abskg.
//______________________________________________________________________
void
RMCRT_Radiation::sched_setBoundaryConditions( const LevelP& level,
                                              SchedulerP& sched,
                                              Task::WhichDW temp_dw,
                                              const bool backoutTemp /* = false */ )
{
  std::string taskname = "RMCRT_radiation::setBoundaryConditions";
  Task* tsk = nullptr;

  if ( m_FLT_DBL == TypeDescription::double_type ) {

    tsk= scinew Task( taskname, this, &RMCRT_Radiation::setBoundaryConditions< double >, temp_dw, backoutTemp );
  } else {
    tsk= scinew Task( taskname, this, &RMCRT_Radiation::setBoundaryConditions< float >, temp_dw, backoutTemp );
  }

  printSchedule(level, dbg, "RMCRT_radiation::sched_setBoundaryConditions");

  if (!backoutTemp) {
    tsk->requires( temp_dw, m_gasTemp_Label, m_gn, 0 );
  }

  tsk->modifies( m_RMCRT->d_sigmaT4Label );
  tsk->modifies( m_RMCRT->d_abskgLabel );

  sched->addTask( tsk, level->eachPatch(), m_matlSet, RMCRT_Radiation::TG_RMCRT );
}

//______________________________________________________________________
//    Task to set boundary conditions for sigmaT4 & sumAbskg.
//______________________________________________________________________
template<class T>
void RMCRT_Radiation::setBoundaryConditions( const ProcessorGroup * pc,
                                             const PatchSubset    * patches,
                                             const MaterialSubset *,
                                                   DataWarehouse  *,
                                                   DataWarehouse  * new_dw,
                                                   Task::WhichDW temp_dw,
                                             const bool backoutTemp )
{
  for (int p=0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

    std::vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    if( bf.size() > 0){

      printTask(patches,patch,dbg,"Doing RMCRT_Radiation::setBoundaryConditions");

      double sigma_over_pi = (m_RMCRT->d_sigma)/M_PI;

      CCVariable<double> temp;
      CCVariable< T > absk;
      CCVariable< T > sigmaT4OverPi;

      new_dw->allocateTemporary(temp,  patch);
      new_dw->getModifiable( absk,          m_RMCRT->d_abskgLabel,    m_matl, patch );
      new_dw->getModifiable( sigmaT4OverPi, m_RMCRT->d_sigmaT4Label,  m_matl, patch );

      //__________________________________
      // loop over boundary faces and backout the temperature
      // one cell from the boundary.  Note that the temperature
      // is not available on all levels but sigmaT4 is.
      if (backoutTemp){
        for( auto itr = bf.cbegin(); itr != bf.cend(); ++itr ){
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
        t_dw->get(varTmp, m_gasTemp_Label, m_matl, patch, m_gn, 0);
        temp.copyData(varTmp);
      }

      //__________________________________
      //  Force absk = 1.0      HARDWIRED
      for( auto itr = bf.cbegin(); itr != bf.cend(); ++itr ){
        Patch::FaceType face = *itr;

        Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;

        for(CellIterator iter=patch->getFaceIterator(face, PEC); !iter.done();iter++) {
          const IntVector& c = *iter;
          absk[c] = (T) 1.0;
        }
      }


      //__________________________________
      // loop over boundary faces and compute sigma T^4
      std::string Temp_name = m_gasTemp_Label->getName();

      BoundaryCondition_new* new_BC = m_boundaryCondition->getNewBoundaryCondition();
      new_BC->setExtraCellScalarValueBC< double >( pc, patch, temp,  Temp_name );

      for( auto itr = bf.cbegin(); itr != bf.cend(); ++itr ){
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
//
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
//    Schedule task to initialize the rad flux array
//______________________________________________________________________
void
RMCRT_Radiation::sched_fluxInit( const LevelP& level,
                                      SchedulerP& sched )
{
  if( level->getIndex() != m_archesLevelIndex){
    throw InternalError("RMCRT_Radiation::sched_fluxInit.  You cannot schedule this task on a non-arches level", __FILE__, __LINE__);
  }

  if( m_RMCRT->d_solveBoundaryFlux ) {
    Task* tsk = scinew Task( "RMCRT_Radiation::fluxInit", this, &RMCRT_Radiation::fluxInit );

    printSchedule( level, dbg, "RMCRT_Radiation::sched_fluxInit" );

    tsk->computes( m_radFluxE_Label );
    tsk->computes( m_radFluxW_Label );
    tsk->computes( m_radFluxN_Label );
    tsk->computes( m_radFluxS_Label );
    tsk->computes( m_radFluxT_Label );
    tsk->computes( m_radFluxB_Label );

    sched->addTask( tsk, level->eachPatch(), m_matlSet );
  }
}
//______________________________________________________________________
//    Task to initialize the rad flux array
//______________________________________________________________________
void
RMCRT_Radiation::fluxInit( const ProcessorGroup *,
                           const PatchSubset    * patches,
                           const MaterialSubset *,
                                 DataWarehouse  * ,
                                 DataWarehouse  * new_dw )
{
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing RMCRT_Radiation::fluxInit");

    CCVariable<double> East, West;
    CCVariable<double> North, South;
    CCVariable<double> Top, Bot;
    new_dw->allocateAndPut( East,  m_radFluxE_Label, m_matl, patch );
    new_dw->allocateAndPut( West,  m_radFluxW_Label, m_matl, patch );
    new_dw->allocateAndPut( North, m_radFluxN_Label, m_matl, patch );
    new_dw->allocateAndPut( South, m_radFluxS_Label, m_matl, patch );
    new_dw->allocateAndPut( Top,   m_radFluxT_Label, m_matl, patch );
    new_dw->allocateAndPut( Bot,   m_radFluxB_Label, m_matl, patch );

      East.initialize(0);
      West.initialize(0);
      North.initialize(0);          // THIS MAPPING MUST BE VERIFIED
      South.initialize(0);
      Top.initialize(0);
      Bot.initialize(0);
  }
}
//______________________________________________________________________
//    Schedule task to convert stencil -> doubles
//______________________________________________________________________
void
RMCRT_Radiation::sched_stencilToDBLs( const LevelP& level,
                                      SchedulerP& sched )
{

  if( level->getIndex() != m_archesLevelIndex){
    throw InternalError("RMCRT_Radiation::sched_stencilToDBLs.  You cannot schedule this task on a non-arches level", __FILE__, __LINE__);
  }

  if( m_RMCRT->d_solveBoundaryFlux ) {
    Task* tsk = scinew Task( "RMCRT_Radiation::stencilToDBLs", this, &RMCRT_Radiation::stencilToDBLs );

    printSchedule( level, dbg, "RMCRT_Radiation::sched_stencilToDBLs" );

    //  only schedule task on arches level
    tsk->requires(Task::NewDW, VarLabel::find("RMCRTboundFlux"), m_gn, 0);

    tsk->computes( m_radFluxE_Label );
    tsk->computes( m_radFluxW_Label );
    tsk->computes( m_radFluxN_Label );
    tsk->computes( m_radFluxS_Label );
    tsk->computes( m_radFluxT_Label );
    tsk->computes( m_radFluxB_Label );

    sched->addTask( tsk, level->eachPatch(), m_matlSet );
  }
}
//______________________________________________________________________
//
//    Task to convert stencil -> doubles
//______________________________________________________________________
void
RMCRT_Radiation::stencilToDBLs( const ProcessorGroup *,
                                const PatchSubset    * patches,
                                const MaterialSubset *,
                                      DataWarehouse  * ,
                                      DataWarehouse  * new_dw )
{
  for (int p = 0; p < patches->size(); ++p) {

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing RMCRT_Radiation::stencilToDBLs");

    constCCVariable<Stencil7>  boundaryFlux;
    new_dw->get( boundaryFlux,     VarLabel::find("RMCRTboundFlux"), m_matl, patch, m_gn, 0 );

    CCVariable<double> East, West;
    CCVariable<double> North, South;
    CCVariable<double> Top, Bot;
    new_dw->allocateAndPut( East,  m_radFluxE_Label, m_matl, patch );
    new_dw->allocateAndPut( West,  m_radFluxW_Label, m_matl, patch );
    new_dw->allocateAndPut( North, m_radFluxN_Label, m_matl, patch );
    new_dw->allocateAndPut( South, m_radFluxS_Label, m_matl, patch );
    new_dw->allocateAndPut( Top,   m_radFluxT_Label, m_matl, patch );
    new_dw->allocateAndPut( Bot,   m_radFluxB_Label, m_matl, patch );

    for (auto iter = patch->getExtraCellIterator();!iter.done();iter++){
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
//    Schedule task to convert rad fluxes doubles -> stencil
//______________________________________________________________________
void
RMCRT_Radiation::sched_DBLsToStencil( const LevelP& level,
                                      SchedulerP& sched )
{

  if( level->getIndex() != m_archesLevelIndex) {
    throw InternalError("RMCRT_Radiation::sched_stencilToDBLs.  You cannot schedule this task on a non-arches level", __FILE__, __LINE__);
  }

  if ( m_RMCRT->d_solveBoundaryFlux ) {
    Task* tsk = scinew Task( "RMCRT_Radiation::DBLsToStencil", this, &RMCRT_Radiation::DBLsToStencil );
    printSchedule( level, dbg, "RMCRT_Radiation::sched_DBLsToStencil" );

    //  only schedule task on arches level
    tsk->requires(Task::NewDW, m_radFluxE_Label, m_gn, 0);
    tsk->requires(Task::NewDW, m_radFluxW_Label, m_gn, 0);
    tsk->requires(Task::NewDW, m_radFluxN_Label, m_gn, 0);
    tsk->requires(Task::NewDW, m_radFluxS_Label, m_gn, 0);
    tsk->requires(Task::NewDW, m_radFluxT_Label, m_gn, 0);
    tsk->requires(Task::NewDW, m_radFluxB_Label, m_gn, 0);

    tsk->computes( m_RMCRT->d_boundFluxLabel );

    sched->addTask( tsk, level->eachPatch(), m_matlSet );
  }
}

//______________________________________________________________________
//    Task to convert rad fluxes oubles -> stencil7
//______________________________________________________________________
void
RMCRT_Radiation::DBLsToStencil( const ProcessorGroup  *,
                                const PatchSubset     * patches,
                                const MaterialSubset  *,
                                      DataWarehouse   * ,
                                      DataWarehouse   * new_dw )
{
  for (int p=0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing RMCRT_Radiation::DBLsToStencil");

    CCVariable<Stencil7>  boundaryFlux;
    new_dw->allocateAndPut( boundaryFlux, m_RMCRT->d_boundFluxLabel, m_matl, patch );

    constCCVariable<double> East, West;
    constCCVariable<double> North, South;
    constCCVariable<double> Top, Bot;

    new_dw->get( East,  m_radFluxE_Label, m_matl, patch, m_gn, 0 );
    new_dw->get( West,  m_radFluxW_Label, m_matl, patch, m_gn, 0 );
    new_dw->get( North, m_radFluxN_Label, m_matl, patch, m_gn, 0 );
    new_dw->get( South, m_radFluxS_Label, m_matl, patch, m_gn, 0 );
    new_dw->get( Top,   m_radFluxT_Label, m_matl, patch, m_gn, 0 );
    new_dw->get( Bot,   m_radFluxB_Label, m_matl, patch, m_gn, 0 );

    for (auto iter = patch->getExtraCellIterator();!iter.done();iter++){
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
