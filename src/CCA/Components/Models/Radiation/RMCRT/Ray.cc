/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

//----- Ray.cc ----------------------------------------------
#include <CCA/Components/Models/Radiation/RMCRT/Ray.h>
#include <CCA/Components/Arches/ArchesStatsEnum.h>

#include <CCA/Ports/ApplicationInterface.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/AMR.h>
#include <Core/Grid/AMR_CoarsenRefine.h>
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Variables/PerPatchVars.h>
#include <Core/Math/MersenneTwister.h>
#include <Core/Util/DOUT.hpp>
#include <Core/Util/Timers/Timers.hpp>

#include <include/sci_defs/uintah_testdefs.h.in>

#include <sci_defs/visit_defs.h>

#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>

// TURN ON debug flag in src/Core/Math/MersenneTwister.h to compare with Ray:CPU
#define DEBUG -9      // 1: divQ, 2: boundFlux, 3: scattering

/*______________________________________________________________________
  TO DO

Optimizations:
  - Create a LevelDB.  Push an pull the communicated variables
  - DO: Reconstruct the fine level using interpolation and coarse level values
  - Investigate why floats are slow.
  - Temperatures af ints?
  - 2L flux coarsening on the boundaries
______________________________________________________________________*/



//______________________________________________________________________
//
using namespace Uintah;
using std::cout;
using std::endl;
using std::vector;
using std::string;

extern Dout g_ray_dbg;
extern Dout g_ray_BC;

//---------------------------------------------------------------------------
// Class: Constructor.
//---------------------------------------------------------------------------
Ray::Ray( const TypeDescription::Type FLT_DBL ) : RMCRTCommon( FLT_DBL)
{
//  d_boundFluxFiltLabel   = VarLabel::create( "boundFluxFilt",    CCVariable<Stencil7>::getTypeDescription() );
//  d_divQFiltLabel        = VarLabel::create( "divQFilt",         CCVariable<double>::getTypeDescription() );

  // Time Step
  m_timeStepLabel = VarLabel::create(timeStep_name, timeStep_vartype::getTypeDescription() );

  // internal variables for RMCRT
  d_flaggedCellsLabel    = VarLabel::create( "flaggedCells",     CCVariable<int>::getTypeDescription() );
  d_ROI_LoCellLabel      = VarLabel::create( "ROI_loCell",       minvec_vartype::getTypeDescription() );
  d_ROI_HiCellLabel      = VarLabel::create( "ROI_hiCell",       maxvec_vartype::getTypeDescription() );

  if ( d_FLT_DBL == TypeDescription::double_type ) {
    d_mag_grad_abskgLabel  = VarLabel::create( "mag_grad_abskg",   CCVariable<double>::getTypeDescription() );
    d_mag_grad_sigmaT4Label= VarLabel::create( "mag_grad_sigmaT4", CCVariable<double>::getTypeDescription() );
  } else {
    d_mag_grad_abskgLabel  = VarLabel::create( "mag_grad_abskg",   CCVariable<float>::getTypeDescription() );
    d_mag_grad_sigmaT4Label= VarLabel::create( "mag_grad_sigmaT4", CCVariable<float>::getTypeDescription() );
  }

  d_PPTimerLabel = VarLabel::create( "Ray_PPTimer", PerPatch<double>::getTypeDescription() );
  d_dbgCells.push_back( IntVector(1,2,2));


  //_____________________________________________
  //   Ordering for Surface Method
  // This block of code is used to properly place ray origins, and orient ray directions
  // onto the correct face.  This is necessary, because by default, the rays are placed
  // and oriented onto a default face, then require adjustment onto the proper face.
  d_dirIndexOrder[EAST]   = IntVector(2, 1, 0);
  d_dirIndexOrder[WEST]   = IntVector(2, 1, 0);
  d_dirIndexOrder[NORTH]  = IntVector(0, 2, 1);
  d_dirIndexOrder[SOUTH]  = IntVector(0, 2, 1);
  d_dirIndexOrder[TOP]    = IntVector(0, 1, 2);
  d_dirIndexOrder[BOT]    = IntVector(0, 1, 2);

  // Ordering is slightly different from 6Flux since here, rays pass through origin cell from the inside faces.
  d_dirSignSwap[EAST]     = IntVector(-1, 1,  1);
  d_dirSignSwap[WEST]     = IntVector( 1, 1,  1);
  d_dirSignSwap[NORTH]    = IntVector( 1, -1, 1);
  d_dirSignSwap[SOUTH]    = IntVector( 1, 1,  1);
  d_dirSignSwap[TOP]      = IntVector( 1, 1, -1);
  d_dirSignSwap[BOT]      = IntVector( 1, 1,  1);
}

//---------------------------------------------------------------------------
// Method: Destructor
//---------------------------------------------------------------------------
Ray::~Ray()
{
  VarLabel::destroy( m_timeStepLabel );

  VarLabel::destroy( d_mag_grad_abskgLabel );
  VarLabel::destroy( d_mag_grad_sigmaT4Label );
  VarLabel::destroy( d_flaggedCellsLabel );
  VarLabel::destroy( d_ROI_LoCellLabel );
  VarLabel::destroy( d_ROI_HiCellLabel );
  VarLabel::destroy( d_PPTimerLabel );

//  VarLabel::destroy( d_divQFiltLabel );
//  VarLabel::destroy( d_boundFluxFiltLabel );

  if( d_radiometer) {
    delete d_radiometer;
  }
}


//---------------------------------------------------------------------------
// Method: Problem setup (access to input file information)
//---------------------------------------------------------------------------
void
Ray::problemSetup( const ProblemSpecP& prob_spec,
                   const ProblemSpecP& rmcrtps,
                   const GridP&        grid)
{
  ProblemSpecP rmcrt_ps = rmcrtps;
  string rayDirSampleAlgo;

  rmcrt_ps->getWithDefault( "nDivQRays" ,       d_nDivQRays ,        10 );             // Number of rays per cell used to compute divQ
  rmcrt_ps->getWithDefault( "Threshold" ,       d_threshold ,      0.01 );             // When to terminate a ray
  rmcrt_ps->getWithDefault( "randomSeed",       d_isSeedRandom,    true );             // random or deterministic seed.
  rmcrt_ps->getWithDefault( "StefanBoltzmann",  d_sigma,           5.67051e-8);        // Units are W/(m^2-K)
  rmcrt_ps->getWithDefault( "solveBoundaryFlux" , d_solveBoundaryFlux, false );
  rmcrt_ps->getWithDefault( "CCRays"    ,       d_CCRays,          false );            // if true, forces rays to always have CC origins
  rmcrt_ps->getWithDefault( "nFluxRays" ,       d_nFluxRays,       1 );                // number of rays per cell for computation of boundary fluxes
  rmcrt_ps->getWithDefault( "sigmaScat"  ,      d_sigmaScat  ,      0 );               // scattering coefficient
  rmcrt_ps->getWithDefault( "allowReflect"   ,  d_allowReflect,     true );            // Allow for ray reflections. Make false for DOM comparisons.
  rmcrt_ps->getWithDefault( "solveDivQ"      ,  d_solveDivQ,        true );            // Allow for solving of divQ for flow cells.
  rmcrt_ps->getWithDefault( "applyFilter"    ,  d_applyFilter,      false );           // Allow filtering of boundFlux and divQ.
  rmcrt_ps->getWithDefault( "rayDirSampleAlgo", rayDirSampleAlgo,   "naive" );         // Change Monte-Carlo Sampling technique for RayDirection.

  if (rayDirSampleAlgo == "LatinHyperCube" ){
    d_rayDirSampleAlgo = LATIN_HYPER_CUBE;
    proc0cout << "  - Using Latin Hyper Cube method for selecting ray directions.\n";
  } else{
    proc0cout << "  - Using traditional Monte-Carlo method for selecting ray directions.\n";
  }

  //__________________________________
  //  Radiometer setup
  ProblemSpecP rad_ps = rmcrt_ps->findBlock("Radiometer");
  if( rad_ps ) {
    d_radiometer = scinew Radiometer( d_FLT_DBL );
    d_radiometer->problemSetup( prob_spec, rmcrtps, grid );
  }

  //__________________________________
  //  Warnings and bulletproofing
#ifndef RAY_SCATTER
  proc0cout<< "    - sigmaScat: " << d_sigmaScat << endl;
  if (d_sigmaScat>0) {
    std::ostringstream warn;
    warn << " ERROR:  To run a scattering case, you must use the following in your configure line..." << endl;
    warn << "                 --enable-ray-scatter" << endl;
    warn << "         To run a non-scattering case, please remove the line containing <sigmaScat> from your input file." << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
#endif

#ifdef RAY_SCATTER
  proc0cout<< "  - Ray scattering is enabled." << endl;
  if(d_sigmaScat<1e-99){
    proc0cout << "    WARNING:  You are running a non-scattering case but the following is in your configure line...\n"
              << "                    --enable-ray-scatter"
              << "            This will run slower than necessary."
              << "            You can remove --enable-ray-scatter from the configure line, re-configure and re-compile" << endl;
  }
#endif

  if( d_nDivQRays == 1 ){
    proc0cout << "    WARNING: You have specified only 1 ray to compute the radiative flux divergence." << endl;
    proc0cout << "              For higher accuracy specify nDivQRays greater than 2." << endl;
  }

  if( d_nFluxRays == 1 && d_whichAlgo != radiometerOnly){
    proc0cout << "    WARNING: You have specified only 1 ray to compute radiative fluxes on the boundaries." << endl;
  }


  //__________________________________
  //  Read in the algorithm section
  bool isMultilevel   = false;
  Algorithm algorithm = singleLevel;

  ProblemSpecP alg_ps = rmcrt_ps->findBlock("algorithm");

  if (alg_ps){

    string type="nullptr";

    if( !alg_ps->getAttribute("type", type) ){
      throw ProblemSetupException("RMCRT: No type specified for algorithm.  Please choose singleLevel, dataOnion or RMCRT_coarseLevel", __FILE__, __LINE__);
    }
    //__________________________________
    //  single level
    if (type == "singleLevel" ) {

      ProblemSpecP ROI_ps = alg_ps->findBlock("ROI_extents");
      ROI_ps->getAttribute( "type", type);
      ROI_ps->get( "length", d_maxRayLength );
      d_ROI_algo = boundedRayLength;
      proc0cout << "  RMCRT: The bounded ray length has been set to: ("<< d_maxRayLength << ")\n";

    //__________________________________
    //  Data Onion
    } else if (type == "dataOnion" ) {

      isMultilevel = true;
      algorithm    = dataOnion;

      alg_ps->getWithDefault( "haloCells",   d_haloCells,  IntVector(10,10,10) );
      alg_ps->get( "haloLength",         d_haloLength );
      alg_ps->get( "coarsenExtraCells" , d_coarsenExtraCells );

      //  Method for determining the extents of the ROI
      ProblemSpecP ROI_ps = alg_ps->findBlock( "ROI_extents" );
      ROI_ps->getAttribute( "type", type );

      if(type == "fixed" ) {

        d_ROI_algo = fixed;
        ROI_ps->get( "min", d_ROI_minPt );
        ROI_ps->get( "max", d_ROI_maxPt );

      } else if ( type == "dynamic" ) {

        d_ROI_algo = dynamic;
        ROI_ps->getWithDefault( "abskgd_threshold",   d_abskg_thld,   DBL_MAX );
        ROI_ps->getWithDefault( "sigmaT4d_threshold", d_sigmaT4_thld, DBL_MAX );

      } else if ( type == "patch_based" ){
        d_ROI_algo = patch_based;
      }

      //experimental virtual ROI.
      const char* roi_str = std::getenv("VIR_ROI");
      if(roi_str){
        if(atoi(roi_str)){
          if( grid->numLevels()>2 ){
            printf("*** Error: Virtual ROI supported only for two levels as of now. At: %s %d  ***\n", __FILE__, __LINE__);
            fflush(stdout);
            exit(1);
          }

          const char* roi_size_str = std::getenv("VIR_ROI_SIZE");
          if(roi_size_str){
            char temp_str[20];
            strcpy(temp_str, roi_size_str);
            const char s[2] = ",";
            char *token;
            token = strtok(temp_str, s);  /* get the first token */
            m_virtual_ROI[0] = atoi(token);
            token = strtok(NULL, s);
            m_virtual_ROI[1] =  atoi(token);
            token = strtok(NULL, s);
            m_virtual_ROI[2] =  atoi(token);

            m_use_virtual_ROI = atoi(roi_str);

            printf("Warning: Using experimental virtual ROI %d %d %d\n", m_virtual_ROI[0], m_virtual_ROI[1], m_virtual_ROI[2]);
          }
        }
      }

    //__________________________________
    //  rmcrt only on the coarse level
    } else if ( type == "RMCRT_coarseLevel" ) {
      isMultilevel = true;
      algorithm    = coarseLevel;
      d_ROI_algo   = entireDomain;
      alg_ps->require( "orderOfInterpolation", d_orderOfInterpolation );
      alg_ps->get( "coarsenExtraCells" , d_coarsenExtraCells );
    }
  }

  //__________________________________
  //  Logic for coarsening of cellType
  if (isMultilevel){
    string tmp = "nullptr";
    rmcrt_ps->get("cellTypeCoarsenLogic", tmp );
    if (tmp == "ROUNDDOWN"){
      d_cellTypeCoarsenLogic = ROUNDDOWN;
      proc0cout << "  RMCRT: When coarsening cellType any partial cell is ROUNDEDDOWN to a FLOW cell" << endl;
    }

    if (tmp == "ROUNDUP") {
      d_cellTypeCoarsenLogic = ROUNDUP;
      proc0cout << "  RMCRT: When coarsening cellType any partial cell is rounded UP to a INTRUSION cell" << endl;
    }
  }

  //__________________________________
  //  bulletproofing

  if ( Parallel::usingDevice() ) {              // GPU
    if( (algorithm == dataOnion && d_ROI_algo != patch_based ) ) {
      std::ostringstream warn;
      warn << "GPU:RMCRT:ERROR: ";
      warn << "At this time only ROI_extents type=\"patch_based\" work on the GPU";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
  }

  // special conditions when using floats and multi-level
  if ( d_FLT_DBL == TypeDescription::float_type && isMultilevel) {

    string abskgName = d_compAbskgLabel->getName();
    if ( rmcrt_ps->isLabelSaved( abskgName ) ){
      std::ostringstream warn;
      warn << "  RMCRT:WARNING: You're saving a variable ("<< abskgName << ") that doesn't exist on all levels."<< endl;
      warn << "  Use either: " << endl;
      warn << "    <save label = 'abskgRMCRT' />             (FLOAT version of abskg, local to RMCRT)" << endl;
      warn << "             or " << endl;
      warn << "    <save label = 'abskg'  levels = -1 />     ( only saved on the finest level )" << endl;
      //throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
  }
  d_sigma_over_pi = d_sigma/M_PI;

  proc0cout << "__________________________________ " << endl;
}

//______________________________________________________________________
//  Method:  Check that the boundary conditions have been set for temperature
//           and abskg
//______________________________________________________________________
void
Ray::BC_bulletproofing( const ProblemSpecP& rmcrtps,
                        const bool chk_temp,
                        const bool chk_absk )
{
  if(d_onOff_SetBCs == false ) {
   return;
  }

  ProblemSpecP rmcrt_ps = rmcrtps;

  //__________________________________
  // BC bulletproofing
  bool ignore_BC_bulletproofing  = false;
  rmcrt_ps->get( "ignore_BC_bulletproofing",  ignore_BC_bulletproofing );

  ProblemSpecP root_ps = rmcrt_ps->getRootNode();
  const MaterialSubset* mss = d_matlSet->getUnion();

  if( ignore_BC_bulletproofing == true ) {
    proc0cout << "\n\n______________________________________________________________________" << endl;
    proc0cout << "  WARNING: bulletproofing of the boundary conditions specs is off!";
    proc0cout << "   You're free to set any BC you'd like " << endl;
    proc0cout << "______________________________________________________________________\n\n" << endl;

  } else {

    if( chk_temp ){
      is_BC_specified(root_ps, d_compTempLabel->getName(), mss);
    }
    if( chk_absk ){
      is_BC_specified(root_ps, d_abskgBC_tag,              mss);
    }

    Vector periodic;
    ProblemSpecP grid_ps  = root_ps->findBlock("Grid");
    ProblemSpecP level_ps = grid_ps->findBlock("Level");
    level_ps->getWithDefault("periodic", periodic, Vector(0,0,0));

    if (periodic.length() != 0 ) {
      throw ProblemSetupException("\nERROR RMCRT:\nPeriodic boundary conditions are not allowed with Reverse Monte-Carlo Ray Tracing.", __FILE__, __LINE__);
    }
  }
}

//---------------------------------------------------------------------------
// Method: Schedule the ray tracer
// This task has both temporal and spatial scheduling and is tricky to follow
// The temporal scheduling is controlled by support for multiple, primary task-graphs
// The spatial scheduling only occurs if the radiometer is used and is specified
// by the radiometerPatchSet.
//---------------------------------------------------------------------------
void
Ray::sched_rayTrace( const LevelP& level,
                     SchedulerP& sched,
                     Task::WhichDW notUsed,
                     Task::WhichDW sigma_dw,
                     Task::WhichDW celltype_dw,
                     bool modifies_divQ )
{
  // Get the application so to record stats.
  m_application = sched->getApplication();

  string taskname = "Ray::rayTrace";
  Task *tsk = nullptr;

  int L = level->getIndex();
  Task::WhichDW abskg_dw = get_abskg_whichDW( L, d_abskgLabel );

  if ( RMCRTCommon::d_FLT_DBL == TypeDescription::double_type ) {
    tsk = scinew Task( taskname, this, &Ray::rayTrace<double>, modifies_divQ, abskg_dw, sigma_dw, celltype_dw );
  } else {
    tsk = scinew Task( taskname, this, &Ray::rayTrace<float>, modifies_divQ, abskg_dw, sigma_dw, celltype_dw );
  }

  printSchedule(level, g_ray_dbg, "Ray::sched_rayTrace");

  //__________________________________
  // Require an infinite number of ghost cells so you can access the entire domain.
  Ghost::GhostType  gac  = Ghost::AroundCells;
  int n_ghostCells = SHRT_MAX;


  //__________________________________
  // logic for determining number of ghostCells/d_halo
  if( d_ROI_algo == boundedRayLength ){

    Vector Dx     = level->dCell();
    Vector nCells = Vector( d_maxRayLength )/Dx;
    double length = nCells.length();
    n_ghostCells  = std::ceil( length );

    // ghost cell can't exceed number of cells on a level
    IntVector lo, hi;
    level->findCellIndexRange(lo,hi);
    IntVector diff = hi-lo;
    int nCellsLevel = Max( diff.x(), diff.y(), diff.z() );

    if (n_ghostCells > SHRT_MAX || n_ghostCells > nCellsLevel ) {
      proc0cout << "\n  WARNING  RMCRT:sched_rayTrace Clamping the number of ghostCells to SHRT_MAX, (n_ghostCells: " << n_ghostCells
                << ") max cells in any direction on a Levels: " << nCellsLevel << "\n\n";
      n_ghostCells = SHRT_MAX;
      d_ROI_algo = entireDomain;
    }
    d_haloCells = IntVector(n_ghostCells, n_ghostCells, n_ghostCells);
  }

  DOUT(g_ray_dbg, "    sched_rayTrace: number of ghost cells for all-to-all variables: (" << n_ghostCells << ")" );

  tsk->requiresVar( abskg_dw ,    d_abskgLabel  ,   gac, n_ghostCells );
  tsk->requiresVar( sigma_dw ,    d_sigmaT4Label,   gac, n_ghostCells );
  tsk->requiresVar( celltype_dw , d_cellTypeLabel , gac, n_ghostCells );


  if( modifies_divQ ) {
    tsk->modifiesVar( d_divQLabel );
    tsk->modifiesVar( d_boundFluxLabel );
    tsk->modifiesVar( d_radiationVolqLabel );
  } else {
    tsk->computesVar( d_divQLabel );
    tsk->computesVar( d_boundFluxLabel );
    tsk->computesVar( d_radiationVolqLabel );
  }

#ifdef USE_TIMER
  if( modifies_divQ ){
    tsk->modifiesVar( d_PPTimerLabel );
  } else {
    tsk->computesVar( d_PPTimerLabel );
  }
  sched->overrideVariableBehavior(d_PPTimerLabel->getName(),
                                  false, false, true, true, true);
#endif

  sched->addTask( tsk, level->eachPatch(), d_matlSet, RMCRTCommon::TG_RMCRT );

}



//---------------------------------------------------------------------------
// Method: The actual work of the ray tracer
//---------------------------------------------------------------------------
template< class T >
void
Ray::rayTrace( const ProcessorGroup* pg,
               const PatchSubset* patches,
               const MaterialSubset* matls,
               DataWarehouse* old_dw,
               DataWarehouse* new_dw,
               bool modifies_divQ,
               Task::WhichDW which_abskg_dw,
               Task::WhichDW which_sigmaT4_dw,
               Task::WhichDW which_celltype_dw )
{
  const Level* level = getLevel(patches);

#ifdef USE_TIMER
    // No carry forward just reset the time to zero.
    PerPatch< double > ppTimer = 0;

    for (int p=0; p<patches->size(); ++p) {
      const Patch* patch = patches->get(p);
      new_dw->put( ppTimer, d_PPTimerLabel, d_matl, patch);
    }
#endif

  //__________________________________
  //
  MTRand mTwister;

  DataWarehouse* abskg_dw    = new_dw->getOtherDataWarehouse(which_abskg_dw);
  DataWarehouse* sigmaT4_dw  = new_dw->getOtherDataWarehouse(which_sigmaT4_dw);
  DataWarehouse* celltype_dw = new_dw->getOtherDataWarehouse(which_celltype_dw);

  constCCVariable< T > sigmaT4OverPi;
  constCCVariable< T > abskg;
  constCCVariable<int>  celltype;

  if ( d_ROI_algo == entireDomain ){
    abskg_dw->getLevel(   abskg ,         d_abskgLabel ,   d_matl , level );
    sigmaT4_dw->getLevel( sigmaT4OverPi , d_sigmaT4Label,  d_matl , level );
    celltype_dw->getLevel( celltype ,     d_cellTypeLabel, d_matl , level );
  }

  // patch loop
  for (int p=0; p < patches->size(); p++){

    Timers::Simple timer;
    timer.start();

    const Patch* patch = patches->get(p);
    printTask(patches,patch,g_ray_dbg,"Doing Ray::rayTrace");


//    if ( d_isSeedRandom == false ){     Disable until the code compares with nightly GS -Todd
//      mTwister.seed(patch->getID());
//    }

    CCVariable<double> divQ;
    CCVariable<Stencil7> boundFlux;
    CCVariable<double> radiationVolq;

    if( modifies_divQ ){
      new_dw->getModifiable( divQ,         d_divQLabel,          d_matl, patch );
      new_dw->getModifiable( boundFlux,    d_boundFluxLabel,     d_matl, patch );
      new_dw->getModifiable( radiationVolq,d_radiationVolqLabel, d_matl, patch );
    }else{
      new_dw->allocateAndPut( divQ,         d_divQLabel,          d_matl, patch );
      new_dw->allocateAndPut( boundFlux,    d_boundFluxLabel,     d_matl, patch );
      new_dw->allocateAndPut( radiationVolq,d_radiationVolqLabel, d_matl, patch );
      divQ.initialize( 0.0 );
      radiationVolq.initialize( 0.0 );

      for (CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        boundFlux[c].initialize(0.0);
      }
    }

    IntVector ROI_Lo = IntVector(-SHRT_MAX,-SHRT_MAX,-SHRT_MAX );
    IntVector ROI_Hi = IntVector( SHRT_MAX, SHRT_MAX, SHRT_MAX );
    //__________________________________
    //  If ray length distance is used
    if ( d_ROI_algo == boundedRayLength ){

      patch->computeVariableExtentsWithBoundaryCheck(CCVariable<double>::getTypeDescription()->getType(), IntVector(0,0,0),
                                                     Ghost::AroundCells, d_haloCells.x(), ROI_Lo, ROI_Hi);
      DOUT(g_ray_dbg, "  ROI: " << ROI_Lo << " "<< ROI_Hi );
      abskg_dw->getRegion(   abskg,          d_abskgLabel ,   d_matl, level, ROI_Lo, ROI_Hi );
      sigmaT4_dw->getRegion( sigmaT4OverPi,  d_sigmaT4Label,  d_matl, level, ROI_Lo, ROI_Hi );
      celltype_dw->getRegion( celltype,      d_cellTypeLabel, d_matl, level, ROI_Lo, ROI_Hi );
    }

    //__________________________________
    //  BULLETPROOFING
//    if ( level->isNonCubic() ){
    if( false) {

      IntVector l = abskg.getLowIndex();
      IntVector h = abskg.getHighIndex();

      CellIterator iterLim = CellIterator(l, h);
      for(CellIterator iter = iterLim; !iter.done();iter++) {
        IntVector c = *iter;

        if ( std::isinf( abskg[c] )         || std::isnan( abskg[c] )  ||
             std::isinf( sigmaT4OverPi[c] ) || std::isnan( sigmaT4OverPi[c] ) ) {
          std::ostringstream warn;
          warn<< "ERROR:Ray::rayTrace   abskg or sigmaT4 is non-physical \n"
              << "     c:   " << c << " location: " << level->getCellPosition(c) << "\n"
              << "     ROI: " << ROI_Lo << " "<< ROI_Hi << "\n"
              << "          " << *patch << "\n"
              << " ( abskg[c]: " << abskg[c] << ", sigmaT4OverPi[c]: " << sigmaT4OverPi[c] << ")\n";

          cout << warn.str() << endl;
          throw InternalError( warn.str(), __FILE__, __LINE__ );
        }
      }
    }

    unsigned long int size = 0;                   // current size of PathIndex
    Vector Dx = patch->dCell();                   // cell spacing

    //______________________________________________________________________
    //           R A D I O M E T E R
    //______________________________________________________________________

    if (d_radiometer) {
      d_radiometer->radiometerFlux< T >( patch, level, new_dw, mTwister, sigmaT4OverPi, abskg, celltype, modifies_divQ );
    }

    //______________________________________________________________________
    //          B O U N D A R Y F L U X
    //______________________________________________________________________
    if( d_solveBoundaryFlux ) {

      //__________________________________
      //
      vector <int> rand_i( d_rayDirSampleAlgo == LATIN_HYPER_CUBE ? d_nFluxRays : 0);  // only needed for LHC scheme

      for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector origin = *iter;

        if (celltype[origin] != d_flowCell) {
          continue;
        }

        // A given flow cell may have 0,1,2,3,4,5, or 6 faces that are adjacent to a wall.
        // boundaryFaces is the vector that contains the list of which faces are adjacent to a wall
        vector<int> boundaryFaces;
        boundaryFaces.clear();

        // determine if origin has one or more boundary faces, and if so, populate boundaryFaces vector
        boundFlux[origin].p = has_a_boundary(origin, celltype, boundaryFaces);

        Point CC_pos = level->getCellPosition(origin);
        //__________________________________
        // Loop over boundary faces of the cell and compute incident radiative flux
        for (vector<int>::iterator it=boundaryFaces.begin() ; it < boundaryFaces.end(); it++ ){

          int RayFace = *it;
          int UintahFace[6] = {WEST,EAST,SOUTH,NORTH,BOT,TOP};

          double sumI         = 0;
          double sumProjI     = 0;
          double sumI_prev    = 0;
          double sumCosTheta  = 0;    // used to force sumCosTheta/nRays == 0.5 or  sum (d_Omega * cosTheta) == pi

          if (d_rayDirSampleAlgo == LATIN_HYPER_CUBE){
            randVector(rand_i, mTwister, origin);
          }


          //__________________________________
          // Flux ray loop
          for (int iRay=0; iRay < d_nFluxRays; iRay++){

            Vector direction_vector;
            Vector rayOrigin;
            double cosTheta;

            if ( d_rayDirSampleAlgo == LATIN_HYPER_CUBE ){        // Latin-Hyper-Cube sampling
              rayDirectionHyperCube_cellFace( mTwister, origin, d_dirIndexOrder[RayFace], d_dirSignSwap[RayFace], iRay,
                                              direction_vector, cosTheta, rand_i[iRay],iRay);
            } else{                                               // Naive Monte-Carlo sampling
              rayDirection_cellFace( mTwister, origin, d_dirIndexOrder[RayFace], d_dirSignSwap[RayFace], iRay,
                                     direction_vector, cosTheta );
            }

            rayLocation_cellFace( mTwister, RayFace, Dx, CC_pos, rayOrigin);

            updateSumI<T>( level, direction_vector, rayOrigin, origin, Dx, sigmaT4OverPi, abskg, celltype, size, sumI, mTwister);

            sumProjI    += cosTheta * (sumI - sumI_prev);              // must subtract sumI_prev, since sumI accumulates intensity

            sumCosTheta += cosTheta;

            sumI_prev    = sumI;

          } // end of flux ray loop

          sumProjI = sumProjI * (double) d_nFluxRays/sumCosTheta/2.0; // This operation corrects for error in the first moment over a half range of the solid angle (Modest Radiative Heat Transfer page 545 1rst edition)

          //__________________________________
          //  Compute Net Flux to the boundary
          int face = UintahFace[RayFace];
          boundFlux[origin][ face ] = sumProjI * 2 *M_PI/ (double) d_nFluxRays;

/*`==========TESTING==========*/
#if (DEBUG == 2)
          if( isDbgCell(origin) ) {
            printf( "\n      [%d, %d, %d]  face: %d sumProjI:  %g BoundaryFlux: %g\n",
                  origin.x(), origin.y(), origin.z(), face, sumProjI, boundFlux[origin][ face ]);
          }
#endif
/*===========TESTING==========`*/

        } // boundary faces loop
      }  // end cell iterator
    }   // end if d_solveBoundaryFlux


    //______________________________________________________________________
    //         S O L V E   D I V Q
    //______________________________________________________________________
    if( d_solveDivQ){

    //__________________________________
    //
      vector <int> rand_i( d_rayDirSampleAlgo == LATIN_HYPER_CUBE ? d_nDivQRays : 0);  // only needed for LHC scheme

      for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector origin = *iter;

        // don't compute in intrusions and walls
        if( celltype[origin] != d_flowCell ){
          continue;
        }

        if (d_rayDirSampleAlgo == LATIN_HYPER_CUBE){
          randVector(rand_i, mTwister, origin);
        }
        double sumI = 0;
        Point CC_pos = level->getCellPosition(origin);

        // ray loop
        for (int iRay=0; iRay < d_nDivQRays; iRay++){

          Vector direction_vector;
          if (d_rayDirSampleAlgo == LATIN_HYPER_CUBE){        // Latin-Hyper-Cube sampling
            direction_vector =findRayDirectionHyperCube(mTwister, origin, iRay, rand_i[iRay],iRay );
          }else{                                              // Naive Monte-Carlo sampling
            direction_vector =findRayDirection(mTwister, origin, iRay );
          }

          Vector rayOrigin;
          ray_Origin( mTwister, CC_pos, Dx, d_CCRays, rayOrigin);

          updateSumI< T >( level, direction_vector, rayOrigin, origin, Dx,  sigmaT4OverPi, abskg, celltype, size, sumI, mTwister);

        }  // Ray loop

        //__________________________________
        //  Compute divQ
        divQ[origin] = -4.0 * M_PI * abskg[origin] * ( sigmaT4OverPi[origin] - (sumI/d_nDivQRays) );

        // radiationVolq is the incident energy per cell (W/m^3) and is necessary when particle heat transfer models (i.e. Shaddix) are used
        radiationVolq[origin] = 4.0 * M_PI * (sumI/d_nDivQRays) ;
        /*`==========TESTING==========*/
#if DEBUG == 1
        if( isDbgCell(origin) ) {
          printf( "\n      [%d, %d, %d]  sumI: %g  divQ: %g radiationVolq: %g  abskg: %g,    sigmaT4: %g \n",
                  origin.x(), origin.y(), origin.z(), sumI,divQ[origin], radiationVolq[origin],abskg[origin], sigmaT4OverPi[origin]);
        }
#endif
/*===========TESTING==========`*/
      }  // end cell iterator
    }  // end of if(_solveDivQ)

    timer.stop();

#ifdef ADD_PERFORMANCE_STATS
    // Add in the patch stat, recording for each patch.
    m_application->getApplicationStats()[ (ApplicationInterface::ApplicationStatsEnum) RMCRTPatchTime ] += timer().milliseconds();
    m_application->getApplicationStats()[ (ApplicationInterface::ApplicationStatsEnum) RMCRTPatchSize ] += size;
    m_application->getApplicationStats()[ (ApplicationInterface::ApplicationStatsEnum) RMCRTPatchEfficiency ] += size / timer().seconds();
    // For each stat recorded increment the count so to get a per patch value.
    m_application->getApplicationStats().incrCount( (ApplicationInterface::ApplicationStatsEnum) RMCRTPatchTime );
    m_application->getApplicationStats().incrCount( (ApplicationInterface::ApplicationStatsEnum) RMCRTPatchSize );
    m_application->getApplicationStats().incrCount( (ApplicationInterface::ApplicationStatsEnum) RMCRTPatchEfficiency );
#endif

    if (patch->getGridIndex() == 0) {
      cout << endl
           << " RMCRT REPORT: Patch 0" << endl
           << " Used " << timer().milliseconds()
           << " milliseconds of CPU time. \n" << endl // Convert time to ms
           << " Size: " << size << endl
           << " Efficiency: " << size / timer().seconds()
           << " steps per sec" << endl
           << endl;
    }
#ifdef USE_TIMER
    PerPatch< double > ppTimer = timer().seconds();
    new_dw->put( ppTimer, d_PPTimerLabel, d_matl, patch);
#endif
  }  //end patch loop
}  // end ray trace method



//---------------------------------------------------------------------------
// Ray tracing using the multilevel data onion scheme
//---------------------------------------------------------------------------
void
Ray::sched_rayTrace_dataOnion( const LevelP& level,
                               SchedulerP& sched,
                               Task::WhichDW notUsed,
                               Task::WhichDW sigma_dw,
                               Task::WhichDW celltype_dw,
                               bool modifies_divQ )
{
  // Get the application so to record stats.
  m_application = sched->getApplication();

  int maxLevels = level->getGrid()->numLevels() - 1;
  int L_indx = level->getIndex();

  if (L_indx != maxLevels) {     // only schedule on the finest level
    return;
  }

  Task* tsk = nullptr;
  string taskname = "";

  Task::WhichDW NotUsed = Task::None;

  taskname = "Ray::rayTrace_dataOnion";
  if (RMCRTCommon::d_FLT_DBL == TypeDescription::double_type) {
    tsk = scinew Task(taskname, this, &Ray::rayTrace_dataOnion<double>, modifies_divQ, NotUsed, sigma_dw, celltype_dw);
  } else {
    tsk = scinew Task(taskname, this, &Ray::rayTrace_dataOnion<float>, modifies_divQ, NotUsed, sigma_dw, celltype_dw);
  }

  printSchedule(level, g_ray_dbg, taskname);

  Task::MaterialDomainSpec  ND  = Task::NormalDomain;
  Ghost::GhostType         gac  = Ghost::AroundCells;
  Task::WhichDW        abskg_dw = get_abskg_whichDW(L_indx, d_abskgLabel);

  // finest level:
  if ( d_ROI_algo == patch_based ) {          // patch_based we know the number of ghostCells
    //__________________________________
    // logic for determining number d_haloCells
    if( d_haloLength > 0 ){
      Vector Dx     = level->dCell();
      Vector nCells = Vector( d_haloLength )/Dx;
      double length = nCells.length();
      int n_Cells   = RoundUp( length );
      d_haloCells   = IntVector( n_Cells, n_Cells, n_Cells );
    }
    if (d_haloCells < IntVector(2,2,2) ){
      std::ostringstream warn;
      warn << "RMCRT:DataOnion ERROR: ";
      warn << "The number of halo cells must be > (1,1,1) ("<< d_haloCells << ")";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    int maxElem = Max( d_haloCells.x(), d_haloCells.y(), d_haloCells.z() );
    tsk->requiresVar( abskg_dw,     d_abskgLabel,     gac, maxElem );
    tsk->requiresVar( sigma_dw,     d_sigmaT4Label,   gac, maxElem );
    tsk->requiresVar( celltype_dw , d_cellTypeLabel , gac, maxElem );
  } else {                                        // we don't know the number of ghostCells so get everything
    tsk->requiresVar( abskg_dw,      d_abskgLabel,     gac, SHRT_MAX );
    tsk->requiresVar( sigma_dw,      d_sigmaT4Label,   gac, SHRT_MAX );
    tsk->requiresVar( celltype_dw ,  d_cellTypeLabel , gac, SHRT_MAX );
  }

  // TODO This is a temporary fix until we can generalize GPU/CPU carry forward functionality.
  if (!(Uintah::Parallel::usingDevice())) {
    // needed for carry Forward
    tsk->requiresVar( Task::OldDW, d_divQLabel,          d_gn, 0 );
    tsk->requiresVar( Task::OldDW, d_boundFluxLabel,     d_gn, 0);
    tsk->requiresVar( Task::OldDW, d_radiationVolqLabel, d_gn, 0 );
  }


  if (d_ROI_algo == dynamic) {
    tsk->requiresVar( Task::NewDW, d_ROI_LoCellLabel );
    tsk->requiresVar( Task::NewDW, d_ROI_HiCellLabel );
  }

  // declare requires for all coarser levels
  for (int l = 0; l < maxLevels; ++l) {
    int offset = maxLevels - l;
    Task::WhichDW abskg_dw_CL = get_abskg_whichDW( l, d_abskgLabel );
    tsk->requiresVar( abskg_dw_CL, d_abskgLabel,    nullptr, Task::CoarseLevel, offset, nullptr, ND, gac, SHRT_MAX );
    tsk->requiresVar( sigma_dw,    d_sigmaT4Label,  nullptr, Task::CoarseLevel, offset, nullptr, ND, gac, SHRT_MAX );
    tsk->requiresVar( celltype_dw, d_cellTypeLabel, nullptr, Task::CoarseLevel, offset, nullptr, ND, gac, SHRT_MAX );

    proc0cout << "WARNING: RMCRT High communication costs on level: " << l
              << ".  Variables from every patch on this level are communicated to every patch on the finest level."
              << endl;
  }

  if( modifies_divQ ){
    tsk->modifiesVar( d_divQLabel );
    tsk->modifiesVar( d_boundFluxLabel );
    tsk->modifiesVar( d_radiationVolqLabel );
  } else {
    tsk->computesVar( d_divQLabel );
    tsk->computesVar( d_boundFluxLabel );
    tsk->computesVar( d_radiationVolqLabel );
  }

#ifdef USE_TIMER
  if( modifies_divQ ){
    tsk->modifiesVar( d_PPTimerLabel );
  } else {
    tsk->computesVar( d_PPTimerLabel );
  }
  sched->overrideVariableBehavior(d_PPTimerLabel->getName(),
                                  false, false, true, true, true);
#endif

  sched->addTask( tsk, level->eachPatch(), d_matlSet, RMCRTCommon::TG_RMCRT );
}


//---------------------------------------------------------------------------
// Ray tracer using the multilevel "data onion" scheme
//---------------------------------------------------------------------------
template< class T>
void
Ray::rayTrace_dataOnion( const ProcessorGroup* pg,
                         const PatchSubset* finePatches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw,
                         bool modifies_divQ,
                         Task::WhichDW notUsed,
                         Task::WhichDW which_sigmaT4_dw,
                         Task::WhichDW which_celltype_dw )
{

  const Level* fineLevel = getLevel(finePatches);

#ifdef USE_TIMER
    // No carry forward just reset the time to zero.
    PerPatch< double > ppTimer = 0;

    for (int p=0; p<finePatches->size(); ++p) {
      const Patch* finePatch = finePatches->get(p);
      new_dw->put( ppTimer, d_PPTimerLabel, d_matl, finePatch );
    }
#endif

  //__________________________________
  //
  int maxLevels    = fineLevel->getGrid()->numLevels();
  int levelPatchID = fineLevel->getPatch(0)->getID();
  LevelP level_0 = new_dw->getGrid()->getLevel(0);
  MTRand mTwister;

  //__________________________________
  // retrieve the coarse level data
  // compute the level dependent variables that are constant
  std::vector< constCCVariable< T > > abskg(maxLevels);
  std::vector< constCCVariable< T > >sigmaT4OverPi(maxLevels);
  std::vector< constCCVariable<int> >cellType(maxLevels);
  constCCVariable< T > abskg_fine;
  constCCVariable< T > sigmaT4OverPi_fine;

  DataWarehouse* sigmaT4_dw  = new_dw->getOtherDataWarehouse(which_sigmaT4_dw);
  DataWarehouse* celltype_dw = new_dw->getOtherDataWarehouse(which_celltype_dw);

  vector<Vector> Dx(maxLevels);

  for(int L = 0; L<maxLevels; L++) {
    LevelP level = new_dw->getGrid()->getLevel(L);

    if (level->hasFinerLevel() ) {                               // coarse level data
      DataWarehouse* abskg_dw = get_abskg_dw(L, d_abskgLabel, new_dw);

      abskg_dw->getLevel(   abskg[L]   ,       d_abskgLabel ,   d_matl , level.get_rep() );
      sigmaT4_dw->getLevel( sigmaT4OverPi[L] , d_sigmaT4Label,  d_matl , level.get_rep() );
      celltype_dw->getLevel( cellType[L] ,     d_cellTypeLabel, d_matl , level.get_rep() );
      DOUT( g_ray_dbg, "    RT DataOnion: getting coarse level data L-" << L );
    }
    Vector dx = level->dCell();
    Dx[L] = dx;
  }

  IntVector fineLevel_ROI_Lo = IntVector( -9,-9,-9 );
  IntVector fineLevel_ROI_Hi = IntVector( -9,-9,-9 );
  vector<IntVector> regionLo( maxLevels );
  vector<IntVector> regionHi( maxLevels );

  //__________________________________
  //  retrieve fine level data & compute the extents (dynamic and fixed )
  if ( d_ROI_algo == fixed || d_ROI_algo == dynamic ) {

    int L = maxLevels - 1;
    DataWarehouse* abskg_dw = get_abskg_dw(L, d_abskgLabel, new_dw);

    const Patch* notUsed = 0;
    computeExtents(level_0, fineLevel, notUsed, maxLevels, new_dw,
                   fineLevel_ROI_Lo, fineLevel_ROI_Hi, regionLo,  regionHi);

    DOUT( g_ray_dbg, "    RT DataOnion:  getting fine level data across L-" << L << " " << fineLevel_ROI_Lo << " " << fineLevel_ROI_Hi );
    abskg_dw->getRegion(   abskg[L]   ,       d_abskgLabel ,   d_matl, fineLevel, fineLevel_ROI_Lo, fineLevel_ROI_Hi );
    sigmaT4_dw->getRegion( sigmaT4OverPi[L] , d_sigmaT4Label,  d_matl, fineLevel, fineLevel_ROI_Lo, fineLevel_ROI_Hi );
    celltype_dw->getRegion( cellType[L] ,     d_cellTypeLabel, d_matl, fineLevel, fineLevel_ROI_Lo, fineLevel_ROI_Hi );
  }

  abskg_fine         = abskg[maxLevels-1];
  sigmaT4OverPi_fine = sigmaT4OverPi[maxLevels-1];

  Timers::Simple timer;
  timer.start();

  // Determine the size of the domain.
  BBox domain_BB;
  level_0->getInteriorSpatialRange(domain_BB);                 // edge of computational domain

  //  patch loop
  for (int p = 0; p < finePatches->size(); p++) {

    const Patch* finePatch = finePatches->get(p);
    printTask(finePatches, finePatch,g_ray_dbg,"Doing Ray::rayTrace_dataOnion");
    if ( d_isSeedRandom == false ){
      mTwister.seed(finePatch->getID());
    }
     //__________________________________
    //  retrieve fine level data ( patch_based )
    if ( d_ROI_algo == patch_based ) {

      computeExtents(level_0, fineLevel, finePatch, maxLevels, new_dw,
                     fineLevel_ROI_Lo, fineLevel_ROI_Hi,
                     regionLo,  regionHi);

      int L = maxLevels - 1;
      DataWarehouse* abskg_dw = get_abskg_dw(L, d_abskgLabel, new_dw);

      DOUT( g_ray_dbg, "    RT DataOnion: getting fine level data across L-" << L );

      abskg_dw->getRegion(   abskg[L]   ,       d_abskgLabel ,  d_matl , fineLevel, fineLevel_ROI_Lo, fineLevel_ROI_Hi );
      sigmaT4_dw->getRegion( sigmaT4OverPi[L] , d_sigmaT4Label, d_matl , fineLevel, fineLevel_ROI_Lo, fineLevel_ROI_Hi );
      celltype_dw->getRegion( cellType[L] ,     d_cellTypeLabel, d_matl ,fineLevel, fineLevel_ROI_Lo, fineLevel_ROI_Hi );
      abskg_fine         = abskg[L];
      sigmaT4OverPi_fine = sigmaT4OverPi[L];
    }

    CCVariable<double> divQ_fine;
    CCVariable<Stencil7> boundFlux_fine;
    CCVariable<double> radiationVolq_fine;

    if( modifies_divQ ){
      old_dw->getModifiable( divQ_fine,         d_divQLabel,          d_matl, finePatch );
      new_dw->getModifiable( boundFlux_fine,    d_boundFluxLabel,     d_matl, finePatch );
      old_dw->getModifiable( radiationVolq_fine,d_radiationVolqLabel, d_matl, finePatch );
    }else{
      new_dw->allocateAndPut( divQ_fine,          d_divQLabel,          d_matl, finePatch );
      new_dw->allocateAndPut( boundFlux_fine,     d_boundFluxLabel,     d_matl, finePatch );
      new_dw->allocateAndPut( radiationVolq_fine, d_radiationVolqLabel, d_matl, finePatch );
      divQ_fine.initialize( 0.0 );
      radiationVolq_fine.initialize( 0.0 );

      for (CellIterator iter = finePatch->getExtraCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        boundFlux_fine[c].initialize(0.0);
      }
    }


    int my_L = maxLevels - 1;
    //______________________________________________________________________
    //          B O U N D A R Y F L U X
    //______________________________________________________________________

    unsigned long int nFluxRaySteps = 0;
    if( d_solveBoundaryFlux){

      //__________________________________
      //
      vector <int> rand_i( d_rayDirSampleAlgo == LATIN_HYPER_CUBE ? d_nFluxRays : 0);  // only needed for LHC scheme

      for (CellIterator iter = finePatch->getCellIterator(); !iter.done(); iter++){
        IntVector origin = *iter;

        // A given flow cell may have 0,1,2,3,4,5, or 6 faces that are adjacent to a wall.
        // boundaryFaces is the vector that contains the list of which faces are adjacent to a wall
        vector<int> boundaryFaces;
        boundaryFaces.clear();

        // determine if origin has one or more boundary faces, and if so, populate boundaryFaces vector
        boundFlux_fine[origin].p = has_a_boundary(origin, cellType[my_L], boundaryFaces);

        Point CC_pos = fineLevel->getCellPosition(origin);
        //__________________________________
        // Loop over boundary faces of the cell and compute incident radiative flux
        for (vector<int>::iterator it=boundaryFaces.begin() ; it < boundaryFaces.end(); it++ ){

          int RayFace = *it;
          int UintahFace[6] = {WEST,EAST,SOUTH,NORTH,BOT,TOP};

          double sumI         = 0;
          double sumProjI     = 0;
          double sumI_prev    = 0;
          double sumCosTheta  = 0;    // used to force sumCosTheta/nRays == 0.5 or  sum (d_Omega * cosTheta) == pi

          if (d_rayDirSampleAlgo == LATIN_HYPER_CUBE){
            randVector(rand_i, mTwister, origin);
          }

          //__________________________________
          // Flux ray loop
          for (int iRay=0; iRay < d_nFluxRays; iRay++){

            Vector direction_vector;
            Vector rayOrigin;
            double cosTheta;

            if ( d_rayDirSampleAlgo == LATIN_HYPER_CUBE ){        // Latin-Hyper-Cube sampling
              rayDirectionHyperCube_cellFace( mTwister, origin, d_dirIndexOrder[RayFace], d_dirSignSwap[RayFace], iRay,
                                              direction_vector, cosTheta, rand_i[iRay],iRay);
            } else{                                               // Naive Monte-Carlo sampling
              rayDirection_cellFace( mTwister, origin, d_dirIndexOrder[RayFace], d_dirSignSwap[RayFace], iRay,
                                     direction_vector, cosTheta );
            }

            rayLocation_cellFace( mTwister, RayFace, Dx[my_L], CC_pos, rayOrigin);

            updateSumI_ML< T >( direction_vector, rayOrigin, origin, Dx, domain_BB, maxLevels, fineLevel,
                         fineLevel_ROI_Lo, fineLevel_ROI_Hi, regionLo, regionHi, sigmaT4OverPi, abskg, cellType,
                         nFluxRaySteps, sumI, mTwister );

            sumProjI    += cosTheta * (sumI - sumI_prev);              // must subtract sumI_prev, since sumI accumulates intensity

            sumCosTheta += cosTheta;

            sumI_prev    = sumI;

          } // end of flux ray loop

          sumProjI = sumProjI * (double) d_nFluxRays/sumCosTheta/2.0; // This operation corrects for error in the first moment over a half range of the solid angle (Modest Radiative Heat Transfer page 545 1rst edition)

          //__________________________________
          //  Compute Net Flux to the boundary
          int face = UintahFace[RayFace];
          boundFlux_fine[origin][ face ] = sumProjI * 2 *M_PI/ (double) d_nFluxRays;

/*`==========TESTING==========*/
#if (DEBUG == 2)
          if( isDbgCell(origin) ) {
            printf( "\n      [%d, %d, %d]  face: %d sumProjI:  %g BoundaryFlux: %g\n",
                  origin.x(), origin.y(), origin.z(), face, sumProjI, boundFlux_fine[origin][ face ]);
          }
#endif
/*===========TESTING==========`*/

        } // boundary faces loop
      }  // end cell iterator
    }   // end if d_solveBoundaryFlux

    unsigned long int nRaySteps = 0;

    //______________________________________________________________________
    //         S O L V E   D I V Q
    //______________________________________________________________________
    if (d_solveDivQ) {

      vector <int> rand_i( d_rayDirSampleAlgo == LATIN_HYPER_CUBE  ? d_nDivQRays : 0);  // only needed for LHC scheme

      for (CellIterator iter = finePatch->getCellIterator(); !iter.done(); iter++){

        IntVector origin = *iter;

        // don't compute in intrusions and walls
        if(cellType[my_L][origin] != d_flowCell ){
          continue;
        }

        Point CC_pos = fineLevel->getCellPosition(origin);

        if (d_rayDirSampleAlgo == LATIN_HYPER_CUBE){
          randVector(rand_i, mTwister, origin);
        }

        double sumI = 0;

        //__________________________________
        //  ray loop
        for (int iRay=0; iRay < d_nDivQRays; iRay++){

          Vector direction_vector;
          if (d_rayDirSampleAlgo== LATIN_HYPER_CUBE){       // Latin-Hyper-Cube sampling
            direction_vector =findRayDirectionHyperCube( mTwister, origin, iRay,rand_i[iRay],iRay );
          }else{                                            // Naive Monte-Carlo sampling
            direction_vector =findRayDirection( mTwister, origin, iRay );
          }

          Vector rayOrigin;
          int my_L = maxLevels - 1;
          ray_Origin( mTwister, CC_pos, Dx[my_L], d_CCRays, rayOrigin );

          updateSumI_ML< T >( direction_vector, rayOrigin, origin, Dx, domain_BB, maxLevels, fineLevel,
                         fineLevel_ROI_Lo, fineLevel_ROI_Hi, regionLo, regionHi, sigmaT4OverPi, abskg, cellType,
                         nRaySteps, sumI, mTwister );


        }  // Ray loop

        //__________________________________
        //  Compute divQ
        divQ_fine[origin] = -4.0 * M_PI * abskg_fine[origin] * ( sigmaT4OverPi_fine[origin] - (sumI/d_nDivQRays) );

        // radiationVolq is the incident energy per cell (W/m^3) and is necessary when particle heat transfer models (i.e. Shaddix) are used
        radiationVolq_fine[origin] = 4.0 * M_PI * (sumI/d_nDivQRays) ;

/*`==========TESTING==========*/
#if DEBUG == 1
    if( isDbgCell(origin) ) {
          printf( "\n      [%d, %d, %d]  sumI: %g  divQ: %g radiationVolq: %g  abskg: %g,    sigmaT4: %g \n\n",
                    origin.x(), origin.y(), origin.z(), sumI,divQ_fine[origin], radiationVolq_fine[origin],abskg_fine[origin], sigmaT4OverPi_fine[origin]);
    }
#endif
/*===========TESTING==========`*/
      }  // end cell iterator
    }  // end of if(_solveDivQ)

    //__________________________________
    //
    timer.stop();

#ifdef ADD_PERFORMANCE_STATS
    // Add in the patch stat, recording for each patch.
    m_application->getApplicationStats()[ (ApplicationInterface::ApplicationStatsEnum) RMCRTPatchTime ] += timer().milliseconds();
    m_application->getApplicationStats()[ (ApplicationInterface::ApplicationStatsEnum) RMCRTPatchSize ] += size;
    m_application->getApplicationStats()[ (ApplicationInterface::ApplicationStatsEnum) RMCRTPatchEfficiency ] += size / timer().seconds();
    // For each stat recorded increment the count so to get a per patch value.
    m_application->getApplicationStats().incrCount( (ApplicationInterface::ApplicationStatsEnum) RMCRTPatchTime );
    m_application->getApplicationStats().incrCount( (ApplicationInterface::ApplicationStatsEnum) RMCRTPatchSize );
    m_application->getApplicationStats().incrCount( (ApplicationInterface::ApplicationStatsEnum) RMCRTPatchEfficiency );
#endif

    if (finePatch->getGridIndex() == levelPatchID) {
      cout << endl
           << " RMCRT REPORT: Patch " << levelPatchID << endl
           << " Used "<< timer().milliseconds()
           << " milliseconds of CPU time. \n" << endl // Convert time to ms
           << " nRaySteps: " << nRaySteps
           << " nFluxRaySteps: " << nFluxRaySteps << endl
           << " Efficiency: " << nRaySteps / timer().seconds()
           << " steps per sec" << endl
           << endl;
    }

#ifdef USE_TIMER
    PerPatch< double > ppTimer = timer().seconds();
    new_dw->put( ppTimer, d_PPTimerLabel, d_matl, finePatch );
#endif
  }  // end finePatch loop
}  // end ray trace method


//______________________________________________________________________
//
void
Ray::computeExtents(LevelP level_0,
                    const Level* fineLevel,
                    const Patch* patch,
                    const int maxLevels,
                    DataWarehouse* new_dw,
                    IntVector& fineLevel_ROI_Lo,
                    IntVector& fineLevel_ROI_Hi,
                    vector<IntVector>& regionLo,
                    vector<IntVector>& regionHi)
{
  //__________________________________
  //   fine level region of interest ROI
  if( d_ROI_algo == dynamic ){

    minvec_vartype lo;
    maxvec_vartype hi;
    new_dw->get( lo, d_ROI_LoCellLabel );
    new_dw->get( hi, d_ROI_HiCellLabel );
    fineLevel_ROI_Lo = roundNearest( Vector(lo) );
    fineLevel_ROI_Hi = roundNearest( Vector(hi) );

  } else if ( d_ROI_algo == fixed ){

    fineLevel_ROI_Lo = fineLevel->getCellIndex( d_ROI_minPt );
    fineLevel_ROI_Hi = fineLevel->getCellIndex( d_ROI_maxPt );

    if( !fineLevel->containsCell( fineLevel_ROI_Lo ) ||
        !fineLevel->containsCell( fineLevel_ROI_Hi ) ){
      std::ostringstream warn;
      warn << "ERROR:  the fixed ROI extents " << d_ROI_minPt << " " << d_ROI_maxPt << " are not contained on the fine level."<< endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
  } else if ( d_ROI_algo == patch_based ){
    IntVector patchLo = patch->getCellLowIndex();
    IntVector patchHi = patch->getCellHighIndex();

    fineLevel_ROI_Lo = patchLo - d_haloCells;
    fineLevel_ROI_Hi = patchHi + d_haloCells;
    DOUT( g_ray_dbg, "    computeExtents: L-"<< fineLevel->getIndex() <<"  patch: ("<<patch->getID() <<") " << patchLo << " " << patchHi <<  " d_haloCells" << d_haloCells );

  }

  // region must be within a finest Level including extraCells.
  IntVector levelLo, levelHi;
  fineLevel->findCellIndexRange(levelLo, levelHi);

  fineLevel_ROI_Lo = Max(fineLevel_ROI_Lo, levelLo);
  fineLevel_ROI_Hi = Min(fineLevel_ROI_Hi, levelHi);
  DOUT (g_ray_dbg, "    computeExtents: fineLevel_ROI: " << fineLevel_ROI_Lo << " "<< fineLevel_ROI_Hi );

  //__________________________________
  // Determine the extents of the regions below the fineLevel

  // finest level
  IntVector finelevel_EC = fineLevel->getExtraCells();
  regionLo[maxLevels-1] = fineLevel_ROI_Lo + finelevel_EC;
  regionHi[maxLevels-1] = fineLevel_ROI_Hi - finelevel_EC;

  // coarsest level
  level_0->findInteriorCellIndexRange(regionLo[0], regionHi[0]);

  for (int L = maxLevels - 2; L > 0; L--) {

    LevelP level = new_dw->getGrid()->getLevel(L);

    if( level->hasCoarserLevel() ){

      regionLo[L] = level->mapCellToCoarser(regionLo[L+1]) - d_haloCells;
      regionHi[L] = level->mapCellToCoarser(regionHi[L+1]) + d_haloCells;

      // region must be within a level
      IntVector levelLo, levelHi;
      level->findInteriorCellIndexRange(levelLo, levelHi);

      regionLo[L] = Max(regionLo[L], levelLo);
      regionHi[L] = Min(regionHi[L], levelHi);
    }
  }
}

//______________________________________________________________________
// Compute the Ray direction from a cell face
void Ray::rayDirection_cellFace( MTRand& mTwister,
                                 const IntVector& origin,
                                 const IntVector& indexOrder,
                                 const IntVector& signOrder,
                                 const int iRay,
                                 Vector& directionVector,
                                 double& cosTheta)
{

  if( d_isSeedRandom == false ){                 // !! This could use a compiler directive for speed-up
    mTwister.seed((origin.x() + origin.y() + origin.z()) * iRay +1);
  }

  // Surface Way to generate a ray direction from the positive z face
  double phi   = 2 * M_PI * mTwister.rand(); // azimuthal angle.  Range of 0 to 2pi
  double theta = acos(mTwister.rand());      // polar angle for the hemisphere
  cosTheta = cos(theta);

  //Convert to Cartesian
  Vector tmp;
  tmp[0] =  sin(theta) * cos(phi);
  tmp[1] =  sin(theta) * sin(phi);
  tmp[2] =  cosTheta;

  // Put direction vector as coming from correct face,
  directionVector[0] = tmp[indexOrder[0]] * signOrder[0];
  directionVector[1] = tmp[indexOrder[1]] * signOrder[1];
  directionVector[2] = tmp[indexOrder[2]] * signOrder[2];
}


//______________________________________________________________________
//  Uses stochastically selected regions in polar and azimuthal space to
//  generate the Monte-Carlo directions. Samples Uniformly on a hemisphere
//  and as hence does not include the cosine in the sample.
//______________________________________________________________________
void
Ray::rayDirectionHyperCube_cellFace(MTRand& mTwister,
                                 const IntVector& origin,
                                 const IntVector& indexOrder,
                                 const IntVector& signOrder,
                                 const int iRay,
                                 Vector& directionVector,
                                 double& cosTheta,
                                 const int bin_i,
                                 const int bin_j)
{
  if( d_isSeedRandom == false ){                 // !! This could use a compiler directive for speed-up
    mTwister.seed((origin.x() + origin.y() + origin.z()) * iRay +1);
  }

 // randomly sample within each randomly selected region (may not be needed, alternatively choose center of subregion)
  cosTheta = (mTwister.randDblExc() + (double) bin_i)/d_nFluxRays;

  double theta = acos(cosTheta);      // polar angle for the hemisphere
  double phi = 2.0 * M_PI * (mTwister.randDblExc() + (double) bin_j)/d_nFluxRays;        // Uniform betwen 0-2Pi

  cosTheta = cos(theta);

  //Convert to Cartesian
  Vector tmp;
  tmp[0] =  sin(theta) * cos(phi);
  tmp[1] =  sin(theta) * sin(phi);
  tmp[2] =  cosTheta;

  //Put direction vector as coming from correct face,
  directionVector[0] = tmp[indexOrder[0]] * signOrder[0];
  directionVector[1] = tmp[indexOrder[1]] * signOrder[1];
  directionVector[2] = tmp[indexOrder[2]] * signOrder[2];

}

//______________________________________________________________________
//  Uses stochastically selected regions in polar and azimuthal space to
//  generate the Monte-Carlo directions.  Samples uniformly on a sphere.
//______________________________________________________________________
Vector
Ray::findRayDirectionHyperCube(MTRand& mTwister,
                               const IntVector& origin,
                               const int iRay,
                               const int bin_i,
                               const int bin_j)
{
  if( d_isSeedRandom == false ){
    mTwister.seed((origin.x() + origin.y() + origin.z()) * iRay +1);
  }

  // Random Points On Sphere
  double plusMinus_one = 2.0 *(mTwister.randDblExc() + (double) bin_i)/d_nDivQRays - 1.0;  // add fuzz to avoid inf in 1/dirVector
  double r = sqrt(1.0 - plusMinus_one * plusMinus_one);     // Radius of circle at z
  double phi = 2.0 * M_PI * (mTwister.randDblExc() + (double) bin_j)/d_nDivQRays;        // Uniform betwen 0-2Pi

  Vector direction_vector;
  direction_vector[0] = r*cos(phi);                       // Convert to cartesian
  direction_vector[1] = r*sin(phi);
  direction_vector[2] = plusMinus_one;

  return direction_vector;
}

//______________________________________________________________________
//
//  Compute the Ray location on a cell face
void Ray::rayLocation_cellFace( MTRand& mTwister,
                                 const int face,
                                 const Vector Dx,
                                 const Point CC_pos,
                                 Vector& rayOrigin)
{
  double cellOrigin[3];
  // left, bottom, back corner of the cell
  cellOrigin[X] = CC_pos.x() - 0.5 * Dx[X];
  cellOrigin[Y] = CC_pos.y() - 0.5 * Dx[Y];
  cellOrigin[Z] = CC_pos.z() - 0.5 * Dx[Z];

  switch(face)
  {
    case WEST:
      rayOrigin[X] = cellOrigin[X];
      rayOrigin[Y] = cellOrigin[Y] + mTwister.rand() * Dx[Y];
      rayOrigin[Z] = cellOrigin[Z] + mTwister.rand() * Dx[Z];
      break;
    case EAST:
      rayOrigin[X] = cellOrigin[X] +  Dx[X];
      rayOrigin[Y] = cellOrigin[Y] + mTwister.rand() * Dx[Y];
      rayOrigin[Z] = cellOrigin[Z] + mTwister.rand() * Dx[Z];
      break;
    case SOUTH:
      rayOrigin[X] = cellOrigin[X] + mTwister.rand() * Dx[X];
      rayOrigin[Y] = cellOrigin[Y];
      rayOrigin[Z] = cellOrigin[Z] + mTwister.rand() * Dx[Z];
      break;
    case NORTH:
      rayOrigin[X] = cellOrigin[X] + mTwister.rand() * Dx[X];
      rayOrigin[Y] = cellOrigin[Y] + Dx[Y];
      rayOrigin[Z] = cellOrigin[Z] + mTwister.rand() * Dx[Z];
      break;
    case BOT:
      rayOrigin[X] = cellOrigin[X] + mTwister.rand() * Dx[X];;
      rayOrigin[Y] = cellOrigin[Y] + mTwister.rand() * Dx[Y];;
      rayOrigin[Z] = cellOrigin[Z];
      break;
    case TOP:
      rayOrigin[X] = cellOrigin[X] + mTwister.rand() * Dx[X];;
      rayOrigin[Y] = cellOrigin[Y] + mTwister.rand() * Dx[Y];;
      rayOrigin[Z] = cellOrigin[Z] + Dx[Z];
      break;
    default:
      throw InternalError("Ray::rayLocation_cellFace,  Invalid FaceType Specified", __FILE__, __LINE__);
      return;
  }
}

//______________________________________________________________________
//
bool
Ray::has_a_boundary( const IntVector      & c,
                     constCCVariable<int> & celltype,
                     vector<int>     & boundaryFaces)
{

  IntVector adjacentCell = c;
  bool hasBoundary = false;

  adjacentCell[0] = c[0] - 1;     // west

  if (celltype[adjacentCell]+1){         // cell type of flow is -1, so when cellType+1 isn't false, we
    boundaryFaces.push_back( WEST );     // know we're at a boundary
    hasBoundary = true;
  }

  adjacentCell[0] += 2;           // east

  if (celltype[adjacentCell]+1){
    boundaryFaces.push_back( EAST );
    hasBoundary = true;
  }

  adjacentCell[0] -= 1;
  adjacentCell[1] = c[1] - 1;     // south

  if (celltype[adjacentCell]+1){
    boundaryFaces.push_back( SOUTH );
    hasBoundary = true;
  }

  adjacentCell[1] += 2;           // north

  if (celltype[adjacentCell]+1){
    boundaryFaces.push_back( NORTH );
    hasBoundary = true;
  }

  adjacentCell[1] -= 1;
  adjacentCell[2] = c[2] - 1;     // bottom

  if (celltype[adjacentCell]+1){
    boundaryFaces.push_back( BOT );
    hasBoundary = true;
  }

  adjacentCell[2] += 2;           // top

  if (celltype[adjacentCell]+1){
    boundaryFaces.push_back( TOP );
    hasBoundary = true;
  }

  // if none of the above returned true, then the current cell must not be adjacent to a wall
  return (hasBoundary);
}




//______________________________________________________________________
inline bool
Ray::containsCell(const IntVector &low,
                  const IntVector &high,
                  const IntVector &cell,
                  const int &dir)
{
  return  low[dir] <= cell[dir] &&
          high[dir] > cell[dir];
}


//---------------------------------------------------------------------------
//   Set the the boundary conditions for sigmaT4 & abskg.
//---------------------------------------------------------------------------
void
Ray::sched_setBoundaryConditions( const LevelP& level,
                                  SchedulerP& sched,
                                  Task::WhichDW temp_dw,
                                  const bool backoutTemp )
{
  string taskname = "Ray::setBoundaryConditions";

  Task* tsk = nullptr;
  if( RMCRTCommon::d_FLT_DBL == TypeDescription::double_type ){

    tsk= scinew Task( taskname, this, &Ray::setBoundaryConditions< double >, temp_dw, backoutTemp );
  } else {
    tsk= scinew Task( taskname, this, &Ray::setBoundaryConditions< float >, temp_dw, backoutTemp );
  }

  printSchedule(level,g_ray_dbg,taskname);

  if(!backoutTemp){
    tsk->requiresVar( temp_dw, d_compTempLabel, Ghost::None, 0 );
  }

  tsk->modifiesVar( d_sigmaT4Label );
  tsk->modifiesVar( d_abskgLabel );

  sched->addTask( tsk, level->eachPatch(), d_matlSet, RMCRTCommon::TG_RMCRT );

  // ______________________________________________________________________

#ifdef HAVE_VISIT
  static bool initialized = false;

  m_application = sched->getApplication();

  // Running with VisIt so add in the variables that the user can
  // modify.
  if( m_application && m_application->getVisIt() && !initialized ) {
    // variable 1 - Must start with the component name and have NO
    // spaces in the var name
    ApplicationInterface::interactiveVar var;
    var.component  = "RMCRT-Ray";
    var.name       = "nDivQRays";
    var.type       = Uintah::TypeDescription::int_type;
    var.value      = (void *) &d_nDivQRays;
    var.range[0]   = 1;
    var.range[1]   = 100;
    var.modifiable = true;
    var.recompile  = false;
    var.modified   = false;
    m_application->getUPSVars().push_back( var );

    // variable 2 - Must start with the component name and have NO
    // spaces in the var name
    var.component  = "RMCRT-Ray";
    var.name       = "nFluxRays";
    var.type       = Uintah::TypeDescription::int_type;
    var.value      = (void *) &d_nFluxRays;
    var.range[0]   = 1;
    var.range[1]   = 100;
    var.modifiable = true;
    var.recompile  = false;
    var.modified   = false;
    m_application->getUPSVars().push_back( var );

    initialized = true;
  }
#endif
}

//---------------------------------------------------------------------------
template<class T>
void Ray::setBoundaryConditions( const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset*,
                                 DataWarehouse*,
                                 DataWarehouse* new_dw,
                                 Task::WhichDW temp_dw,
                                 const bool backoutTemp )
{
  if ( d_onOff_SetBCs == false ) {
    return;
  }

  for (int p=0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);

    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    if( bf.size() > 0){

      printTask(patches,patch,g_ray_dbg,"Doing Ray::setBoundaryConditions");

      double sigma_over_pi = d_sigma/M_PI;

      CCVariable<double> temp;
      CCVariable< T > abskg;
      CCVariable< T > sigmaT4OverPi;

      new_dw->allocateTemporary(temp,  patch);
      new_dw->getModifiable( abskg,         d_abskgLabel,    d_matl, patch );
      new_dw->getModifiable( sigmaT4OverPi, d_sigmaT4Label,  d_matl, patch );
      //__________________________________
      // loop over boundary faces and backout the temperature
      // one cell from the boundary.  Note that the temperature
      // is not available on all levels but sigmaT4 is.
      if (backoutTemp){
        for ( vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ) {
          Patch::FaceType face = *itr;

          Patch::FaceIteratorType IFC = Patch::InteriorFaceCells;

          for (CellIterator iter=patch->getFaceIterator(face, IFC); !iter.done();iter++) {
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
        t_dw->get(varTmp, d_compTempLabel,   d_matl, patch, Ghost::None, 0);
        temp.copyData(varTmp);
      }


      //__________________________________
      // set the boundary conditions
      setBC< T, double >  (abskg,    d_abskgBC_tag,               patch, d_matl);
      setBC<double,double>(temp,     d_compTempLabel->getName(),  patch, d_matl);

      //__________________________________
      // loop over boundary faces and compute sigma T^4
      for ( vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ) {
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
//  Set Boundary conditions
//  We're using 2 template types.  "T" is for the CCVariable type and the
//  "V" is used to create a specialization.  The infrastructure only
//  allows int and double BC so we're using the template type "V" to
//  fake it out.
//______________________________________________________________________
template<class T, class V>
void Ray::setBC(CCVariable< T >& Q_CC,
                const string& desc,
                const Patch* patch,
                const int mat_id)
{
  if(patch->hasBoundaryFaces() == false || d_onOff_SetBCs == false){
    return;
  }

  if( g_ray_BC.active() ){
    const Level* level = patch->getLevel();
    DOUT( g_ray_BC, "setBC \t"<< desc <<" "
           << " mat_id = " << mat_id <<  ", Patch: "<< patch->getID() << " L-" <<level->getIndex() );
  }

  // Iterate over the faces encompassing the domain
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);

  for( vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ){
    Patch::FaceType face = *iter;
    int nCells = 0;
    string bc_kind = "NotSet";

    IntVector dir= patch->getFaceAxes(face);
    Vector cell_dx = patch->dCell();
    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id);

    // iterate over each geometry object along that face
    for (int child = 0;  child < numChildren; child++) {
      V bc_value = -9;                          // see comments above
      Iterator bound_ptr;

      bool foundIterator = getIteratorBCValueBCKind( patch, face, child, desc, mat_id,
                                                     bc_value, bound_ptr,bc_kind);

      if(foundIterator) {
        // cast the value to the same type as the CCVariable needed when T = float
        T value = (T) bc_value;

        //__________________________________
        // Dirichlet
        if(bc_kind == "Dirichlet"){
          nCells += setDirichletBC_CC< T >( Q_CC, bound_ptr, value);
        }
        //__________________________________
        // Neumann
        else if(bc_kind == "Neumann"){
          nCells += setNeumannBC_CC< T >( patch, face, Q_CC, bound_ptr, value, cell_dx);
        }
        //__________________________________
        //  Symmetry
        else if ( bc_kind == "symmetry" || bc_kind == "zeroNeumann" ) {
          bc_value = 0.0;
          nCells += setNeumannBC_CC<T> ( patch, face, Q_CC, bound_ptr, value, cell_dx);
        }

        //__________________________________
        //  debugging
        if( g_ray_BC.active() ) {
          bound_ptr.reset();
          DOUT( g_ray_BC, "Face: "<< patch->getFaceName(face) <<" numCellsTouched " << nCells
             <<"\t child " << child  <<" NumChildren "<<numChildren
             <<"\t BC kind "<< bc_kind <<" \tBC value "<< bc_value
             <<"\t bound limits = "<< bound_ptr );
        }
      }  // if iterator found
    }  // child loop

    if( g_ray_BC.active() ){
      DOUT( g_ray_BC, "    "<< patch->getFaceName(face) << " \t " << bc_kind << " numChildren: " << numChildren
             << " nCellsTouched: " << nCells );
    }
    //__________________________________
    //  bulletproofing
#if 0
    Patch::FaceIteratorType type = Patch::ExtraPlusEdgeCells;
    int nFaceCells = numFaceCells(patch,  type, face);

    if(nCells != nFaceCells){
      ostringstream warn;
      warn << "ERROR: ICE: setSpecificVolBC Boundary conditions were not set correctly ("<< desc<< ", "
           << patch->getFaceName(face) << ", " << bc_kind  << " numChildren: " << numChildren
           << " nCells Touched: " << nCells << " nCells on boundary: "<< nFaceCells<<") " << endl;
      throw InternalError(warn.str(), __FILE__, __LINE__);
    }
#endif
  }  // faces loop
}

//______________________________________________________________________
//
void Ray::sched_Refine_Q( SchedulerP& sched,
                          const PatchSet* patches,
                          const MaterialSet* matls )
{
  const Level* fineLevel = getLevel(patches);
  int L_indx = fineLevel->getIndex();

  if(L_indx > 0 ){
     printSchedule(patches,g_ray_dbg,"Ray::scheduleRefine_Q (divQ)");

    Task* task = scinew Task("Ray::refine_Q",this, &Ray::refine_Q);

    Task::MaterialDomainSpec  ND  = Task::NormalDomain;
    #define allPatches 0
    #define allMatls 0
    task->requiresVar( Task::NewDW, d_divQLabel,          allPatches, Task::CoarseLevel, allMatls, ND, d_gac,1 );
    task->requiresVar( Task::NewDW, d_boundFluxLabel,     allPatches, Task::CoarseLevel, allMatls, ND, d_gac,1 );
    task->requiresVar( Task::NewDW, d_radiationVolqLabel, allPatches, Task::CoarseLevel, allMatls, ND, d_gac,1 );

    task->computesVar( d_divQLabel );
    task->computesVar( d_boundFluxLabel );
    task->computesVar( d_radiationVolqLabel );
    sched->addTask( task, patches, matls, RMCRTCommon::TG_RMCRT );
  }
}

//______________________________________________________________________
//
void Ray::refine_Q( const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw )
{

  const Level* fineLevel = getLevel(patches);
  const Level* coarseLevel = fineLevel->getCoarserLevel().get_rep();

  //__________________________________
  //
  for(int p=0;p<patches->size();p++){
    const Patch* finePatch = patches->get(p);
    printTask(patches, finePatch,g_ray_dbg,"Doing refineQ");

    Level::selectType coarsePatches;
    finePatch->getCoarseLevelPatches(coarsePatches);

    CCVariable<double> divQ_fine;
    CCVariable<double> radVolQ_fine;
    CCVariable<Stencil7> boundFlux_fine;

    new_dw->allocateAndPut(divQ_fine,      d_divQLabel,        d_matl, finePatch);
    new_dw->allocateAndPut(radVolQ_fine, d_radiationVolqLabel, d_matl, finePatch);
    new_dw->allocateAndPut(boundFlux_fine, d_boundFluxLabel,   d_matl, finePatch);

    divQ_fine.initialize( 0.0 );
    radVolQ_fine.initialize( 0.0 );

    for (CellIterator iter = finePatch->getExtraCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      boundFlux_fine[c].initialize( 0.0 );
    }

    IntVector refineRatio = fineLevel->getRefinementRatio();

    // region of fine space that will correspond to the coarse we need to get
    IntVector cl, ch, fl, fh;
    IntVector bl(0,0,0);  // boundary layer or padding
    int nghostCells = 1;
    bool returnExclusiveRange=true;

    getCoarseLevelRange(finePatch, coarseLevel, cl, ch, fl, fh, bl,
                        nghostCells, returnExclusiveRange);

    DOUT( g_ray_dbg, " refineQ: "
              <<" finePatch  "<< finePatch->getID() << " fl " << fl << " fh " << fh
              <<" coarseRegion " << cl << " " << ch );

    //__________________________________DivQ
    constCCVariable<double> divQ_coarse;
    new_dw->getRegion( divQ_coarse, d_divQLabel, d_matl, coarseLevel, cl, ch );

    selectInterpolator(divQ_coarse, d_orderOfInterpolation, coarseLevel, fineLevel,
                       refineRatio, fl, fh, divQ_fine);

    //__________________________________raditionVolQ
    constCCVariable<double> radVolQ_coarse;
    new_dw->getRegion( radVolQ_coarse, d_radiationVolqLabel, d_matl, coarseLevel, cl, ch );

    selectInterpolator(radVolQ_coarse, d_orderOfInterpolation, coarseLevel, fineLevel,
                       refineRatio, fl, fh, radVolQ_fine);

    //__________________________________boundary Flux
    constCCVariable<Stencil7> boundFlux_coarse;
    new_dw->getRegion( boundFlux_coarse, d_boundFluxLabel, d_matl, coarseLevel, cl, ch );
#if 0               // ----------------------------------------------------------------TO BE FILLED IN   Todd
    selectInterpolator(boundFlux_coarse, d_orderOfInterpolation, coarseLevel, fineLevel,
                       refineRatio, fl, fh, boundFlux_fine);
#endif

  }  // fine patch loop
}

//______________________________________________________________________
// This task computes the extents of the fine level region of interest
void Ray::sched_ROI_Extents ( const LevelP& level,
                              SchedulerP& scheduler )
{
  int maxLevels = level->getGrid()->numLevels() -1;
  int L_indx = level->getIndex();

  if( (L_indx != maxLevels ) || ( d_ROI_algo != dynamic ) ){     // only schedule on the finest level and dynamic
    return;
  }

  printSchedule(level,g_ray_dbg,"Ray::ROI_Extents");

  Task* tsk = nullptr;
  if( RMCRTCommon::d_FLT_DBL == TypeDescription::double_type ){
    tsk= scinew Task( "Ray::ROI_Extents", this, &Ray::ROI_Extents< double >);
  } else {
    tsk= scinew Task( "Ray::ROI_Extents", this, &Ray::ROI_Extents< float >);
  }

  tsk->requiresVar( Task::NewDW, d_abskgLabel,    d_gac, 1 );
  tsk->requiresVar( Task::NewDW, d_sigmaT4Label,  d_gac, 1 );
  tsk->computesVar( d_mag_grad_abskgLabel );
  tsk->computesVar( d_mag_grad_sigmaT4Label );
  tsk->computesVar( d_flaggedCellsLabel );

  tsk->computesVar(d_ROI_LoCellLabel);
  tsk->computesVar(d_ROI_HiCellLabel);

  scheduler->addTask( tsk, level->eachPatch(), d_matlSet, RMCRTCommon::TG_RMCRT );
}

//______________________________________________________________________
//
template< class T >
void Ray::ROI_Extents ( const ProcessorGroup*,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw)
{
  IntVector ROI_hi(-SHRT_MAX,-SHRT_MAX,-SHRT_MAX );
  IntVector ROI_lo(SHRT_MAX,  SHRT_MAX, SHRT_MAX);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,g_ray_dbg,"Doing ROI_Extents");

    //__________________________________
    constCCVariable< T > abskg;
    constCCVariable< T > sigmaT4;

    CCVariable< T > mag_grad_abskg;
    CCVariable< T > mag_grad_sigmaT4;
    CCVariable<int> flaggedCells;

    new_dw->get(abskg,    d_abskgLabel ,     d_matl , patch, d_gac,1);
    new_dw->get(sigmaT4,  d_sigmaT4Label ,  d_matl , patch, d_gac,1);

    new_dw->allocateAndPut(mag_grad_abskg,   d_mag_grad_abskgLabel,    0, patch);
    new_dw->allocateAndPut(mag_grad_sigmaT4, d_mag_grad_sigmaT4Label,  0, patch);
    new_dw->allocateAndPut(flaggedCells,     d_flaggedCellsLabel,      0, patch);

    mag_grad_abskg.initialize(0.0);
    mag_grad_sigmaT4.initialize(0.0);
    flaggedCells.initialize(0);

    //__________________________________
    //  compute the magnitude of the gradient of abskg & sigmatT4
    //  useful to visualize and set the thresholds
    compute_Mag_gradient(abskg,   mag_grad_abskg,   patch);
    compute_Mag_gradient(sigmaT4, mag_grad_sigmaT4, patch);
    bool flaggedPatch = false;


    for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;

      if( mag_grad_abskg[c] > d_abskg_thld || mag_grad_sigmaT4[c] > d_sigmaT4_thld ){
        flaggedCells[c] = true;
        flaggedPatch = true;
      }
    }

    // compute ROI lo & hi
    if(flaggedPatch){
      IntVector lo = patch->getExtraCellLowIndex();  // include BCs
      IntVector hi = patch->getExtraCellHighIndex();

      ROI_lo = Min(ROI_lo, lo);
      ROI_hi = Max(ROI_hi, hi);
    }
  }  // patches loop

  new_dw->put(minvec_vartype(ROI_lo.asVector()), d_ROI_LoCellLabel);
  new_dw->put(maxvec_vartype(ROI_hi.asVector()), d_ROI_HiCellLabel);
}


//______________________________________________________________________
void Ray::sched_CoarsenAll( const LevelP& coarseLevel,
                            SchedulerP& sched,
                            const bool modifies_abskg,
                            const bool modifies_sigmaT4)
{
  if(coarseLevel->hasFinerLevel()){
    printSchedule(coarseLevel,g_ray_dbg,"Ray::sched_CoarsenAll");

    int L = coarseLevel->getIndex();
    Task::WhichDW fineLevel_abskg_dw = get_abskg_whichDW( L+1, d_abskgLabel);

    sched_Coarsen_Q(coarseLevel, sched, fineLevel_abskg_dw, modifies_abskg,     d_abskgLabel );
    sched_Coarsen_Q(coarseLevel, sched, Task::NewDW,        modifies_sigmaT4,  d_sigmaT4Label );
  }
}

//______________________________________________________________________
void Ray::sched_Coarsen_Q ( const LevelP& coarseLevel,
                            SchedulerP& sched,
                            Task::WhichDW fineLevel_Q_dw,
                            const bool modifies,
                            const VarLabel* variable )
{
  string taskName  = "Ray::coarsen_Q_" + variable->getName();
  string schedName = "Ray::sched_coarsen_Q_" + variable->getName();
  printSchedule(coarseLevel,g_ray_dbg,schedName);

  const Uintah::TypeDescription* td = variable->typeDescription();
  const Uintah::TypeDescription::Type subtype = td->getSubType()->getType();

  Task* tsk = nullptr;
  switch( subtype ) {
    case TypeDescription::double_type:
      tsk = scinew Task( taskName, this, &Ray::coarsen_Q< double >, variable, modifies, fineLevel_Q_dw );
      break;
    case TypeDescription::float_type:
      tsk = scinew Task( taskName, this, &Ray::coarsen_Q< float >, variable, modifies, fineLevel_Q_dw );
      break;
    default:
      throw InternalError("Ray::sched_Coarsen_Q: (CCVariable) invalid data type", __FILE__, __LINE__);
  }

  if(modifies){
    tsk->requiresVar( fineLevel_Q_dw, variable, 0, Task::FineLevel, 0, Task::NormalDomain, d_gn, 0 );
    tsk->modifiesVar( variable);
  }else{
    tsk->requiresVar( fineLevel_Q_dw, variable, 0, Task::FineLevel, 0, Task::NormalDomain, d_gn, 0 );
    tsk->computesVar( variable );
  }

  sched->addTask( tsk, coarseLevel->eachPatch(), d_matlSet, RMCRTCommon::TG_RMCRT );
}

//______________________________________________________________________
//
template < class T >
void Ray::coarsen_Q ( const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw,
                      const VarLabel* variable,
                      const bool modifies,
                      Task::WhichDW which_dw )
{
  const Level* coarseLevel = getLevel(patches);
  const Level* fineLevel   = coarseLevel->getFinerLevel().get_rep();
  DataWarehouse* fineLevel_Q_dw = new_dw->getOtherDataWarehouse( which_dw );

  //__________________________________
  //
  for(int p=0;p<patches->size();p++){
    const Patch* coarsePatch = patches->get(p);

    printTask(patches, coarsePatch,g_ray_dbg,"Doing coarsen: " + variable->getName());

    // Find the overlapping regions...
    Level::selectType finePatches;
    coarsePatch->getFineLevelPatches(finePatches);

    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      CCVariable< T > Q_coarse;
      if(modifies){
        new_dw->getModifiable(Q_coarse,  variable, matl, coarsePatch);
      }else{
        new_dw->allocateAndPut(Q_coarse, variable, matl, coarsePatch);
      }
      Q_coarse.initialize(0.0);

      bool computesAve = true;

      // coarsen the coarse patch interior cells
      fineToCoarseOperator(Q_coarse,   computesAve,
                           variable,   matl, fineLevel_Q_dw,
                           coarsePatch, coarseLevel, fineLevel);

      //__________________________________
      //  Coarsen along the edge of the computational domain
      if( d_coarsenExtraCells && coarsePatch->hasBoundaryFaces() ){

        for(size_t i=0;i<finePatches.size();i++){
          const Patch* finePatch = finePatches[i];

          if( finePatch->hasBoundaryFaces() ){


            IntVector refineRatio = fineLevel->getRefinementRatio();

            // used for extents tests
            IntVector finePatchLo = finePatch->getExtraCellLowIndex();
            IntVector finePatchHi = finePatch->getExtraCellHighIndex();

            IntVector coarsePatchLo = coarsePatch->getExtraCellLowIndex();
            IntVector coarsePatchHi = coarsePatch->getExtraCellHighIndex();

            constCCVariable<T> fine_q_CC;
            fineLevel_Q_dw->get(fine_q_CC, variable,   matl, finePatch, d_gn, 0);

            //__________________________________
            //  loop over boundary faces for the fine patch

            vector<Patch::FaceType> bf;
            finePatch->getBoundaryFaces( bf );

            for ( vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ) {
              Patch::FaceType face = *iter;

              IntVector faceRefineRatio = refineRatio;
              int P_dir = coarsePatch->getFaceAxes(face)[0];  //principal dir.
              faceRefineRatio[P_dir]=1;

              double inv_RR = 1.0;
              if(computesAve){
                inv_RR = 1.0/( (double)(faceRefineRatio.x() * faceRefineRatio.y() * faceRefineRatio.z()) );
              }

              // for this  fine patch find the extents of the boundary face
              CellIterator iter_tmp = finePatch->getFaceIterator(face, Patch::ExtraMinusEdgeCells);
              IntVector fl = iter_tmp.begin();
              IntVector fh = iter_tmp.end();

              IntVector cl  = fineLevel->mapCellToCoarser(fl);
              IntVector ch  = fineLevel->mapCellToCoarser(fh+refineRatio - IntVector(1,1,1));

              // don't exceed the coarse patch
              cl = Max(cl, coarsePatchLo);
              ch = Min(ch, coarsePatchHi);

              //cout << "    " << finePatch->getFaceName(face) << endl;
              //cout << "    fl: " << fl << " fh: " << fh;
              //cout << "   " <<  " cl: " << cl << " ch: " << ch << endl;
              //__________________________________
              //  iterate over coarse patch cells that overlapp this fine patch
              T zero(0.0);

              for(CellIterator iter(cl, ch); !iter.done(); iter++){
                IntVector c = *iter;
                T q_CC_tmp(zero);
                IntVector fineStart = coarseLevel->mapCellToFiner(c);

                // don't exceed fine patch boundaries
                fineStart = Max(finePatchLo, fineStart);
                fineStart = Min(finePatchHi, fineStart);

                double count = 0;

                // for each coarse level cell iterate over the fine level cells
                for(CellIterator inside(IntVector(0,0,0),faceRefineRatio );
                    !inside.done(); inside++){
                  IntVector fc = fineStart + *inside;

                  if( fc.x() >= fl.x() && fc.y() >= fl.y() && fc.z() >= fl.z() &&
                      fc.x() <= fh.x() && fc.y() <= fh.y() && fc.z() <= fh.z() ) {
                    q_CC_tmp += fine_q_CC[fc];
                    count +=1.0;
                  }
                }
                Q_coarse[c] =q_CC_tmp*inv_RR;

                //__________________________________
                //  bulletproofing
    //          #if SCI_ASSERTION_LEVEL > 0     enable this when you're 100% confident it's working correctly for different domains. -Todd
                if ( (fabs(inv_RR - 1.0/count) > 2 * DBL_EPSILON) && inv_RR != 1 ) {
                  std::ostringstream msg;
                  msg << " ERROR:  coarsen_Q: coarse cell " << c << "\n"
                      <<  "Only (" << count << ") fine level cells were used to compute the coarse cell value."
                      << " There should have been ("<< 1/inv_RR << ") cells used";

                  throw InternalError(msg.str(),__FILE__,__LINE__);
                }
    //          #endif
              }  // boundary face iterator
            }  // boundary face loop
          }  // has boundaryFace
        }  // fine patches
      }  // coarsen ExtraCells
    }  // matl loop
  }  // course patch loop
}

//______________________________________________________________________
//
void Ray::sched_computeCellType ( const LevelP& level,
                                  SchedulerP& sched,
                                  modifiesComputes which)
{
  string taskname = "Ray::computeCellType";
  Task* tsk = scinew Task( taskname, this, &Ray::computeCellType, which );

  printSchedule(level, g_ray_dbg, taskname);

  if ( which == Ray::modifiesVar ){
    tsk->requiresVar(Task::NewDW, d_cellTypeLabel, 0, Task::FineLevel, 0, Task::NormalDomain, d_gn, 0);
    tsk->modifiesVar( d_cellTypeLabel );
  }else if ( which == Ray::computesVar ){
    tsk->computesVar( d_cellTypeLabel );
  }
  sched->addTask( tsk, level->eachPatch(), d_matlSet, RMCRTCommon::TG_RMCRT );
}

//______________________________________________________________________
//    Initialize cellType on the coarser Levels
void Ray::computeCellType( const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw,
                           const modifiesComputes which )
{
  const Level* coarseLevel = getLevel(patches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  IntVector r_Ratio = fineLevel->getRefinementRatio();

  double inv_RR = 1.0/( (double)(r_Ratio.x() * r_Ratio.y() * r_Ratio.z()) );

  int FLOWCELL = -1;             // HARDWIRED to match Arches definitiion.  This
  int INTRUSION = 10;            // is temporary and will change when moving to
                                 // volFraction
  //__________________________________
  //  For each coarse level patch coarsen the fine level data
  for(int p=0;p<patches->size();p++){
    const Patch* coarsePatch = patches->get(p);

    printTask(patches, coarsePatch,g_ray_dbg,"Doing computeCellType: " + d_cellTypeLabel->getName());

    // Find the overlapping regions...
    Level::selectType finePatches;
    coarsePatch->getFineLevelPatches(finePatches);

    CCVariable< int > cellType_coarse;

    if ( which == Ray::modifiesVar ){
      new_dw->getModifiable(  cellType_coarse, d_cellTypeLabel, d_matl, coarsePatch);
    }else if ( which == Ray::computesVar ){
      new_dw->allocateAndPut( cellType_coarse, d_cellTypeLabel, d_matl, coarsePatch);
    }

    //__________________________________
    // coarsen
    bool computesAve = false;
    fineToCoarseOperator(cellType_coarse,   computesAve,
                         d_cellTypeLabel,   d_matl, new_dw,
                         coarsePatch, coarseLevel, fineLevel);


    for (CellIterator iter = coarsePatch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;

      double tmp = (double) cellType_coarse[c] * inv_RR;
      // Default
      cellType_coarse[c] = (int) tmp;

      if (tmp != FLOWCELL){
        //__________________________________
        // ROUND DOWN
        if ( d_cellTypeCoarsenLogic == ROUNDDOWN ) {
          if( fabs(tmp)/(double)INTRUSION <= (1 - inv_RR) ) {
            cellType_coarse[c] = FLOWCELL;
          } else {
            cellType_coarse[c] = INTRUSION;
          }
        }
        //__________________________________
        // ROUND UP
        if ( d_cellTypeCoarsenLogic == ROUNDUP ) {
          if( fabs(tmp)/(double)INTRUSION >= inv_RR  ){
            cellType_coarse[c] = INTRUSION;
          } else {
            cellType_coarse[c] = FLOWCELL;
          }
        }
      }
    }

  }  // coarse patch loop
}

//______________________________________________________________________
//  Multi-level
 template< class T>
 void Ray::updateSumI_ML ( Vector& ray_direction,
                           Vector& ray_origin,
                           const IntVector& origin,
                           const vector<Vector>& Dx,
                           const BBox& domain_BB,
                           const int maxLevels,
                           const Level* fineLevel,
                           const IntVector& fineLevel_ROI_Lo1,
                           const IntVector& fineLevel_ROI_Hi1,
                           vector<IntVector>& regionLo,
                           vector<IntVector>& regionHi,
                           std::vector< constCCVariable< T > >& sigmaT4OverPi,
                           std::vector< constCCVariable< T > >& abskg,
                           std::vector< constCCVariable< int > >& cellType,
                           unsigned long int& nRaySteps,
                           double& sumI,
                           MTRand& mTwister)
{
  IntVector fineLevel_ROI_Lo = fineLevel_ROI_Lo1, fineLevel_ROI_Hi = fineLevel_ROI_Hi1;

  if(m_use_virtual_ROI){
    //dont have % operator overloaded in IntVector. So doing it manually now.
    //origin[0] % m_virtual_ROI[0] gives additional cells over "virtual" lo boundary that will be defined if the patch size is set to m_virtual_ROI
    //subtract that number from origin to get virtual Lo boundary. Subtract ghost cells d_haloCells to get fineLevel_ROI_Lo.
    //To get fineLevel_ROI_Lo, compute virtual Lo boundary and add m_virtual_ROI to get virtual Hi of the patch assuming the patch size of m_virtual_ROI.
    //Add d_haloCells. Exactly the same logic as "patch based", except uses manually defined ROI size rather than patch size to determine ROI

    fineLevel_ROI_Lo[0] = (origin[0] - (origin[0] % m_virtual_ROI[0])) - d_haloCells[0];
    fineLevel_ROI_Lo[1] = (origin[1] - (origin[1] % m_virtual_ROI[1])) - d_haloCells[1];
    fineLevel_ROI_Lo[2] = (origin[2] - (origin[2] % m_virtual_ROI[2])) - d_haloCells[2];

    fineLevel_ROI_Hi[0] = (origin[0] - (origin[0] % m_virtual_ROI[0])) + m_virtual_ROI[0] + d_haloCells[0];
    fineLevel_ROI_Hi[1] = (origin[1] - (origin[1] % m_virtual_ROI[1])) + m_virtual_ROI[1] + d_haloCells[1];
    fineLevel_ROI_Hi[2] = (origin[2] - (origin[2] % m_virtual_ROI[2])) + m_virtual_ROI[2] + d_haloCells[2];
  }
//  printf("ROI: %d %d %d %d %d %d\n", fineLevel_ROI_Lo[0], fineLevel_ROI_Lo[1], fineLevel_ROI_Lo[2], fineLevel_ROI_Hi[0], fineLevel_ROI_Hi[1], fineLevel_ROI_Hi[2]);

  int L       = maxLevels -1;  // finest level
  int prevLev = L;

  IntVector cur      = origin;
  IntVector prevCell = cur;

  int step[3];                                           // Gives +1 or -1 based on sign
  double sign[3];

  Vector inv_direction = Vector(1.0)/ray_direction;

/*`==========TESTING==========*/
#if DEBUG == 1
  if( isDbgCell(origin) ) {
    printf("        updateSumI_ML: [%d,%d,%d] ray_dir [%g,%g,%g] ray_loc [%g,%g,%g]\n", origin.x(), origin.y(), origin.z(),ray_direction.x(), ray_direction.y(), ray_direction.z(), ray_origin.x(), ray_origin.y(), ray_origin.z());
  }
#endif
/*===========TESTING==========`*/

  raySignStep(sign, step, ray_direction);

  //__________________________________
  // define tMax & tDelta on all levels
  // go from finest to coarsest level so you can compare
  // with 1L rayTrace results.

  Point CC_posOrigin = fineLevel->getCellPosition(origin);

  // rayDx is the distance from bottom, left, back, corner of cell to ray
  Vector rayDx;
  rayDx[0] = ray_origin.x() - ( CC_posOrigin.x() - 0.5*Dx[L][0] );
  rayDx[1] = ray_origin.y() - ( CC_posOrigin.y() - 0.5*Dx[L][1] );
  rayDx[2] = ray_origin.z() - ( CC_posOrigin.z() - 0.5*Dx[L][2] );

  // tMax is the physical distance from the ray origin to each of the respective planes of intersection
  Vector tMaxV;
  tMaxV[0] = (sign[0] * Dx[L][0] - rayDx[0]) * inv_direction.x();
  tMaxV[1] = (sign[1] * Dx[L][1] - rayDx[1]) * inv_direction.y();
  tMaxV[2] = (sign[2] * Dx[L][2] - rayDx[2]) * inv_direction.z();

  vector<Vector> tDelta(maxLevels);
  for(int Lev = maxLevels-1; Lev>-1; Lev--){
    //Length of t to traverse one cell
    tDelta[Lev].x( std::fabs(inv_direction[0]) * Dx[Lev][0]);
    tDelta[Lev].y( std::fabs(inv_direction[1]) * Dx[Lev][1] );
    tDelta[Lev].z( std::fabs(inv_direction[2]) * Dx[Lev][2] );
  }

  //Initializes the following values for each ray
  bool   in_domain      = true;
  Vector tMaxV_prev     = Vector(0,0,0);
  double old_length     = 0.0;

  double intensity      = 1.0;
  double fs             = 1.0;
  int    nReflect       = 0;             // Number of reflections
  bool   onFineLevel    = true;
  const  Level* level   = fineLevel;
  double optical_thickness     = 0;
  double expOpticalThick_prev  = 1.0;    // exp(-opticalThick_prev)
  double rayLength             = 0.0;
  Vector ray_location          = ray_origin;
  Point CC_pos                 = CC_posOrigin;

  //______________________________________________________________________
  //  Threshold  loop
  while (intensity > d_threshold){
    DIR dir = NONE;
    while (in_domain){

      prevCell = cur;
      prevLev  = L;

      //__________________________________
      //  Determine the principal direction the ray is traveling
      //
      dir = NONE;
      if ( tMaxV[0] < tMaxV[1] ){    // X < Y
        if ( tMaxV[0] < tMaxV[2] ){  // X < Z
          dir = X;
        } else {
          dir = Z;
        }
      } else {
        if(tMaxV[1] <tMaxV[2] ){     // Y < Z
          dir = Y;
        } else {
          dir = Z;
        }
      }

      // next cell index and position
      cur[dir]  = cur[dir] + step[dir];

      //__________________________________
      // Logic for moving between levels
      // - Currently you can only move from fine to coarse level
      // - Don't jump levels if ray is at edge of domain

      CC_pos = level->getCellPosition(cur);           // position could be outside of domain
      in_domain = domain_BB.inside(CC_pos);

      bool ray_outside_ROI    = ( containsCell( fineLevel_ROI_Lo, fineLevel_ROI_Hi, cur, dir ) == false );
      bool ray_outside_Region = ( containsCell( regionLo[L], regionHi[L], cur, dir ) == false );

      bool jumpFinetoCoarserLevel   = ( onFineLevel &&  ray_outside_ROI && in_domain );
      bool jumpCoarsetoCoarserLevel = ( (onFineLevel == false) && ray_outside_Region && (L > 0) && in_domain );

//#define ML_DEBUG
#if ( (DEBUG == 1 || DEBUG == 4) && defined(ML_DEBUG) )
      if( isDbgCell(origin) ) {
        printf( "        Ray: [%i,%i,%i] **jumpFinetoCoarserLevel %i jumpCoarsetoCoarserLevel %i containsCell: %i ", cur.x(), cur.y(), cur.z(), jumpFinetoCoarserLevel, jumpCoarsetoCoarserLevel,
            containsCell(fineLevel_ROI_Lo, fineLevel_ROI_Hi, cur, dir));
        printf( " onFineLevel: %i ray_outside_ROI: %i ray_outside_Region: %i in_domain: %i\n", onFineLevel, ray_outside_ROI, ray_outside_Region, in_domain );
        printf( " L: %i regionLo: [%i,%i,%i], regionHi: [%i,%i,%i]\n",L,regionLo[L].x(),regionLo[L].y(),regionLo[L].z(),regionHi[L].x(),regionHi[L].y(),regionHi[L].z());
      }
#endif

      if( jumpFinetoCoarserLevel ){
        cur   = level->mapCellToCoarser(cur);
        level = level->getCoarserLevel().get_rep();      // move to a coarser level
        L     = level->getIndex();
        onFineLevel = false;

#if ( (DEBUG == 1 || DEBUG == 4) && defined(ML_DEBUG) )
        if( isDbgCell(origin) ) {
          printf( "        ** Jumping off fine patch switching Levels:  prev L: %i, L: %i, cur: [%i,%i,%i] \n",prevLev, L, cur.x(), cur.y(), cur.z());
        }
#endif
      } else if ( jumpCoarsetoCoarserLevel ) {

        IntVector c_old = cur;                          // needed for debugging
        cur   = level->mapCellToCoarser(cur);
        level = level->getCoarserLevel().get_rep();
        L     = level->getIndex();
#if ( (DEBUG == 1 || DEBUG == 4) && defined(ML_DEBUG) )
        if( isDbgCell(origin) ) {
          printf( "        ** Switching Levels:  prev L: %i, L: %i, cur: [%i,%i,%i], c_old: [%i,%i,%i]\n",prevLev, L, cur.x(), cur.y(), cur.z(), c_old.x(), c_old.y(), c_old.z());
        }
#endif
      }




      //__________________________________
      //  update marching variables
      double distanceTraveled = (tMaxV[dir] - old_length);
      old_length     = tMaxV[dir];
      tMaxV_prev     = tMaxV;
      tMaxV[dir]     = tMaxV[dir] + tDelta[L][dir];

      ray_location[0] = ray_location[0] + ( distanceTraveled  * ray_direction[0] );
      ray_location[1] = ray_location[1] + ( distanceTraveled  * ray_direction[1] );
      ray_location[2] = ray_location[2] + ( distanceTraveled  * ray_direction[2] );

      //__________________________________
      // when moving to a coarse level tmax will change only in the direction the ray is moving
      if ( jumpFinetoCoarserLevel || jumpCoarsetoCoarserLevel ) {
        double rayDx_Level = ray_location[dir] - ( CC_pos(dir) - 0.5*Dx[L][dir] );
        double tMax_tmp    = ( sign[dir] * Dx[L][dir] - rayDx_Level ) * inv_direction[dir];
        tMaxV        = tMaxV_prev;
        tMaxV[dir]  += tMax_tmp;
      }


      // if the cell isn't a flow cell then terminate the ray
      in_domain = in_domain && (cellType[L][cur] == d_flowCell);

      rayLength         += distanceTraveled;
      optical_thickness += abskg[prevLev][prevCell]*distanceTraveled;
      nRaySteps++;

/*`==========TESTING==========*/
#ifdef FAST_EXP
      double expOpticalThick = fast_exp(-optical_thickness);
#else
      double expOpticalThick = exp(-optical_thickness);
#endif
/*===========TESTING==========`*/

 /*`==========TESTING==========*/
#if DEBUG == 1
  if( isDbgCell( origin ) ){
    printf( "            cur [%d,%d,%d] prev [%d,%d,%d]", cur.x(), cur.y(), cur.z(), prevCell.x(), prevCell.y(), prevCell.z());
    printf( " dir %d ", dir );
    printf( " cellType: %i ", cellType[L][cur] );
//    printf( " stepSize [%i,%i,%i] ",step[0],step[1],step[2]);
    printf( "tMax [%g,%g,%g] ", tMaxV[0],tMaxV[1], tMaxV[2]);
    printf( "rayLoc [%4.5f,%4.5f,%4.5f] ",ray_location.x(),ray_location.y(), ray_location.z());
    printf( "\tdistanceTraveled %4.5f tMax[dir]: %g tMax_prev[dir]: %g, Dx[dir]: %g\n",distanceTraveled, tMaxV[dir], tMaxV_prev[dir], Dx[L][dir]);
    printf( "                tDelta [%g,%g,%g] \n",tDelta[L].x(),tDelta[L].y(), tDelta[L].z());

//    printf( "inv_dir [%g,%g,%g] ",inv_direction.x(),inv_direction.y(), inv_direction.z());
//    printf( "            abskg[prev] %g  \t sigmaT4OverPi[prev]: %g \n",abskg[prevLev][prevCell],  sigmaT4OverPi[prevLev][prevCell]);
//    printf( "            abskg[cur]  %g  \t sigmaT4OverPi[cur]:  %g  \t  cellType: %i \n",abskg[L][cur], sigmaT4OverPi[L][cur], cellType[L][cur]);
//    printf( "            Dx[prevLev].x  %g \n",  Dx[prevLev].x() );
    printf( "                optical_thickkness %g \t rayLength: %g\n", optical_thickness, rayLength);
  }
#endif
/*===========TESTING==========`*/

      sumI += sigmaT4OverPi[prevLev][prevCell] * ( expOpticalThick_prev - expOpticalThick ) * fs;

      expOpticalThick_prev = expOpticalThick;

    } //end domain while loop.  ++++++++++++++

    double wallEmissivity = abskg[L][cur];

    if (wallEmissivity > 1.0){       // Ensure wall emissivity doesn't exceed one.
      wallEmissivity = 1.0;
    }

    intensity = exp(-optical_thickness);

    sumI += wallEmissivity * sigmaT4OverPi[L][cur] * intensity;

    intensity = intensity * fs;

    // when a ray reaches the end of the domain, we force it to terminate.
    if(!d_allowReflect){
      intensity = 0;
    }

/*`==========TESTING==========*/
#if DEBUG == 1
  if( isDbgCell(origin) ){
    printf( "        intensity: %g OptThick: %g, fs: %g allowReflect: %i\n", intensity, optical_thickness, fs, d_allowReflect );
  }
#endif
/*===========TESTING==========`*/

    //__________________________________
    //  Reflections
    if (intensity > d_threshold && d_allowReflect ){
      ++nReflect;
      reflect( fs, cur, prevCell, abskg[L][cur], in_domain, step[dir], sign[dir], ray_direction[dir] );
    }
  }  // threshold while loop.
}

#if 0
//---------------------------------------------------------------------------
//
//---------------------------------------------------------------------------
void
Ray::sched_filter( const LevelP& level,
                    SchedulerP& sched,
                    Task::WhichDW which_divQ_dw,
                    const bool includeEC,
                    bool modifies_divQFilt )
{
  string taskname = "Ray::filter";
  Task* tsk= scinew Task( taskname, this, &Ray::filter, which_divQ_dw, includeEC, modifies_divQFilt );

  printSchedule(level,g_ray_dbg,taskname);

  tsk->requiresVar( which_divQ_dw, d_divQLabel,      d_gn, 0 );
  tsk->requiresVar( which_divQ_dw, d_boundFluxLabel, d_gn, 0 );
  tsk->computesVar(                d_divQFiltLabel);
  tsk->computesVar(                d_boundFluxFiltLabel);

  sched->addTask( tsk, level->eachPatch(), d_matlSet );
}
//---------------------------------------------------------------------------
// Filter divQ values.  In future will also filter boundFlux
//---------------------------------------------------------------------------
void
Ray::filter( const ProcessorGroup*,
              const PatchSubset* patches,
              const MaterialSubset*,
              DataWarehouse* old_dw,
              DataWarehouse* new_dw,
              Task::WhichDW which_divQ_dw,
              const bool includeEC,
              bool modifies_divQFilt)
{
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,g_ray_dbg,"Doing Ray::filt");

    constCCVariable<double> divQ;
    CCVariable<double>      divQFilt;
    constCCVariable<Stencil7> boundFlux;
    constCCVariable<Stencil7> boundFluxFilt;

    DataWarehouse* divQ_dw = new_dw->getOtherDataWarehouse(which_divQ_dw);
    divQ_dw->get(divQ,               d_divQLabel,        d_matl, patch, d_gn, 0);
    divQ_dw->get(boundFlux,          d_boundFluxLabel,   d_matl, patch, d_gn, 0);

    new_dw->allocateAndPut(divQFilt, d_boundFluxLabel,   d_matl, patch); // !! This needs to be fixed.  I need to create boundFluxFilt variable
    new_dw->allocateAndPut(divQFilt, d_divQLabel,        d_matl, patch);

    if( modifies_divQFilt ){
       old_dw->getModifiable(  divQFilt,  d_divQFiltLabel,  d_matl, patch );
     }else{
       new_dw->allocateAndPut( divQFilt,  d_divQFiltLabel,  d_matl, patch );
       divQFilt.initialize( 0.0 );
     }

    // set the cell iterator
    CellIterator iter = patch->getCellIterator();
    if(includeEC){
      iter = patch->getExtraCellIterator();
    }

    for (;!iter.done();iter++){
      const IntVector& c = *iter;
      int i = c.x();
      int j = c.y();
      int k = c.z();

      // if (i>=113 && i<=115 && j>=233 && j<=235 && k>=0 && k<=227 ){ // 3x3 extrusion test in z direction

      // box filter of origin plus 6 adjacent cells
      divQFilt[c] = (divQ[c]
                        + divQ[IntVector(i-1,j,k)] + divQ[IntVector(i+1,j,k)]
                        + divQ[IntVector(i,j-1,k)] + divQ[IntVector(i,j+1,k)]
                        + divQ[IntVector(i,j,k-1)] + divQ[IntVector(i,j,k+1)]) / 7;

      // 3D box filter, filter width=3
      /* divQFilt[c] = (  divQ[IntVector(i-1,j-1,k-1)] + divQ[IntVector(i,j-1,k-1)] + divQ[IntVector(i+1,j-1,k-1)]
                          + divQ[IntVector(i-1,j,k-1)]   + divQ[IntVector(i,j,k-1)]   + divQ[IntVector(i+1,j,k-1)]
                          + divQ[IntVector(i-1,j+1,k-1)] + divQ[IntVector(i,j+1,k-1)] + divQ[IntVector(i+1,j+1,k-1)]
                          + divQ[IntVector(i-1,j-1,k)]   + divQ[IntVector(i,j-1,k)]   + divQ[IntVector(i+1,j-1,k)]
                          + divQ[IntVector(i-1,j,k)]     + divQ[IntVector(i,j,k)]     + divQ[IntVector(i+1,j,k)]
                          + divQ[IntVector(i-1,j+1,k)]   + divQ[IntVector(i,j+1,k)]   + divQ[IntVector(i+1,j+1,k)]
                          + divQ[IntVector(i-1,j-1,k+1)] + divQ[IntVector(i,j-1,k+1)] + divQ[IntVector(i+1,j-1,k+1)]
                          + divQ[IntVector(i-1,j,k+1)]   + divQ[IntVector(i,j,k+1)]   + divQ[IntVector(i+1,j,k+1)]
                          + divQ[IntVector(i-1,j+1,k+1)] + divQ[IntVector(i,j+1,k+1)] + divQ[IntVector(i+1,j+1,k+1)]) / 27;
      */

    //} // end 3x3 extrusion test
    }
  }
}
#endif


//______________________________________________________________________
// Explicit template instantiations:
template void Ray::setBC<int, int>(       CCVariable<int>&    Q_CC, const string& desc, const Patch* patch, const int mat_id);
template void Ray::setBC<double,double>(  CCVariable<double>& Q_CC, const string& desc, const Patch* patch, const int mat_id);
template void Ray::setBC<float, double>(  CCVariable<float>&  Q_CC, const string& desc, const Patch* patch, const int mat_id);

template void Ray::setBoundaryConditions< double >( const ProcessorGroup*,
                                                    const PatchSubset* ,
                                                    const MaterialSubset*,
                                                    DataWarehouse*,
                                                    DataWarehouse* ,
                                                    Task::WhichDW ,
                                                    const bool );

template void Ray::setBoundaryConditions< float >( const ProcessorGroup*,
                                                   const PatchSubset* ,
                                                   const MaterialSubset*,
                                                   DataWarehouse*,
                                                   DataWarehouse* ,
                                                   Task::WhichDW ,
                                                   const bool );

template void  Ray::updateSumI_ML< double> ( Vector&,
                                             Vector&,
                                             const IntVector&,
                                             const vector<Vector>&,
                                             const BBox&,
                                             const int,
                                             const Level* ,
                                             const IntVector&,
                                             const IntVector&,
                                             vector<IntVector>&,
                                             vector<IntVector>&,
                                             std::vector< constCCVariable< double > >& sigmaT4OverPi,
                                             std::vector< constCCVariable<double> >& abskg,
                                             std::vector< constCCVariable< int > >& cellType,
                                             unsigned long int& ,
                                             double& ,
                                             MTRand&);

template void  Ray::updateSumI_ML< float> ( Vector&,
                                            Vector&,
                                            const IntVector&,
                                            const vector<Vector>&,
                                            const BBox&,
                                            const int,
                                            const Level* ,
                                            const IntVector&,
                                            const IntVector&,
                                            vector<IntVector>&,
                                            vector<IntVector>&,
                                            std::vector< constCCVariable< float > >& sigmaT4OverPi,
                                            std::vector< constCCVariable< float > >& abskg,
                                            std::vector< constCCVariable< int > >& cellType,
                                            unsigned long int& ,
                                            double& ,
                                            MTRand&);
