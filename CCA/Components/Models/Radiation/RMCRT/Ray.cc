/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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
#include <CCA/Components/Regridder/PerPatchVars.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/AMR.h>
#include <Core/Grid/AMR_CoarsenRefine.h>
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Math/MersenneTwister.h>

#ifdef UINTAH_ENABLE_KOKKOS
#include <Core/Grid/Variables/BlockRange.h>
#include <Kokkos_Core.hpp>
#endif // end UINTAH_ENABLE_KOKKOS

#include <time.h>
#include <fstream>

#include <include/sci_defs/uintah_testdefs.h.in>

// TURN ON debug flag in src/Core/Math/MersenneTwister.h to compare with Ray:CPU
#define DEBUG -9     // 1: divQ, 2: boundFlux, 3: scattering
#define CUDA_PRINTF   // increase the printf buffer

/*______________________________________________________________________
  TO DO
  create a vector (isComputedVarLabels) that contains boundryFlux, VRFlux, radiationVolq
  and use it for scheduling.  Add logic to what is actually in that vector.

Critical:
  - How do you coarsen cell type and retain information?
  - Memory issues.
  - Fix getRegion() for L-shaped domains.


Optimizations:
  - Spatial scheduling, used by radiometer.  Only need to execute & communicate
    variables on a subset of patches.
  - Temporal scheduling:  To reduce task graph recompilations create two task graphs
    that can be swapped while runninng.
  - Create a LevelDB.  Push an pull the communicated variables
  - DO: Reconstruct the fine level using interpolation and coarse level values
  - Investigate why floats are slow.
  -
______________________________________________________________________*/



//______________________________________________________________________
//
using namespace Uintah;
using namespace std;
static Uintah::DebugStream dbg("RAY",       false);
static Uintah::DebugStream dbg2("RAY_DEBUG",false);
static Uintah::DebugStream dbg_BC("RAY_BC", false);


//---------------------------------------------------------------------------
// Class: Constructor.
//---------------------------------------------------------------------------
Ray::Ray( const TypeDescription::Type FLT_DBL ) : RMCRTCommon( FLT_DBL)
{
//  d_boundFluxFiltLabel   = VarLabel::create( "boundFluxFilt",    CCVariable<Stencil7>::getTypeDescription() );
//  d_divQFiltLabel        = VarLabel::create( "divQFilt",         CCVariable<double>::getTypeDescription() );

  // internal variables for RMCRT
  d_flaggedCellsLabel    = VarLabel::create( "flaggedCells",     CCVariable<int>::getTypeDescription() );
  d_ROI_LoCellLabel      = VarLabel::create( "ROI_loCell",       minvec_vartype::getTypeDescription() );
  d_ROI_HiCellLabel      = VarLabel::create( "ROI_hiCell",       maxvec_vartype::getTypeDescription() );

  if( d_FLT_DBL == TypeDescription::double_type ){
    d_mag_grad_abskgLabel  = VarLabel::create( "mag_grad_abskg",   CCVariable<double>::getTypeDescription() );
    d_mag_grad_sigmaT4Label= VarLabel::create( "mag_grad_sigmaT4", CCVariable<double>::getTypeDescription() );
  } else {
    d_mag_grad_abskgLabel  = VarLabel::create( "mag_grad_abskg",   CCVariable<float>::getTypeDescription() );
    d_mag_grad_sigmaT4Label= VarLabel::create( "mag_grad_sigmaT4", CCVariable<float>::getTypeDescription() );
  }

  d_matlSet       = 0;
  d_isDbgOn       = dbg2.active();

  d_gac           = Ghost::AroundCells;
  d_gn            = Ghost::None;
  d_orderOfInterpolation = -9;
  d_onOff_SetBCs   = true;
  d_radiometer     = NULL;
  d_dbgCells.push_back( IntVector(36,0,0) );
  d_halo          = IntVector(-9,-9,-9);
  d_rayDirSampleAlgo = NAIVE;
  d_cellTypeCoarsenLogic = ROUNDUP;

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

  d_locationIndexOrder[EAST]  = IntVector(1,0,2);
  d_locationIndexOrder[WEST]  = IntVector(1,0,2);
  d_locationIndexOrder[NORTH] = IntVector(0,1,2);
  d_locationIndexOrder[SOUTH] = IntVector(0,1,2);
  d_locationIndexOrder[TOP]   = IntVector(0,2,1);
  d_locationIndexOrder[BOT]   = IntVector(0,2,1);

  d_locationShift[EAST]   = IntVector(1, 0, 0);
  d_locationShift[WEST]   = IntVector(0, 0, 0);
  d_locationShift[NORTH]  = IntVector(0, 1, 0);
  d_locationShift[SOUTH]  = IntVector(0, 0, 0);
  d_locationShift[TOP]    = IntVector(0, 0, 1);
  d_locationShift[BOT]    = IntVector(0, 0, 0);
}

//---------------------------------------------------------------------------
// Method: Destructor
//---------------------------------------------------------------------------
Ray::~Ray()
{
  VarLabel::destroy( d_mag_grad_abskgLabel );
  VarLabel::destroy( d_mag_grad_sigmaT4Label );
  VarLabel::destroy( d_flaggedCellsLabel );
  VarLabel::destroy( d_ROI_LoCellLabel );
  VarLabel::destroy( d_ROI_HiCellLabel );
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
                   const GridP& grid,
                   SimulationStateP&   sharedState)
{

  d_sharedState = sharedState;
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

  proc0cout << "__________________________________ " << endl;

#ifdef UINTAH_ENABLE_KOKKOS
  proc0cout << "  RMCRT:  Using the Kokkos-based implementation of RMCRT." << endl;
#endif

  if (rayDirSampleAlgo == "LatinHyperCube" ){
    d_rayDirSampleAlgo = LATIN_HYPER_CUBE;
    proc0cout << "  RMCRT:  Using Latin Hyper Cube method for selecting ray directions.";
  } else{
    proc0cout << "  RMCRT:  Using traditional Monte-Carlo method for selecting ray directions.";
  }

  //__________________________________
  //  Radiometer setup
  ProblemSpecP rad_ps = rmcrt_ps->findBlock("Radiometer");
  if( rad_ps ) {
    d_radiometer = scinew Radiometer( d_FLT_DBL );
    bool getExtraInputs = false;
    d_radiometer->problemSetup( prob_spec, rad_ps, grid, sharedState, getExtraInputs );
  }

  //__________________________________
  //  Warnings and bulletproofing
#ifndef RAY_SCATTER
  proc0cout<< "sigmaScat: " << d_sigmaScat << endl;
  if(d_sigmaScat>0){
    ostringstream warn;
    warn << " ERROR:  To run a scattering case, you must use the following in your configure line..." << endl;
    warn << "                 --enable-ray-scatter" << endl;
    warn << "         To run a non-scattering case, please remove the line containing <sigmaScat> from your input file." << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
#endif

#ifdef RAY_SCATTER
  proc0cout<< endl << "  RMCRT:  Ray scattering is enabled." << endl;
  if(d_sigmaScat<1e-99){
    proc0cout << "  WARNING:  You are running a non-scattering case, and you have the following in your configure line..." << endl;
    proc0cout << "                    --enable-ray-scatter" << endl;
    proc0cout << "            This will run slower than necessary." << endl;
    proc0cout << "            You can remove --enable-ray-scatter from your configure line and re-configure and re-compile" << endl;
  }
#endif

  if( d_nDivQRays == 1 ){
    proc0cout << "  RMCRT: WARNING: You have specified only 1 ray to compute the radiative flux divergence." << endl;
    proc0cout << "                  For better accuracy and stability, specify nDivQRays greater than 2." << endl;
  }

  if( d_nFluxRays == 1 ){
    proc0cout << "  RMCRT: WARNING: You have specified only 1 ray to compute radiative fluxes." << endl;
  }


  //__________________________________
  //  Read in the algorithm section
  bool isMultilevel   = false;
  Algorithm algorithm = singleLevel;

  ProblemSpecP alg_ps = rmcrt_ps->findBlock("algorithm");

  if (alg_ps){

    string type="NULL";

    if( !alg_ps->getAttribute("type", type) ){
      throw ProblemSetupException("RMCRT: No type specified for algorithm.  Please choose dataOnion on RMCRT_coarseLevel", __FILE__, __LINE__);
    }

    //__________________________________
    //  Data Onion
    if (type == "dataOnion" ) {

      isMultilevel = true;
      algorithm    = dataOnion;
      alg_ps->getWithDefault( "halo",  d_halo,  IntVector(10,10,10));

      //  Method for deteriming the extents of the ROI
      ProblemSpecP ROI_ps = alg_ps->findBlock("ROI_extents");
      ROI_ps->getAttribute("type", type);

      if(type == "fixed" ) {

        d_whichROI_algo = fixed;
        ROI_ps->get("min", d_ROI_minPt );
        ROI_ps->get("max", d_ROI_maxPt );

      } else if ( type == "dynamic" ) {

        d_whichROI_algo = dynamic;
        ROI_ps->getWithDefault( "abskgd_threshold",   d_abskg_thld,   DBL_MAX);
        ROI_ps->getWithDefault( "sigmaT4d_threshold", d_sigmaT4_thld, DBL_MAX);

      } else if ( type == "patch_based" ){
        d_whichROI_algo = patch_based;
      }

    //__________________________________
    //  rmcrt only on the coarse level
    } else if ( type == "RMCRT_coarseLevel" ) {
      isMultilevel = true;
      algorithm    = coarseLevel;
      alg_ps->require( "orderOfInterpolation", d_orderOfInterpolation);
    }
  }
  
  //__________________________________
  //  Logic for coarsening of cellType
  if (isMultilevel){
    string tmp = "NULL";
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

  if( Parallel::usingDevice() ){              // GPU
    if( (algorithm == dataOnion && d_whichROI_algo != patch_based ) ){
      ostringstream warn;
      warn << "GPU:RMCRT:ERROR: ";
      warn << "At this time only ROI_extents type=\"patch_based\" work on the GPU";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
  }

  // special conditions when using floats and multi-level
  if ( d_FLT_DBL == TypeDescription::float_type && isMultilevel) {

    string abskgName = d_compAbskgLabel->getName();
    if ( rmcrt_ps->isLabelSaved( abskgName ) ){
      ostringstream warn;
      warn << "  RMCRT:ERROR: You're saving a variable ("<< abskgName << ") that doesn't exist on all levels."<< endl;
      warn << "  Use either: " << endl;
      warn << "    <save label = 'abskgRMCRT' />             (float version of abskg, local to RMCRT)" << endl;
      warn << "             or " << endl;
      warn << "    <save label = 'abskg'  levels = -1 />     ( only saved on the finest level )" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
  }
  d_sigma_over_pi = d_sigma/M_PI;


//__________________________________
// Increase the printf buffer size only once!
#ifdef HAVE_CUDA
  #ifdef CUDA_PRINTF
  if( Parallel::usingDevice() && Parallel::getMPIRank() == 0){
    size_t size;
    CUDA_RT_SAFE_CALL( cudaDeviceGetLimit(&size,cudaLimitPrintfFifoSize) );
    CUDA_RT_SAFE_CALL( cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10*size ) );
    printf("RMCRT: CUDA: Increasing the size of the print buffer from %lu to %lu bytes\n",(long uint) size, ((long uint)10 * size) );
  }
  #endif
#endif
  proc0cout << "__________________________________ " << endl;

}

//______________________________________________________________________
//  Method:  Check that the boundary conditions have been set for temperature
//           and abskg
//______________________________________________________________________
void
Ray::BC_bulletproofing( const ProblemSpecP& rmcrtps )
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
    is_BC_specified(root_ps, d_compTempLabel->getName(), mss);
    is_BC_specified(root_ps, d_abskgBC_tag,              mss);

    Vector periodic;
    ProblemSpecP grid_ps  = root_ps->findBlock("Grid");
    ProblemSpecP level_ps = grid_ps->findBlock("Level");
    level_ps->getWithDefault("periodic", periodic, Vector(0,0,0));

    if (periodic.length() != 0 ){
      throw ProblemSetupException("\nERROR RMCRT:\nPeriodic boundary conditions are not allowed with Reverse Monte-Carlo Ray Tracing.", __FILE__, __LINE__);
    }
  }
}

//---------------------------------------------------------------------------
// Method: Schedule the ray tracer
// This task has both temporal and spatial scheduling and is tricky to follow
// The temporal scheduling is controlled by doCarryForward() and doRecompileTaskgraph()
// The spatial scheduling only occurs if the radiometer is used and is specified
// by the radiometerPatchSet.
//---------------------------------------------------------------------------
void
Ray::sched_rayTrace( const LevelP& level,
                     SchedulerP& sched,
                     Task::WhichDW abskg_dw,
                     Task::WhichDW sigma_dw,
                     Task::WhichDW celltype_dw,
                     bool modifies_divQ,
                     const int radCalc_freq )
{
  std::string taskname = "Ray::rayTrace";
  Task *tsk = NULL;

  if (Parallel::usingDevice()) {          // G P U

    if(radCalc_freq != 1){                // FIXME
       ostringstream warn;
       warn << "RMCRT:GPU  A radiation calculation frequency > 1 is not supported\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    if ( RMCRTCommon::d_FLT_DBL == TypeDescription::double_type ) {
      tsk = scinew Task( taskname, this, &Ray::rayTraceGPU< double >,
                           modifies_divQ, abskg_dw, sigma_dw, celltype_dw, radCalc_freq );
    } else {
      tsk = scinew Task( taskname, this, &Ray::rayTraceGPU< float >,
                           modifies_divQ, abskg_dw, sigma_dw, celltype_dw, radCalc_freq );

    }
    tsk->usesDevice(true);
  } else {                                // C P U

    if ( RMCRTCommon::d_FLT_DBL == TypeDescription::double_type ) {
      tsk = scinew Task( taskname, this, &Ray::rayTrace<double>,
                          modifies_divQ, abskg_dw, sigma_dw, celltype_dw, radCalc_freq );
    } else {
      tsk = scinew Task( taskname, this, &Ray::rayTrace<float>,
                         modifies_divQ, abskg_dw, sigma_dw, celltype_dw, radCalc_freq );
    }
  }

  printSchedule(level,dbg,taskname);

  //__________________________________
  // Require an infinite number of ghost cells so you can access the entire domain.
  //
  // THIS IS VERY EXPENSIVE.  THIS EXPENSE IS INCURRED ON NON-CALCULATION TIMESTEPS,
  // ONLY REQUIRE THESE VARIABLES ON A CALCULATION TIMESTEPS.
  //
  // For temporal scheduling the taskgraph must be recompiled to detect a change in the conditional.
  // The taskgraph recompilation is controlled with RMCRTCommon:doRecompileTaskgraph()
  if ( !doCarryForward( radCalc_freq) ) {
    Ghost::GhostType  gac  = Ghost::AroundCells;
    dbg << "    sched_rayTrace: adding requires for all-to-all variables " << endl;
    tsk->requires( abskg_dw ,    d_abskgLabel  ,   gac, SHRT_MAX);
    tsk->requires( sigma_dw ,    d_sigmaT4Label,   gac, SHRT_MAX);
    tsk->requires( celltype_dw , d_cellTypeLabel , gac, SHRT_MAX);
  }

  // TODO This is a temporary fix until we can generalize GPU/CPU carry forward functionality.
  if (!(Uintah::Parallel::usingDevice())) {
    // needed for carry Forward
    tsk->requires(Task::OldDW, d_divQLabel,          d_gn, 0);
    tsk->requires(Task::OldDW, d_boundFluxLabel,     d_gn, 0);
    tsk->requires(Task::OldDW, d_radiationVolqLabel, d_gn, 0);
  }

  if( modifies_divQ ){
    tsk->modifies( d_divQLabel );
    tsk->modifies( d_boundFluxLabel );
    tsk->modifies( d_radiationVolqLabel );
  } else {
    tsk->computes( d_divQLabel );
    tsk->computes( d_boundFluxLabel );
    tsk->computes( d_radiationVolqLabel );
  }

  //__________________________________
  // Radiometer
  if ( d_radiometer ){
    const VarLabel* VRFluxLabel = d_radiometer->getRadiometerLabel();
    if (!(Uintah::Parallel::usingDevice())) {
      // needed for carry Forward                       CUDA HACK
      tsk->requires(Task::OldDW, VRFluxLabel, d_gn, 0);
    }

    tsk->modifies( VRFluxLabel );
  }
  sched->addTask( tsk, level->eachPatch(), d_matlSet );

}

#ifdef UINTAH_ENABLE_KOKKOS

namespace {

template <typename T>
struct solveDivQFunctor {

    bool                   m_latinHyperCube;
    int                    m_d_nDivQRays;
    bool                   m_d_isSeedRandom;
    bool                   m_d_CCRays;
    double                 m_DyDx;
    double                 m_DzDx;
    double                 m_d_sigmaScat;
    double                 m_d_threshold;
    KokkosView3<const T>   m_abskg;
    KokkosView3<const T>   m_sigmaT4OverPi;
    KokkosView3<const int> m_celltype;
    int                    m_d_flowCell;
    double                 m_Dx[3];
    bool                   m_d_allowReflect;
    KokkosView3<double>    m_divQ;
    KokkosView3<double>    m_radiationVolq;
    BlockRange             m_range;

    solveDivQFunctor( bool                 & latinHyperCube,
                      int                  & d_nDivQRays,
                      bool                 & d_isSeedRandom,
                      bool                 & d_CCRays,
                      double               & DyDx,
                      double               & DzDx,
                      double               & d_sigmaScat,
                      double               & d_threshold,
                      constCCVariable< T > & abskg,
                      constCCVariable< T > & sigmaT4OverPi,
                      constCCVariable<int> & celltype,
                      int                  & d_flowCell,
                      Vector               & Dx,
                      bool                 & d_allowReflect,
                      CCVariable<double>   & divQ,
                      CCVariable<double>   & radiationVolq,
                      BlockRange           & range)
      : m_latinHyperCube ( latinHyperCube )
      , m_d_nDivQRays    ( d_nDivQRays )
      , m_d_isSeedRandom ( d_isSeedRandom )
      , m_d_CCRays       ( d_CCRays )
      , m_DyDx           ( DyDx )
      , m_DzDx           ( DzDx )
      , m_d_sigmaScat    ( d_sigmaScat )
      , m_d_threshold    ( d_threshold )
      , m_abskg          ( abskg.getKokkosView() )
      , m_sigmaT4OverPi  ( sigmaT4OverPi.getKokkosView() )
      , m_celltype       ( celltype.getKokkosView() )
      , m_d_flowCell     ( d_flowCell )
      , m_d_allowReflect ( d_allowReflect )
      , m_divQ           ( divQ.getKokkosView() )
      , m_radiationVolq  ( radiationVolq.getKokkosView() )
      , m_range          ( range )
    {
        m_Dx[0] = Dx.x();
        m_Dx[1] = Dx.y();
        m_Dx[2] = Dx.z();
    }

    // This operator() replaces the cellIterator loop used to solve DivQ
    void operator() ( int i, int j, int k, unsigned long int & m_nRaySteps ) const {

      MTRand mTwister;
      vector <int> m_rand_i( m_latinHyperCube ? m_d_nDivQRays : 0 );  // Only needed for LHC scheme

      //for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){

      //IntVector origin = *iter;

      //if (d_rayDirSampleAlgo == LATIN_HYPER_CUBE){
      //  randVector(rand_i, mTwister, origin);
      //}
      if ( m_latinHyperCube == true ) {

        //____________________________________________________//
        //==== START randVector(rand_i, mTwister, origin) ====//

        //int max= int_array.size();
        int max = m_rand_i.size();

        //for (int i=0; i<max; i++){   // populate sequential array from 0 to max-1
        //  int_array[i] = i;
        //}
        for ( int index = 0; index < max; index++ ) {  // Populate sequential array from 0 to max-1
          m_rand_i[index] = index;
        }

        //if( d_isSeedRandom == false ){
        //  mTwister.seed((cell.x() + cell.y() + cell.z()));
        //}
        if ( m_d_isSeedRandom == false ) {
          mTwister.seed( i + j + k );
        }

        //for (int i=max-1; i>0; i--){  // fisher-yates shuffle starting with max-1
        //  int rand_int =  mTwister.randInt(i);
        //  int swap = int_array[i];
        //  int_array[i] = int_array[rand_int];
        //  int_array[rand_int] = swap;
        //}
        for ( int index = max - 1; index > 0; index-- ) {  // Fisher-yates shuffle starting with max-1
          int rand_int       = mTwister.randInt( index );
          int swap           = m_rand_i[index];
          m_rand_i[index]    = m_rand_i[rand_int];
          m_rand_i[rand_int] = swap;
        }

        //__________________________________________________//
        //==== END randVector(rand_i, mTwister, origin) ====//

      }

      double sumI = 0;

      // Ray loop
      //for (int iRay=0; iRay < d_nDivQRays; iRay++){
      for ( int iRay = 0; iRay < m_d_nDivQRays; iRay++ ) {

        double direction_vector[3];

        //if (d_rayDirSampleAlgo == LATIN_HYPER_CUBE){        // Latin-Hyper-Cube sampling
        if ( m_latinHyperCube == true ) {  // Latin-Hyper-Cube sampling

          //direction_vector =findRayDirectionHyperCube(mTwister, origin, iRay, rand_i[iRay],iRay );

          //_____________________________________________________________________________________//
          //==== START findRayDirectionHyperCube(mTwister, origin, iRay, rand_i[iRay],iRay ) ====//

          //if( d_isSeedRandom == false ){
          //  m_mTwister.seed((i + j + k) * iRay + 1);
          //}
          if ( m_d_isSeedRandom == false ) {
            mTwister.seed( (i + j + k) * iRay + 1 );
          }

          // Random Points On Sphere
          //double plusMinus_one = 2.0 *(mTwister.randDblExc() + (double) rand_i[iRay])/d_nDivQRays - 1.0;  // add fuzz to avoid inf in 1/dirVector
          //double r = sqrt(1.0 - plusMinus_one * plusMinus_one);     // Radius of circle at z
          //double phi = 2.0 * M_PI * (mTwister.randDblExc() + (double) iRay)/d_nDivQRays;        // Uniform between 0-2Pi
          double plusMinus_one = 2.0 * ( mTwister.randDblExc() + (double) m_rand_i[iRay] ) / m_d_nDivQRays - 1.0;  // Add fuzz to avoid inf in 1/dirVector
          double r             = sqrt( 1.0 - plusMinus_one * plusMinus_one );                                      // Radius of circle at z
          double phi           = 2.0 * M_PI * ( mTwister.randDblExc() + (double) iRay ) / m_d_nDivQRays;           // Uniform between 0-2Pi

          direction_vector[0] = r * cos( phi );  // Convert to cartesian
          direction_vector[1] = r * sin( phi );
          direction_vector[2] = plusMinus_one;

          //___________________________________________________________________________________//
          //==== END findRayDirectionHyperCube(mTwister, origin, iRay, rand_i[iRay],iRay ) ====//

        } else {  // Naive Monte-Carlo sampling

          //direction_vector =findRayDirection(mTwister, origin, iRay );

          //_________________________________________________________//
          //==== START findRayDirection(mTwister, origin, iRay ) ====//

          //if( d_isSeedRandom == false ){
          //  mTwister.seed((origin.x() + origin.y() + origin.z()) * iRay +1);
          //}
          if ( m_d_isSeedRandom == false ) {
            mTwister.seed( (i + j + k) * iRay + 1 );
          }

          // Random Points On Sphere
          //double plusMinus_one = 2.0 * mTwister.randDblExc() - 1.0 + DBL_EPSILON;  // add fuzz to avoid inf in 1/dirVector
          //double r = sqrt(1.0 - plusMinus_one * plusMinus_one);     // Radius of circle at z
          //double theta = 2.0 * M_PI * mTwister.randDblExc();        // Uniform between 0-2Pi
          double plusMinus_one = 2.0 * mTwister.randDblExc() - 1.0 + DBL_EPSILON;  // Add fuzz to avoid inf in 1/dirVector
          double r             = sqrt( 1.0 - plusMinus_one * plusMinus_one );      // Radius of circle at z
          double theta         = 2.0 * M_PI * mTwister.randDblExc();               // Uniform between 0-2Pi

          direction_vector[0] = r * cos( theta );  // Convert to cartesian
          direction_vector[1] = r * sin( theta );
          direction_vector[2] = plusMinus_one;

          //_______________________________________________________//
          //==== END findRayDirection(mTwister, origin, iRay ) ====//

        }  // end m_latinHyperCube

        //Vector rayOrigin;
        double rayOrigin[3];

        //_______________________________________________________________________________//
        //==== START ray_Origin( mTwister, origin, DyDx,  DzDx, d_CCRays, rayOrigin) ====//

        //if( d_CCRays == false ){
        if ( m_d_CCRays == false ) {

          //rayOrigin[0] =   origin[0] +  mTwister.rand() ;             // FIX ME!!! This is not the physical location of the ray.
          //rayOrigin[1] =   origin[1] +  mTwister.rand() * DyDx ;      // this is index space.
          //rayOrigin[2] =   origin[2] +  mTwister.rand() * DzDx ;
          rayOrigin[0] = i +  mTwister.rand() ;           // FIX ME!!! This is not the physical location of the ray.
          rayOrigin[1] = j +  mTwister.rand() * m_DyDx ;  // This is index space.
          rayOrigin[2] = k +  mTwister.rand() * m_DzDx ;

        } else {

          //rayOrigin[0] =   origin[0] +  0.5 ;
          //rayOrigin[1] =   origin[1] +  0.5 * DyDx ;
          //rayOrigin[2] =   origin[2] +  0.5 * DzDx ;
          rayOrigin[0] = i +  0.5 ;
          rayOrigin[1] = j +  0.5 * m_DyDx ;
          rayOrigin[2] = k +  0.5 * m_DzDx ;

        }

        //_____________________________________________________________________________//
        //==== END ray_Origin( mTwister, origin, DyDx,  DzDx, d_CCRays, rayOrigin) ====//

        //________________________________________________________________________________________________________________________________//
        //==== START updateSumI< T >( direction_vector, rayOrigin, origin, Dx,  sigmaT4OverPi, abskg, celltype, size, sumI, mTwister) ====//

        //IntVector cur = origin;
        //IntVector prevCell = cur;
        int cur[3]      = { i, j, k };
        int prevCell[3] = { cur[0], cur[1], cur[2] };

        // Step and sign for ray marching
        int  step[3];  // Gives +1 or -1 based on sign
        bool sign[3];

        //Vector inv_ray_direction = Vector(1.0)/direction_vector;
        double inv_ray_direction[3] = { 1.0/direction_vector[0],
                                        1.0/direction_vector[1],
                                        1.0/direction_vector[2]  };

        //findStepSize( step, sign, inv_ray_direction);

        //____________________________________________________________//
        //==== START findStepSize( step, sign, inv_ray_direction) ====//

        (inv_ray_direction[0] > 0) ? (step[0] = 1, sign[0] = 1) : (step[0] = -1, sign[0] = 0);
        (inv_ray_direction[1] > 0) ? (step[1] = 1, sign[1] = 1) : (step[1] = -1, sign[1] = 0);
        (inv_ray_direction[2] > 0) ? (step[2] = 1, sign[2] = 1) : (step[2] = -1, sign[2] = 0);

        //____________________________________________________________//
        //==== END findStepSize( step, sign, inv_ray_direction) ====//

        //Vector D_DxRatio(1, Dx.y()/Dx.x(), Dx.z()/Dx.x() );
        double D_DxRatio[3] = { 1, m_DyDx, m_DzDx };

        //Vector tMax;         // (mixing bools, ints and doubles)
        //tMax.x( (origin[0] + sign[0]                - rayOrigin[0]) * inv_ray_direction[0] );
        //tMax.y( (origin[1] + sign[1] * D_DxRatio[1] - rayOrigin[1]) * inv_ray_direction[1] );
        //tMax.z( (origin[2] + sign[2] * D_DxRatio[2] - rayOrigin[2]) * inv_ray_direction[2] );
        double tMax[3] = { (i + sign[0]                - rayOrigin[0]) * inv_ray_direction[0],     // Mixing bools, ints and doubles
                           (j + sign[1] * D_DxRatio[1] - rayOrigin[1]) * inv_ray_direction[1],
                           (k + sign[2] * D_DxRatio[2] - rayOrigin[2]) * inv_ray_direction[2]  };

        // Length of t to traverse one cell
        //Vector tDelta = Abs(inv_ray_direction) * D_DxRatio;
        double tDelta[3] = { Abs( inv_ray_direction[0] ) * D_DxRatio[0],
                             Abs( inv_ray_direction[1] ) * D_DxRatio[1],
                             Abs( inv_ray_direction[2] ) * D_DxRatio[2]  };

        // Initializes the following values for each ray
        bool   in_domain            = true;
        double tMax_prev            = 0;
        double intensity            = 1.0;
        double fs                   = 1.0;
        int    nReflect             = 0;    // Number of reflections
        double optical_thickness    = 0;
        double expOpticalThick_prev = 1.0;

#ifdef RAY_SCATTER

        //double scatCoeff = d_sigmaScat;       // [m^-1]  !! HACK !! This needs to come from data warehouse
        double scatCoeff = m_d_sigmaScat;       // [m^-1]  !! HACK !! This needs to come from data warehouse

        if ( scatCoeff == 0 ) scatCoeff = 1e-99;  // Avoid division by zero

        // Determine the length at which scattering will occur
        // See CCA/Components/Arches/RMCRT/PaulasAttic/MCRT/ArchesRMCRT/ray.cc
        double scatLength = -log( mTwister.randDblExc() ) / scatCoeff;
        double curLength  = 0;

#endif  // end RAY_SCATTER

        //+++++++Begin ray tracing+++++++++++++++++++
        //Threshold while loop
        //while ( intensity > d_threshold ){
        while ( intensity > m_d_threshold ) {

          //DIR dir = NONE;
          int dir = -9;

          while ( in_domain ) {

            //prevCell = cur;
            prevCell[0] = cur[0];
            prevCell[1] = cur[1];
            prevCell[2] = cur[2];

            double disMin = -9;  // Represents ray segment length.

            //T abskg_prev         = abskg[prevCell];  // optimization
            //T sigmaT4OverPi_prev = sigmaT4OverPi[prevCell];
            T abskg_prev         = m_abskg(         prevCell[0], prevCell[1], prevCell[2] );  // Optimization
            T sigmaT4OverPi_prev = m_sigmaT4OverPi( prevCell[0], prevCell[1], prevCell[2] );

            //__________________________________
            //  Determine which cell the ray will enter next
            if ( tMax[0] < tMax[1] ) {    // X < Y
              if ( tMax[0] < tMax[2] ) {  // X < Z
                //dir = X;
                dir = 0;
              } else {
                //dir = Z;
                dir = 2;
              }
            } else {
              if ( tMax[1] < tMax[2] ) {   // Y < Z
                //dir = Y;
                dir = 1;
              } else {
                //dir = Z;
                dir = 2;
              }
            }

            //__________________________________
            //  Update marching variables
            cur[dir]  = cur[dir]  + step[dir];
            disMin    = ( tMax[dir] - tMax_prev );
            tMax_prev = tMax[dir];
            tMax[dir] = tMax[dir] + tDelta[dir];

            rayOrigin[0] = rayOrigin[0] + (disMin * direction_vector[0]);
            rayOrigin[1] = rayOrigin[1] + (disMin * direction_vector[1]);
            rayOrigin[2] = rayOrigin[2] + (disMin * direction_vector[2]);

            //in_domain = (celltype[cur] == d_flowCell);
            in_domain = ( m_celltype( cur[0], cur[1], cur[2] ) == m_d_flowCell );

            //optical_thickness += Dx[0] * abskg_prev*disMin; // as long as tDeltaY,Z tMax.y(),Z and ray_location[1],[2]..
            optical_thickness += m_Dx[0] * abskg_prev * disMin; // As long as tDeltaY,Z tMax.y(),Z and ray_location[1],[2]
                                                                // were adjusted by DyDx  or DzDx, this line is now correct
                                                                // for noncubic domains.

            //nRaySteps++;
            m_nRaySteps++;

            // Eqn 3-15(see below reference) while
            // Third term inside the parentheses is accounted for in Inet. Chi is accounted for in Inet calc.
            double expOpticalThick = exp( -optical_thickness );

            sumI += sigmaT4OverPi_prev * ( expOpticalThick_prev - expOpticalThick ) * fs;

            expOpticalThick_prev = expOpticalThick;

#ifdef RAY_SCATTER

            //curLength += disMin * Dx[0];
            curLength += disMin * m_Dx[0];

            if ( curLength > scatLength && in_domain ) {

              // Get new scatLength for each scattering event
              scatLength = -log( mTwister.randDblExc() ) / scatCoeff;

              //direction_vector     =  findRayDirection( mTwister, cur );

              //_________________________________________________//
              //==== START findRayDirection( mTwister, cur ) ====//

              //if( d_isSeedRandom == false ){
              //  mTwister.seed((cur.x() + cur.y() + cur.z()) * -9 +1);
              //}
              if ( m_d_isSeedRandom == false ) {
                mTwister.seed( (cur[0] + cur[1] + cur[2]) * -9 + 1 );
              }

              // Random Points On Sphere
              //double plusMinus_one = 2.0 * mTwister.randDblExc() - 1.0 + DBL_EPSILON;  // add fuzz to avoid inf in 1/dirVector
              //double r = sqrt(1.0 - plusMinus_one * plusMinus_one);     // Radius of circle at z
              //double theta = 2.0 * M_PI * mTwister.randDblExc();        // Uniform between 0-2Pi
              double plusMinus_one = 2.0 * mTwister.randDblExc() - 1.0 + DBL_EPSILON;  // Add fuzz to avoid inf in 1/dirVector
              double r             = sqrt( 1.0 - plusMinus_one * plusMinus_one );      // Radius of circle at z
              double theta         = 2.0 * M_PI * mTwister.randDblExc();               // Uniform between 0-2Pi

              //Vector direction_vector;
              direction_vector[0] = r * cos( theta );  // Convert to cartesian
              direction_vector[1] = r * sin( theta );
              direction_vector[2] = plusMinus_one;

              //_______________________________________________//
              //==== END findRayDirection( mTwister, cur ) ====//

              //inv_ray_direction = Vector(1.0)/direction_vector;
              inv_ray_direction[0] = 1.0 / direction_vector[0];
              inv_ray_direction[1] = 1.0 / direction_vector[1];
              inv_ray_direction[2] = 1.0 / direction_vector[2];

              // Get new step and sign
              int stepOld = step[dir];
              //findStepSize( step, sign, inv_ray_direction);

              //____________________________________________________________//
              //==== START findStepSize( step, sign, inv_ray_direction) ====//

              (inv_ray_direction[0] > 0) ? (step[0] = 1, sign[0] = 1) : (step[0] = -1, sign[0] = 0);
              (inv_ray_direction[1] > 0) ? (step[1] = 1, sign[1] = 1) : (step[1] = -1, sign[1] = 0);
              (inv_ray_direction[2] > 0) ? (step[2] = 1, sign[2] = 1) : (step[2] = -1, sign[2] = 0);

              //__________________________________________________________//
              //==== END findStepSize( step, sign, inv_ray_direction) ====//

              // If sign[dir] changes sign, put ray back into prevCell (back scattering)
              // A sign change only occurs when the product of old and new is negative
              if ( step[dir] * stepOld < 0 ) {
                //cur = prevCell;
                cur[0] = prevCell[0];
                cur[1] = prevCell[1];
                cur[2] = prevCell[2];
              }

              // Get new tMax (mixing bools, ints and doubles)
              //tMax.x( ( cur[0] + sign[0]                - rayOrigin[0]) * inv_ray_direction[0] );
              //tMax.y( ( cur[1] + sign[1] * D_DxRatio[1] - rayOrigin[1]) * inv_ray_direction[1] );
              //tMax.z( ( cur[2] + sign[2] * D_DxRatio[2] - rayOrigin[2]) * inv_ray_direction[2] );
              tMax[0] = (cur[0] + sign[0]                - rayOrigin[0]) * inv_ray_direction[0];  // Mixing bools, ints and doubles
              tMax[1] = (cur[1] + sign[1] * D_DxRatio[1] - rayOrigin[1]) * inv_ray_direction[1];
              tMax[2] = (cur[2] + sign[2] * D_DxRatio[2] - rayOrigin[2]) * inv_ray_direction[2];

              // Length of t to traverse one cell
              //tDelta    = Abs(inv_ray_direction) * D_DxRatio;
              tDelta[0] = Abs( inv_ray_direction[0] ) * D_DxRatio[0];
              tDelta[1] = Abs( inv_ray_direction[1] ) * D_DxRatio[1];
              tDelta[2] = Abs( inv_ray_direction[2] ) * D_DxRatio[2];

              tMax_prev = 0;
              curLength = 0;  // Allow for multiple scattering events per ray
            }

#endif // end RAY_SCATTER

          } // end domain while loop.  ++++++++++++++

          //T wallEmissivity = abskg[cur];

          //if (wallEmissivity > 1.0){       // Ensure wall emissivity doesn't exceed one
          //  wallEmissivity = 1.0;
          //}

          T wallEmissivity = (m_abskg( cur[0], cur[1], cur[2] ) > 1.0) ? 1.0: m_abskg( cur[0], cur[1], cur[2] );  // Ensure wall emissivity doesn't exceed one

          intensity = exp( -optical_thickness );

          //sumI += wallEmissivity * sigmaT4OverPi[cur] * intensity;
          sumI += wallEmissivity * m_sigmaT4OverPi( cur[0], cur[1], cur[2] ) * intensity;

          //intensity = intensity * fs;

          // When a ray reaches the end of the domain, we force it to terminate.
          //if(!d_allowReflect) intensity = 0;
          intensity = ( !m_d_allowReflect ) ? 0 : ( intensity * fs );

          //__________________________________
          //  Reflections
          //if ( (intensity > d_threshold) && d_allowReflect){
          if ( (intensity > m_d_threshold) && m_d_allowReflect ) {

            //reflect( fs, cur, prevCell, abskg[cur], in_domain, step[dir], sign[dir], direction_vector[dir]);

            //_______________________________________________________________________________________________________________//
        	//==== START reflect( fs, cur, prevCell, abskg[cur], in_domain, step[dir], sign[dir], direction_vector[dir]) ====//

            //fs *= (1 - abskg[cur]);
            fs *= ( 1 - m_abskg( cur[0], cur[1], cur[2] ) );

            // Put cur back inside the domain
            //cur = prevCell;
            cur[0] = prevCell[0];
            cur[1] = prevCell[1];
            cur[2] = prevCell[2];

            in_domain = true;

            // Apply reflection condition
            step[dir]             *= -1;              // Begin stepping in opposite direction
            sign[dir]              =  1 - sign[dir];  // Swap sign from 1 to 0 or vice versa
            //sign[dir] = (sign[dir]==1) ? 0 : 1;  //  swap sign from 1 to 0 or vice versa
            direction_vector[dir] *= -1;

            //_____________________________________________________________________________________________________________//
            //==== END reflect( fs, cur, prevCell, abskg[cur], in_domain, step[dir], sign[dir], direction_vector[dir]) ====//

            ++nReflect;

          }

        }  // end threshold while loop.

        //______________________________________________________________________________________________________________________________//
        //==== END updateSumI< T >( direction_vector, rayOrigin, origin, Dx,  sigmaT4OverPi, abskg, celltype, size, sumI, mTwister) ====//

      }  // end Ray loop

      //__________________________________
      //  Compute divQ
      //divQ[origin] = -4.0 * M_PI * abskg[origin] * ( sigmaT4OverPi[origin] - (sumI/d_nDivQRays) );
      m_divQ( i, j, k ) = -4.0 * M_PI * m_abskg( i, j, k ) * ( m_sigmaT4OverPi( i, j, k ) - (sumI / m_d_nDivQRays) );

      // radiationVolq is the incident energy per cell (W/m^3) and is necessary when particle heat transfer models (i.e. Shaddix) are used
      //radiationVolq[origin] = 4.0 * M_PI * abskg[origin] *  (sumI/d_nDivQRays) ;
      m_radiationVolq( i, j, k ) = 4.0 * M_PI * m_abskg( i, j, k ) * (sumI / m_d_nDivQRays);

    }  // end operator()
};  // end solveDivQFunctor
}   // end namespace

#endif // end UINTAH_ENABLE_KOKKOS

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
               Task::WhichDW which_celltype_dw,
               const int radCalc_freq )
{

  const Level* level = getLevel(patches);

  // Control code to recompile the task graph
  // This provides temporal scheduling mechanism
  doRecompileTaskgraph( radCalc_freq );

  if ( doCarryForward( radCalc_freq ) ) {
    printTask(patches,patches->get(0), dbg,"Doing Ray::rayTrace (carryForward)");
    bool replaceVar = true;
    new_dw->transferFrom( old_dw, d_divQLabel,          patches, matls, replaceVar );
    new_dw->transferFrom( old_dw, d_boundFluxLabel,     patches, matls, replaceVar );
    new_dw->transferFrom( old_dw, d_radiationVolqLabel, patches, matls, replaceVar );
    return;
  }

  //__________________________________
  //
  MTRand mTwister;

  DataWarehouse* abskg_dw    = new_dw->getOtherDataWarehouse(which_abskg_dw);
  DataWarehouse* sigmaT4_dw  = new_dw->getOtherDataWarehouse(which_sigmaT4_dw);
  DataWarehouse* celltype_dw = new_dw->getOtherDataWarehouse(which_celltype_dw);

  constCCVariable< T > sigmaT4OverPi;
  constCCVariable< T > abskg;
  constCCVariable<int>  celltype;

  abskg_dw->getLevel(   abskg   ,       d_abskgLabel ,   d_matl , level );
  sigmaT4_dw->getLevel( sigmaT4OverPi , d_sigmaT4Label,  d_matl , level );
  celltype_dw->getLevel( celltype ,     d_cellTypeLabel, d_matl , level );


  double start=clock();

  // patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing Ray::rayTrace");

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
        IntVector origin = *iter;
        boundFlux[origin].initialize(0.0);
      }
   }
    unsigned long int size = 0;                   // current size of PathIndex
    Vector Dx = patch->dCell();                   // cell spacing
    double DyDx = Dx.y() / Dx.x();                //noncubic
    double DzDx = Dx.z() / Dx.x();                //noncubic

    //______________________________________________________________________
    //           R A D I O M E T E R
    //______________________________________________________________________

    if (d_radiometer) {
      d_radiometer->radiometerFlux< T >( patch, level, new_dw, mTwister, sigmaT4OverPi, abskg, celltype, modifies_divQ );
    }

    //______________________________________________________________________
    //          B O U N D A R Y F L U X
    //______________________________________________________________________
    if( d_solveBoundaryFlux){

      //__________________________________
      //
      vector <int> rand_i( d_rayDirSampleAlgo == LATIN_HYPER_CUBE ? d_nFluxRays : 0);  // only needed for LHC scheme

      for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector origin = *iter;

        // A given flow cell may have 0,1,2,3,4,5, or 6 faces that are adjacent to a wall.
        // boundaryFaces is the vector that contains the list of which faces are adjacent to a wall
        vector<int> boundaryFaces;
        boundaryFaces.clear();

        // determine if origin has one or more boundary faces, and if so, populate boundaryFaces vector
        boundFlux[origin].p = has_a_boundary(origin, celltype, boundaryFaces);

        //__________________________________
        // Loop over boundary faces of the cell and compute incident radiative flux
        for (vector<int>::iterator it=boundaryFaces.begin() ; it < boundaryFaces.end(); it++ ){

          int RayFace = *it;
          int UintahFace[6] = {WEST,EAST,SOUTH,NORTH,BOT,TOP};

          double sumI     = 0;
          double sumProjI = 0;
          double sumI_prev= 0;
          double sumCos=0;    // used to force sumCostheta/nRays == 0.5 or  sum (d_Omega * cosTheta) == pi

          if (d_rayDirSampleAlgo == LATIN_HYPER_CUBE){
            randVector(rand_i, mTwister, origin);
          }


          //__________________________________
          // Flux ray loop
          for (int iRay=0; iRay < d_nFluxRays; iRay++){

            Vector direction_vector, ray_location;
            double cosTheta;

            if ( d_rayDirSampleAlgo == LATIN_HYPER_CUBE ){    // Latin-Hyper-Cube sampling
              rayDirectionHyperCube_cellFace( mTwister, origin, d_dirIndexOrder[RayFace], d_dirSignSwap[RayFace], iRay,
                                              direction_vector, cosTheta, rand_i[iRay],iRay);
            } else{                                          // Naive Monte-Carlo sampling
              rayDirection_cellFace( mTwister, origin, d_dirIndexOrder[RayFace], d_dirSignSwap[RayFace], iRay,
                                     direction_vector, cosTheta );
            }

              rayLocation_cellFace( mTwister, origin, d_locationIndexOrder[RayFace], d_locationShift[RayFace],
                                    DyDx, DzDx, ray_location);

            updateSumI<T>( direction_vector, ray_location, origin, Dx, sigmaT4OverPi, abskg, celltype, size, sumI, mTwister);

            sumProjI += cosTheta * (sumI - sumI_prev);   // must subtract sumI_prev, since sumI accumulates intensity
            sumCos += cosTheta;

            sumI_prev = sumI;

          } // end of flux ray loop

          sumProjI = sumProjI * d_nFluxRays/sumCos/2.0; // This operation corrects for error in the first moment over a half range of the solid angle (Modest Radiative Heat Transfer page 545 1rst edition)

          //__________________________________
          //  Compute Net Flux to the boundary
          int face = UintahFace[RayFace];
          boundFlux[origin][ face ] = sumProjI * 2 *M_PI/d_nFluxRays;

/*`==========TESTING==========*/
#if DEBUG == 2
          printf( "\n      [%d, %d, %d]  face: %d sumProjI:  %g BF: %g\n",
                    origin.x(), origin.y(), origin.z(), face, sumProjI, boundFlux[origin][ face ]);
#endif
/*===========TESTING==========`*/
        } // boundary faces loop
      }  // end cell iterator
    }   // end if d_solveBoundaryFlux


    //______________________________________________________________________
    //         S O L V E   D I V Q
    //______________________________________________________________________
  if( d_solveDivQ){

#ifdef UINTAH_ENABLE_KOKKOS

    bool latinHyperCube = ( d_rayDirSampleAlgo == LATIN_HYPER_CUBE ) ? true : false;

    IntVector lo = patch->getCellLowIndex();
    IntVector hi = patch->getCellHighIndex();

    Uintah::BlockRange range(lo, hi);

    solveDivQFunctor<T> functor( latinHyperCube,
                                 d_nDivQRays,
                                 d_isSeedRandom,
                                 d_CCRays,
                                 DyDx,
                                 DzDx,
                                 d_sigmaScat,
                                 d_threshold,
                                 abskg,
                                 sigmaT4OverPi,
                                 celltype,
                                 d_flowCell,
                                 Dx,
                                 d_allowReflect,
                                 divQ,
                                 radiationVolq,
                                 range           );

    // This parallel_reduce replaces the cellIterator loop used to solve DivQ
    Uintah::parallel_reduce( range, functor, size );

#else // else UINTAH_ENABLE_KOKKOS

    vector <int> rand_i( d_rayDirSampleAlgo == LATIN_HYPER_CUBE ? d_nDivQRays : 0);  // only needed for LHC scheme

    for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      IntVector origin = *iter;

      if (d_rayDirSampleAlgo == LATIN_HYPER_CUBE){
        randVector(rand_i, mTwister, origin);
      }
      double sumI = 0;

      // ray loop
      for (int iRay=0; iRay < d_nDivQRays; iRay++){

        Vector direction_vector;
        if (d_rayDirSampleAlgo == LATIN_HYPER_CUBE){        // Latin-Hyper-Cube sampling
          direction_vector =findRayDirectionHyperCube(mTwister, origin, iRay, rand_i[iRay],iRay );
        }else{                                              // Naive Monte-Carlo sampling
          direction_vector =findRayDirection(mTwister, origin, iRay );
        }

        Vector rayOrigin;
        ray_Origin( mTwister, origin, DyDx,  DzDx, d_CCRays, rayOrigin);

        updateSumI< T >( direction_vector, rayOrigin, origin, Dx,  sigmaT4OverPi, abskg, celltype, size, sumI, mTwister);

      }  // Ray loop

      //__________________________________
      //  Compute divQ
      divQ[origin] = -4.0 * M_PI * abskg[origin] * ( sigmaT4OverPi[origin] - (sumI/d_nDivQRays) );

      // radiationVolq is the incident energy per cell (W/m^3) and is necessary when particle heat transfer models (i.e. Shaddix) are used
      radiationVolq[origin] = 4.0 * M_PI * abskg[origin] *  (sumI/d_nDivQRays) ;
/*`==========TESTING==========*/
#if DEBUG == 1
    if( isDbgCell(origin) ) {
          printf( "\n      [%d, %d, %d]  sumI: %g  divQ: %g radiationVolq: %g  abskg: %g,    sigmaT4: %g \n",
                    origin.x(), origin.y(), origin.z(), sumI,divQ[origin], radiationVolq[origin],abskg[origin], sigmaT4OverPi[origin]);
    }
#endif
/*===========TESTING==========`*/
    }  // end cell iterator

#endif // end UINTAH_ENABLE_KOKKOS

  }  // end of if(_solveDivQ)


    double end =clock();
    double efficiency = size/((end-start)/ CLOCKS_PER_SEC);
    if (patch->getGridIndex() == 0) {
      cout<< endl;
      cout << " RMCRT REPORT: Patch 0" << endl;
      cout << " Used "<< (end-start) * 1000 / CLOCKS_PER_SEC<< " milliseconds of CPU time. \n" << endl;// Convert time to ms
      cout << " Size: " << size << endl;
      cout << " Efficiency: " << efficiency << " steps per sec" << endl;
      cout << endl;
    }
  }  //end patch loop
}  // end ray trace method



//---------------------------------------------------------------------------
// Ray tracing using the multilevel data onion scheme
//---------------------------------------------------------------------------
void
Ray::sched_rayTrace_dataOnion( const LevelP& level,
                               SchedulerP& sched,
                               Task::WhichDW abskg_dw,
                               Task::WhichDW sigma_dw,
                               Task::WhichDW celltype_dw,
                               bool modifies_divQ,
                               const int radCalc_freq )
{
  int maxLevels = level->getGrid()->numLevels() - 1;
  int L_indx = level->getIndex();

  if (L_indx != maxLevels) {     // only schedule on the finest level
    return;
  }
  std::string taskname = "";
  Task* tsk = NULL;

  if (Parallel::usingDevice()) {          // G P U
    taskname = "Ray::rayTraceDataOnionGPU";

    if(radCalc_freq != 1){                // FIXME
       ostringstream warn;
       warn << "RMCRT:GPU  A radiation calculation frequency > 1 is not supported\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }


    if ( RMCRTCommon::d_FLT_DBL == TypeDescription::double_type ){
      tsk = scinew Task( taskname, this, &Ray::rayTraceDataOnionGPU< double >,
                         modifies_divQ, abskg_dw, sigma_dw, celltype_dw, radCalc_freq );
    } else {
      tsk = scinew Task( taskname, this, &Ray::rayTraceDataOnionGPU< float >,
                         modifies_divQ, abskg_dw, sigma_dw, celltype_dw, radCalc_freq );
    }
    tsk->usesDevice(true);
  } else {                                // CPU
    taskname = "Ray::rayTrace_dataOnion";
    if (RMCRTCommon::d_FLT_DBL == TypeDescription::double_type) {
      tsk = scinew Task(taskname, this, &Ray::rayTrace_dataOnion<double>, modifies_divQ, abskg_dw, sigma_dw, celltype_dw,
                        radCalc_freq);
    }
    else {
      tsk = scinew Task(taskname, this, &Ray::rayTrace_dataOnion<float>, modifies_divQ, abskg_dw, sigma_dw, celltype_dw,
                        radCalc_freq);
    }
  }

  printSchedule(level,dbg,taskname);

  Task::MaterialDomainSpec  ND  = Task::NormalDomain;
  #define allPatches 0
  #define allMatls 0
  Ghost::GhostType  gac  = Ghost::AroundCells;

  // finest level:
  if ( d_whichROI_algo == patch_based ) {          // patch_based we know the number of ghostCells

    int maxElem = Max( d_halo.x(), d_halo.y(), d_halo.z() );
    tsk->requires( abskg_dw,     d_abskgLabel,     gac, maxElem);
    tsk->requires( sigma_dw,     d_sigmaT4Label,   gac, maxElem);
    tsk->requires( celltype_dw , d_cellTypeLabel , gac, maxElem);
  } else {                                        // we don't know the number of ghostCells so get everything
    tsk->requires( abskg_dw,      d_abskgLabel,     gac, SHRT_MAX);
    tsk->requires( sigma_dw,      d_sigmaT4Label,   gac, SHRT_MAX);
    tsk->requires( celltype_dw ,  d_cellTypeLabel , gac, SHRT_MAX);
  }

  // TODO This is a temporary fix until we can generalize GPU/CPU carry forward functionality.
  if (!(Uintah::Parallel::usingDevice())) {
    // needed for carry Forward
    tsk->requires( Task::OldDW, d_divQLabel,           d_gn, 0 );
    tsk->requires( Task::OldDW, d_radiationVolqLabel,  d_gn, 0 );
  }


  if( d_whichROI_algo == dynamic ){
    tsk->requires(Task::NewDW, d_ROI_LoCellLabel);
    tsk->requires(Task::NewDW, d_ROI_HiCellLabel);
  }

  // coarser level
  int nCoarseLevels = maxLevels;
  for (int l=1; l<=nCoarseLevels; ++l){
    tsk->requires(abskg_dw,    d_abskgLabel,    allPatches, Task::CoarseLevel,l,allMatls, ND, gac, SHRT_MAX);
    tsk->requires(sigma_dw,    d_sigmaT4Label,  allPatches, Task::CoarseLevel,l,allMatls, ND, gac, SHRT_MAX);
    tsk->requires(celltype_dw, d_cellTypeLabel, allPatches, Task::CoarseLevel,l,allMatls, ND, gac, SHRT_MAX);
    proc0cout << "WARNING: RMCRT High communication costs on level:" << l
              << ".  Variables from every patch on this level are communicated to every patch on the finest level."<< endl;
  }

  if( modifies_divQ ){
    tsk->modifies( d_divQLabel );
    tsk->modifies( d_boundFluxLabel );
    tsk->modifies( d_radiationVolqLabel );

  } else {

    tsk->computes( d_divQLabel );
    tsk->computes( d_boundFluxLabel );
    tsk->computes( d_radiationVolqLabel );

  }
  sched->addTask( tsk, level->eachPatch(), d_matlSet );
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
                         Task::WhichDW which_abskg_dw,
                         Task::WhichDW which_sigmaT4_dw,
                         Task::WhichDW which_celltype_dw,
                         const int radCalc_freq )
{

  const Level* fineLevel = getLevel(finePatches);
   //__________________________________
  //  Carry Forward (old_dw -> new_dw)
  if ( doCarryForward( radCalc_freq ) ) {
    printTask( finePatches, dbg, "Doing Ray::rayTrace_dataOnion carryForward ( divQ )" );

    new_dw->transferFrom( old_dw, d_divQLabel,          finePatches, matls, true );
    new_dw->transferFrom( old_dw, d_radiationVolqLabel, finePatches, matls, true );
    return;
  }

  //__________________________________
  //
  int maxLevels    = fineLevel->getGrid()->numLevels();
  int levelPatchID = fineLevel->getPatch(0)->getID();
  LevelP level_0 = new_dw->getGrid()->getLevel(0);
  MTRand mTwister;

  //__________________________________
  // retrieve the coarse level data
  // compute the level dependent variables that are constant
  StaticArray< constCCVariable< T > > abskg(maxLevels);
  StaticArray< constCCVariable< T > >sigmaT4OverPi(maxLevels);
  StaticArray< constCCVariable<int> >cellType(maxLevels);
  constCCVariable< T > abskg_fine;
  constCCVariable< T > sigmaT4OverPi_fine;

  DataWarehouse* abskg_dw    = new_dw->getOtherDataWarehouse(which_abskg_dw);
  DataWarehouse* sigmaT4_dw  = new_dw->getOtherDataWarehouse(which_sigmaT4_dw);
  DataWarehouse* celltype_dw = new_dw->getOtherDataWarehouse(which_celltype_dw);

  vector<Vector> Dx(maxLevels);
  double DyDx[maxLevels];
  double DzDx[maxLevels];

  for(int L = 0; L<maxLevels; L++) {
    LevelP level = new_dw->getGrid()->getLevel(L);

    if (level->hasFinerLevel() ) {                               // coarse level data

      abskg_dw->getLevel(   abskg[L]   ,       d_abskgLabel ,   d_matl , level.get_rep() );
      sigmaT4_dw->getLevel( sigmaT4OverPi[L] , d_sigmaT4Label,  d_matl , level.get_rep() );
      celltype_dw->getLevel( cellType[L] ,     d_cellTypeLabel, d_matl , level.get_rep() );
      dbg << " getting coarse level data L-" << L << endl;
    }
    Vector dx = level->dCell();
    DyDx[L] = dx.y() / dx.x();
    DzDx[L] = dx.z() / dx.x();
    Dx[L] = dx;
  }

  IntVector fineLevel_ROI_Lo = IntVector(-9,-9,-9);
  IntVector fineLevel_ROI_Hi = IntVector(-9,-9,-9);
  vector<IntVector> regionLo(maxLevels);
  vector<IntVector> regionHi(maxLevels);

  //__________________________________
  //  retrieve fine level data & compute the extents (dynamic and fixed )
  if ( d_whichROI_algo == fixed || d_whichROI_algo == dynamic ) {
    int L = maxLevels - 1;

    const Patch* notUsed=0;
    computeExtents(level_0, fineLevel, notUsed, maxLevels, new_dw,
                   fineLevel_ROI_Lo, fineLevel_ROI_Hi,
                   regionLo,  regionHi);

    dbg << " getting fine level data across L-" << L << " " << fineLevel_ROI_Lo << " " << fineLevel_ROI_Hi << endl;
    abskg_dw->getRegion(   abskg[L]   ,       d_abskgLabel ,   d_matl , fineLevel, fineLevel_ROI_Lo, fineLevel_ROI_Hi);
    sigmaT4_dw->getRegion( sigmaT4OverPi[L] , d_sigmaT4Label,  d_matl , fineLevel, fineLevel_ROI_Lo, fineLevel_ROI_Hi);
    celltype_dw->getRegion( cellType[L] ,     d_cellTypeLabel, d_matl , fineLevel, fineLevel_ROI_Lo, fineLevel_ROI_Hi);
  }

  abskg_fine         = abskg[maxLevels-1];
  sigmaT4OverPi_fine = sigmaT4OverPi[maxLevels-1];

  double start=clock();

  // Determine the size of the domain.
  BBox domain_BB;
  level_0->getInteriorSpatialRange(domain_BB);                 // edge of computational domain

  //  patch loop
  for (int p=0; p < finePatches->size(); p++) {

    const Patch* finePatch = finePatches->get(p);
    printTask(finePatches, finePatch,dbg,"Doing Ray::rayTrace_dataOnion");

     //__________________________________
    //  retrieve fine level data ( patch_based )
    if ( d_whichROI_algo == patch_based ){

      computeExtents(level_0, fineLevel, finePatch, maxLevels, new_dw,
                     fineLevel_ROI_Lo, fineLevel_ROI_Hi,
                     regionLo,  regionHi);

      int L = maxLevels - 1;
      dbg << " getting fine level data across L-" << L << endl;

      abskg_dw->getRegion(   abskg[L]   ,       d_abskgLabel ,  d_matl , fineLevel, fineLevel_ROI_Lo, fineLevel_ROI_Hi);
      sigmaT4_dw->getRegion( sigmaT4OverPi[L] , d_sigmaT4Label, d_matl , fineLevel, fineLevel_ROI_Lo, fineLevel_ROI_Hi);
      celltype_dw->getRegion( cellType[L] ,     d_cellTypeLabel, d_matl ,fineLevel, fineLevel_ROI_Lo, fineLevel_ROI_Hi);
      abskg_fine         = abskg[L];
      sigmaT4OverPi_fine = sigmaT4OverPi[L];
    }

    CCVariable<double> divQ_fine;
    CCVariable<double> radiationVolq_fine;

    if( modifies_divQ ){
      old_dw->getModifiable( divQ_fine,         d_divQLabel,          d_matl, finePatch );
      old_dw->getModifiable( radiationVolq_fine,d_radiationVolqLabel, d_matl, finePatch );
    }else{
      new_dw->allocateAndPut( divQ_fine,          d_divQLabel,         d_matl, finePatch );
      new_dw->allocateAndPut( radiationVolq_fine, d_radiationVolqLabel, d_matl,finePatch );
      divQ_fine.initialize( 0.0 );
      radiationVolq_fine.initialize( 0.0 );
    }

    unsigned long int nRaySteps = 0;

    //__________________________________
    //

    vector <int> rand_i( d_rayDirSampleAlgo == LATIN_HYPER_CUBE  ? d_nDivQRays : 0);  // only needed for LHC scheme

    for (CellIterator iter = finePatch->getCellIterator(); !iter.done(); iter++){

      IntVector origin = *iter;

      if (d_rayDirSampleAlgo == LATIN_HYPER_CUBE){
        randVector(rand_i, mTwister, origin);
      }
/*`==========TESTING==========*/
#if 0
      if( isDbgCell(origin) && d_isDbgOn ){
        dbg2.setActive(true);
      }else{
        dbg2.setActive(false);
      }

//      if( origin != d_dbgCell ){
//        return;
//     }
#endif
/*===========TESTING==========`*/

      double sumI = 0;

      //__________________________________
      //  ray loop
      for (int iRay=0; iRay < d_nDivQRays; iRay++){

        Vector direction_vector;
        if (d_rayDirSampleAlgo== LATIN_HYPER_CUBE){       // Latin-Hyper-Cube sampling
          direction_vector =findRayDirectionHyperCube(mTwister, origin, iRay,rand_i[iRay],iRay );
        }else{                                            // Naive Monte-Carlo sampling
          direction_vector =findRayDirection(mTwister, origin, iRay );
        }

        Vector rayOrigin;
        int my_L = maxLevels - 1;
        ray_Origin( mTwister, origin, DyDx[my_L],  DzDx[my_L], d_CCRays, rayOrigin);

        updateSumI_ML< T >( direction_vector, rayOrigin, origin, Dx, domain_BB, maxLevels, fineLevel, DyDx,DzDx,
                       fineLevel_ROI_Lo, fineLevel_ROI_Hi, regionLo, regionHi, sigmaT4OverPi, abskg, cellType,
                       nRaySteps, sumI, mTwister);


      }  // Ray loop

      //__________________________________
      //  Compute divQ
      divQ_fine[origin] = -4.0 * M_PI * abskg_fine[origin] * ( sigmaT4OverPi_fine[origin] - (sumI/d_nDivQRays) );

      // radiationVolq is the incident energy per cell (W/m^3) and is necessary when particle heat transfer models (i.e. Shaddix) are used
      radiationVolq_fine[origin] = 4.0 * M_PI * abskg_fine[origin] *  (sumI/d_nDivQRays) ;

/*`==========TESTING==========*/
#if DEBUG == 1
    if( isDbgCell(origin) ) {
          printf( "\n      [%d, %d, %d]  sumI: %g  divQ: %g radiationVolq: %g  abskg: %g,    sigmaT4: %g \n\n",
                    origin.x(), origin.y(), origin.z(), sumI,divQ_fine[origin], radiationVolq_fine[origin],abskg_fine[origin], sigmaT4OverPi_fine[origin]);
    }
#endif
/*===========TESTING==========`*/
    }  // end cell iterator

    //__________________________________
    //
    double end =clock();
    double efficiency = nRaySteps/((end-start)/ CLOCKS_PER_SEC);
    if (finePatch->getGridIndex() == levelPatchID) {
      cout<< endl;
      cout << " RMCRT REPORT: Patch " << levelPatchID <<endl;
      cout << " Used "<< (end-start) * 1000 / CLOCKS_PER_SEC<< " milliseconds of CPU time. \n" << endl;// Convert time to ms
      cout << " Size: " << nRaySteps << endl;
      cout << " Efficiency: " << efficiency << " steps per sec" << endl;
      cout << endl;
    }
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
  if( d_whichROI_algo == dynamic ){

    minvec_vartype lo;
    maxvec_vartype hi;
    new_dw->get( lo, d_ROI_LoCellLabel );
    new_dw->get( hi, d_ROI_HiCellLabel );
    fineLevel_ROI_Lo = roundNearest( Vector(lo) );
    fineLevel_ROI_Hi = roundNearest( Vector(hi) );

  } else if ( d_whichROI_algo == fixed ){

    fineLevel_ROI_Lo = fineLevel->getCellIndex( d_ROI_minPt );
    fineLevel_ROI_Hi = fineLevel->getCellIndex( d_ROI_maxPt );

    if( !fineLevel->containsCell( fineLevel_ROI_Lo ) ||
        !fineLevel->containsCell( fineLevel_ROI_Hi ) ){
      ostringstream warn;
      warn << "ERROR:  the fixed ROI extents " << d_ROI_minPt << " " << d_ROI_maxPt << " are not contained on the fine level."<< endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
  } else if ( d_whichROI_algo == patch_based ){

    IntVector patchLo = patch->getExtraCellLowIndex();
    IntVector patchHi = patch->getExtraCellHighIndex();

    fineLevel_ROI_Lo = patchLo - d_halo;
    fineLevel_ROI_Hi = patchHi + d_halo;
    dbg << "  L-"<< fineLevel->getIndex() <<"  patch: ("<<patch->getID() <<") " << patchLo << " " << patchHi << endl;

  }

  // region must be within a finest Level including extraCells.
  IntVector levelLo, levelHi;
  fineLevel->findCellIndexRange(levelLo, levelHi);

  fineLevel_ROI_Lo = Max(fineLevel_ROI_Lo, levelLo);
  fineLevel_ROI_Hi = Min(fineLevel_ROI_Hi, levelHi);
  dbg << "  fineLevel_ROI: " << fineLevel_ROI_Lo << " "<< fineLevel_ROI_Hi << endl;

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

      regionLo[L] = level->mapCellToCoarser(regionLo[L+1]) - d_halo;
      regionHi[L] = level->mapCellToCoarser(regionHi[L+1]) + d_halo;

      // region must be within a level
      IntVector levelLo, levelHi;
      level->findInteriorCellIndexRange(levelLo, levelHi);

      regionLo[L] = Max(regionLo[L], levelLo);
      regionHi[L] = Min(regionHi[L], levelHi);
    }
  }

  // debugging
  if(dbg2.active()){
    for(int L = 0; L<maxLevels; L++){
      //dbg2 << "L-"<< L << " regionLo " << regionLo[L] << " regionHi " << regionHi[L] << endl;
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
//  Compute the Ray location from a cell face
void Ray::rayLocation_cellFace( MTRand& mTwister,
                                const IntVector& origin,
                                const IntVector &indexOrder,
                                const IntVector &shift,
                                const double &DyDx,
                                const double &DzDx,
                                Vector& location)
{
  Vector tmp;
  tmp[0] =  mTwister.rand() ;
  tmp[1] =  0;
  tmp[2] =  mTwister.rand() ;

  // Put point on correct face
  location[0] = tmp[indexOrder[0]] + shift[0];
  location[1] = (tmp[indexOrder[1]] + shift[1]) * DyDx;
  location[2] = (tmp[indexOrder[2]] + shift[2]) * DzDx;

  location[0] += origin.x();
  location[1] += origin.y();
  location[2] += origin.z();
}

//______________________________________________________________________
//
bool Ray::has_a_boundary(const IntVector &c,
                         constCCVariable<int> &celltype,
                         vector<int> &boundaryFaces){

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
                                  const int radCalc_freq,
                                  const bool backoutTemp )
{

  std::string taskname = "Ray::setBoundaryConditions";

  Task* tsk = NULL;
  if( RMCRTCommon::d_FLT_DBL == TypeDescription::double_type ){

    tsk= scinew Task( taskname, this, &Ray::setBoundaryConditions< double >,
                      temp_dw, radCalc_freq, backoutTemp );
  } else {
    tsk= scinew Task( taskname, this, &Ray::setBoundaryConditions< float >,
                      temp_dw, radCalc_freq, backoutTemp );
  }

  printSchedule(level,dbg,taskname);

  if(!backoutTemp){
    tsk->requires( temp_dw, d_compTempLabel, Ghost::None,0 );
  }

  tsk->modifies( d_sigmaT4Label );
  tsk->modifies( d_abskgLabel );

  sched->addTask( tsk, level->eachPatch(), d_matlSet );
}
//---------------------------------------------------------------------------
template<class T>
void Ray::setBoundaryConditions( const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset*,
                                 DataWarehouse*,
                                 DataWarehouse* new_dw,
                                 Task::WhichDW temp_dw,
                                 const int radCalc_freq,
                                 const bool backoutTemp )
{
  // Only run if it's time
  if ( doCarryForward( radCalc_freq ) ) {
    return;
  }

  if ( d_onOff_SetBCs == false )
    return;

  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);

    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    if( bf.size() > 0){

      printTask(patches,patch,dbg,"Doing Ray::setBoundaryConditions");

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
        t_dw->get(varTmp, d_compTempLabel,   d_matl, patch, Ghost::None, 0);
        temp.copyData(varTmp);
      }


      //__________________________________
      // set the boundary conditions
      setBC< T, double >  (abskg,    d_abskgBC_tag,               patch, d_matl);
      setBC<double,double>(temp,     d_compTempLabel->getName(),  patch, d_matl);

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

  if( dbg_BC.active() ){
    const Level* level = patch->getLevel();
    dbg_BC << "setBC \t"<< desc <<" "
           << " mat_id = " << mat_id <<  ", Patch: "<< patch->getID() << " L-" <<level->getID()<< endl;
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
        if( dbg_BC.active() ) {
          bound_ptr.reset();
          dbg_BC <<"Face: "<< patch->getFaceName(face) <<" numCellsTouched " << nCells
             <<"\t child " << child  <<" NumChildren "<<numChildren
             <<"\t BC kind "<< bc_kind <<" \tBC value "<< bc_value
             <<"\t bound limits = "<< bound_ptr << std::endl;
        }
      }  // if iterator found
    }  // child loop

    if( dbg_BC.active() ){
      dbg_BC << "    "<< patch->getFaceName(face) << " \t " << bc_kind << " numChildren: " << numChildren
             << " nCellsTouched: " << nCells << endl;
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
void Ray::sched_Refine_Q(SchedulerP& sched,
                         const PatchSet* patches,
                         const MaterialSet* matls,
                         const int radCalc_freq)
{
  const Level* fineLevel = getLevel(patches);
  int L_indx = fineLevel->getIndex();

  if(L_indx > 0 ){
     printSchedule(patches,dbg,"Ray::scheduleRefine_Q (divQ)");

    Task* task = scinew Task("Ray::refine_Q",this,
                             &Ray::refine_Q,  radCalc_freq);

    Task::MaterialDomainSpec  ND  = Task::NormalDomain;
    #define allPatches 0
    #define allMatls 0
    task->requires(Task::NewDW, d_divQLabel, allPatches, Task::CoarseLevel, allMatls, ND, d_gac,1);

    // when carryforward is needed
    task->requires( Task::OldDW, d_divQLabel, d_gn, 0 );

    task->computes(d_divQLabel);
    sched->addTask(task, patches, matls);
  }
}

//______________________________________________________________________
//
void Ray::refine_Q(const ProcessorGroup*,
                   const PatchSubset* patches,
                   const MaterialSubset* matls,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw,
                   const int radCalc_freq)
{

  const Level* fineLevel = getLevel(patches);
  const Level* coarseLevel = fineLevel->getCoarserLevel().get_rep();

  //__________________________________
  //  Carry Forward (old_dw -> new_dw)
  if ( doCarryForward( radCalc_freq ) ) {
    printTask( fineLevel->getPatch(0), dbg, "Doing Ray::refine_Q carryForward ( divQ )" );

    new_dw->transferFrom( old_dw, d_divQLabel, patches, matls, true );
    return;
  }

  //__________________________________
  //
  for(int p=0;p<patches->size();p++){
    const Patch* finePatch = patches->get(p);
    printTask(patches, finePatch,dbg,"Doing refineQ");

    Level::selectType coarsePatches;
    finePatch->getCoarseLevelPatches(coarsePatches);

    CCVariable<double> divQ_fine;
    new_dw->allocateAndPut(divQ_fine, d_divQLabel, d_matl, finePatch);
    divQ_fine.initialize(0);

    IntVector refineRatio = fineLevel->getRefinementRatio();

    // region of fine space that will correspond to the coarse we need to get
    IntVector cl, ch, fl, fh;
    IntVector bl(0,0,0);  // boundary layer or padding
    int nghostCells = 1;
    bool returnExclusiveRange=true;

    getCoarseLevelRange(finePatch, coarseLevel, cl, ch, fl, fh, bl,
                        nghostCells, returnExclusiveRange);

    dbg <<" refineQ: "
        <<" finePatch  "<< finePatch->getID() << " fl " << fl << " fh " << fh
        <<" coarseRegion " << cl << " " << ch <<endl;

    constCCVariable<double> divQ_coarse;
    new_dw->getRegion( divQ_coarse, d_divQLabel, d_matl, coarseLevel, cl, ch );

    selectInterpolator(divQ_coarse, d_orderOfInterpolation, coarseLevel, fineLevel,
                       refineRatio, fl, fh, divQ_fine);

  }  // fine patch loop
}

//______________________________________________________________________
// This task computes the extents of the fine level region of interest
void Ray::sched_ROI_Extents ( const LevelP& level,
                              SchedulerP& scheduler )
{
  int maxLevels = level->getGrid()->numLevels() -1;
  int L_indx = level->getIndex();

  if( (L_indx != maxLevels ) || ( d_whichROI_algo != dynamic ) ){     // only schedule on the finest level and dynamic
    return;
  }

  printSchedule(level,dbg,"Ray::ROI_Extents");

  Task* tsk = NULL;
  if( RMCRTCommon::d_FLT_DBL == TypeDescription::double_type ){
    tsk= scinew Task( "Ray::ROI_Extents", this, &Ray::ROI_Extents< double >);
  } else {
    tsk= scinew Task( "Ray::ROI_Extents", this, &Ray::ROI_Extents< float >);
  }

  tsk->requires( Task::NewDW, d_abskgLabel,    d_gac, 1 );
  tsk->requires( Task::NewDW, d_sigmaT4Label,  d_gac, 1 );
  tsk->computes( d_mag_grad_abskgLabel );
  tsk->computes( d_mag_grad_sigmaT4Label );
  tsk->computes( d_flaggedCellsLabel );

  tsk->computes(d_ROI_LoCellLabel);
  tsk->computes(d_ROI_HiCellLabel);

  scheduler->addTask( tsk, level->eachPatch(), d_matlSet );
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
    printTask(patches, patch,dbg,"Doing ROI_Extents");

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
                            const bool modifiesd_sigmaT4,
                            const int radCalc_freq)
{
  if(coarseLevel->hasFinerLevel()){
    printSchedule(coarseLevel,dbg,"Ray::sched_CoarsenAll");
    sched_Coarsen_Q(coarseLevel, sched, Task::NewDW, modifies_abskg,     d_abskgLabel,   radCalc_freq );
    sched_Coarsen_Q(coarseLevel, sched, Task::NewDW, modifiesd_sigmaT4,  d_sigmaT4Label, radCalc_freq );
  }
}

//______________________________________________________________________
void Ray::sched_Coarsen_Q ( const LevelP& coarseLevel,
                            SchedulerP& sched,
                            Task::WhichDW this_dw,
                            const bool modifies,
                            const VarLabel* variable,
                            const int radCalc_freq)
{
  string taskname = "        Coarsen_Q_" + variable->getName();
  printSchedule(coarseLevel,dbg,taskname);

  const Uintah::TypeDescription* td = variable->typeDescription();
  const Uintah::TypeDescription::Type subtype = td->getSubType()->getType();

  Task* tsk = NULL;
  switch( subtype ) {
    case TypeDescription::double_type:
      tsk = scinew Task( taskname, this, &Ray::coarsen_Q< double >,
                         variable, modifies, this_dw, radCalc_freq );
      break;
    case TypeDescription::float_type:
      tsk = scinew Task( taskname, this, &Ray::coarsen_Q< float >,
                         variable, modifies, this_dw, radCalc_freq );
      break;
    default:
      throw InternalError("Ray::sched_Coarsen_Q: (CCVariable) invalid data type", __FILE__, __LINE__);
  }

  if(modifies){
    tsk->requires(this_dw, variable, 0, Task::FineLevel, 0, Task::NormalDomain, d_gn, 0);
    tsk->modifies(variable);
  }else{
    tsk->requires(this_dw, variable, 0, Task::FineLevel, 0, Task::NormalDomain, d_gn, 0);
    tsk->requires(Task::OldDW, variable, d_gn, 0);  // needed for carryForward
    tsk->computes(variable);
  }

  sched->addTask( tsk, coarseLevel->eachPatch(), d_matlSet );
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
                      Task::WhichDW which_dw,
                      const int radCalc_freq )
{
  if ( doCarryForward( radCalc_freq ) ) {
    bool replaceVar = true;
    new_dw->transferFrom( old_dw, variable, patches, matls, replaceVar );
    printTask(patches,dbg,"Doing Ray::coarsen_Q carryForward : " + variable->getName());
    return;
  }

  const Level* coarseLevel = getLevel(patches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();

  //__________________________________
  //
  for(int p=0;p<patches->size();p++){
    const Patch* coarsePatch = patches->get(p);

    printTask(patches, coarsePatch,dbg,"Doing coarsen: " + variable->getName());

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

      // coarsen
      bool computesAve = true;
      fineToCoarseOperator(Q_coarse,   computesAve,
                           variable,   matl, new_dw,
                           coarsePatch, coarseLevel, fineLevel);
    }
  }  // course patch loop
}

//______________________________________________________________________
//
void Ray::sched_computeCellType ( const LevelP& level,
                                  SchedulerP& sched,
                                  modifiesComputes which)
{
  Task* tsk = scinew Task( "Ray::computeCellType", this,
                           &Ray::computeCellType, which );

  if ( which == Ray::modifiesVar ){
    tsk->requires(Task::NewDW, d_cellTypeLabel, 0, Task::FineLevel, 0, Task::NormalDomain, d_gn, 0);
    tsk->modifies( d_cellTypeLabel );
  }else if ( which == Ray::computesVar ){
    tsk->computes( d_cellTypeLabel );
  }
  sched->addTask( tsk, level->eachPatch(), d_matlSet );
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

    printTask(patches, coarsePatch,dbg,"Doing computeCellType: " + d_cellTypeLabel->getName());

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
                           Vector& ray_location,
                           const IntVector& origin,
                           const vector<Vector>& Dx,
                           const BBox& domain_BB,
                           const int maxLevels,
                           const Level* fineLevel,
                           double DyDx[],
                           double DzDx[],
                           const IntVector& fineLevel_ROI_Lo,
                           const IntVector& fineLevel_ROI_Hi,
                           vector<IntVector>& regionLo,
                           vector<IntVector>& regionHi,
                           StaticArray< constCCVariable< T > >& sigmaT4OverPi,
                           StaticArray< constCCVariable< T > >& abskg,
                           StaticArray< constCCVariable< int > >& cellType,
                           unsigned long int& nRaySteps,
                           double& sumI,
                           MTRand& mTwister)
{


/*`==========TESTING==========*/
#if DEBUG == 1
  if( isDbgCell(origin) ) {
    printf("        A) updateSumI_ML: [%d,%d,%d] ray_dir [%g,%g,%g] ray_loc [%g,%g,%g]\n", origin.x(), origin.y(), origin.z(),ray_direction.x(), ray_direction.y(), ray_direction.z(), ray_location.x(), ray_location.y(), ray_location.z());
  }
#endif
/*===========TESTING==========`*/
  int L       = maxLevels -1;  // finest level
  int prevLev = L;

  IntVector cur      = origin;
  IntVector prevCell = cur;

  int step[3];                                           // Gives +1 or -1 based on sign
  bool sign[3];

  Vector inv_direction = Vector(1.0)/ray_direction;
  findStepSize( step, sign, inv_direction );

  //__________________________________
  // define tMax & tDelta on all levels
  // go from finest to coarset level so you can compare
  // with 1L rayTrace results.

  Vector tMax;
  tMax.x( (origin[0] + sign[0]           - ray_location[0]) * inv_direction[0] );
  tMax.y( (origin[1] + sign[1] * DyDx[L] - ray_location[1]) * inv_direction[1] );
  tMax.z( (origin[2] + sign[2] * DzDx[L] - ray_location[2]) * inv_direction[2] );

  vector<Vector> tDelta(maxLevels);
  for(int Lev = maxLevels-1; Lev>-1; Lev--){
    //Length of t to traverse one cell
    tDelta[Lev].x( fabs(inv_direction[0]) );
    tDelta[Lev].y( fabs(inv_direction[1]) * DyDx[Lev] );
    tDelta[Lev].z( fabs(inv_direction[2]) * DzDx[Lev] );
  }

  //Initializes the following values for each ray
  bool   in_domain      = true;
  double tMax_prev      = 0;
  double intensity      = 1.0;
  double fs             = 1.0;
  int    nReflect       = 0;             // Number of reflections
  bool   onFineLevel    = true;
  const  Level* level   = fineLevel;
  double optical_thickness     = 0;
  double expOpticalThick_prev  = 1.0;    // exp(-opticalThick_prev)

  //______________________________________________________________________
  //  Threshold  loop
  while (intensity > d_threshold){
    DIR dir = NONE;
    while (in_domain){

      prevCell = cur;
      prevLev  = L;

      double disMin = -9;   // Ray segment length.

      //__________________________________
      //  Determine the princple direction the ray is traveling
      //
      if ( tMax[0] < tMax[1] ){    // X < Y
        if ( tMax[0] < tMax[2] ){  // X < Z
          dir = X;
        } else {
          dir = Z;
        }
      } else {
        if(tMax[1] <tMax[2] ){     // Y < Z
          dir = Y;
        } else {
          dir = Z;
        }
      }

      // next cell index and position
      cur[dir]  = cur[dir] + step[dir];
      Vector dx_prev = Dx[L];           //  Used to compute coarsenRatio  FIX ME

      //__________________________________
      // Logic for moving between levels
      // - Currently you can only move from fine to coarse level
      // - Don't jump levels if ray is at edge of domain

      Point pos = level->getCellPosition(cur);           // position could be outside of domain
      in_domain = domain_BB.inside(pos);
      //in_domain = (cellType[L][cur] == d_flowCell);    // use this when direct comparison with 1L resullts

      bool ray_outside_ROI    = ( containsCell( fineLevel_ROI_Lo, fineLevel_ROI_Hi, cur, dir ) == false );
      bool ray_outside_Region = ( containsCell( regionLo[L], regionHi[L], cur, dir ) == false );

      bool jumpFinetoCoarserLevel   = ( onFineLevel &&  ray_outside_ROI && in_domain );
      bool jumpCoarsetoCoarserLevel = ( (onFineLevel == false) && ray_outside_Region && (L > 0) && in_domain );

#if (DEBUG == 1 || DEBUG == 4)
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

#if (DEBUG == 1 || DEBUG == 4)
        if( isDbgCell(origin) ) {
          printf( "        ** Jumping off fine patch switching Levels:  prev L: %i, L: %i, cur: [%i,%i,%i] \n",prevLev, L, cur.x(), cur.y(), cur.z());
        }
#endif
      } else if ( jumpCoarsetoCoarserLevel ){

        IntVector c_old = cur;                          // needed for debugging
        cur   = level->mapCellToCoarser(cur);
        level = level->getCoarserLevel().get_rep();
        L     = level->getIndex();
#if (DEBUG == 1 || DEBUG == 4)
        if( isDbgCell(origin) ) {
          printf( "        ** Switching Levels:  prev L: %i, L: %i, cur: [%i,%i,%i], c_old: [%i,%i,%i]\n",prevLev, L, cur.x(), cur.y(), cur.z(), c_old.x(), c_old.y(), c_old.z());
        }
#endif
      }



      //__________________________________
      //  update marching variables
      disMin        = (tMax[dir] - tMax_prev);        // Todd:   replace tMax[dir]
      tMax_prev     = tMax[dir];
      tMax[dir]     = tMax[dir] + tDelta[L][dir];

      ray_location[0] = ray_location[0] + (disMin  * ray_direction[0]);
      ray_location[1] = ray_location[1] + (disMin  * ray_direction[1]);
      ray_location[2] = ray_location[2] + (disMin  * ray_direction[2]);

      //__________________________________
      // Account for uniqueness of first step after reaching a new level
      Vector dx = level->dCell();
      IntVector coarsenRatio = IntVector(1,1,1);

      coarsenRatio[0] = dx[0]/dx_prev[0];
      coarsenRatio[1] = dx[1]/dx_prev[1];
      coarsenRatio[2] = dx[2]/dx_prev[2];

      Vector lineup;
      for (int ii=0; ii<3; ii++){
        if (sign[ii]) {
          lineup[ii] = -(cur[ii] % coarsenRatio[ii] - (coarsenRatio[ii] - 1 ));
        }
        else {
          lineup[ii] = cur[ii] % coarsenRatio[ii];
        }
      }

      tMax += lineup * tDelta[prevLev];

 /*`==========TESTING==========*/
#if DEBUG == 1
  if( isDbgCell( origin) ){
    printf( "        B) cur [%d,%d,%d] prev [%d,%d,%d]", cur.x(), cur.y(), cur.z(), prevCell.x(), prevCell.y(), prevCell.z());
    printf( " dir %d ", dir );
    printf( " stepSize [%i,%i,%i] ",step[0],step[1],step[2]);
    printf( " tMax [%g,%g,%g] ",tMax.x(),tMax.y(), tMax.z());
    printf( "rayLoc [%g,%g,%g] ", ray_location.x(),ray_location.y(), ray_location.z());
    printf( "inv_dir [%g,%g,%g] ",inv_direction.x(),inv_direction.y(), inv_direction.z());
    printf( "disMin %g inDomain %i\n",disMin, in_domain );

    printf( "            abskg[prev] %g  \t sigmaT4OverPi[prev]: %g \n",abskg[prevLev][prevCell],  sigmaT4OverPi[prevLev][prevCell]);
    printf( "            abskg[cur]  %g  \t sigmaT4OverPi[cur]:  %g  \t  cellType: %i \n",abskg[L][cur], sigmaT4OverPi[L][cur], cellType[L][cur]);
    printf( "            Dx[prevLev].x  %g \n",  Dx[prevLev].x() );
  }
#endif
/*===========TESTING==========`*/

      optical_thickness += Dx[prevLev].x() * abskg[prevLev][prevCell]*disMin;
      nRaySteps++;

/*`==========TESTING==========*/
#ifdef FAST_EXP
      double expOpticalThick = fast_exp(-optical_thickness); 
#else
      double expOpticalThick = exp(-optical_thickness);
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
    if(!d_allowReflect) intensity = 0;


/*`==========TESTING==========*/
#if DEBUG == 1
  if( isDbgCell(origin) ){
    printf( "        C) intensity: %g OptThick: %g, fs: %g allowReflect: %i\n", intensity, optical_thickness, fs, d_allowReflect );

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
  std::string taskname = "Ray::filter";
  Task* tsk= scinew Task( taskname, this, &Ray::filter, which_divQ_dw, includeEC, modifies_divQFilt );

  printSchedule(level,dbg,taskname);

  tsk->requires( which_divQ_dw, d_divQLabel,      d_gn, 0 );
  tsk->requires( which_divQ_dw, d_boundFluxLabel, d_gn, 0 );
  tsk->computes(                d_divQFiltLabel);
  tsk->computes(                d_boundFluxFiltLabel);

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
    printTask(patches,patch,dbg,"Doing Ray::filt");

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
                                                    const int ,
                                                    const bool );

template void Ray::setBoundaryConditions< float >( const ProcessorGroup*,
                                                   const PatchSubset* ,
                                                   const MaterialSubset*,
                                                   DataWarehouse*,
                                                   DataWarehouse* ,
                                                   Task::WhichDW ,
                                                   const int ,
                                                   const bool );

template void  Ray::updateSumI_ML< double> ( Vector&,
                                             Vector&,
                                             const IntVector&,
                                             const vector<Vector>&,
                                             const BBox&,
                                             const int,
                                             const Level* ,
                                             double DyDx[],
                                             double DzDx[],
                                             const IntVector&,
                                             const IntVector&,
                                             vector<IntVector>&,
                                             vector<IntVector>&,
                                             StaticArray< constCCVariable< double > >& sigmaT4OverPi,
                                             StaticArray< constCCVariable<double> >& abskg,
                                             StaticArray< constCCVariable< int > >& cellType,
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
                                             double DyDx[],
                                             double DzDx[],
                                             const IntVector&,
                                             const IntVector&,
                                             vector<IntVector>&,
                                             vector<IntVector>&,
                                             StaticArray< constCCVariable< float > >& sigmaT4OverPi,
                                             StaticArray< constCCVariable< float > >& abskg,
                                             StaticArray< constCCVariable< int > >& cellType,
                                             unsigned long int& ,
                                             double& ,
                                             MTRand&);
