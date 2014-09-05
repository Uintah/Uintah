/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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
#include <CCA/Components/Models/Radiation/RMCRT/MersenneTwister.h>
#include <CCA/Components/Models/Radiation/RMCRT/Ray.h>
#include <CCA/Components/Regridder/PerPatchVars.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/BBox.h>
#include <Core/Grid/AMR.h>
#include <Core/Grid/AMR_CoarsenRefine.h>
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <time.h>
#include <fstream>
#include <include/sci_defs/uintah_testdefs.h.in>
#include <CCA/Components/Arches/BoundaryCondition.h>


//--------------------------------------------------------------
//
using namespace Uintah;
using namespace std;
static DebugStream dbg("RAY",       false);
static DebugStream dbg2("RAY_DEBUG",false);
static DebugStream dbg_BC("RAY_BC", false);


//______________________________________________________________________
//
//______________________________________________________________________
void Ray::constructor(){
  _pi = acos(-1); 

  d_sigmaT4_label        = VarLabel::create( "sigmaT4",          CCVariable<double>::getTypeDescription() );
  d_mag_grad_abskgLabel  = VarLabel::create( "mag_grad_abskg",   CCVariable<double>::getTypeDescription() );
  d_mag_grad_sigmaT4Label= VarLabel::create( "mag_grad_sigmaT4", CCVariable<double>::getTypeDescription() );
  d_flaggedCellsLabel    = VarLabel::create( "flaggedCells",     CCVariable<int>::getTypeDescription() );
  d_ROI_LoCellLabel      = VarLabel::create( "ROI_loCell",       minvec_vartype::getTypeDescription() );
  d_ROI_HiCellLabel      = VarLabel::create( "ROI_hiCell",       maxvec_vartype::getTypeDescription() );
  d_VRFluxLabel          = VarLabel::create( "VRFlux",           CCVariable<double>::getTypeDescription() );
  d_boundFluxLabel       = VarLabel::create( "boundFlux",        CCVariable<Stencil7>::getTypeDescription() );
  d_boundFluxFiltLabel   = VarLabel::create( "boundFluxFilt",    CCVariable<Stencil7>::getTypeDescription() );
  d_divQFiltLabel        = VarLabel::create( "divQFilt",         CCVariable<double>::getTypeDescription() );
  d_cellTypeLabel        = VarLabel::create( "cellType",         CCVariable<int>::getTypeDescription() );
  d_radiationVolqLabel   = VarLabel::create( "radiationVolq",    CCVariable<double>::getTypeDescription() );
   
  d_matlSet       = 0;
  _isDbgOn        = dbg2.active();
  
  d_gac           = Ghost::AroundCells;
  d_gn            = Ghost::None;
  d_orderOfInterpolation = -9;
  _onOff_SetBCs   = true;
}



//---------------------------------------------------------------------------
// Class: Constructor. 
//---------------------------------------------------------------------------
Ray::Ray()
{
  constructor();  // put everything that is common to cpu & gpu inside the function above
}

//---------------------------------------------------------------------------
// Method: Constructor for GPU Version.
//---------------------------------------------------------------------------
#ifdef HAVE_CUDA
Ray::Ray(UnifiedScheduler* scheduler)
{
  _scheduler = scheduler;
  constructor();
}
#endif

//---------------------------------------------------------------------------
// Method: Destructor
//---------------------------------------------------------------------------
Ray::~Ray()
{
  VarLabel::destroy( d_sigmaT4_label );
  VarLabel::destroy( d_mag_grad_abskgLabel );
  VarLabel::destroy( d_mag_grad_sigmaT4Label );
  VarLabel::destroy( d_flaggedCellsLabel );
  VarLabel::destroy( d_ROI_LoCellLabel );
  VarLabel::destroy( d_ROI_HiCellLabel );
  VarLabel::destroy( d_VRFluxLabel );
  VarLabel::destroy( d_boundFluxLabel );
  VarLabel::destroy( d_divQFiltLabel );
  VarLabel::destroy( d_boundFluxFiltLabel );
  VarLabel::destroy( d_cellTypeLabel );
  VarLabel::destroy( d_radiationVolqLabel );

  if(d_matlSet && d_matlSet->removeReference()) {
    delete d_matlSet;
  }
}

//______________________________________________________________________
//  Logic for determing when to carry forward
bool doCarryForward( const int timestep,
                     const int radCalc_freq){
  bool test = (timestep%radCalc_freq != 0 && timestep != 1);
  return test;
}

//---------------------------------------------------------------------------
// Method: Problem setup (access to input file information)
//---------------------------------------------------------------------------
void
Ray::problemSetup( const ProblemSpecP& prob_spec,
                   const ProblemSpecP& rmcrtps,
                   SimulationStateP&   sharedState) 
{

  d_sharedState = sharedState;
  ProblemSpecP rmcrt_ps = rmcrtps;
  rmcrt_ps->getWithDefault( "NoOfRays"  ,       _NoOfRays  ,      10 );
  rmcrt_ps->getWithDefault( "Threshold" ,       _Threshold ,      0.01 );       // When to terminate a ray
  rmcrt_ps->getWithDefault( "randomSeed",       _isSeedRandom,    true );       // random or deterministic seed.
  rmcrt_ps->getWithDefault( "benchmark" ,       _benchmark,       0 );  
  rmcrt_ps->getWithDefault( "StefanBoltzmann",  _sigma,           5.67051e-8);  // Units are W/(m^2-K)
  rmcrt_ps->getWithDefault( "solveBoundaryFlux" , _solveBoundaryFlux, false );
  rmcrt_ps->getWithDefault( "CCRays"    ,       _CCRays,          false );      // if true, forces rays to always have CC origins
  rmcrt_ps->getWithDefault( "VirtRadiometer" ,  _virtRad,         false );             // if true, at least one virtual radiometer exists
  rmcrt_ps->getWithDefault( "VRViewAngle"    ,  _viewAng,         180 );               // view angle of the radiometer in degrees
  rmcrt_ps->getWithDefault( "VROrientation"  ,  _orient,          Vector(0,0,1) );     // Normal vector of the radiometer orientation (Cartesian)
  rmcrt_ps->getWithDefault( "VRLocationsMin" ,  _VRLocationsMin,  IntVector(0,0,0) );  // minimum extent of the string or block of virtual radiometers
  rmcrt_ps->getWithDefault( "VRLocationsMax" ,  _VRLocationsMax,  IntVector(0,0,0) );  // maximum extent of the string or block or virtual radiometers
  rmcrt_ps->getWithDefault( "nRadRays"  ,      _nRadRays  ,      1000 );
  rmcrt_ps->getWithDefault( "nFluxRays" ,       _nFluxRays,     500 );                 // number of rays per cell for computation of boundary fluxes
  rmcrt_ps->getWithDefault( "sigmaScat"  ,      _sigmaScat  ,      0 );                // scattering coefficient
  rmcrt_ps->getWithDefault( "abskgBench4"  ,    _abskgBench4,      1 );                // absorption coefficient specific to Bench4
  rmcrt_ps->get(             "shouldSetBCs" ,   _onOff_SetBCs );                       // ignore applying boundary conditions
  rmcrt_ps->getWithDefault( "allowReflect"   ,  _allowReflect,     true );             // Allow for ray reflections. Make false for DOM comparisons.
  rmcrt_ps->getWithDefault( "solveDivQ"      ,  _solveDivQ,        true );             // Allow for solving of divQ for flow cells.
  rmcrt_ps->getWithDefault( "applyFilter"    ,  _applyFilter,      false );            // Allow filtering of boundFlux and divQ.



  //__________________________________
  //  Warnings and bulletproofing

#ifndef RAY_SCATTER
  proc0cout<< "sigmaScat: " << _sigmaScat << endl;
  if(_sigmaScat>0){
    ostringstream warn;
    warn << "ERROR:  In order to run a scattering case, you must use the following in your configure line..." << endl;
    warn << "--enable-ray-scatter" << endl;
    warn << "If you wish to run a scattering case, please modify your configure line and re-configure and re-compile." << endl;
    warn << "If you wish to run a non-scattering case, please remove the line containing <sigmaScat> from your input file." << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
#endif

#ifdef RAY_SCATTER
  if(_sigmaScat<1e-99){
    ostringstream warn;
    warn << "WARNING:  You are running a non-scattering case, yet you have the following in your configure line..." << endl;
    warn << "--enable-ray-scatter" << endl;
    warn << "As such, this task will run slower than is necessary." << endl;
    warn << "If you wish to run a scattering case, please specify a positive value greater than 1e-99 for the scattering coefficient." << endl;
    warn << "If you wish to run a non-scattering case, please remove --enable-ray-scatter from your configure line and re-configure and re-compile" << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
#endif

#ifdef RAY_SCATTER
cout<< endl << "RAY_SCATTER IS DEFINED" << endl; 
#endif



  if (_benchmark > 5 || _benchmark < 0  ){
    ostringstream warn;
    warn << "ERROR:  Benchmark value ("<< _benchmark <<") not set correctly." << endl;
    warn << "Specify a value of 1 through 5 to run a benchmark case, or 0 otherwise." << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

  if ( _viewAng > 360 ){
    ostringstream warn;
    warn << "ERROR:  VRViewAngle ("<< _viewAng <<") exceeds the maximum acceptable value of 360 degrees." << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

  if (_virtRad && _nRadRays < int(15 + pow(5.4, _viewAng/40) ) ){
    ostringstream warn;
    warn << "Number of rays:  ("<< _nRadRays <<") is less than the recommended number of ("<< int(15 + pow(5.4, _viewAng/40) ) <<"). Errors will exceed 1%. " << endl;
  } 

  //__________________________________
  //  Read in the algorithm section
  ProblemSpecP alg_ps = rmcrt_ps->findBlock("algorithm");
  if (alg_ps){
  
    string type="NULL";

    if( !alg_ps->getAttribute("type", type) ){
      throw ProblemSetupException("RMCRT: No type specified for algorithm.  Please choose dataOnion on RMCRT_coarseLevel", __FILE__, __LINE__);
    }
  
    //__________________________________
    //  Data Onion
    if (type == "dataOnion" ) {

      alg_ps->getWithDefault( "halo",  _halo,  IntVector(10,10,10));
      
      //  Method for deteriming the extents of the ROI
      ProblemSpecP ROI_ps = alg_ps->findBlock("ROI_extents");
      ROI_ps->getAttribute("type", type);

      if(type == "fixed" ) {
        
        _whichROI_algo = fixed;
        ROI_ps->get("min", _ROI_minPt );
        ROI_ps->get("max", _ROI_maxPt );
        
      } else if ( type == "dynamic" ) {
        
        _whichROI_algo = dynamic;
        ROI_ps->getWithDefault( "abskg_threshold",   _abskg_thld,   DBL_MAX);
        ROI_ps->getWithDefault( "sigmaT4_threshold", _sigmaT4_thld, DBL_MAX);
        
      } else if ( type == "patch_based" ){
        _whichROI_algo = patch_based;
      }
      
    //__________________________________
    //  rmcrt only on the coarse level  
    } else if ( type == "RMCRT_coarseLevel" ) {
      alg_ps->require( "orderOfInterpolation", d_orderOfInterpolation);
    }
  }

  _sigma_over_pi = _sigma/_pi;

  //__________________________________
  // BC bulletproofing  
  bool ignore_BC_bulletproofing  = false;
  rmcrt_ps->get( "ignore_BC_bulletproofing",  ignore_BC_bulletproofing );
  
  ProblemSpecP root_ps = rmcrt_ps->getRootNode();
  const MaterialSubset* mss = d_matlSet->getUnion();
  
  if( ignore_BC_bulletproofing == true || _onOff_SetBCs == false) {
    proc0cout << "\n\n______________________________________________________________________" << endl;
    proc0cout << "  WARNING: bulletproofing of the boundary conditions specs is off!";
    proc0cout << "   You're free to set any BC you'd like " << endl;
    proc0cout << "______________________________________________________________________\n\n" << endl;
  
  } else {  
    is_BC_specified(root_ps, d_temperatureLabel->getName(), mss);
    is_BC_specified(root_ps, d_abskgLabel->getName(),       mss);
  }
}

//______________________________________________________________________
// Register the material index and label names
void
Ray::registerVarLabels(int   matlIndex,
                       const VarLabel* abskg,
                       const VarLabel* absorp,
                       const VarLabel* temperature,
                       const VarLabel* celltype, 
                       const VarLabel* divQ )
{
  d_matl             = matlIndex;
  d_abskgLabel       = abskg;
  d_absorpLabel      = absorp;
  d_temperatureLabel = temperature;
  d_cellTypeLabel    = celltype; 
  d_divQLabel        = divQ;

  //__________________________________
  //  define the materialSet
  d_matlSet = scinew MaterialSet();
  vector<int> m;
  m.push_back(matlIndex);
  d_matlSet->addAll(m);
  d_matlSet->addReference();
}
//---------------------------------------------------------------------------
//
void 
Ray::sched_initProperties( const LevelP& level, 
                           SchedulerP& sched,
                           const int radCalc_freq )
{

  if(_benchmark != 0){
    Task* tsk = scinew Task( "Ray::initProperties", this, 
                             &Ray::initProperties, radCalc_freq);
                              
    printSchedule(level,dbg,"Ray::initProperties");

    tsk->modifies( d_temperatureLabel );
    tsk->modifies( d_abskgLabel );
    tsk->modifies( d_cellTypeLabel );


    sched->addTask( tsk, level->eachPatch(), d_matlSet ); 
  }
}
//______________________________________________________________________
//
void
Ray::initProperties( const ProcessorGroup* pc,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw,
                     const int radCalc_freq )
{

  // Only run if it's time
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();
  if ( doCarryForward( timestep, radCalc_freq) ) {
    return;
  }
  
  const Level* level = getLevel(patches);

  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing Ray::InitProperties");

    CCVariable<double> abskg; 
    CCVariable<double> absorp; 
    CCVariable<double> celltype;

    new_dw->getModifiable( abskg,    d_abskgLabel,     d_matl, patch );  
    abskg.initialize  ( 0.0 ); 



    IntVector pLow;
    IntVector pHigh;
    level->findInteriorCellIndexRange(pLow, pHigh);

    int Nx = pHigh[0] - pLow[0];
    int Ny = pHigh[1] - pLow[1];
    int Nz = pHigh[2] - pLow[2];

    Vector Dx = patch->dCell(); 
    
    BBox L_BB;
    level->getInteriorSpatialRange(L_BB);                 // edge of computational domain
    Vector L_length = Abs(L_BB.max() - L_BB.min());
    
    //__________________________________
    //  Benchmark initializations
    if ( _benchmark == 1 || _benchmark == 3 ) {
  
      // bulletproofing
      Vector valid_length(1,1,1);
      if (L_length != valid_length){
        ostringstream msg;
        msg << "\n RMCRT:ERROR: the benchmark problem selected is only valid on the domain \n";
        msg << valid_length << ".  Your domain is " << L_BB << endl; 
        throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
      }
    
      for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ){
        IntVector c = *iter;
        abskg[c] = 0.90 * ( 1.0 - 2.0 * fabs( ( c[0] - (Nx - 1.0) /2.0) * Dx[0]) )
                        * ( 1.0 - 2.0 * fabs( ( c[1] - (Ny - 1.0) /2.0) * Dx[1]) )
                        * ( 1.0 - 2.0 * fabs( ( c[2] - (Nz - 1.0) /2.0) * Dx[2]) ) 
                        + 0.1;                  
      }     
    } else if (_benchmark == 2) {
      
      for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ){ 
        IntVector c = *iter;
        abskg[c] = 1;
      }
    }
    
    if(_benchmark == 3) {
      CCVariable<double> temp;
      new_dw->getModifiable(temp, d_temperatureLabel, d_matl, patch);
      
      for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ){ 
        IntVector c = *iter; 
        temp[c] = 1000 * abskg[c];

      }
    }

    if(_benchmark == 4 || _benchmark == 5) {  // Siegel isotropic scattering
      for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ){
        IntVector c = *iter;
        abskg[c] = _abskgBench4;
      }
    }

    if(_benchmark == 5 ) {  // Siegel isotropic scattering for specific abskg and sigma_scat
      for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ){
        IntVector c = *iter;
        abskg[c] = 2;
        _sigmaScat = 8;
      }
    }
  }
}

//---------------------------------------------------------------------------
// 
//---------------------------------------------------------------------------
void
Ray::sched_sigmaT4( const LevelP& level, 
                    SchedulerP& sched,
                    Task::WhichDW temp_dw,
                    const int radCalc_freq,
                    const bool includeEC )
{
  std::string taskname = "Ray::sigmaT4";
  Task* tsk= scinew Task( taskname, this, &Ray::sigmaT4, temp_dw, radCalc_freq, includeEC );

  printSchedule(level,dbg,taskname);
  
  tsk->requires( temp_dw, d_temperatureLabel,  d_gn, 0 );
  tsk->requires( Task::OldDW, d_sigmaT4_label, d_gn, 0 ); 
  tsk->computes(d_sigmaT4_label); 

  sched->addTask( tsk, level->eachPatch(), d_matlSet );
}
//---------------------------------------------------------------------------
// Compute total intensity over all wave lengths (sigma * Temperature^4/pi)
//---------------------------------------------------------------------------
void
Ray::sigmaT4( const ProcessorGroup*,
              const PatchSubset* patches,           
              const MaterialSubset* matls,                
              DataWarehouse* old_dw, 
              DataWarehouse* new_dw,
              Task::WhichDW which_temp_dw,
              const int radCalc_freq,
              const bool includeEC )               
{
  //__________________________________
  //  Carry Forward
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();
  if ( doCarryForward( timestep, radCalc_freq) ) {
    printTask( patches, patches->get(0), dbg, "Doing Ray::sigmaT4 carryForward (sigmaT4)" );
    
    new_dw->transferFrom( old_dw, d_sigmaT4_label, patches, matls );
    return;
  }
  
  //__________________________________
  //  do the work
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing Ray::sigmaT4");

    double sigma_over_pi = _sigma/M_PI;

    constCCVariable<double> temp;
    CCVariable<double> sigmaT4;             // sigma T ^4/pi

    DataWarehouse* temp_dw = new_dw->getOtherDataWarehouse(which_temp_dw);
    temp_dw->get(temp,              d_temperatureLabel,   d_matl, patch, Ghost::None, 0);
    new_dw->allocateAndPut(sigmaT4, d_sigmaT4_label,      d_matl, patch);
    
    // set the cell iterator
    CellIterator iter = patch->getCellIterator();
    if(includeEC){
      iter = patch->getExtraCellIterator();
    }

    for (;!iter.done();iter++){
      const IntVector& c = *iter;
      double T_sqrd = temp[c] * temp[c];
      sigmaT4[c] = sigma_over_pi * T_sqrd * T_sqrd;
    }
  }
}

//---------------------------------------------------------------------------
// Method: Schedule the ray tracer
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
#ifdef HAVE_CUDA
  std::string gputaskname = "Ray::rayTraceGPU";
  Task* tsk = scinew Task( &Ray::rayTraceGPU, gputaskname, taskname, this,
                           &Ray::rayTrace, modifies_divQ, abskg_dw, sigma_dw, celltype_dw, radCalc_freq );
#else
  Task* tsk= scinew Task( taskname, this, &Ray::rayTrace,
                         modifies_divQ, abskg_dw, sigma_dw, celltype_dw, radCalc_freq );
#endif

  printSchedule(level,dbg,taskname);

  // require an infinite number of ghost cells so  you can access the entire domain.
  Ghost::GhostType  gac  = Ghost::AroundCells;
  tsk->requires( abskg_dw ,    d_abskgLabel  ,   gac, SHRT_MAX);
  tsk->requires( sigma_dw ,    d_sigmaT4_label,  gac, SHRT_MAX);
  
  // when carryforward is needed
  tsk->requires( Task::OldDW, d_divQLabel,           d_gn, 0 );
  tsk->requires( Task::OldDW, d_VRFluxLabel,         d_gn, 0 );
  tsk->requires( Task::OldDW, d_boundFluxLabel,      d_gn, 0 ); 
  tsk->requires( Task::OldDW, d_radiationVolqLabel,  d_gn, 0 );
  
  if (!tsk->usesDevice()) {
    tsk->requires( celltype_dw , d_cellTypeLabel , gac, SHRT_MAX);
  }
  if( modifies_divQ ){
    tsk->modifies( d_divQLabel ); 
    if (!tsk->usesDevice()) {
      tsk->modifies( d_VRFluxLabel );
      tsk->modifies( d_boundFluxLabel );
      tsk->modifies( d_radiationVolqLabel );
    }
  } else {
    tsk->computes( d_divQLabel );
    if (!tsk->usesDevice()) {
      tsk->computes( d_VRFluxLabel );
      tsk->computes( d_boundFluxLabel );
      tsk->computes( d_radiationVolqLabel );
    }
  }
  sched->addTask( tsk, level->eachPatch(), d_matlSet );
  
}

//---------------------------------------------------------------------------
// Method: The actual work of the ray tracer
//---------------------------------------------------------------------------
void
Ray::rayTrace( const ProcessorGroup* pc,
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
   //__________________________________
  //  Carry Forward (old_dw -> new_dw)
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();
  if ( doCarryForward( timestep, radCalc_freq) ) {
    printTask( level->getPatch(0), dbg, "Doing Ray::rayTrace carryForward (divQ, VRFlux, boundFlux, radiationVolq )" );
    
    new_dw->transferFrom( old_dw, d_divQLabel,          patches, matls );
    new_dw->transferFrom( old_dw, d_VRFluxLabel,        patches, matls );
    new_dw->transferFrom( old_dw, d_boundFluxLabel,     patches, matls );
    new_dw->transferFrom( old_dw, d_radiationVolqLabel, patches, matls );
    return;
  }
  
  //__________________________________
  //
  MTRand _mTwister;
  
  // Determine the size of the domain.
  IntVector domainLo, domainHi;
  IntVector domainLo_EC, domainHi_EC;
  
  level->findInteriorCellIndexRange(domainLo, domainHi);     // excluding extraCells
  level->findCellIndexRange(domainLo_EC, domainHi_EC);       // including extraCells
  
  DataWarehouse* abskg_dw    = new_dw->getOtherDataWarehouse(which_abskg_dw);
  DataWarehouse* sigmaT4_dw  = new_dw->getOtherDataWarehouse(which_sigmaT4_dw);
  DataWarehouse* celltype_dw = new_dw->getOtherDataWarehouse(which_celltype_dw);


  constCCVariable<double> sigmaT4OverPi;
  constCCVariable<double> abskg;
  constCCVariable<int>    celltype;

  abskg_dw->getRegion(   abskg   ,       d_abskgLabel ,   d_matl , level, domainLo_EC, domainHi_EC);
  sigmaT4_dw->getRegion( sigmaT4OverPi , d_sigmaT4_label, d_matl , level, domainLo_EC, domainHi_EC);
  celltype_dw->getRegion( celltype ,     d_cellTypeLabel, d_matl , level, domainLo_EC, domainHi_EC);
  
  double start=clock();

  // patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing Ray::rayTrace");

    CCVariable<double> divQ;
    CCVariable<double> VRFlux;
    CCVariable<Stencil7> boundFlux;
    CCVariable<double> radiationVolq;

    if( modifies_divQ ){
      old_dw->getModifiable( divQ,         d_divQLabel,          d_matl, patch );
      old_dw->getModifiable( VRFlux,       d_VRFluxLabel,        d_matl, patch );
      old_dw->getModifiable( boundFlux,    d_boundFluxLabel,     d_matl, patch );
      old_dw->getModifiable( radiationVolq,d_radiationVolqLabel, d_matl, patch );
    }else{
      new_dw->allocateAndPut( divQ,      d_divQLabel,      d_matl, patch );
      divQ.initialize( 0.0 ); 
      new_dw->allocateAndPut( VRFlux,    d_VRFluxLabel,    d_matl, patch );
      VRFlux.initialize( 0.0 );
      new_dw->allocateAndPut( boundFlux,    d_boundFluxLabel, d_matl, patch );
      new_dw->allocateAndPut( radiationVolq, d_radiationVolqLabel, d_matl, patch );
      radiationVolq.initialize( 0.0 );
      
      for (CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
        IntVector origin = *iter;

        boundFlux[origin].p = 0.0;
        boundFlux[origin].w = 0.0;
        boundFlux[origin].e = 0.0;
        boundFlux[origin].s = 0.0;
        boundFlux[origin].n = 0.0;
        boundFlux[origin].b = 0.0;
        boundFlux[origin].t = 0.0;
      }
   }
    unsigned long int size = 0;                        // current size of PathIndex
    Vector Dx = patch->dCell();                        // cell spacing
 
    //______________________________________________________________________
    //           R A D I O M E T E R
    //______________________________________________________________________
    if (_virtRad){

      for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){ 
        
        IntVector origin = *iter; 
        int i = origin.x();
        int j = origin.y();
        int k = origin.z();
        // loop over the VR extents
        if ( i >= _VRLocationsMin.x() && i <= _VRLocationsMax.x() &&
             j >= _VRLocationsMin.y() && j <= _VRLocationsMax.y() &&
             k >= _VRLocationsMin.z() && k <= _VRLocationsMax.z() ){
 
          double sumI      = 0;
          double sumProjI  = 0; // for virtual radiometer
          double sumI_prev = 0; // used for VR
          double sldAngl   = 0; // solid angle of VR
          double VRTheta   = 0; // the polar angle of each ray from the radiometer normal
          // ray loop
          for (int iRay=0; iRay < _nRadRays; iRay++){
            
            if(_isSeedRandom == false){
              _mTwister.seed((i + j +k) * iRay +1);
            }

            Vector direction_vector;
            Vector inv_direction_vector;

            double DyDxRatio = Dx.y() / Dx.x(); //noncubic
            double DzDxRatio = Dx.z() / Dx.x(); //noncubic
            
            Vector ray_location;
            ray_location[0] =   i +  0.5 ;
            ray_location[1] =   j +  0.5 * DyDxRatio ; //noncubic
            ray_location[2] =   k +  0.5 * DzDxRatio ; //noncubic

            double deltaTheta = _viewAng/360*_pi;//divides view angle by two and converts to radians

            //_orient[0,1,2] represent the user specified vector normal of the radiometer.
            // These will be converted to rotations about the x,y, and z axes, respectively.
            //Each rotation is counterclockwise when the observer is looking from the
            //positive axis about which the rotation is occurring.

            // Avoid division by zero by re-assigning orientations of 0

            if (_orient[0] == 0)
              _orient[0] = 1e-16;
            if (_orient[1] == 0)
              _orient[1] = 1e-16;
            if (_orient[2] == 0)
              _orient[2] = 1e-16;

            //  In spherical coordinates, the polar angle, theta_rot,
            //  represents the counterclockwise rotation about the y axis,
            //  The azimuthal angle represents the negative of the
            //  counterclockwise rotation about the z axis.
            // Convert the user specified radiometer vector normal into three axial
            // rotations about the x,y, and z axes.
            double thetaRot = acos(_orient[2]/sqrt(_orient[0]*_orient[0]+_orient[1]*_orient[1] +_orient[2]*_orient[2]));
            double psiRot   = acos(_orient[0]/sqrt(_orient[0]*_orient[0]+_orient[1]*_orient[1]));
            const double phiRot = 0;
            //  phiRot is alsays  0. There will never be a need for a rotation about the x axis.  All
            // possible rotations can be accomplished using the other two.
           
            //  The calculated rotations must be adjusted if the x and y components of the normal vector
            //  are in the 3rd or 4th quadrants due to the constraints on arccos
            if (_orient[0] < 0 && _orient[1] < 0) //quadrant 3
              psiRot = (_pi/2 + psiRot);
              
            if (_orient[0] > 0 && _orient[1] < 0) //quadrant 4
              psiRot = (2*_pi - psiRot);
          
            // x,y, and z represent the pre-rotated direction vector of a ray
            double x;
            double y;
            double z;

            double phi = 0; // the azimuthal angle of each ray
            double range = 1 - cos(deltaTheta); // cos(0) to cos(deltaTheta) gives the range of possible vals
            
            sldAngl = 2*_pi*range; //the solid angle that the radiometer can view
            //testProjI = 0; // used to test a view factor and give the user an approximate rayError

            // Generate two uniformly-distributed-over-the-solid-angle random numbers
            // Used in determining the ray direction
            phi = 2 * _pi * _mTwister.randDblExc(); //azimuthal angle.  Range of 0 to 2pi
            // This guarantees that the polar angle of the ray is within the delta_theta
            VRTheta = acos(cos(deltaTheta)+range*_mTwister.randDblExc());
            
            //Convert to Cartesian
            x = sin(VRTheta)*cos(phi);
            y = sin(VRTheta)*sin(phi);
            z = cos(VRTheta);

            // ++++++++ Apply the rotational offsets ++++++
            direction_vector[0] = 
              x*cos(thetaRot)*cos(psiRot) +
              y*(-cos(phiRot)*sin(psiRot) + sin(phiRot)*sin(thetaRot)*cos(psiRot)) +
              z*( sin(phiRot)*sin(psiRot) + cos(phiRot)*sin(thetaRot)*cos(psiRot));
            
            direction_vector[1] = 
              x*cos(thetaRot)*sin(psiRot) +
              y *( cos(phiRot)*cos(psiRot) + sin(phiRot)*sin(thetaRot)*sin(psiRot)) +
              z *(-sin(phiRot)*cos(psiRot) + cos(phiRot)*sin(thetaRot)*sin(psiRot));
            
            direction_vector[2] = 
              x*(-sin(thetaRot)) +
              y*sin(phiRot)*cos(thetaRot) +
              z*cos(phiRot)*cos(thetaRot);
          
             inv_direction_vector = Vector(1.0)/direction_vector;       
            
            // get the intensity for this ray
            updateSumI(inv_direction_vector, ray_location, origin, Dx, domainLo, domainHi, sigmaT4OverPi, abskg, celltype, size, sumI, &_mTwister);
            sumProjI += cos(VRTheta) * (sumI - sumI_prev); // must subtract sumI_prev, since sumI accumulates intensity
                                                           // from all the rays up to that point
            sumI_prev = sumI;

          } // end VR ray loop
       
          //__________________________________
          //  Compute VRFlux
          VRFlux[origin] = sumProjI * sldAngl/_nRadRays;
        } // end of VR extents
      } // end if _virtRad
    } // end VR cell iterator


    //______________________________________________________________________
    //          B O U N D A R Y F L U X
    //______________________________________________________________________
    if( _solveBoundaryFlux){
      vector<Patch::FaceType> bf;

      IntVector pLow;
      IntVector pHigh;
      level->findInteriorCellIndexRange(pLow, pHigh);

      //_____________________________________________
      //   Ordering for Surface Method
      // This block of code is used to properly place ray origins, and orient ray directions
      // onto the correct face.  This is necessary, because by default, the rays are placed
      // and oriented onto a default face, then require adjustment onto the proper face.
      vector <IntVector> dirIndexOrder(6);
      vector <IntVector> dirSignSwap(6);
      vector <IntVector> locationIndexOrder(6);
      vector <IntVector> locationShift(6);

      dirIndexOrder[0]  = IntVector(2, 1, 0);
      dirIndexOrder[1]  = IntVector(2, 1, 0);
      dirIndexOrder[2]  = IntVector(0, 2, 1);
      dirIndexOrder[3]  = IntVector(0, 2, 1);
      dirIndexOrder[4]  = IntVector(0, 1, 2);
      dirIndexOrder[5]  = IntVector(0, 1, 2);

      // Ordering is slightly different from 6Flux since here, rays pass through origin cell from the inside faces.
      dirSignSwap[0]  = IntVector(-1, 1, 1);
      dirSignSwap[1]  = IntVector(1, 1, 1);
      dirSignSwap[2]  = IntVector(1, -1, 1);
      dirSignSwap[3]  = IntVector(1, 1, 1);
      dirSignSwap[4]  = IntVector(1, 1, -1);
      dirSignSwap[5]  = IntVector(1, 1, 1);


      locationIndexOrder[0] = IntVector(1,0,2);
      locationIndexOrder[1] = IntVector(1,0,2);
      locationIndexOrder[2] = IntVector(0,1,2);
      locationIndexOrder[3] = IntVector(0,1,2);
      locationIndexOrder[4] = IntVector(0,2,1);
      locationIndexOrder[5] = IntVector(0,2,1);

      locationShift[0] = IntVector(1, 0, 0);
      locationShift[1] = IntVector(0, 0, 0);
      locationShift[2] = IntVector(0, 1, 0);
      locationShift[3] = IntVector(0, 0, 0);
      locationShift[4] = IntVector(0, 0, 1);
      locationShift[5] = IntVector(0, 0, 0);


      for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector origin = *iter;

        // quick flux debug test
        //if(face==3 && j==Ny-1 && k==Nz/2)  // Burns flux locations
        //if(face==5 && j==Nx/2 && k==Nx-1){  // benchmark4, benchmark5: Siegel top surface flux locations
        //if ( origin.x()==0 && origin.y()==234 ){    // ifrf restart case. face should be 0 for these cells.

        // A given flow cell may have 0,1,2,3,4,5, or 6 faces that are adjacent to a wall.
        // boundaryFaces is the vector that contains the list of which faces are adjacent to a wall
        vector<int> boundaryFaces;
        boundaryFaces.clear();
        if(_benchmark==4 || _benchmark==5) boundaryFaces.push_back(5); // Benchmark4 benchmark5


        // determine if origin has one or more boundary faces, and if so, populate boundaryFaces vector
        boundFlux[origin].p = has_a_boundary(origin, celltype, boundaryFaces);


        /*  Benchmark4
        // Loop over 40 kappa and sigma_s values

        // open sigma_s
      char inputFilename[] = "sigma_s.txt";
      ifstream inFile;
      inFile.open(inputFilename, ios::in);
      if (!inFile) {
        cerr << "Can't open input file " << inputFilename << endl;
        exit(1);
      }
      double sigma_s[40];
      double kappa[40];

      // open kappa
      char inputFilename2[] = "kappa.txt";
      ifstream inFile2;
      inFile2.open(inputFilename2, ios::in);
      if (!inFile2) {
        cerr << "Can't open input file 2" << inputFilename << endl;
        exit(1);
      }

      //assign kappa and sigma_s and LOOP over 40 values
      int i_s=0;
      while (!inFile.eof()) {
        inFile >> sigma_s[i_s];
        i_s++;
      }
      i_s = 0;
      while(!inFile2.eof()) {
        inFile2 >> kappa[i_s];
        i_s++;
      }

       i_s=0;
       while(i_s<40) {

        _abskgBench4 = kappa[i_s];
        _sigmaScat = sigma_s[i_s];
        i_s++;


        //cout << _sigmaScat << endl;
        //cout << _abskgBench4 << endl;
*/
        FILE * f = NULL;
        if(_benchmark==5){
          f=fopen("benchmark5.txt", "w");
        }
    //__________________________________
    // Loop over boundary faces of the cell at hand and compute incident radiative flux
        for (vector<int>::iterator it=boundaryFaces.begin() ; it < boundaryFaces.end(); it++ ){  // 5/25

          int face = *it;  // face uses Uintah ordering
          int UintahFace[6] = {1,0,3,2,5,4}; //Uintah face iterator is an enum with the order WESNBT
          int RayFace = UintahFace[face];    // All the Ray functions are based on the face order of EWNSTB
          //  IntVector origin = IntVector(i,j,k);

          double sumI     = 0;
          double sumProjI = 0;
          double sumI_prev= 0;

          //__________________________________
          // Flux ray loop
          for (int iRay=0; iRay < _nFluxRays; iRay++){

            IntVector cur = origin;

            if(_isSeedRandom == false){           // !! This could use a compiler directive for speed-up
              _mTwister.seed((origin.x() + origin.y() + origin.z()) * iRay +1);
            }

            Vector direction_vector;

            // Surface Way to generate a ray direction from the positive z face
            double phi   = 2 * M_PI * _mTwister.rand(); //azimuthal angle.  Range of 0 to 2pi
            double theta = acos(_mTwister.rand());      // polar angle for the hemisphere
          
            //Convert to Cartesian
            direction_vector[0] =  sin(theta) * cos(phi);
            direction_vector[1] =  sin(theta) * sin(phi);
            direction_vector[2] =  cos(theta);
          
            // Put direction vector as coming from correct face
            adjustDirection(direction_vector, dirIndexOrder[RayFace], dirSignSwap[RayFace]);
            Vector inv_direction_vector = Vector(1.0)/direction_vector;

            double DyDxRatio = Dx.y() / Dx.x(); //noncubic
            double DzDxRatio = Dx.z() / Dx.x(); //noncubic

            Vector ray_location;

            // Surface way to generate a ray location from the negative y face
            ray_location[0] =  _mTwister.rand() ;
            ray_location[1] =  0;
            ray_location[2] =  _mTwister.rand() * DzDxRatio ;
          
            // Put point on correct face
            adjustLocation(ray_location, locationIndexOrder[RayFace],  locationShift[RayFace], DyDxRatio, DzDxRatio);
            ray_location[0] += origin.x();
            ray_location[1] += origin.y();
            ray_location[2] += origin.z();
            updateSumI(inv_direction_vector, ray_location, origin, Dx, domainLo, domainHi, sigmaT4OverPi, abskg, celltype, size, sumI, &_mTwister);

            sumProjI += cos(theta) * (sumI - sumI_prev); // must subtract sumI_prev, since sumI accumulates intensity


            // from all the rays up to that point
            sumI_prev = sumI;

          } // end of flux ray loop

          //__________________________________
          //  Compute Net Flux to the boundary
          //itr->second.net = sumProjI * 2*_pi/_nFluxRays - abskg[origin] * sigmaT4OverPi[origin] * _pi; // !!origin is a flow cell, not a wall
          double fluxIn = sumProjI * 2 *_pi/_nFluxRays;
          switch(face){
          case 0 : boundFlux[origin].w = fluxIn; break;
          case 1 : boundFlux[origin].e = fluxIn; break;
          case 2 : boundFlux[origin].s = fluxIn; break;
          case 3 : boundFlux[origin].n = fluxIn; break;
          case 4 : boundFlux[origin].b = fluxIn; break;
          case 5 : boundFlux[origin].t = fluxIn; break;
          }
          if(_benchmark==5)fprintf(f, "%lf \n",sumProjI * 2*_pi/_nFluxRays);


        } // end of looping through the vector boundaryFaces

        if(_benchmark==5) fclose(f);

      //}// end of file for benchmark4 verification test
      //} // end of quick flux debug

      }// end cell iterator

      // if(_applyFilter)
      // Put a cell iterator here
      // Implement fancy 2D filtering for boundFluxFilt here
      // Will need a smart algorithm to determine in which plane to do the filtering

    }   // end if _solveBoundaryFlux
        
         
    //______________________________________________________________________
    //         S O L V E   D I V Q
    //______________________________________________________________________
  if( _solveDivQ){
    for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){ 
      IntVector origin = *iter; 
      int i = origin.x();
      int j = origin.y();
      int k = origin.z();

      // Allow for quick debugging test
       IntVector pLow;
       IntVector pHigh;
       level->findInteriorCellIndexRange(pLow, pHigh);
       //int Nx = pHigh[0] - pLow[0];
       //if (j==Nx/2 && k==Nx/2){

      double sumI = 0;
      
      // ray loop
      for (int iRay=0; iRay < _NoOfRays; iRay++){

        if(_isSeedRandom == false){
          _mTwister.seed((i + j +k) * iRay +1);
        }

        // see http://www.cgafaq.info/wiki/aandom_Points_On_Sphere for explanation

        double plusMinus_one = 2 * _mTwister.randDblExc() - 1;
        double r = sqrt(1 - plusMinus_one * plusMinus_one);    // Radius of circle at z
        double theta = 2 * M_PI * _mTwister.randDblExc();            // Uniform betwen 0-2Pi

        Vector direction_vector;
        direction_vector[0] = r*cos(theta);                   // Convert to cartesian
        direction_vector[1] = r*sin(theta);
        direction_vector[2] = plusMinus_one;                  
        Vector inv_direction_vector = Vector(1.0)/direction_vector;

        double DyDxRatio = Dx.y() / Dx.x(); //noncubic
        double DzDxRatio = Dx.z() / Dx.x(); //noncubic
        
        Vector ray_location;
        Vector ray_location_prev;

        if(_CCRays){
          ray_location[0] =   i +  0.5 ;
          ray_location[1] =   j +  0.5 * DyDxRatio ;
          ray_location[2] =   k +  0.5 * DzDxRatio ;
        } else{
          ray_location[0] =   i +  _mTwister.rand() ;
          ray_location[1] =   j +  _mTwister.rand() * DyDxRatio ;
          ray_location[2] =   k +  _mTwister.rand() * DzDxRatio ;
        }
        updateSumI(inv_direction_vector, ray_location, origin, Dx, domainLo, domainHi, sigmaT4OverPi, abskg, celltype, size, sumI, &_mTwister);
        
      }  // Ray loop
      
      //__________________________________
      //  Compute divQ
      divQ[origin] = 4.0 * _pi * abskg[origin] * ( sigmaT4OverPi[origin] - (sumI/_NoOfRays) );
      radiationVolq[origin] = -divQ[origin] / abskg[origin]; // The minus sign is necessary due to the way the enthalpy source term is defined
      //} // end quick debug testing
    }  // end cell iterator
  } // end of if(_solveDivQ)
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
                               bool modifies_divQ,
                               const int radCalc_freq )
{  
  int maxLevels = level->getGrid()->numLevels() -1;
  int L_indx = level->getIndex();
  
  if(L_indx != maxLevels){     // only schedule on the finest level
    return;
  }
  std::string taskname = "Ray::rayTrace_dataOnion";
  Task* tsk= scinew Task( taskname, this, &Ray::rayTrace_dataOnion,
                          modifies_divQ, abskg_dw, sigma_dw, radCalc_freq );
                          
  printSchedule(level,dbg,taskname);

  // used when carryforward is needed
  tsk->requires( Task::OldDW, d_divQLabel,           d_gn, 0 );

  Task::MaterialDomainSpec  ND  = Task::NormalDomain;
  #define allPatches 0
  #define allMatls 0
  Ghost::GhostType  gac  = Ghost::AroundCells;

  // finest level:
  tsk->requires(abskg_dw, d_abskgLabel,     gac, SHRT_MAX);
  tsk->requires(sigma_dw, d_sigmaT4_label,  gac, SHRT_MAX);
  
  if( _whichROI_algo == dynamic ){
    tsk->requires(Task::NewDW, d_ROI_LoCellLabel);
    tsk->requires(Task::NewDW, d_ROI_HiCellLabel);
  }
  
  // coarser level
  int nCoarseLevels = maxLevels;
  for (int l=1; l<=nCoarseLevels; ++l){
    tsk->requires(abskg_dw, d_abskgLabel,     allPatches, Task::CoarseLevel,l,allMatls, ND, gac, SHRT_MAX);
    tsk->requires(sigma_dw, d_sigmaT4_label,  allPatches, Task::CoarseLevel,l,allMatls, ND, gac, SHRT_MAX);
  }
  
  if( modifies_divQ ){
    tsk->modifies( d_divQLabel );
    tsk->modifies( d_VRFluxLabel );
    tsk->modifies( d_boundFluxLabel );
    tsk->modifies( d_radiationVolqLabel );

  } else {
    
    tsk->computes( d_divQLabel );
    tsk->computes( d_VRFluxLabel );
    tsk->computes( d_boundFluxLabel );
    tsk->computes( d_radiationVolqLabel );

  }
  sched->addTask( tsk, level->eachPatch(), d_matlSet );
}


//---------------------------------------------------------------------------
// Ray tracer using the multilevel "data onion" scheme
//---------------------------------------------------------------------------
void
Ray::rayTrace_dataOnion( const ProcessorGroup* pc,
                         const PatchSubset* finePatches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw,
                         bool modifies_divQ,
                         Task::WhichDW which_abskg_dw,
                         Task::WhichDW which_sigmaT4_dw,
                         const int radCalc_freq )
{ 

  const Level* fineLevel = getLevel(finePatches);
   //__________________________________
  //  Carry Forward (old_dw -> new_dw)
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();
  if ( doCarryForward( timestep, radCalc_freq) ) {
    printTask( fineLevel->getPatch(0), dbg, "Coing Ray::rayTrace_dataOnion carryForward ( divQ )" );
    
    new_dw->transferFrom( old_dw, d_divQLabel,      finePatches, matls );    
    return;
  } 
  
  //__________________________________
  //
  int maxLevels    = fineLevel->getGrid()->numLevels();
  int levelPatchID = fineLevel->getPatch(0)->getID();
  LevelP level_0 = new_dw->getGrid()->getLevel(0);
  MTRand _mTwister;

  //__________________________________
  //retrieve the coarse level data
  StaticArray< constCCVariable<double> > abskg(maxLevels);
  StaticArray< constCCVariable<double> >sigmaT4OverPi(maxLevels);
  constCCVariable<double> abskg_fine;
  constCCVariable<double> sigmaT4OverPi_fine;
    
  DataWarehouse* abskg_dw   = new_dw->getOtherDataWarehouse(which_abskg_dw);
  DataWarehouse* sigmaT4_dw = new_dw->getOtherDataWarehouse(which_sigmaT4_dw);
  
  vector<Vector> Dx(maxLevels);
  double DyDx[maxLevels];
  double DzDx[maxLevels];
  
  for(int L = 0; L<maxLevels; L++){
    LevelP level = new_dw->getGrid()->getLevel(L);
    
    if (level->hasFinerLevel() ) {                               // coarse level data
      IntVector domainLo_EC, domainHi_EC;
      level->findCellIndexRange(domainLo_EC, domainHi_EC);       // including extraCells

      abskg_dw->getRegion(   abskg[L]   ,       d_abskgLabel ,   d_matl , level.get_rep(), domainLo_EC, domainHi_EC);
      sigmaT4_dw->getRegion( sigmaT4OverPi[L] , d_sigmaT4_label, d_matl , level.get_rep(), domainLo_EC, domainHi_EC);
      dbg << " getting coarse level data L-" <<L<< endl;
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
  if ( _whichROI_algo == fixed || _whichROI_algo == dynamic ){
    int L = maxLevels - 1;
    
    const Patch* notUsed=0;
    computeExtents(level_0, fineLevel, notUsed, maxLevels, new_dw,
                   fineLevel_ROI_Lo, fineLevel_ROI_Hi,  
                   regionLo,  regionHi);
    
    dbg << " getting fine level data across L-" <<L<< " " << fineLevel_ROI_Lo << " " << fineLevel_ROI_Hi<<endl;
    abskg_dw->getRegion(   abskg[L]   ,       d_abskgLabel ,   d_matl , fineLevel, fineLevel_ROI_Lo, fineLevel_ROI_Hi);
    sigmaT4_dw->getRegion( sigmaT4OverPi[L] , d_sigmaT4_label, d_matl , fineLevel, fineLevel_ROI_Lo, fineLevel_ROI_Hi);
  }
  
  abskg_fine         = abskg[maxLevels-1];
  sigmaT4OverPi_fine = sigmaT4OverPi[maxLevels-1];
  
  // Determine the size of the domain.
  BBox domain_BB;
  level_0->getInteriorSpatialRange(domain_BB);                 // edge of computational domain

  double start=clock();

  //__________________________________
  //  patch loop
  for (int p=0; p < finePatches->size(); p++){

    const Patch* finePatch = finePatches->get(p);
    printTask(finePatches, finePatch,dbg,"Doing Ray::rayTrace_dataOnion");

     //__________________________________
    //  retrieve fine level data ( patch_based )
    if ( _whichROI_algo == patch_based ){
    
      computeExtents(level_0, fineLevel, finePatch, maxLevels, new_dw,        
                     fineLevel_ROI_Lo, fineLevel_ROI_Hi,  
                     regionLo,  regionHi);
    
      int L = maxLevels - 1;
      dbg << " getting fine level data across L-" <<L<< endl;
           
      abskg_dw->getRegion(   abskg[L]   ,       d_abskgLabel ,   d_matl , fineLevel, fineLevel_ROI_Lo, fineLevel_ROI_Hi);
      sigmaT4_dw->getRegion( sigmaT4OverPi[L] , d_sigmaT4_label, d_matl , fineLevel, fineLevel_ROI_Lo, fineLevel_ROI_Hi);
      abskg_fine         = abskg[L];
      sigmaT4OverPi_fine = sigmaT4OverPi[L];
    }
    
    CCVariable<double> divQ_fine;
    
    if( modifies_divQ ){
      old_dw->getModifiable( divQ_fine,  d_divQLabel, d_matl, finePatch );
    }else{
      new_dw->allocateAndPut( divQ_fine, d_divQLabel, d_matl, finePatch );
      divQ_fine.initialize( 0.0 );
    }

    unsigned long int size = 0;                             // current size of PathIndex

    //__________________________________
    //
    for (CellIterator iter = finePatch->getCellIterator(); !iter.done(); iter++){ 

      IntVector origin = *iter; 
      int i = origin.x();
      int j = origin.y();
      int k = origin.z();
      
      // Allow for quick debugging test
     /*  IntVector pLow;
       IntVector pHigh;
       level->findInteriorCellIndexRange(pLow, pHigh);
       int Nx = pHigh[0] - pLow[0];
       if (i==Nx/2 && k==Nx/2){
     */
      
      
/*`==========TESTING==========*/
      if(origin == IntVector(10,10,0) && _isDbgOn ){
        dbg2.setActive(true);
      }else{
        dbg2.setActive(false);
      } 
/*===========TESTING==========`*/
      
      
      double sumI = 0;
      
      Vector tMax;
      vector<Vector> tDelta(maxLevels);

      //__________________________________
      //  ray loop
      for (int iRay=0; iRay < _NoOfRays; iRay++){
        IntVector cur      = origin;
        IntVector prevCell = cur;
        
        int L       = maxLevels -1;  // finest level
        int prevLev = L;
        
        if(_isSeedRandom == false){
          _mTwister.seed((i + j +k) * iRay +1);
        }

        //__________________________________
        //  Ray direction      
        // see http://www.cgafaq.info/wiki/aandom_Points_On_Sphere for explanation

        double plusMinus_one = 2 * _mTwister.randDblExc() - 1;
        double r = sqrt(1 - plusMinus_one * plusMinus_one);    // Radius of circle at z
        double theta = 2 * M_PI * _mTwister.randDblExc();      // Uniform betwen 0-2Pi

        // dbg2 << " plusMinus_one " << plusMinus_one << " r " << r << " theta " << theta << endl;

        Vector direction;
        direction[0] = r*cos(theta);                           // Convert to cartesian
        direction[1] = r*sin(theta);
        direction[2] = plusMinus_one;
        
/*`==========TESTING==========*/
 //       direction = Vector(0,1,0);                   // Debug:: shoot ray in 1 directon
/*===========TESTING==========`*/
        
        Vector inv_direction = Vector(1.0)/direction;

        int step[3];                                           // Gives +1 or -1 based on sign
        bool sign[3];
        for ( int ii= 0; ii<3; ii++){
          if (inv_direction[ii]>0){
            step[ii] = 1;
            sign[ii] = 1;
          }
          else{
            step[ii] = -1;
            sign[ii] = 0;
          }
        }
        
        //__________________________________
        // define tMax & tDelta on all levels
        // go from finest to coarset level so you can compare 
        // with 1L rayTrace results.
        
        tMax.x( (sign[0]  - _mTwister.rand())            * inv_direction[0] );  
        tMax.y( (sign[1]  - _mTwister.rand()) * DyDx[L]  * inv_direction[1] );  
        tMax.z( (sign[2]  - _mTwister.rand()) * DzDx[L]  * inv_direction[2] );  
        
        for(int Lev = maxLevels-1; Lev>-1; Lev--){
          //Length of t to traverse one cell
          tDelta[Lev].x( abs(inv_direction[0]) );
          tDelta[Lev].y( abs(inv_direction[1]) * DyDx[Lev] );
          tDelta[Lev].z( abs(inv_direction[2]) * DzDx[Lev] );
        }

        //Initializes the following values for each ray
        bool   in_domain      = true;
        double tMax_prev      = 0;
        double intensity      = 1.0;
        double fs             = 1.0;
        int    nReflect       = 0;             // Number of reflections
        double optical_thickness = 0;
        bool   onFineLevel    = true;
        const Level* level    = fineLevel;


        dbg2 << "  fineLevel_ROI_Lo: " <<  fineLevel_ROI_Lo << " fineLevel_ROI_HI: " << fineLevel_ROI_Hi << endl;
         
        //______________________________________________________________________
        //  Threshold  loop
        while (intensity > _Threshold){
          
          DIR dir = NONE;
          while (in_domain){
            
            prevCell = cur;
            prevLev  = L;
            
            double disMin = -9;   // Ray segment length.
            
            //__________________________________
            //  Determine the princple direction the ray is traveling
            //  
            if (tMax.x() < tMax.y()){
              if (tMax.x() < tMax.z()){
                dir = X;
              } else {
                dir = Z;
              }
            } else {
              if(tMax.y() <tMax.z()){
                dir = Y;
              } else {
                dir = Z;
              }
            }
            
            // next cell index and position
            cur[dir]  = cur[dir] + step[dir];
            Point pos = level->getCellPosition(cur);
            Vector dx_prev = level->dCell();  //  Used to compute coarsenRatio

            
            //__________________________________
            // Logic for moving between levels
            // currently you can only move from fine to coarse level
            
            //bool jumpFinetoCoarserLevel   = ( onFineLevel && finePatch->containsCell(cur) == false );
            bool jumpFinetoCoarserLevel   = ( onFineLevel && containsCell(fineLevel_ROI_Lo, fineLevel_ROI_Hi, cur, dir) == false );
            bool jumpCoarsetoCoarserLevel = ( onFineLevel == false && containsCell(regionLo[L], regionHi[L], cur, dir) == false && L > 0 );
            
            dbg2 << cur << " jumpFinetoCoarserLevel " << jumpFinetoCoarserLevel << " jumpCoarsetoCoarserLevel " << jumpCoarsetoCoarserLevel
                 << " containsCell: " << containsCell(fineLevel_ROI_Lo, fineLevel_ROI_Hi, cur, dir) << endl; 
                 
            if( jumpFinetoCoarserLevel ){
              cur   = level->mapCellToCoarser(cur); 
              level = level->getCoarserLevel().get_rep();      // move to a coarser level
              L     = level->getIndex();
              onFineLevel = false;
              
              dbg2 << " Jumping off fine patch switching Levels:  prev L: " << prevLev << " cur L " << L << " cur " << cur << endl;
            } else if ( jumpCoarsetoCoarserLevel ){
              
              IntVector c_old = cur;
              cur   = level->mapCellToCoarser(cur); 
              level = level->getCoarserLevel().get_rep();
              L     = level->getIndex();
              
              dbg2 << " Switching Levels:  prev L: " << prevLev << " cur L " << L << " cur " << cur << " c_old " << c_old << endl;
            }
            
            //__________________________________
            // Account for uniqueness of first step after reaching a new level

            //__________________________________
            //  update marching variables
            disMin        = (tMax[dir] - tMax_prev);        // Todd:   replace tMax[dir]
            tMax_prev     = tMax[dir];
            tMax[dir]     = tMax[dir] + tDelta[L][dir];

            Vector dx = level->dCell();


            IntVector coarsenRatio = IntVector(1,1,1);
            coarsenRatio[0] = dx[0]/dx_prev[0];
            coarsenRatio[1] = dx[1]/dx_prev[1];
            coarsenRatio[2] = dx[2]/dx_prev[2];

            // Update DyDx and DzDx ratios in the event that coarsening is not uniform in each dir.
            DyDx[L] = dx.y() / dx.x();
            DzDx[L] = dx.z() / dx.x();
            Dx[L] = dx;

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
            
            in_domain = domain_BB.inside(pos);

            //__________________________________
            //  Update the ray location
            //this is necessary to find the absorb_coef at the endpoints of each step if doing interpolations
            //ray_location_prev = ray_location;
            //ray_location      = ray_location + (disMin * direction_vector);// If this line is used,  make sure that direction_vector is adjusted after a reflection

            // The running total of alpha*length
            double optical_thickness_prev = optical_thickness;
            optical_thickness += Dx[prevLev].x() * abskg[prevLev][prevCell]*disMin;
            size++;

            //Eqn 3-15(see below reference) while
            //Third term inside the parentheses is accounted for in Inet. Chi is accounted for in Inet calc.
            sumI += sigmaT4OverPi[prevLev][prevCell] * ( exp(-optical_thickness_prev) - exp(-optical_thickness) ) * fs;
            
            dbg2 << "    origin " << origin << "dir " << dir << " cur " << cur <<" prevCell " << prevCell << " sumI " << sumI << " in_domain " << in_domain << endl;
            //dbg2 << "    tmaxX " << tMax.x() << " tmaxY " << tMax.y() << " tmaxZ " << tMax.z() << endl;
            //dbg2 << "    direction " << direction << endl;
         
          } //end domain while loop.  ++++++++++++++

          intensity = exp(-optical_thickness);

          //  wall emission
          sumI += abskg[L][cur] * sigmaT4OverPi[L][cur] * intensity;

          intensity = intensity * fs;  
           
          //__________________________________
          //  Reflections
          if (intensity > _Threshold){

            ++nReflect;
            fs = fs * (1 - abskg[L][cur]);

            //put cur back inside the domain
            cur = prevCell;
            in_domain = 1;

            // apply reflection condition
            step[dir] *= -1;                      // begin stepping in opposite direction
            sign[dir] = (sign[dir]==1) ? 0 : 1;  //  swap sign from 1 to 0 or vice versa
            
            dbg2 << " REFLECTING " << endl;
          }  // if reflection
        }  // threshold while loop.
      }  // Ray loop

      //__________________________________
      //  Compute divQ
      divQ_fine[origin] = 4.0 * _pi * abskg_fine[origin] * ( sigmaT4OverPi_fine[origin] - (sumI/_NoOfRays) );

      dbg2 << origin << "    divQ: " << divQ_fine[origin] << " term2 " << abskg_fine[origin] << " sumI term " << (sumI/_NoOfRays) << endl;
       // } // end quick debug testing
    }  // end cell iterator

    double end =clock();
    double efficiency = size/((end-start)/ CLOCKS_PER_SEC);
    if (finePatch->getGridIndex() == levelPatchID) {
      cout<< endl;
      cout << " RMCRT REPORT: Patch " << levelPatchID <<endl;
      cout << " Used "<< (end-start) * 1000 / CLOCKS_PER_SEC<< " milliseconds of CPU time. \n" << endl;// Convert time to ms
      cout << " Size: " << size << endl;
      cout << " Efficiency: " << efficiency << " steps per sec" << endl;
      cout << endl;
    }
  }  //end finePatch loop
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
  if( _whichROI_algo == dynamic ){
  
    minvec_vartype lo;
    maxvec_vartype hi;
    new_dw->get( lo, d_ROI_LoCellLabel );
    new_dw->get( hi, d_ROI_HiCellLabel );
    fineLevel_ROI_Lo = roundNearest( Vector(lo) );
    fineLevel_ROI_Hi = roundNearest( Vector(hi) );
    
  } else if ( _whichROI_algo == fixed ){
  
    fineLevel_ROI_Lo = fineLevel->getCellIndex( _ROI_minPt );
    fineLevel_ROI_Hi = fineLevel->getCellIndex( _ROI_maxPt );
    
    if( !fineLevel->containsCell( fineLevel_ROI_Lo ) || 
        !fineLevel->containsCell( fineLevel_ROI_Hi ) ){
      ostringstream warn;
      warn << "ERROR:  the fixed ROI extents " << _ROI_minPt << " " << _ROI_maxPt << " are not contained on the fine level."<< endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);    
    }
  } else if ( _whichROI_algo == patch_based ){
  
    IntVector patchLo = patch->getCellLowIndex();
    IntVector patchHi = patch->getCellHighIndex();
    
    fineLevel_ROI_Lo = patchLo - _halo;
    fineLevel_ROI_Hi = patchHi + _halo; 
    dbg << "  patch: " << patchLo << " " << patchHi << endl;

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

      regionLo[L] = level->mapCellToCoarser(regionLo[L+1]) - _halo;
      regionHi[L] = level->mapCellToCoarser(regionHi[L+1]) + _halo;

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
      dbg2 << "L-"<< L << " regionLo " << regionLo[L] << " regionHi " << regionHi[L] << endl;
    }
  }  
}



//______________________________________________________________________
void Ray::adjustDirection(Vector &directionVector, 
                          const IntVector &indexOrder, 
                          const IntVector &signOrder){

  Vector tmpry = directionVector;

  directionVector[0] = tmpry[indexOrder[0]] * signOrder[0];
  directionVector[1] = tmpry[indexOrder[1]] * signOrder[1];
  directionVector[2] = tmpry[indexOrder[2]] * signOrder[2];

}

//______________________________________________________________________
//
void Ray::adjustLocation(Vector &location, 
                        const IntVector &indexOrder, 
                        const IntVector &shift, 
                        const double &DyDxRatio, 
                        const double &DzDxRatio){

  Vector tmpry = location;

  location[0] = tmpry[indexOrder[0]] + shift[0];
  location[1] = tmpry[indexOrder[1]] + shift[1] * DyDxRatio;
  location[2] = tmpry[indexOrder[2]] + shift[2] * DzDxRatio;

}

//______________________________________________________________________
//
bool Ray::has_a_boundary(const IntVector &c, 
                         constCCVariable<int> &celltype, 
                         vector<int> &boundaryFaces){

  IntVector adjacentCell = c;
  bool hasBoundary = false;

  adjacentCell = c;
  adjacentCell[0] = c[0] - 1; // west

  if (celltype[adjacentCell]+1){ // cell type of flow is -1, so when cellType+1 isn't false, we
    boundaryFaces.push_back(0);     // know we're at a boundary
    hasBoundary = true;
  }

  adjacentCell[0] += 2; // east

  if (celltype[adjacentCell]+1){
    boundaryFaces.push_back(1);
    hasBoundary = true;
  }

  adjacentCell[0] -= 1;
  adjacentCell[1] = c[1] - 1; // south

  if (celltype[adjacentCell]+1){
    boundaryFaces.push_back(2);
    hasBoundary = true;
  }

  adjacentCell[1] += 2; // north

  if (celltype[adjacentCell]+1){
    boundaryFaces.push_back(3);
    hasBoundary = true;
  }

  adjacentCell[1] -= 1;
  adjacentCell[2] = c[2] - 1; // bottom

  if (celltype[adjacentCell]+1){
    boundaryFaces.push_back(4);
    hasBoundary = true;
  }

  adjacentCell[2] += 2; // top

  if (celltype[adjacentCell]+1){
    boundaryFaces.push_back(5);
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
  Task* tsk= scinew Task( taskname, this, &Ray::setBoundaryConditions, 
                          temp_dw, radCalc_freq, backoutTemp );

  printSchedule(level,dbg,taskname);

  if(!backoutTemp){
    tsk->requires( temp_dw, d_temperatureLabel, Ghost::None,0 );
  }
  
  tsk->modifies( d_sigmaT4_label ); 
  tsk->modifies( d_abskgLabel );
  tsk->modifies( d_cellTypeLabel );

  sched->addTask( tsk, level->eachPatch(), d_matlSet );
}
//---------------------------------------------------------------------------
void
Ray::setBoundaryConditions( const ProcessorGroup*,
                            const PatchSubset* patches,           
                            const MaterialSubset*,                
                            DataWarehouse*,                
                            DataWarehouse* new_dw,
                            Task::WhichDW temp_dw,
                            const int radCalc_freq,
                            const bool backoutTemp )               
{
  // Only run if it's time
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();
  if ( doCarryForward( timestep, radCalc_freq) ) {
    return;
  }
  
  if ( _onOff_SetBCs == false )
    return;

  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);
    
    if( bf.size() > 0){
    
      printTask(patches,patch,dbg,"Doing Ray::setBoundaryConditions");

      double sigma_over_pi = _sigma/M_PI;
      
      CCVariable<double> temp;
      CCVariable<double> abskg;
      CCVariable<double> sigmaT4OverPi;
      CCVariable<int> cellType;
      
      new_dw->allocateTemporary(temp,  patch);
      new_dw->getModifiable( abskg,         d_abskgLabel,     d_matl, patch );
      new_dw->getModifiable( sigmaT4OverPi, d_sigmaT4_label,  d_matl, patch );
      new_dw->getModifiable( cellType,      d_cellTypeLabel,  d_matl, patch );
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
        t_dw->get(varTmp, d_temperatureLabel,   d_matl, patch, Ghost::None, 0);
        temp.copyData(varTmp);
      }
      
      
      //__________________________________
      // set the boundary conditions
      setBC(abskg,    d_abskgLabel->getName(),       patch, d_matl);
      setBC(temp,     d_temperatureLabel->getName(), patch, d_matl);
      setBC(cellType, d_cellTypeLabel->getName(),    patch, d_matl);


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
template<class T>
void Ray::setBC(CCVariable<T>& Q_CC,
                const string& desc,
                const Patch* patch,
                const int mat_id)
{
  if(patch->hasBoundaryFaces() == false || _onOff_SetBCs == false){
    return;
  }

  dbg_BC << "setBC \t"<< desc <<" "
        << " mat_id = " << mat_id <<  ", Patch: "<< patch->getID() << endl;

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
      T bc_value = -9;
      Iterator bound_ptr;

      bool foundIterator = 
        getIteratorBCValueBCKind( patch, face, child, desc, mat_id,
                        bc_value, bound_ptr,bc_kind); 

      if(foundIterator) {

        //__________________________________
        // Dirichlet
        if(bc_kind == "Dirichlet"){
          nCells += setDirichletBC_CC<T>( Q_CC, bound_ptr, bc_value);
        }
        //__________________________________
        // Neumann
        else if(bc_kind == "Neumann"){
          nCells += setNeumannBC_CC<T>( patch, face, Q_CC, bound_ptr, bc_value, cell_dx);
        }                                   
        //__________________________________
        //  Symmetry
        else if ( bc_kind == "symmetry" || bc_kind == "zeroNeumann" ) {
          bc_value = 0.0;
          nCells += setNeumannBC_CC<T> ( patch, face, Q_CC, bound_ptr, bc_value, cell_dx);
        }

        //__________________________________
        //  debugging
        if( dbg_BC.active() ) {
          bound_ptr.reset();
          dbg_BC <<"Face: "<< patch->getFaceName(face) <<" numCellsTouched " << nCells
             <<"\t child " << child  <<" NumChildren "<<numChildren 
             <<"\t BC kind "<< bc_kind <<" \tBC value "<< bc_value
             <<"\t bound limits = "<< bound_ptr << endl;
        }
      }  // if iterator found
    }  // child loop

    dbg_BC << "    "<< patch->getFaceName(face) << " \t " << bc_kind << " numChildren: " << numChildren 
               << " nCellsTouched: " << nCells << endl;
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
    task->requires(Task::NewDW, d_divQLabel, allPatches, Task::CoarseLevel, allMatls, ND, d_gn,0);
     
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
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();
  if ( doCarryForward( timestep, radCalc_freq) ) {
    printTask( fineLevel->getPatch(0), dbg, "Doing Ray::refine_Q carryForward ( divQ )" );
    
    new_dw->transferFrom( old_dw, d_divQLabel, patches, matls );
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
  
  if( (L_indx != maxLevels ) || ( _whichROI_algo != dynamic ) ){     // only schedule on the finest level and dynamic
    return;
  }
  
  printSchedule(level,dbg,"Ray::ROI_Extents");

  Task* tsk = scinew Task( "Ray::ROI_Extents", this, 
                           &Ray::ROI_Extents);

  tsk->requires( Task::NewDW, d_abskgLabel,     d_gac, 1 );
  tsk->requires( Task::NewDW, d_sigmaT4_label,  d_gac, 1 );
  tsk->computes( d_mag_grad_abskgLabel );
  tsk->computes( d_mag_grad_sigmaT4Label );
  tsk->computes( d_flaggedCellsLabel );

  tsk->computes(d_ROI_LoCellLabel);
  tsk->computes(d_ROI_HiCellLabel);

  scheduler->addTask( tsk, level->eachPatch(), d_matlSet );
}

//______________________________________________________________________
// 
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
    constCCVariable<double> abskg;
    constCCVariable<double> sigmaT4;

    CCVariable<double> mag_grad_abskg;
    CCVariable<double> mag_grad_sigmaT4;
    CCVariable<int> flaggedCells;

    new_dw->get(abskg,    d_abskgLabel ,     d_matl , patch, d_gac,1);
    new_dw->get(sigmaT4,  d_sigmaT4_label ,  d_matl , patch, d_gac,1);

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

      if( mag_grad_abskg[c] > _abskg_thld || mag_grad_sigmaT4[c] > _sigmaT4_thld ){
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
                            const bool modifies_sigmaT4,
                            const int radCalc_freq)
{
  if(coarseLevel->hasFinerLevel()){
    printSchedule(coarseLevel,dbg,"Ray::sched_CoarsenAll");
    sched_Coarsen_Q(coarseLevel, sched, Task::NewDW, modifies_abskg,   d_abskgLabel,    radCalc_freq );
    sched_Coarsen_Q(coarseLevel, sched, Task::NewDW, modifies_sigmaT4, d_sigmaT4_label, radCalc_freq );
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

  Task* t = scinew Task( taskname, this, &Ray::coarsen_Q, 
                         variable, modifies, this_dw, radCalc_freq );
  
  if(modifies){
    t->requires(this_dw, variable, 0, Task::FineLevel, 0, Task::NormalDomain, d_gn, 0);
    t->modifies(variable);
  }else{
    t->requires(this_dw, variable, 0, Task::FineLevel, 0, Task::NormalDomain, d_gn, 0);
    t->computes(variable);
  }
  
  sched->addTask( t, coarseLevel->eachPatch(), d_matlSet );
}

//______________________________________________________________________
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

      CCVariable<double> Q_coarse;
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
// Utility task:  move variable from old_dw -> new_dw
void Ray::sched_CarryForward ( const LevelP& level, 
                               SchedulerP& sched,
                               const VarLabel* variable)
{ 
  string taskname = "        carryForward_" + variable->getName();
  printSchedule(level, dbg, taskname);

  Task* tsk = scinew Task( taskname, this, &Ray::carryForward, variable );
  
  tsk->requires(Task::OldDW, variable,   d_gn, 0);
  tsk->computes(variable);
 
  sched->addTask( tsk, level->eachPatch(), d_matlSet );
}

//______________________________________________________________________
void Ray::carryForward ( const ProcessorGroup*,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw, 
                         DataWarehouse* new_dw,
                         const VarLabel* variable)
{
  new_dw->transferFrom(old_dw, variable, patches, matls);
}

//______________________________________________________________________
void Ray::updateSumI ( Vector& inv_direction_vector,
                       Vector& ray_location,
                       const IntVector& origin,
                       const Vector& Dx,
                       const IntVector& domainLo,
                       const IntVector& domainHi,
                       constCCVariable<double>& sigmaT4OverPi,
                       constCCVariable<double>& abskg,
                       constCCVariable<int>& celltype,
                       unsigned long int& size,
                       double& sumI,
                       MTRand * _mTwister)

{

  IntVector cur = origin;
  IntVector prevCell = cur;
  // Step and sign for ray marching
   int step[3];                                          // Gives +1 or -1 based on sign
   bool sign[3];
   for ( int ii= 0; ii<3; ii++){
     if (inv_direction_vector[ii]>0){
       step[ii] = 1;
       sign[ii] = 1;
     }
     else{
       step[ii] = -1;
       sign[ii] = 0;
     }
   }

   double DyDxRatio = Dx.y() / Dx.x(); //noncubic
   double DzDxRatio = Dx.z() / Dx.x(); //noncubic

   double tMaxX = (origin[0] + sign[0]             - ray_location[0]) * inv_direction_vector[0];
   double tMaxY = (origin[1] + sign[1] * DyDxRatio - ray_location[1]) * inv_direction_vector[1];
   double tMaxZ = (origin[2] + sign[2] * DzDxRatio - ray_location[2]) * inv_direction_vector[2];

   //Length of t to traverse one cell
   double tDeltaX = abs(inv_direction_vector[0]);
   double tDeltaY = abs(inv_direction_vector[1]) * DyDxRatio;
   double tDeltaZ = abs(inv_direction_vector[2]) * DzDxRatio;
   double tMax_prev = 0;
   bool in_domain = true;

   //Initializes the following values for each ray
   double intensity = 1.0;
   double fs = 1.0;
   double optical_thickness = 0;


   //#define RAY_SCATTER 1
#ifdef RAY_SCATTER
   double scatCoeff = _sigmaScat; //[m^-1]  !! HACK !! This needs to come from data warehouse
   if (scatCoeff == 0) scatCoeff = 1e-99;  // avoid division by zero

   // Determine the length at which scattering will occur
   // See CCA/Components/Arches/RMCRT/PaulasAttic/MCRT/ArchesRMCRT/ray.cc
   double scatLength = -log(_mTwister->randDblExc() ) / scatCoeff;
   double curLength = 0;
#endif

   //+++++++Begin ray tracing+++++++++++++++++++
   int nReflect = 0; // Number of reflections that a ray has undergone
   //Threshold while loop
   while (intensity > _Threshold){

     int face = -9;

     while (in_domain){

       prevCell = cur;
       double disMin = -9;  // Common variable name in ray tracing. Represents ray segment length.

       //__________________________________
       //  Determine which cell the ray will enter next
       if (tMaxX < tMaxY){
         if (tMaxX < tMaxZ){
           cur[0]    = cur[0] + step[0];
           disMin    = tMaxX - tMax_prev;
           tMax_prev = tMaxX;
           tMaxX     = tMaxX + tDeltaX;
           face      = 0;
         }
         else {
           cur[2]    = cur[2] + step[2];
           disMin    = tMaxZ - tMax_prev;
           tMax_prev = tMaxZ;
           tMaxZ     = tMaxZ + tDeltaZ;
           face      = 2;
         }
       }
       else {
         if(tMaxY <tMaxZ){
           cur[1]    = cur[1] + step[1];
           disMin    = tMaxY - tMax_prev;
           tMax_prev = tMaxY;
           tMaxY     = tMaxY + tDeltaY;
           face      = 1;
         }
         else {
           cur[2]    = cur[2] + step[2];
           disMin    = tMaxZ - tMax_prev;
           tMax_prev = tMaxZ;
           tMaxZ     = tMaxZ + tDeltaZ;
           face      =2;
         }
       }

       ray_location[0] = ray_location[0] + (disMin  / inv_direction_vector[0]);
       ray_location[1] = ray_location[1] + (disMin  / inv_direction_vector[1]);
       ray_location[2] = ray_location[2] + (disMin  / inv_direction_vector[2]);

       in_domain = (celltype[cur]==-1);  //cellType of -1 is flow
       // The running total of alpha*length
       double optical_thickness_prev = optical_thickness;
       optical_thickness += Dx.x() * abskg[prevCell]*disMin; // as long as tDeltaY,Z tMaxY,Z and ray_location[1],[2]..
       // were adjusted by DyDxRatio or DzDxRatio, this line is now correct for noncubic domains.
       //optical_thickness += Dx.x() * _abskgBench4*disMin; // Use this line for Benchmark4 rather than the above line


       size++;

       //Eqn 3-15(see below reference) while
       //Third term inside the parentheses is accounted for in Inet. Chi is accounted for in Inet calc.
       sumI += sigmaT4OverPi[prevCell] * ( exp(-optical_thickness_prev) - exp(-optical_thickness) ) * fs;

#ifdef RAY_SCATTER
       curLength += disMin * Dx.x(); // July 18
       if (curLength > scatLength && in_domain){

         // get new scatLength for each scattering event
         scatLength = -log(_mTwister->randDblExc() ) / scatCoeff; 
         //store old step
         int stepOld = step[face];

         // Get new direction (below is isotropic scatteirng)
         double plusMinus_one = 2 * _mTwister->randDblExc() - 1;
         double r = sqrt(1 - plusMinus_one * plusMinus_one);    // Radius of circle at z
         double theta = 2 * M_PI * _mTwister->randDblExc();            // Uniform betwen 0-2Pi

         Vector direction_vector;
         direction_vector[0] = r*cos(theta);                   // Convert to cartesian
         direction_vector[1] = r*sin(theta);
         direction_vector[2] = plusMinus_one;

         inv_direction_vector = Vector(1.0)/direction_vector;

         // get new step and sign
         for ( int ii= 0; ii<3; ii++){
           if (inv_direction_vector[ii]>0){
             step[ii] = 1;
             sign[ii] = 1;
           }
           else{
             step[ii] = -1;
             sign[ii] = 0;
           }
         }

         // if sign[face] changes sign, put ray back into prevCell (back scattering)
         // a sign change only occurs when the product of old and new is negative
         if( step[face] * stepOld < 0 ){
           cur = prevCell;
         }
         // get new tMax
         tMaxX = (cur[0] + sign[0]             - ray_location[0]) * inv_direction_vector[0];
         tMaxY = (cur[1] + sign[1] * DyDxRatio - ray_location[1]) * inv_direction_vector[1];
         tMaxZ = (cur[2] + sign[2] * DzDxRatio - ray_location[2]) * inv_direction_vector[2];

         // get new tDelta
         //Length of t to traverse one cell
         tDeltaX = abs(inv_direction_vector[0]);
         tDeltaY = abs(inv_direction_vector[1]) * DyDxRatio;
         tDeltaZ = abs(inv_direction_vector[2]) * DzDxRatio;
         tMax_prev = 0;

         curLength = 0;  // allow for multiple scattering events per ray
         if(_benchmark == 4 || _benchmark ==5) scatLength = 1e16; // only for Siegel Benchmark4 benchmark5. Only allows 1 scatter event.
       }

#endif

     } //end domain while loop.  ++++++++++++++

     intensity = exp(-optical_thickness);

     //  wall emission 12/15/11
     sumI += abskg[cur]*sigmaT4OverPi[cur] * intensity;

     intensity = intensity * fs;  

     // for DOM comparisons, we don't allow for reflections, so 
     // when a ray reaches the end of the domain, we force it to terminate. 
     if(!_allowReflect) intensity = 0; //9-21-12
                                            
     //__________________________________
     //  Reflections
     if (intensity > _Threshold){
       
       ++nReflect;
       fs = fs * (1-abskg[cur]);

       //put cur back inside the domain
       cur = prevCell;

       // apply reflection condition
       step[face] *= -1;                      // begin stepping in opposite direction
       sign[face] = (sign[face]==1) ? 0 : 1; //  swap sign from 1 to 0 or vice versa
       inv_direction_vector[face] *= -1;

       in_domain = 1;


     }  // if reflection
   }  // threshold while loop.
} // end of updateSumI function


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
    
    new_dw->allocateAndPut(divQFilt, d_boundFluxLabel,   d_matl, patch);
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







//______________________________________________________________________
// Explicit template instantiations:

template void Ray::setBC<int>(    CCVariable<int>&    Q_CC, const string& desc, const Patch* patch, const int mat_id);
template void Ray::setBC<double>( CCVariable<double>& Q_CC, const string& desc, const Patch* patch, const int mat_id);

//______________________________________________________________________
// ISAAC's NOTES: 
//Jan 6. Began work on solving for boundary fluxes
//Jan 5. Changed containsCell method to only need to compare two faces rather than 6
//Dec 15. Now uses interactive BCs correctly from input file
//Dec 1. Clean up (removed ray viz stuff.
//Nov 30. Modified so user can specify in the input file either benchmark_13pt2, benchmark_1, or no benchmark (real case)
//Nov 18. Put in visualization stuff. It worked well.
//Nov 16. Realized that the "correct" method for reflections is no different from Paula's (using fs). Reverted back to Paula's
//Nov 9, 2011.  Added in reflections based on correct method of attenuating I for each reflection.
//Jun 9. Ray_noncubic.cc now handles non-cubic cells. Created from Ray.cc as it was in the repository on Jun 9, 2011.
//May 18. cleaned up comments
//May 6. Changed to cell iterator
//Created Jan 31. Cleaned up comments, removed hard coding of T and abskg 
// Jan 19// I changed cx to be lagging.  This changed nothing in the RMS error, but may be important...
//when referencing a non-uniform temperature.
//Created Jan13. //  Ray_PW_const.cc Making this piecewise constant by using CC values. not interpolating
//Removed symmetry test. 
//Has a new equation for absorb_coef for chi and optical thickness calculations...
//I did this based on my findings in my intepolator
//Just commented out a few unnecessary vars
//No more hitch!  Fixed cx to not be incremented the first march, and...
//fixed the formula for absorb_coef and chi which reference ray_location
//Now can do a DelDotqline in each of the three coordinate directions, through the center
//Ray Visualization works, and is correct
//To plot out the rays in matlab
//Now we use an average of two values for a more precise value of absorb_coef rather...
//than using the cell centered absorb_coef
//Using the exact absorb_coef for chi by using formula.Beautiful results...
//see chi_is_exact_absorb_coef.eps in runcases folder
//FIXED THE VARIANCE REDUCTION PROBLEM BY GETTING NEW CHI FOR EACH RAY goes with Chi Fixed folder
//BENCHMARK CASE 99. 
//with error msg if slice is too big
//Based on Ray_bak_Oct15.cc which was Created Oct 13.
// Using Woo (and Amanatides) method//and it works!
//efficiency of approx 20Msteps/sec
//I try to wait to declare each variable until I need it
//Incorporates Steve's sperical way of generating a direction vector
//Back to ijk from cell iterator
//Now absorb_coef is hard coded in because abskg in DW is simply zero
//Now gets abskg from Dw
// with capability to print temperature profile to a file
//Now gets T from DW.  accounts for intensity coming back from surfaces. calculates
// the net Intensity for each cell. Still needs to send rays out from surfaces.Chi is inside while 
//loop. I took out the double domain while loop simply for readability.  I should put it back in 
//when running cases. if(sign[xyorz]) else. See Ray_bak_Aug10.cc for correct implementation. ix 
//is just (NxNyNz) rather than (xNxxNyxNz).  absorbing media. reflections, and simplified while 
//(w/precompute) loop. ray_location now represents what was formally called emiss_point.  It is by
// cell index, not by physical location.
//NOTE equations are from the dissertation of Xiaojing Sun... 
//"REVERSE MONTE CARLO RAY-TRACING FOR RADIATIVE HEAT TRANSFER IN COMBUSTION SYSTEMS 
