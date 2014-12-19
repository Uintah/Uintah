/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

//______________________________________________________________________
//
#include <CCA/Components/Models/Radiation/RMCRT/Radiometer.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/BBox.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Math/MersenneTwister.h>
#include <time.h>
#include <fstream>

#include <include/sci_defs/uintah_testdefs.h.in>


//______________________________________________________________________
//
using namespace Uintah;
using namespace std;
static DebugStream dbg("RAY", false);

//______________________________________________________________________
// Class: Constructor.
//______________________________________________________________________
//
Radiometer::Radiometer(const TypeDescription::Type FLT_DBL ) : RMCRTCommon( FLT_DBL)
{
  if ( FLT_DBL == TypeDescription::double_type ){
    d_VRFluxLabel = VarLabel::create( "VRFlux", CCVariable<double>::getTypeDescription() );
    proc0cout << "__________________________________ USING DOUBLE VERSION OF RADIOMETER" << endl;
  } else {
    d_VRFluxLabel = VarLabel::create( "VRFlux", CCVariable<float>::getTypeDescription() );
    proc0cout << "__________________________________ USING FLOAT VERSION OF RADIOMETER" << endl;
  }

}

//______________________________________________________________________
// Method: Destructor
//______________________________________________________________________
//
Radiometer::~Radiometer()
{
  VarLabel::destroy( d_VRFluxLabel );
}

//______________________________________________________________________
// Method: Problem setup (access to input file information)
//______________________________________________________________________
void
Radiometer::problemSetup( const ProblemSpecP& prob_spec,
                          const ProblemSpecP& radps,
                          const GridP&        grid,
                          SimulationStateP&   sharedState,
                          const bool getExtraInputs)
{
  d_sharedState = sharedState;
  ProblemSpecP rad_ps = radps;
  Vector orient;
  rad_ps->getWithDefault( "VRViewAngle"    ,    d_viewAng,         180 );              // view angle of the radiometer in degrees
  rad_ps->getWithDefault( "VROrientation"  ,    orient,          Vector(0,0,1) );      // Normal vector of the radiometer orientation (Cartesian)
  rad_ps->getWithDefault( "nRadRays"  ,         d_nRadRays ,       1000 );
  rad_ps->get(            "VRLocationsMin" ,    d_VRLocationsMin );                    // minimum extent of the string or block of virtual radiometers in physical units
  rad_ps->get(            "VRLocationsMax" ,    d_VRLocationsMax );                    // maximum extent

  if( getExtraInputs ){
    rad_ps->getWithDefault( "sigmaScat"  ,      d_sigmaScat  ,      0 );                // scattering coefficient
    rad_ps->getWithDefault( "Threshold" ,       d_threshold ,      0.01 );              // When to terminate a ray
    rad_ps->getWithDefault( "randomSeed",       d_isSeedRandom,    true );              // random or deterministic seed.
    rad_ps->getWithDefault( "StefanBoltzmann",  d_sigma,           5.67051e-8);         // Units are W/(m^2-K)
    rad_ps->getWithDefault( "allowReflect"   ,  d_allowReflect,     true );             // Allow for ray reflections. Make false for DOM comparisons.
  } else {
                   // bulletproofing.
    for( ProblemSpecP n = rad_ps->getFirstChild(); n != 0; n=n->getNextSibling() ){
      string me = n->getNodeName();
      if( ( me == "sigmaScat"  ||  me == "Threshold" || me == "randomSeed" ||  me == "StefanBoltzmann" || me == "allowReflect" ) && me !="text" ){
        ostringstream warn;
        warn << "\n ERROR:Radiometer::problemSetup: You've specified the variable (" << me << ")"
             << " which will be ignored.  You should set the variable outside <Radiometer> section. \n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
    }
  }

  //__________________________________
  //  Warnings and bulletproofing
  //
  //the VRLocations can't exceed computational domain
  BBox compDomain;
  grid->getSpatialRange(compDomain);
  Point start = d_VRLocationsMin;
  Point end   = d_VRLocationsMax;

  Point min = compDomain.min();
  Point max = compDomain.max();

  if(start.x() < min.x() || start.y() < min.y() || start.z() < min.z() ||
     end.x() > max.x()   || end.y() > max.y()   || end.z() > max.z() ) {
    ostringstream warn;
    warn << "\n ERROR:Radiometer::problemSetup: the radiometer that you've specified " << start
         << " " << end << " begins or ends outside of the computational domain. \n" << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

  if( start.x() > end.x() || start.y() > end.y() || start.z() > end.z() ) {
    ostringstream warn;
    warn << "\n ERROR:Radiometer::problemSetup: the radiometerthat you've specified " << start
         << " " << end << " the starting point is > than the ending point \n" << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

  // you need at least one cell that's a radiometer
  if( start.x() == end.x() || start.y() == end.y() || start.z() == end.z() ){
    ostringstream warn;
    warn << "\n ERROR:Radiometer::problemSetup: The specified radiometer has the same "
         << "starting and ending points.  All of the directions must differ by one cell\n "
         << "                                start: " << start << " end: " << end << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

#ifndef RAY_SCATTER
  proc0cout<< "sigmaScat: " << d_sigmaScat << endl;
  if(d_sigmaScat>0){
    ostringstream warn;
    warn << "ERROR:  In order to run a scattering case, you must use the following in your configure line..." << endl;
    warn << "--enable-ray-scatter" << endl;
    warn << "If you wish to run a scattering case, please modify your configure line and re-configure and re-compile." << endl;
    warn << "If you wish to run a non-scattering case, please remove the line containing <sigmaScat> from your input file." << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
#endif

#ifdef RAY_SCATTER
  if(d_sigmaScat<1e-99){
    proc0cout << "WARNING:  You are running a non-scattering case, yet you have the following in your configure line..." << endl;
    proc0cout << "--enable-ray-scatter" << endl;
    proc0cout << "As such, this task will run slower than is necessary." << endl;
    proc0cout << "If you wish to run a scattering case, please specify a positive value greater than 1e-99 for the scattering coefficient." << endl;
    proc0cout << "If you wish to run a non-scattering case, please remove --enable-ray-scatter from your configure line and re-configure and re-compile" << endl;
  }
  proc0cout<< endl << "RAY_SCATTER IS DEFINED" << endl;
#endif

  if ( d_viewAng > 360 ){
    ostringstream warn;
    warn << "ERROR:  VRViewAngle ("<< d_viewAng <<") exceeds the maximum acceptable value of 360 degrees." << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

  if ( d_nRadRays < int(15 + pow(5.4, d_viewAng/40) ) ){
    proc0cout << "RMCRT: WARNING Number of radiometer rays:  ("<< d_nRadRays <<") is less than the recommended number of ("<< int(15 + pow(5.4, d_viewAng/40) ) <<"). Errors will exceed 1%. " << endl;
  }

  // orient[0,1,2] represent the user specified vector normal of the radiometer.
  // These will be converted to rotations about the x,y, and z axes, respectively.
  // Each rotation is counterclockwise when the observer is looking from the
  // positive axis about which the rotation is occurring. d
  for(int d = 0; d<3; d++){
    if(orient[d] == 0){      // WARNING WARNING this conditional only works for integers, not doubles, and should be fixed.
      orient[d] =1e-16;      // to avoid divide by 0.
    }
  }

  //__________________________________
  //  CONSTANT VR VARIABLES
  //  In spherical coordinates, the polar angle, theta_rot,
  //  represents the counterclockwise rotation about the y axis,
  //  The azimuthal angle represents the negative of the
  //  counterclockwise rotation about the z axis.
  //  Convert the user specified radiometer vector normal into three axial
  //  rotations about the x, y, and z axes.
  d_VR.thetaRot = acos( orient[2] / orient.length() );
  double psiRot = acos( orient[0] / sqrt( orient[0]*orient[0] + orient[1]*orient[1] ) );

  // The calculated rotations must be adjusted if the x and y components of the normal vector
  // are in the 3rd or 4th quadrants due to the constraints on arccos
  if (orient[0] < 0 && orient[1] < 0){       // quadrant 3
    psiRot = (M_PI/2 + psiRot);
  }
  if (orient[0] > 0 && orient[1] < 0){       // quadrant 4
    psiRot = (2*M_PI - psiRot);
  }

  d_VR.psiRot = psiRot;
  //  phiRot is always  0. There will never be a need for a rotation about the x axis. All
  //  possible rotations can be accomplished using the other two.
  d_VR.phiRot = 0;

  double deltaTheta = d_viewAng/360*M_PI;       // divides view angle by two and converts to radians
  double range      = 1 - cos(deltaTheta);      // cos(0) to cos(deltaTheta) gives the range of possible vals
  d_VR.sldAngl      = 2*M_PI*range;             // the solid angle that the radiometer can view
  d_VR.deltaTheta   = deltaTheta;
  d_VR.range        = range;
  d_sigma_over_pi   = d_sigma/M_PI;

  //__________________________________
  // bulletproofing
  ProblemSpecP root_ps = rad_ps->getRootNode();

  Vector periodic;
  ProblemSpecP grid_ps  = root_ps->findBlock("Grid");
  ProblemSpecP level_ps = grid_ps->findBlock("Level");
  level_ps->getWithDefault("periodic", periodic, Vector(0,0,0));

  if (periodic.length() != 0 ){
    throw ProblemSetupException("\nERROR RMCRT:\nPeriodic boundary conditions are not allowed with Radiometer.", __FILE__, __LINE__);
  }
}


//______________________________________________________________________
//
//______________________________________________________________________
void
Radiometer::sched_initializeRadVars( const LevelP& level,
                                     SchedulerP& sched,
                                     const int radCalc_freq )
{

  std::string taskname = "Radiometer::initializeRadVars";

  Task* tsk = NULL;
  if ( RMCRTCommon::d_FLT_DBL == TypeDescription::double_type ){
    tsk= scinew Task( taskname, this, &Radiometer::initializeRadVars< double >, radCalc_freq );
  }else{
    tsk= scinew Task( taskname, this, &Radiometer::initializeRadVars< float >, radCalc_freq );
  }
  
  printSchedule(level,dbg,taskname);
  tsk->requires(Task::OldDW, d_VRFluxLabel, d_gn, 0);
  tsk->computes( d_VRFluxLabel );

  sched->addTask( tsk, level->eachPatch(), d_matlSet );
}

//______________________________________________________________________
//  - Initialize the flux on all patches or move that variable foward
//    The flux is modified downstream.
//  - Determine if the taskgraph should be recompiled
//______________________________________________________________________
template< class T >
void
Radiometer::initializeRadVars( const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw,
                               const int radCalc_freq )
{
  // Recompile taskgraph on calc timesteps
  // This controls the temporal task scheduling
  doRecompileTaskgraph( radCalc_freq );  // this must be at the top of the taskrecomp

  if ( doCarryForward( radCalc_freq) ) {
    printTask(patches,patches->get(0), dbg,"Doing Radiometer::initializeVars (carryForward)");
    bool replaceVar = true;
    new_dw->transferFrom( old_dw, d_VRFluxLabel,    patches, matls, replaceVar );

    return;
  }

  //__________________________________
  //  Initialize the flux.
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing Radiometer::initializeVars");

    CCVariable< T > VRFlux;
    new_dw->allocateAndPut( VRFlux, d_VRFluxLabel, d_matl, patch );
    VRFlux.initialize( 0.0 );
  }
}
//______________________________________________________________________
// Method: Schedule the virtual radiometer.  Only use temporal scheduling 
//  This is a HACK until spatial scheduling working.  Each patch is 
//  performing all-to-all communication even if they don't have
//  radiometers.
//______________________________________________________________________
void
Radiometer::sched_radiometer( const LevelP& level,
                              SchedulerP& sched,
                              Task::WhichDW abskg_dw,
                              Task::WhichDW sigma_dw,
                              Task::WhichDW celltype_dw,
                              const int radCalc_freq )
{
  // return if it's a carryforward timestep
  // Temporal task scheduling
  if ( doCarryForward( radCalc_freq) ) {
    return;
  }
  
  vector<const Patch*> myPatches = getPatchSet( sched, level );
  bool hasRadiometers = false;
  int nGhostCells = SHRT_MAX;
  
  //__________________________________
  //  If this processor owns any patches with radiometers
  if( myPatches.size() !=  0 ){
    hasRadiometers = true;
    nGhostCells = SHRT_MAX; 
  }
  
  std::string taskname = "Radiometer::radiometer";
  Task *tsk;

  if ( RMCRTCommon::d_FLT_DBL == TypeDescription::double_type ){
    tsk = scinew Task( taskname, this, &Radiometer::radiometer< double >, abskg_dw, sigma_dw, celltype_dw, radCalc_freq, hasRadiometers );
  } else {
    tsk = scinew Task( taskname, this, &Radiometer::radiometer< float >, abskg_dw, sigma_dw, celltype_dw, radCalc_freq, hasRadiometers );
  }

  printSchedule( level,dbg,"Radiometer::sched_radiometer" );

  //__________________________________
  // Require an infinite number of ghost cells so you can access the entire domain.
  //
  // THIS IS VERY EXPENSIVE.  THIS EXPENSE IS INCURRED ON NON-CALCULATION TIMESTEPS,
  // ONLY REQUIRE THESE VARIABLES ON A CALCULATION TIMESTEPS.
  //
  // The taskgraph must be recompiled to detect a change in the conditional.
  // The taskgraph recompilation is activated from RMCRTCommon:doRecompileTaskgraph()
  Ghost::GhostType  gac  = Ghost::AroundCells;
  tsk->requires( abskg_dw ,    d_abskgLabel  ,   gac, SHRT_MAX);
  tsk->requires( sigma_dw ,    d_sigmaT4Label,   gac, SHRT_MAX);
  tsk->requires( celltype_dw , d_cellTypeLabel , gac, SHRT_MAX);

  tsk->modifies( d_VRFluxLabel );

  sched->addTask( tsk, level->eachPatch(), d_matlSet );
}

#if 0
//______________________________________________________________________
// Method: Schedule the virtual radiometer.  This task has both 
// temporal and spatial scheduling.
//______________________________________________________________________
void
Radiometer::sched_radiometer( const LevelP& level,
                              SchedulerP& sched,
                              Task::WhichDW abskg_dw,
                              Task::WhichDW sigma_dw,
                              Task::WhichDW celltype_dw,
                              const int radCalc_freq )
{

  // return if it's a carryforward timestep
  // Temporal task scheduling
  if ( doCarryForward( radCalc_freq) ) {
    return;
  }

  vector<const Patch*> myPatches = getPatchSet( sched, level );
  bool hasRadiometers = false;
  
  //__________________________________
  //  If this processor owns any patches with radiometers
  if( myPatches.size() !=  0 ){
    hasRadiometers = true;
  }

  std::string taskname = "Radiometer::radiometer";
  Task *tsk;
  
  if ( RMCRTCommon::d_FLT_DBL == TypeDescription::double_type ){
    tsk = scinew Task( taskname, this, &Radiometer::radiometer< double >, abskg_dw, sigma_dw, celltype_dw, radCalc_freq, hasRadiometers );
  } else {
    tsk = scinew Task( taskname, this, &Radiometer::radiometer< float >, abskg_dw, sigma_dw, celltype_dw, radCalc_freq, hasRadiometers );
  }
  
  tsk->setType(Task::Spatial);

  printSchedule( level,dbg,"Radiometer::sched_radiometer" );

  //__________________________________
  // Require an infinite number of ghost cells so you can access the entire domain.
  //
  // THIS IS VERY EXPENSIVE.  THIS EXPENSE IS INCURRED ON NON-CALCULATION TIMESTEPS,
  // ONLY REQUIRE THESE VARIABLES ON A CALCULATION TIMESTEPS.
  //
  // The taskgraph must be recompiled to detect a change in the conditional.
  // The taskgraph recompilation is activated from RMCRTCommon:doRecompileTaskgraph()
  dbg << "    sched_radiometer: adding requires for all-to-all variables " << endl;
  Ghost::GhostType  gac  = Ghost::AroundCells;
  tsk->requires( abskg_dw ,    d_abskgLabel  ,   gac, SHRT_MAX);
  tsk->requires( sigma_dw ,    d_sigmaT4_label,  gac, SHRT_MAX);
  tsk->requires( celltype_dw , d_cellTypeLabel , gac, SHRT_MAX);
  
  tsk->modifies( d_VRFluxLabel );

  // only schedule on the patches that contain radiometers
  // Spatial task scheduling
  PatchSet* radiometerPatchSet;
  radiometerPatchSet = scinew PatchSet();
  radiometerPatchSet->addReference();

  radiometerPatchSet->addAll( myPatches );
  
  sched->addTask( tsk, radiometerPatchSet, d_matlSet );
  
  if( radiometerPatchSet && radiometerPatchSet->removeReference() ){ 
    delete radiometerPatchSet;
  }
}
#endif
//______________________________________________________________________
// Method: The actual work of the ray tracer
//______________________________________________________________________
template < class T >
void
Radiometer::radiometer( const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw,
                        Task::WhichDW which_abskg_dw,
                        Task::WhichDW whichd_sigmaT4_dw,
                        Task::WhichDW which_celltype_dw,
                        const int radCalc_freq,
                        const bool hasRadiometers )
{
  // return if it's a carryforward timestep or there are no
  // radiometers.
  if ( doCarryForward( radCalc_freq) || hasRadiometers == false) {
    return;
  }

  const Level* level = getLevel(patches);

  //__________________________________
  //
  MTRand mTwister;

  // Determine the size of the domain.
  IntVector domainLo, domainHi;
  IntVector domainLo_EC, domainHi_EC;

  level->findInteriorCellIndexRange(domainLo, domainHi);     // excluding extraCells
  level->findCellIndexRange(domainLo_EC, domainHi_EC);       // including extraCells

  DataWarehouse* abskg_dw    = new_dw->getOtherDataWarehouse(which_abskg_dw);
  DataWarehouse* sigmaT4_dw  = new_dw->getOtherDataWarehouse(whichd_sigmaT4_dw);
  DataWarehouse* celltype_dw = new_dw->getOtherDataWarehouse(which_celltype_dw);

  constCCVariable< T > sigmaT4OverPi;
  constCCVariable< T > abskg;
  constCCVariable<int>    celltype;

  abskg_dw->getRegion(   abskg   ,       d_abskgLabel ,   d_matl , level, domainLo_EC, domainHi_EC );
  sigmaT4_dw->getRegion( sigmaT4OverPi , d_sigmaT4Label,  d_matl , level, domainLo_EC, domainHi_EC );
  celltype_dw->getRegion( celltype ,     d_cellTypeLabel, d_matl , level, domainLo_EC, domainHi_EC );

  //__________________________________
  // patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing Radiometer::radiometer");

    bool modifiesFlux= true;
    radiometerFlux < T > ( patch, level, new_dw, mTwister, sigmaT4OverPi, abskg, celltype, modifiesFlux );

  }  // end patch loop
}  // end radiometer

//______________________________________________________________________
//    Compute the radiometer flux.
//______________________________________________________________________
template< class T >
void
Radiometer::radiometerFlux( const Patch* patch,
                            const Level* level,
                            DataWarehouse* new_dw,
                            MTRand& mTwister,
                            constCCVariable< T > sigmaT4OverPi,
                            constCCVariable< T > abskg,
                            constCCVariable<int> celltype,
                            const bool modifiesFlux )
{

  if ( dbg.active() ){
    printTask(patch, dbg,"Doing Radiometer::radiometerFlux");
  }

  CCVariable< T > VRFlux;
  if( modifiesFlux ){
    new_dw->getModifiable( VRFlux,  d_VRFluxLabel,  d_matl, patch );
  }else{
    new_dw->allocateAndPut( VRFlux, d_VRFluxLabel, d_matl, patch );
    VRFlux.initialize( 0.0 );
  }

  unsigned long int size = 0;                   // current size of PathIndex
  Vector Dx = patch->dCell();                   // cell spacing
  double DyDx = Dx.y() / Dx.x();                //noncubic
  double DzDx = Dx.z() / Dx.x();                //noncubic

  IntVector lo = patch->getCellLowIndex();
  IntVector hi = patch->getCellHighIndex();

  IntVector VR_posLo  = level->getCellIndex( d_VRLocationsMin );
  IntVector VR_posHi  = level->getCellIndex( d_VRLocationsMax );

  if ( doesIntersect( VR_posLo, VR_posHi, lo, hi ) ){

    lo = Max(lo, VR_posLo);  // form an iterator for this patch
    hi = Min(hi, VR_posHi);  // this is an intersection

    for(CellIterator iter(lo,hi); !iter.done(); iter++){

      IntVector c = *iter;

      double sumI      = 0;
      double sumProjI  = 0;
      double sumI_prev = 0;

      //__________________________________
      // ray loop
      for (int iRay=0; iRay < d_nRadRays; iRay++){

        Vector ray_location;
        bool useCCRays = true;
        rayLocation(mTwister, c, DyDx, DzDx, useCCRays, ray_location);


        double cosVRTheta;
        Vector direction_vector;
        rayDirection_VR( mTwister, c, iRay, d_VR, DyDx, DzDx, direction_vector, cosVRTheta);

        // get the intensity for this ray
        updateSumI< T >( direction_vector, ray_location, c, Dx, sigmaT4OverPi, abskg, celltype, size, sumI, mTwister);

        sumProjI += cosVRTheta * (sumI - sumI_prev); // must subtract sumI_prev, since sumI accumulates intensity
                                                     // from all the rays up to that point
        sumI_prev = sumI;

      } // end VR ray loop

      //__________________________________
      //  Compute VRFlux
      VRFlux[c] = (T) sumProjI * d_VR.sldAngl/d_nRadRays;

    }  // end VR cell iterator
  }  // is radiometer on this patch
}

//______________________________________________________________________
//    Compute the Ray direction for Virtual Radiometer
//______________________________________________________________________
void
Radiometer::rayDirection_VR( MTRand& mTwister,
                             const IntVector& origin,
                             const int iRay,
                             VR_variables& VR,
                             const double DyDx,
                             const double DzDx,
                             Vector& direction_vector,
                             double& cosVRTheta)
{
  if( d_isSeedRandom == false ){
    mTwister.seed((origin.x() + origin.y() + origin.z()) * iRay +1);
  }

  // to help code readability
  double thetaRot   = VR.thetaRot;
  double deltaTheta = VR.deltaTheta;
  double psiRot     = VR.psiRot;
  double phiRot     = VR.phiRot;
  double range      = VR.range;

  // Generate two uniformly-distributed-over-the-solid-angle random numbers
  // Used in determining the ray direction
  double phi = 2 * M_PI * mTwister.randDblExc(); //azimuthal angle. Range of 0 to 2pi

  // This guarantees that the polar angle of the ray is within the delta_theta
  double VRTheta = acos(cos(deltaTheta)+range*mTwister.randDblExc());
  cosVRTheta = cos(VRTheta);

  // Convert to Cartesian x,y, and z represent the pre-rotated direction vector of a ray
  double x = sin(VRTheta)*cos(phi);
  double y = sin(VRTheta)*sin(phi);
  double z = cosVRTheta;

  // ++++++++ Apply the rotational offsets ++++++
  direction_vector[0] =                       // Why re-compute cos/sin(phiRot) when phiRot = 0? -Todd
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
}

//______________________________________________________________________
//  Return the patchSet that contains radiometers a process owns
//______________________________________________________________________
std::vector< const Patch* >
Radiometer::getPatchSet( SchedulerP& sched,
                        const LevelP& level )
{
  //__________________________________
  //
  //  that contain radiometers and that this processor owns
  vector< const Patch* > myPatches;
  LoadBalancer* lb = sched->getLoadBalancer();
  const PatchSet* procPatches = lb->getPerProcessorPatchSet(level);

  for(int m=0; m<procPatches->size(); m++) {
   const PatchSubset* patches = procPatches->getSubset(m);

    for (int p=0; p < patches->size(); p++){

      const Patch* patch = patches->get(p);

      IntVector lo = patch->getCellLowIndex();
      IntVector hi = patch->getCellHighIndex();

      IntVector VR_posLo  = level->getCellIndex( d_VRLocationsMin );
      IntVector VR_posHi  = level->getCellIndex( d_VRLocationsMax );

      if ( doesIntersect( VR_posLo, VR_posHi, lo, hi ) ){
        // add to patchSet if proc owns this patch
        int proc = lb->getPatchwiseProcessorAssignment(patch);
        if( proc ==  Uintah::Parallel::getMPIRank() ) {
          myPatches.push_back( patch );
        }
      }
    }
  }
  
  return myPatches;
}
//______________________________________________________________________
// Explicit template instantiations:

template void
Radiometer::radiometerFlux( const Patch*, const Level*, DataWarehouse*, MTRand&,
                            constCCVariable< double >, constCCVariable<double>, constCCVariable<int>,
                            const bool );
template void
Radiometer::radiometerFlux( const Patch*, const Level*, DataWarehouse*, MTRand&,
                            constCCVariable< float >, constCCVariable< float >, constCCVariable<int>,
                            const bool );
