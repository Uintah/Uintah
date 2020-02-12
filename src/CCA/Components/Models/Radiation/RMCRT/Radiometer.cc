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

//______________________________________________________________________
//
#include <CCA/Components/Models/Radiation/RMCRT/Radiometer.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/BBox.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Math/MersenneTwister.h>
#include <Core/Util/DOUT.hpp>

#include <fstream>

#include <include/sci_defs/uintah_testdefs.h.in>


//______________________________________________________________________
//  References:
//  1) I.L. Hunsaker, D.J. Glaze, J.N. Thornock, and P.J. Smith,
// "A New Model For Virtual Radiometers, Proceedings of the ASME 1012 International
//  Heat Transfer Conference, HT2012-58093, 2012.
//
//  2) I.L. Hunsaker, D.J. Glaze, J.N. Thornock, and P.J. Smith,
//  "Virtual Raiometers for Parallel Architectures," Technical Report,
//  http://hdl.handle.net/123456789/11198
//______________________________________________________________________

using namespace Uintah;
extern Dout g_ray_dbg;

//______________________________________________________________________
// Class: Constructor.
//______________________________________________________________________
//
Radiometer::Radiometer(const TypeDescription::Type FLT_DBL ) : RMCRTCommon( FLT_DBL)
{
  if ( FLT_DBL == TypeDescription::double_type ){
    d_VRFluxLabel      = VarLabel::create( "VRFlux",      CCVariable<double>::getTypeDescription() );
    d_VRIntensityLabel = VarLabel::create( "VRIntensity", CCVariable<double>::getTypeDescription() );
    proc0cout << "  - Using double implementation of the radiometer" << std::endl;
  } else {
    d_VRFluxLabel      = VarLabel::create( "VRFlux",      CCVariable<float>::getTypeDescription() );
    d_VRIntensityLabel = VarLabel::create( "VRIntensity", CCVariable<float>::getTypeDescription() );
    proc0cout << "  - Using float implementation of the radiometer" << std::endl;
  }
}

//______________________________________________________________________
// Method: Destructor
//______________________________________________________________________
//
Radiometer::~Radiometer()
{
  VarLabel::destroy( d_VRFluxLabel );
  VarLabel::destroy( d_VRIntensityLabel );
 
  for( auto iter  = d_radiometers.begin();iter != d_radiometers.end(); iter++){
    delete *iter;
  }
}

//______________________________________________________________________
// Method: Problem setup (access to input file information)
//______________________________________________________________________
void
Radiometer::problemSetup( const ProblemSpecP& prob_spec,
                          const ProblemSpecP& rmcrtps,
                          const GridP&        grid)
{
  ProblemSpecP rmcrt_ps = rmcrtps;

  for( ProblemSpecP rad_ps = rmcrt_ps->findBlock( "Radiometer" ); rad_ps != nullptr; rad_ps = rad_ps->findNextBlock( "Radiometer" ) ) {

    Vector notInitialized( DBL_MAX );
    Vector unitNormalVec( notInitialized );
    Vector normalVec( notInitialized );
    
    double viewAngle;
    int    nRays;
    Point  start;
    Point  end;
    
    rad_ps->require( "viewAngle"    ,    viewAngle );   // view angle of the radiometer in degrees
    rad_ps->require( "nRays"  ,          nRays );       // number of rays per radiometer
    rad_ps->require( "locationsMin" ,    start );       // minimum extent of radiometer(s) in physical units
    rad_ps->require( "locationsMax" ,    end );         // maximum extent of radiometer(s) in physical units
    
    rad_ps->get( "unitNormalVector",  unitNormalVec );  // non-dimensional orientation of radiometer
    rad_ps->get( "normalVector",      normalVec);       // radiometer orientation in cartesian coordinates

    if( normalVec != notInitialized ){
      unitNormalVec = normalVec/normalVec.length();
    }
    
    //__________________________________
    //  Warnings and bulletproofing

    BBox compDomain;
    grid->getSpatialRange(compDomain);

    Point min = compDomain.min();
    Point max = compDomain.max();

    if(start.x() < min.x() || start.y() < min.y() || start.z() < min.z() ||
       end.x() > max.x()   || end.y() > max.y()   || end.z() > max.z() ) {
      std::ostringstream warn;
      warn << "\n ERROR:Radiometer::problemSetup: the radiometer that you've specified " << start
           << " " << end << " begins or ends outside of the computational domain. \n" << std::endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    if( start.x() > end.x() || start.y() > end.y() || start.z() > end.z() ) {
      std::ostringstream warn;
      warn << "\n ERROR:Radiometer::problemSetup: the radiometer that you've specified " << start
           << " " << end << " the starting point is > than the ending point \n" << std::endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    // you need at least one cell that's a radiometer
    if( start.x() == end.x() || start.y() == end.y() || start.z() == end.z() ){
      std::ostringstream warn;
      warn << "\n ERROR:Radiometer::problemSetup: The specified radiometer has the same "
           << "starting and ending points.  All of the directions must differ by one cell\n "
           << "                                start: " << start << " end: " << end << std::endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    for(int L = 0; L<grid->numLevels(); L++){
      LevelP level = grid->getLevel(L);
      IntVector lo = level->getCellIndex( start );
      IntVector hi = level->getCellIndex( end );
      IntVector nCells = hi - lo;

      proc0cout << "  - radiometer: lower cell " << lo << " upper cell " << hi << " nCells " << nCells << "\n";

      if ( nCells.x() <= 0 || nCells.y() <= 0 || nCells.z() <= 0){
        std::ostringstream warn;
        warn << "\n ERROR:Radiometer::problemSetup: The specified radiometer has the same "
             << "starting and ending points.  There must be at least 1 cell between locationMin & locationMax in each direction:\n "
             << "                                startCell: " << lo << " endCell: " << hi;
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
    }

    //__________________________________
    // 
    if ( unitNormalVec == notInitialized && normalVec == notInitialized){
      std::ostringstream warn;
      warn << "\n ERROR:Radiometer::problemSetup: You must specify either: "
           << "  <unitNormalVector> or <normalVector> \n"
           << " to set the direction of the radiometer";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    //__________________________________
    //
    if ( viewAngle > 360 ){
      std::ostringstream warn;
      warn << "ERROR:  VRViewAngle ("<< viewAngle <<") exceeds the maximum acceptable value of 360 degrees." << std::endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    if ( nRays < int(15 + pow(5.4, viewAngle/40) ) ){
      proc0cout << "    WARNING Number of radiometer rays:  ("<< nRays <<") is less than the recommended number of ("<< int(15 + pow(5.4, viewAngle/40) ) <<"). Errors will exceed 1%. " << std::endl;
    }

    //__________________________________
    //    compute the rotation angles
    for(int d = 0; d<3; d++){
      if( unitNormalVec[d] == 0 ){      // WARNING WARNING this conditional only works for integers, not doubles, and should be fixed.
        unitNormalVec[d] = 1e-16;      // to avoid divide by 0.
      }
    }

    //__________________________________
    //  Convert the user specified radiometer normal vector into three axial
    //  rotations about the x, y, and z axes.
    //  Each rotation is counterclockwise when the observer is looking from the
    //  positive axis about which the rotation is occurring. d
    //  This follows eqs, 14-17 in reference 1.
    //
    //  counter-clockwise rotation:
    // phi_rotate  :   x axis
    // theta_rotate:   y axis
    // xi_rotate   :   z axis

    //  phi_rotate is always  0. There will never be a need for a rotation about the x axis. All
    //  possible rotations can be accomplished using the other two.
    double phi_rotate = 0;

    double theta_rotate = acos( unitNormalVec[2] / unitNormalVec.length() );

    double xi_rotate  = acos( unitNormalVec[0] / sqrt( unitNormalVec[0]*unitNormalVec[0] + unitNormalVec[1]*unitNormalVec[1] ) );

    // The rotations must be adjusted if the x and y components of the normal vector
    // are in the 3rd or 4th quadrants due to the constraints on arccos
    if ( unitNormalVec[0] < 0 && unitNormalVec[1] < 0 ){       // quadrant 3
      xi_rotate = (M_PI/2 + xi_rotate);
    }
    if ( unitNormalVec[0] > 0 && unitNormalVec[1] < 0 ){       // quadrant 4
      xi_rotate = (2*M_PI - xi_rotate);
    }

    proc0cout << "                xi_rotate: " << phi_rotate << " theta_rotate: " << theta_rotate << " xi_rotate: " << xi_rotate << std::endl;

    //__________________________________
    //
    double theta_viewAngle = viewAngle/360*M_PI;       // divides view angle by two and converts to radians
    double range           = 1 - cos(theta_viewAngle); // cos(0) to cos(theta_viewAngle) gives the range of possible vals
    double solidAngle      = 2*M_PI*range;             // the solid angle that the radiometer can view

    // Store in the vector
    radiometer* r   = scinew radiometer;
    r->locationsMin    = start;
    r->locationsMax    = end;
    r->nRays           = nRays;
    r->solidAngle      = solidAngle;
    r->theta_viewAngle = theta_viewAngle;
    r->range           = range;
    r->phi_rotate      = phi_rotate;
    r->theta_rotate    = theta_rotate;
    r->xi_rotate       = xi_rotate;

    d_radiometers.push_back( r );
  }  // loop over radiometers
  
  
  //__________________________________
  // bulletproofing
  ProblemSpecP root_ps = rmcrt_ps->getRootNode();

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
Radiometer::sched_initialize_VRFlux( const LevelP& level,
                                     SchedulerP& sched )
{

  std::string taskname = "Radiometer::initialize_VRFlux";

  Task* tsk = nullptr;
  if ( RMCRTCommon::d_FLT_DBL == TypeDescription::double_type ){
    tsk= scinew Task( taskname, this, &Radiometer::initialize_VRFlux< double > );
  }else{
    tsk= scinew Task( taskname, this, &Radiometer::initialize_VRFlux< float > );
  }

  printSchedule(level, g_ray_dbg, taskname);

  tsk->computes( d_VRFluxLabel );
  tsk->computes( d_VRIntensityLabel );
  sched->addTask( tsk, level->eachPatch(), d_matlSet, RMCRTCommon::TG_RMCRT );
}

//______________________________________________________________________
//
//______________________________________________________________________
template< class T >
void
Radiometer::initialize_VRFlux( const ProcessorGroup *,
                               const PatchSubset    * patches,
                               const MaterialSubset * matls,
                               DataWarehouse        * old_dw,
                               DataWarehouse        * new_dw )
{
  //__________________________________
  //  Initialize the flux and intensity.
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);

    printTask(patches, patch, g_ray_dbg, "Doing Radiometer::initialize_VRFlux");

    CCVariable< T > flux;
    CCVariable< T > intensity;
    new_dw->allocateAndPut( flux,      d_VRFluxLabel,       d_matl, patch );
    new_dw->allocateAndPut( intensity, d_VRIntensityLabel, d_matl, patch );

    flux.initialize( 0.0 );
    intensity.initialize( 0.0 );
  }
}

//______________________________________________________________________
// Method: Schedule the virtual radiometer.  This task has both
// temporal and spatial scheduling.
//______________________________________________________________________
void
Radiometer::sched_radiometer( const LevelP & level,
                              SchedulerP   & sched,
                              Task::WhichDW notUsed,
                              Task::WhichDW sigma_dw,
                              Task::WhichDW celltype_dw )
{
  //__________________________________
  //  There has to be a value of VRFlux VRIntensity on all patches for output
  sched_initialize_VRFlux( level, sched );


  //__________________________________
  //  Compute VRFlux on subset of patches
  // only schedule on the patches that contain radiometers - Spatial task scheduling
  //   we want a PatchSet like: { {19}, {22}, {25} } (singleton subsets like level->eachPatch())
  //     NOT -> { {19,22,25} }, as one proc isn't guaranteed to own the entire, 3-element subset.
  PatchSet* radiometerPatchSet = scinew PatchSet();
  radiometerPatchSet->addReference();
  getPatchSet(sched, level, d_radiometers, radiometerPatchSet);

  int L = level->getIndex();
  Task::WhichDW abskg_dw = get_abskg_whichDW( L, d_abskgLabel );

  std::string taskname = "Radiometer::radiometerTask";
  Task *tsk;

  if (RMCRTCommon::d_FLT_DBL == TypeDescription::double_type) {
    tsk = scinew Task(taskname, this, &Radiometer::radiometerTask<double>, abskg_dw, sigma_dw, celltype_dw);
  }
  else {
    tsk = scinew Task(taskname, this, &Radiometer::radiometerTask<float>, abskg_dw, sigma_dw, celltype_dw);
  }

  tsk->setType(Task::Spatial);

  printSchedule(level, g_ray_dbg, "Radiometer::sched_radiometer on patches ");

  //__________________________________
  // Require an infinite number of ghost cells so you can access the entire domain.
  DOUT(g_ray_dbg, "    sched_radiometer: adding requires for all-to-all variables ");

  Ghost::GhostType gac = Ghost::AroundCells;

  tsk->requires( abskg_dw,    d_abskgLabel,    gac, SHRT_MAX);
  tsk->requires( sigma_dw,    d_sigmaT4Label,  gac, SHRT_MAX);
  tsk->requires( celltype_dw, d_cellTypeLabel, gac, SHRT_MAX);

  tsk->modifies( d_VRFluxLabel );
  tsk->modifies( d_VRIntensityLabel );

  sched->addTask(tsk, radiometerPatchSet, d_matlSet, RMCRTCommon::TG_RMCRT);

  if (radiometerPatchSet && radiometerPatchSet->removeReference()) {
    delete radiometerPatchSet;
  }
}

//______________________________________________________________________
// Method: The actual work of the radiometer
//______________________________________________________________________
template < class T >
void
Radiometer::radiometerTask( const ProcessorGroup  * pg,
                            const PatchSubset     * patches,
                            const MaterialSubset  * matls,
                            DataWarehouse         * old_dw,
                            DataWarehouse         * new_dw,
                            Task::WhichDW which_abskg_dw,
                            Task::WhichDW which_sigmaT4_dw,
                            Task::WhichDW which_celltype_dw )
{
  const Level* level = getLevel(patches);
  MTRand mTwister;

  DataWarehouse* abskg_dw    = new_dw->getOtherDataWarehouse(which_abskg_dw);
  DataWarehouse* sigmaT4_dw  = new_dw->getOtherDataWarehouse(which_sigmaT4_dw);
  DataWarehouse* celltype_dw = new_dw->getOtherDataWarehouse(which_celltype_dw);

  constCCVariable< T > sigmaT4OverPi;
  constCCVariable< T > abskg;
  constCCVariable<int> celltype;

  abskg_dw->getLevel(    abskg        , d_abskgLabel   , d_matl, level );
  sigmaT4_dw->getLevel(  sigmaT4OverPi, d_sigmaT4Label , d_matl, level );
  celltype_dw->getLevel( celltype     , d_cellTypeLabel, d_matl, level );

  //__________________________________
  // patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);

    bool modifiesFlux= true;
    radiometerFlux < T > ( patch, level, new_dw, mTwister, sigmaT4OverPi, abskg, celltype, modifiesFlux );
  }
}

//______________________________________________________________________
//    Compute the radiometer flux.
//______________________________________________________________________
template< class T >
void
Radiometer::radiometerFlux( const Patch       *  patch,
                            const Level       *  level,
                            DataWarehouse     *  new_dw,
                            MTRand            &  mTwister,
                            constCCVariable< T > sigmaT4OverPi,
                            constCCVariable< T > abskg,
                            constCCVariable<int> celltype,
                            const bool modifiesFlux )
{
  printTask(patch, g_ray_dbg, "Doing Radiometer::radiometerFlux");

  CCVariable< T > VRFlux;
  CCVariable< T > intensity;
  if( modifiesFlux ){
    new_dw->getModifiable( VRFlux,     d_VRFluxLabel,      d_matl, patch );
    new_dw->getModifiable( intensity,  d_VRIntensityLabel, d_matl, patch );
  }else{
    new_dw->allocateAndPut( VRFlux,    d_VRFluxLabel,      d_matl, patch );
    new_dw->allocateAndPut( intensity, d_VRIntensityLabel, d_matl, patch );
    VRFlux.initialize( 0.0 );
    intensity.initialize( 0.0 );
  }

  unsigned long int size = 0;                   // current size of PathIndex
  Vector dx     = patch->dCell();               // cell spacing
  IntVector pLo = patch->getCellLowIndex();
  IntVector pHi = patch->getCellHighIndex();

  //__________________________________
  // loop over each radiometer
  for (unsigned int r =0 ; r < d_radiometers.size(); r++) {

    const radiometer* rad = d_radiometers[r];
    IntVector rLo  = level->getCellIndex( rad->locationsMin );
    IntVector rHi  = level->getCellIndex( rad->locationsMax );

    if ( doesIntersect( rLo, rHi, pLo, pHi ) ){

      IntVector lo = Max( pLo, rLo );  // form an iterator for this patch
      IntVector hi = Min( pHi, rHi );  // this is an intersection

      for(CellIterator iter(lo,hi); !iter.done(); iter++){

        IntVector c = *iter;

        double sumI      = 0;
        double sumProjI  = 0;
        double sumI_prev = 0;
        Point CC_pos     = level->getCellPosition(c);

        const int nRays = rad->nRays;
        //__________________________________
        // ray loop
        for (int iRay=0; iRay < nRays; iRay++){

          Vector rayOrigin;
          bool useCCRays = true;
          ray_Origin( mTwister, CC_pos, dx, useCCRays, rayOrigin);


          double cosVRTheta;
          Vector direction_vector;
          rayDirection_VR( mTwister, c, iRay, rad, direction_vector, cosVRTheta);

          // get the intensity for this ray
          updateSumI< T >(level, direction_vector, rayOrigin, c, dx, sigmaT4OverPi, abskg, celltype, size, sumI, mTwister);

          sumProjI += cosVRTheta * (sumI - sumI_prev); // must subtract sumI_prev, since sumI accumulates intensity
                                                       // from all the rays up to that point
          sumI_prev = sumI;

        } // end ray loop

        //__________________________________
        //  Compute  flux and intensity
        const double solidAngle = rad->solidAngle;
        VRFlux[c]    = (T) sumProjI * solidAngle/nRays;
        intensity[c] = (T) sumProjI/nRays;
      }  // end VR cell iterator
    }  // is radiometer on this patch
  }
}

//______________________________________________________________________
//    Compute the ray direction
//______________________________________________________________________
void
Radiometer::rayDirection_VR( MTRand           & mTwister,
                             const IntVector  & origin,
                             const int          iRay,
                             const radiometer * rad,
                             Vector           & direction_vector,
                             double           & cosTheta_ray)
{
  if( d_isSeedRandom == false ){
    mTwister.seed((origin.x() + origin.y() + origin.z()) * iRay +1);
  }

  // to help code readability
  double theta_rotate    = rad->theta_rotate;
  double theta_viewAngle = rad->theta_viewAngle;
  double xi_rotate       = rad->xi_rotate;
  double phi_rotate      = rad->phi_rotate;
  double range           = rad->range;
  double R1              = mTwister.randDblExc();
  double R2              = mTwister.randDblExc();

  // Eq. 11, ref 1.
  double phi_ray = 2 * M_PI * R1; //azimuthal angle. Range of 0 to 2pi

  // This guarantees that the polar angle of the ray is within the theta_viewAngle
  // Eq. 12, ref 1.
  double theta_ray = acos( cos(theta_viewAngle) + range * R2 );
  cosTheta_ray     = cos(theta_ray);

  // This is the pre-rotated direction vector of the ray in cartesian coordinates, eq. 18 ref 1.
  double x = sin(theta_ray) * cos(phi_ray);
  double y = sin(theta_ray) * sin(phi_ray);
  double z = cosTheta_ray;

  // Equation 13, ref 1.
  direction_vector[0] =
    x * cos(theta_rotate) * cos(xi_rotate) +
    y * (-cos(phi_rotate) * sin(xi_rotate) + sin(phi_rotate) * sin(theta_rotate) * cos(xi_rotate)) +
    z * ( sin(phi_rotate) * sin(xi_rotate) + cos(phi_rotate) * sin(theta_rotate) * cos(xi_rotate));

  direction_vector[1] =
    x * cos(theta_rotate) * sin(xi_rotate) +
    y * ( cos(phi_rotate) * cos(xi_rotate) + sin(phi_rotate) * sin(theta_rotate) * sin(xi_rotate)) +
    z * (-sin(phi_rotate) * cos(xi_rotate) + cos(phi_rotate) * sin(theta_rotate) * sin(xi_rotate));

  direction_vector[2] =
    x * -sin(theta_rotate) +
    y * sin(phi_rotate)*cos(theta_rotate) +
    z * cos(phi_rotate)*cos(theta_rotate);

  //std::cout << " direction_vector: " << direction_vector << " x: " << x << " y: " << y << " z: " << z << std::endl;

}

//______________________________________________________________________
//  Return the patchSet that contains radiometers
//______________________________________________________________________
void
Radiometer::getPatchSet( SchedulerP   & sched,
                         const LevelP & level,
                         std::vector<radiometer* > radiometers,
                         PatchSet     * ps )
{
  IntVector L_lo;
  IntVector L_hi;
  level->findInteriorCellIndexRange( L_lo, L_hi );
  std::set<const Patch*> rad_patches;

  for (unsigned int r =0 ; r < d_radiometers.size(); r++) {

    const radiometer* rad = d_radiometers[r];
    IntVector rLo  = level->getCellIndex( rad->locationsMin );
    IntVector rHi  = level->getCellIndex( rad->locationsMax );


    if( !doesIntersect (rLo, rHi, L_lo, L_hi ) ){
      std::ostringstream warn;
      warn << "\n ERROR:Radiometer::problemSetup: The VR locations:\n"
           << "     VR min " << rad->locationsMin  << "  cell index: " << rLo << "\n"
           << "     VR max " << rad->locationsMax << "  cell index: " << rHi << "\n"
           << "     Number of cells: " << rHi - rLo << "\n"
           << " do not overlap with the domain, " << L_lo << " -> " << L_hi << "\n"
           << " There must be at least 1 cell in each direction" ;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    //__________________________________
    // find patches that contain radiometers

    LoadBalancer * lb = sched->getLoadBalancer();
    const PatchSet * procPatches = lb->getPerProcessorPatchSet(level);

    for (int m = 0; m < procPatches->size(); m++) {
      const PatchSubset* patches = procPatches->getSubset(m);

      for (int p = 0; p < patches->size(); p++) {
        const Patch* patch = patches->get(p);
        IntVector pLo = patch->getCellLowIndex();
        IntVector pHi = patch->getCellHighIndex();

        if (doesIntersect(rLo, rHi, pLo, pHi)) {
          rad_patches.emplace(patch);
        }
      }
    }
  }  // loop over radiometers
  
  for (auto it=rad_patches.begin(); it!=rad_patches.end(); ++it){
    ps->add( *it );
  }

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
