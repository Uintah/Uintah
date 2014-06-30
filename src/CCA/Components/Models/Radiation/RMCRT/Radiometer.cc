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
Radiometer::Radiometer()
{
  d_sigmaT4_label  = VarLabel::create( "sigmaT4",  CCVariable<double>::getTypeDescription() );              
  d_VRFluxLabel    = VarLabel::create( "VRFlux",   CCVariable<double>::getTypeDescription() );              
  d_cellTypeLabel  = VarLabel::create( "cellType", CCVariable<int>::getTypeDescription() );                 
   
  d_matlSet = 0;       
  d_gac     = Ghost::AroundCells;      
  d_gn      = Ghost::None;             
}

//______________________________________________________________________
// Method: Destructor
//______________________________________________________________________
//
Radiometer::~Radiometer()
{
  VarLabel::destroy( d_sigmaT4_label );
  VarLabel::destroy( d_VRFluxLabel );
  VarLabel::destroy( d_cellTypeLabel );

  if(d_matlSet && d_matlSet->removeReference()) {
    delete d_matlSet;
  }
}

//______________________________________________________________________
//  Logic for determing when to carry forward
//______________________________________________________________________
bool 
Radiometer::doCarryForward( const int timestep,
                            const int radCalc_freq){
  bool test = (timestep%radCalc_freq != 0 && timestep != 1);
  return test;
}

//______________________________________________________________________
// Method: Problem setup (access to input file information)
//______________________________________________________________________
void
Radiometer::problemSetup( const ProblemSpecP& prob_spec,
                          const ProblemSpecP& rmcrtps,
                          SimulationStateP&   sharedState) 
{

  d_sharedState = sharedState;
  ProblemSpecP rmcrt_ps = rmcrtps;
  Vector orient;
  rmcrt_ps->getWithDefault( "Threshold" ,       d_threshold ,      0.01 );             // When to terminate a ray
  rmcrt_ps->getWithDefault( "randomSeed",       d_isSeedRandom,    true );             // random or deterministic seed. 
  rmcrt_ps->getWithDefault( "StefanBoltzmann",  d_sigma,           5.67051e-8);        // Units are W/(m^2-K)
  rmcrt_ps->getWithDefault( "VRViewAngle"    ,  d_viewAng,         180 );              // view angle of the radiometer in degrees
  rmcrt_ps->getWithDefault( "VROrientation"  ,  orient,          Vector(0,0,1) );       // Normal vector of the radiometer orientation (Cartesian)
  rmcrt_ps->getWithDefault( "nRadRays"  ,       d_nRadRays ,       1000 );
  rmcrt_ps->getWithDefault( "sigmaScat"  ,      d_sigmaScat  ,      0 );                // scattering coefficient
  rmcrt_ps->getWithDefault( "allowReflect"   ,  d_allowReflect,     true );             // Allow for ray reflections. Make false for DOM comparisons.  
  rmcrt_ps->get(            "VRLocationsMin" ,  d_VRLocationsMin );                     // minimum extent of the string or block of virtual radiometers in physical units
  rmcrt_ps->get(            "VRLocationsMax" ,  d_VRLocationsMax );                     // maximum extent


  //__________________________________
  //  Warnings and bulletproofing
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

  if (d_virtRad && d_nRadRays < int(15 + pow(5.4, d_viewAng/40) ) ){
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
  d_VR.deltaTheta = deltaTheta;
  d_VR.range      = range;
  d_sigma_over_pi = d_sigma/M_PI;

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
// Register the material index and label names
//______________________________________________________________________
void
Radiometer::registerVarLabels(int   matlIndex,
                              const VarLabel* abskg,
                              const VarLabel* absorp,
                              const VarLabel* temperature,
                              const VarLabel* celltype)
{
  d_matl             = matlIndex;
  d_abskgLabel       = abskg;
  d_temperatureLabel = temperature;
  d_cellTypeLabel    = celltype;

  //__________________________________
  //  define the materialSet
  d_matlSet = scinew MaterialSet();
  vector<int> m;
  m.push_back(matlIndex);
  d_matlSet->addAll(m);
  d_matlSet->addReference();
}


//______________________________________________________________________
//
//______________________________________________________________________
void
Radiometer::sched_sigmaT4( const LevelP& level, 
                           SchedulerP& sched,
                           Task::WhichDW temp_dw,
                           const int radCalc_freq,
                           const bool includeEC )
{
  std::string taskname = "Radiometer::sigmaT4";
  Task* tsk= scinew Task( taskname, this, &Radiometer::sigmaT4, temp_dw, radCalc_freq, includeEC );

  printSchedule(level,dbg,taskname);
  
  tsk->requires( temp_dw, d_temperatureLabel,  d_gn, 0 );
  tsk->requires( Task::OldDW, d_sigmaT4_label, d_gn, 0 ); 
  tsk->computes(d_sigmaT4_label); 

  sched->addTask( tsk, level->eachPatch(), d_matlSet );
}
//______________________________________________________________________
// Compute total intensity over all wave lengths (sigma * Temperature^4/pi)
//______________________________________________________________________
void
Radiometer::sigmaT4( const ProcessorGroup*,
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
    printTask( patches, patches->get(0), dbg, "Doing Radiometer::sigmaT4 carryForward (sigmaT4)" );
    
    new_dw->transferFrom( old_dw, d_sigmaT4_label, patches, matls, true );
    return;
  }
  
  //__________________________________
  //  do the work
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing Radiometer::sigmaT4");

    double sigma_over_pi = d_sigma/M_PI;

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

//______________________________________________________________________
// Method: Schedule the virtual radiometer
//______________________________________________________________________
void
Radiometer::sched_radiometer( const LevelP& level, 
                              SchedulerP& sched,
                              Task::WhichDW abskg_dw,
                              Task::WhichDW sigma_dw,
                              Task::WhichDW celltype_dw,
                              const int radCalc_freq )
{
  std::string taskname = "Radiometer::radiometer";
  Task *tsk;
  tsk = scinew Task( taskname, this, &Radiometer::radiometer, abskg_dw, sigma_dw, celltype_dw, radCalc_freq );

  printSchedule(level,dbg,taskname);

  // require an infinite number of ghost cells so you can access the entire domain.
  Ghost::GhostType  gac  = Ghost::AroundCells;
  tsk->requires( abskg_dw ,    d_abskgLabel  ,   gac, SHRT_MAX);
  tsk->requires( sigma_dw ,    d_sigmaT4_label,  gac, SHRT_MAX);
  tsk->requires( celltype_dw , d_cellTypeLabel , gac, SHRT_MAX);
  
  tsk->requires(Task::OldDW, d_VRFluxLabel, d_gn, 0);
  tsk->computes( d_VRFluxLabel );
  sched->addTask( tsk, level->eachPatch(), d_matlSet );
  
}

//---------------------------------------------------------------------------
// Method: The actual work of the ray tracer
//______________________________________________________________________
void
Radiometer::radiometer( const ProcessorGroup* pc,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw,
                        Task::WhichDW which_abskg_dw,
                        Task::WhichDW whichd_sigmaT4_dw,
                        Task::WhichDW which_celltype_dw,
                        const int radCalc_freq )
{ 

  const Level* level = getLevel(patches);
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();
  
  if ( doCarryForward( timestep, radCalc_freq) ) {
    printTask(patches,patches->get(0), dbg,"Doing Radiometer::rayTrace (carryForward)");
    bool replaceVar = true;
    new_dw->transferFrom( old_dw, d_VRFluxLabel, patches, matls, replaceVar );
    return;
  }
  
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

  constCCVariable<double> sigmaT4OverPi;
  constCCVariable<double> abskg;
  constCCVariable<int>    celltype;

  abskg_dw->getRegion(   abskg   ,       d_abskgLabel ,   d_matl , level, domainLo_EC, domainHi_EC);
  sigmaT4_dw->getRegion( sigmaT4OverPi , d_sigmaT4_label, d_matl , level, domainLo_EC, domainHi_EC);
  celltype_dw->getRegion( celltype ,     d_cellTypeLabel, d_matl , level, domainLo_EC, domainHi_EC);

  //__________________________________
  // patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing Radiometer::rayTrace");

    CCVariable<double> VRFlux;
    new_dw->allocateAndPut( VRFlux, d_VRFluxLabel, d_matl, patch );
    VRFlux.initialize( 0.0 );

    unsigned long int size = 0;                   // current size of PathIndex
    Vector Dx = patch->dCell();                   // cell spacing
    double DyDx = Dx.y() / Dx.x();                //noncubic
    double DzDx = Dx.z() / Dx.x();                //noncubic 
    
    //______________________________________________________________________
    //           R A D I O M E T E R 
    //______________________________________________________________________
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
          updateSumI( direction_vector, ray_location, c, Dx, sigmaT4OverPi, abskg, celltype, size, sumI, mTwister);

          sumProjI += cosVRTheta * (sumI - sumI_prev); // must subtract sumI_prev, since sumI accumulates intensity
                                                       // from all the rays up to that point
          sumI_prev = sumI;

        } // end VR ray loop

        //__________________________________
        //  Compute VRFlux
        VRFlux[c] = sumProjI * d_VR.sldAngl/d_nRadRays;

      }  // end VR cell iterator
    }  // end on Patch 
  }  //end patch loop
}  // end radiometer

//______________________________________________________________________
//
//______________________________________________________________________
void 
Radiometer::findStepSize(int step[],
                         bool sign[],                             
                         const Vector& inv_direction_vector){     
  // get new step and sign
  for ( int d= 0; d<3; d++){
    if (inv_direction_vector[d]>0){
      step[d] = 1;
      sign[d] = 1;
    }
    else{
      step[d] = -1;
      sign[d] = 0;
    }
  }
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
//  Compute the ray direction
//______________________________________________________________________
Vector 
Radiometer::findRayDirection(MTRand& mTwister,
                             const bool isSeedRandom,
                             const IntVector& origin,
                             const int iRay )
{
  if( isSeedRandom == false ){
    mTwister.seed((origin.x() + origin.y() + origin.z()) * iRay +1);
  }

  // Random Points On Sphere
  double plusMinus_one = 2.0 * mTwister.randDblExc() - 1.0 + DBL_EPSILON;  // add fuzz to avoid inf in 1/dirVector
  double r = sqrt(1.0 - plusMinus_one * plusMinus_one);     // Radius of circle at z
  double theta = 2.0 * M_PI * mTwister.randDblExc();        // Uniform betwen 0-2Pi

  Vector direction_vector;
  direction_vector[0] = r*cos(theta);                       // Convert to cartesian
  direction_vector[1] = r*sin(theta);
  direction_vector[2] = plusMinus_one;
  
  return direction_vector;
}
//______________________________________________________________________
//  Compute the physical location of the ray
//______________________________________________________________________
void 
Radiometer::rayLocation( MTRand& mTwister,
                         const IntVector origin,
                         const double DyDx, 
                         const double DzDx,
                         const bool useCCRays,
                         Vector& location)
{
  if( useCCRays == false ){
    location[0] =   origin[0] +  mTwister.rand() ;
    location[1] =   origin[1] +  mTwister.rand() * DyDx ;
    location[2] =   origin[2] +  mTwister.rand() * DzDx ;
  }else{
    location[0] =   origin[0] +  0.5 ;
    location[1] =   origin[1] +  0.5 * DyDx ;
    location[2] =   origin[2] +  0.5 * DzDx ;
  }
}

//______________________________________________________________________
//    Core function:  
//______________________________________________________________________
void 
Radiometer::reflect(double& fs,
                    IntVector& cur,
                    IntVector& prevCell,
                    const double abskg,
                    bool& in_domain,
                    int& step,
                    bool& sign,
                    double& ray_direction)
{
  fs = fs * (1 - abskg);

  //put cur back inside the domain
  cur = prevCell;
  in_domain = true;

  // apply reflection condition
  step *= -1;                // begin stepping in opposite direction
  sign = (sign==1) ? 0 : 1;  //  swap sign from 1 to 0 or vice versa
  ray_direction *= -1;
  //dbg2 << " REFLECTING " << endl;
}

//______________________________________________________________________
//    Core function:  Integrate the intensity
//______________________________________________________________________
void 
Radiometer::updateSumI ( Vector& ray_direction,
                         Vector& ray_location,
                         const IntVector& origin,
                         const Vector& Dx,
                         constCCVariable<double>& sigmaT4OverPi,
                         constCCVariable<double>& abskg,
                         constCCVariable<int>& celltype,
                         unsigned long int& nRaySteps,
                         double& sumI,
                         MTRand& mTwister)

{
/*`==========TESTING==========*/
#if DEBUG == 1
  printf("        updateSumI: [%d,%d,%d] ray_dir [%g,%g,%g] ray_loc [%g,%g,%g]\n", origin.x(), origin.y(), origin.z(),ray_direction.x(), ray_direction.y(), ray_direction.z(), ray_location.x(), ray_location.y(), ray_location.z());
#endif 
/*===========TESTING==========`*/
  
  IntVector cur = origin;
  IntVector prevCell = cur;
  // Step and sign for ray marching
   int step[3];                                          // Gives +1 or -1 based on sign
   bool sign[3];
   
   Vector inv_ray_direction = Vector(1.0)/ray_direction;
   findStepSize(step, sign, inv_ray_direction);
   Vector D_DxRatio(1, Dx.y()/Dx.x(), Dx.z()/Dx.x() );

   Vector tMax;         // (mixing bools, ints and doubles)
   tMax.x( (origin[0] + sign[0]                - ray_location[0]) * inv_ray_direction[0] );
   tMax.y( (origin[1] + sign[1] * D_DxRatio[1] - ray_location[1]) * inv_ray_direction[1] );
   tMax.z( (origin[2] + sign[2] * D_DxRatio[2] - ray_location[2]) * inv_ray_direction[2] );

   //Length of t to traverse one cell
   Vector tDelta = Abs(inv_ray_direction) * D_DxRatio;
   
   //Initializes the following values for each ray
   bool in_domain     = true;
   double tMax_prev   = 0;
   double intensity   = 1.0;
   double fs          = 1.0;
   int nReflect       = 0;                 // Number of reflections
   double optical_thickness      = 0;
   double expOpticalThick_prev   = 1.0;


#ifdef RAY_SCATTER
   double scatCoeff = d_sigmaScat;          //[m^-1]  !! HACK !! This needs to come from data warehouse
   if (scatCoeff == 0) scatCoeff = 1e-99;  // avoid division by zero

   // Determine the length at which scattering will occur
   // See CCA/Components/Arches/RMCRT/PaulasAttic/MCRT/ArchesRMCRT/ray.cc
   double scatLength = -log(mTwister.randDblExc() ) / scatCoeff;
   double curLength = 0;
#endif

   //+++++++Begin ray tracing+++++++++++++++++++
   //Threshold while loop
   while ( intensity > d_threshold ){
    
     DIR face = NONE;

     while (in_domain){

       prevCell = cur;
       double disMin = -9;          // Represents ray segment length.
       
       double abskg_prev = abskg[prevCell];  // optimization
       double sigmaT4OverPi_prev = sigmaT4OverPi[prevCell];
       //__________________________________
       //  Determine which cell the ray will enter next
       if ( tMax[0] < tMax[1] ){        // X < Y
         if ( tMax[0] < tMax[2] ){      // X < Z
           face = X;
         } else {
           face = Z;
         }
       } else {
         if( tMax[1] < tMax[2] ){       // Y < Z
           face = Y;
         } else {
           face = Z;
         }
       }

       //__________________________________
       //  update marching variables
       cur[face]  = cur[face] + step[face];
       disMin     = (tMax[face] - tMax_prev);
       tMax_prev  = tMax[face];
       tMax[face] = tMax[face] + tDelta[face];

       ray_location[0] = ray_location[0] + (disMin  * ray_direction[0]);
       ray_location[1] = ray_location[1] + (disMin  * ray_direction[1]);
       ray_location[2] = ray_location[2] + (disMin  * ray_direction[2]);
  
 /*`==========TESTING==========*/
#if DEBUG == 1
if(origin.x() == 0 && origin.y() == 0 && origin.z() ==0){
    printf( "            cur [%d,%d,%d] prev [%d,%d,%d] ", cur.x(), cur.y(), cur.z(), prevCell.x(), prevCell.y(), prevCell.z());
    printf( " face %d ", face ); 
    printf( "tMax [%g,%g,%g] ",tMax.x(),tMax.y(), tMax.z());
    printf( "rayLoc [%g,%g,%g] ",ray_location.x(),ray_location.y(), ray_location.z());
    printf( "inv_dir [%g,%g,%g] ",inv_ray_direction.x(),inv_ray_direction.y(), inv_ray_direction.z()); 
    printf( "disMin %g \n",disMin ); 
   
    printf( "            abskg[prev] %g  \t sigmaT4OverPi[prev]: %g \n",abskg[prevCell],  sigmaT4OverPi[prevCell]);
    printf( "            abskg[cur]  %g  \t sigmaT4OverPi[cur]:  %g  \t  cellType: %i \n",abskg[cur], sigmaT4OverPi[cur], celltype[cur]);
} 
#endif
/*===========TESTING==========`*/           
//cout << "cur " << cur << " face " << face << " tmax " << tMax << " rayLoc " << ray_location << 
//        " inv_dir: " << inv_ray_direction << " disMin: " << disMin << endl;
       
       in_domain = (celltype[cur]==-1);  //cellType of -1 is flow


       optical_thickness += Dx.x() * abskg_prev*disMin; // as long as tDeltaY,Z tMax.y(),Z and ray_location[1],[2]..
       // were adjusted by DyDx  or DzDx, this line is now correct for noncubic domains.
       
       nRaySteps++;

       //Eqn 3-15(see below reference) while
       //Third term inside the parentheses is accounted for in Inet. Chi is accounted for in Inet calc.
       double expOpticalThick = exp(-optical_thickness);
       
       sumI += sigmaT4OverPi_prev * ( expOpticalThick_prev - expOpticalThick ) * fs;
       
       expOpticalThick_prev = expOpticalThick;

#ifdef RAY_SCATTER
       curLength += disMin * Dx.x();
       
       if (curLength > scatLength && in_domain){

         // get new scatLength for each scattering event
         scatLength = -log(mTwister.randDblExc() ) / scatCoeff; 
         
         ray_direction     =  findRayDirection( mTwister, d_isSeedRandom, cur ); 
         inv_ray_direction = Vector(1.0)/ray_direction;

         // get new step and sign
         int stepOld = step[face];
         findStepSize( step, sign, inv_ray_direction);
         
         // if sign[face] changes sign, put ray back into prevCell (back scattering)
         // a sign change only occurs when the product of old and new is negative
         if( step[face] * stepOld < 0 ){
           cur = prevCell;
         }
         
         // get new tMax (mixing bools, ints and doubles)
         tMax.x( ( cur[0] + sign[0]                - ray_location[0]) * inv_ray_direction[0] );
         tMax.y( ( cur[1] + sign[1] * D_DxRatio[1] - ray_location[1]) * inv_ray_direction[1] );
         tMax.z( ( cur[2] + sign[2] * D_DxRatio[2] - ray_location[2]) * inv_ray_direction[2] );

         // Length of t to traverse one cell
         tDelta    = Abs(inv_ray_direction) * D_DxRatio;
         tMax_prev = 0;
         curLength = 0;  // allow for multiple scattering events per ray
/*`==========TESTING==========*/
#if DEBUG == 3        
  printf( "%i, %i, %i, tmax: %g, %g, %g  tDelta: %g, %g, %g \n", cur.x(), cur.y(), cur.z(), tMax.x(), tMax.y(), tMax.z(), tDelta.x(), tDelta.y() , tDelta.z());         
#endif 
/*===========TESTING==========`*/
         //if(_benchmark == 4 || _benchmark ==5) scatLength = 1e16; // only for Siegel Benchmark4 benchmark5. Only allows 1 scatter event.
       }
#endif

     } //end domain while loop.  ++++++++++++++
     
     double wallEmissivity = abskg[cur];
     
     if (wallEmissivity > 1.0){       // Ensure wall emissivity doesn't exceed one. 
       wallEmissivity = 1.0;
     } 
     
     intensity = exp(-optical_thickness);
     
     sumI += wallEmissivity * sigmaT4OverPi[cur] * intensity;

     intensity = intensity * fs;
     
     // when a ray reaches the end of the domain, we force it to terminate. 
     if(!d_allowReflect) intensity = 0;
                                 
/*`==========TESTING==========*/
#if DEBUG == 1
if(origin.x() == 0 && origin.y() == 0 && origin.z() ==0 ){
    printf( "            cur [%d,%d,%d] intensity: %g expOptThick: %g, fs: %g allowReflect: %i\n", 
           cur.x(), cur.y(), cur.z(), intensity,  exp(-optical_thickness), fs, d_allowReflect );
    
} 
#endif 
/*===========TESTING==========`*/     
     //__________________________________
     //  Reflections
     if ( (intensity > d_threshold) && d_allowReflect){
       reflect( fs, cur, prevCell, abskg[cur], in_domain, step[face], sign[face], ray_direction[face]);
       ++nReflect;
     }
   }  // threshold while loop.
} // end of updateSumI function



//______________________________________________________________________
// Utility task:  move variable from old_dw -> new_dw
//______________________________________________________________________
void 
Radiometer::sched_CarryForward_Var ( const LevelP& level, 
                                     SchedulerP& sched,
                                     const VarLabel* variable)
{ 
  string taskname = "        carryForward_Var" + variable->getName();
  printSchedule(level, dbg, taskname);

  Task* tsk = scinew Task( taskname, this, &Radiometer::carryForward_Var, variable );
  
  tsk->requires(Task::OldDW, variable,   d_gn, 0);
  tsk->computes(variable);
 
  sched->addTask( tsk, level->eachPatch(), d_matlSet );
}

//______________________________________________________________________
void 
Radiometer::carryForward_Var ( const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw, 
                               DataWarehouse* new_dw,
                               const VarLabel* variable)
{
  new_dw->transferFrom(old_dw, variable, patches, matls, true);
}
