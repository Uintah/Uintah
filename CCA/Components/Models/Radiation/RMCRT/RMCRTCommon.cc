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

//______________________________________________________________________
//
#include <CCA/Components/Models/Radiation/RMCRT/RMCRTCommon.h>

#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Math/MersenneTwister.h>
#include <time.h>
#include <fstream>

#include <include/sci_defs/uintah_testdefs.h.in>

#define DEBUG -9         // 1: divQ, 2: boundFlux, 3: scattering
#define FIXED_RAY_DIR -9 // Sets ray direction.  1: (0.7071,0.7071, 0), 2: (0.7071, 0, 0.7071), 3: (0, 0.7071, 0.7071)

//#define FAST_EXP       // This uses a fast approximate exp() function that is 
                         // significantly faster.

//______________________________________________________________________
//
using namespace Uintah;
using namespace std;
static DebugStream dbg("RAY", false);

//______________________________________________________________________
// Static variable declarations
// This class is instantiated by ray() and radiometer().
// You only want 1 instance of each of these variables thus we use
// static variables
//______________________________________________________________________

double RMCRTCommon::d_threshold;
double RMCRTCommon::d_sigma;
double RMCRTCommon::d_sigmaScat;
bool   RMCRTCommon::d_isSeedRandom;
bool   RMCRTCommon::d_allowReflect;
int    RMCRTCommon::d_matl;
string RMCRTCommon::d_abskgBC_tag;
vector<IntVector> RMCRTCommon::d_dbgCells;



MaterialSet* RMCRTCommon::d_matlSet = 0;
const VarLabel* RMCRTCommon::d_sigmaT4Label;
const VarLabel* RMCRTCommon::d_abskgLabel;
const VarLabel* RMCRTCommon::d_divQLabel;
const VarLabel* RMCRTCommon::d_boundFluxLabel;
const VarLabel* RMCRTCommon::d_radiationVolqLabel;

const VarLabel* RMCRTCommon::d_compAbskgLabel;
const VarLabel* RMCRTCommon::d_compTempLabel;
const VarLabel* RMCRTCommon::d_cellTypeLabel;

//______________________________________________________________________
// Class: Constructor.
//______________________________________________________________________
//
RMCRTCommon::RMCRTCommon( TypeDescription::Type FLT_DBL )
    : d_FLT_DBL(FLT_DBL)
{
  if (RMCRTCommon::d_FLT_DBL == TypeDescription::double_type){
    d_sigmaT4Label = VarLabel::create( "sigmaT4", CCVariable<double>::getTypeDescription() );
    proc0cout << "__________________________________ USING DOUBLE VERSION OF RMCRT" << endl;
  } else {
    d_sigmaT4Label = VarLabel::create( "sigmaT4",    CCVariable<float>::getTypeDescription() );
    d_abskgLabel   = VarLabel::create( "abskgRMCRT", CCVariable<float>::getTypeDescription() );
    proc0cout << "__________________________________ USING FLOAT VERSION OF RMCRT" << endl;
  }

  d_boundFluxLabel     = VarLabel::create( "RMCRTboundFlux",   CCVariable<Stencil7>::getTypeDescription() );
  d_radiationVolqLabel = VarLabel::create( "radiationVolq",    CCVariable<double>::getTypeDescription() );

  d_gac     = Ghost::AroundCells;
  d_gn      = Ghost::None;
  d_flowCell = -1; //<----HARD CODED FLOW CELL
  
#ifdef FAST_EXP  
  d_fastExp.populateExp_int(-2, 2);
#endif
}

//______________________________________________________________________
// Method: Destructor
//______________________________________________________________________
//
RMCRTCommon::~RMCRTCommon()
{
  VarLabel::destroy( d_sigmaT4Label );
  VarLabel::destroy( d_boundFluxLabel );
  VarLabel::destroy( d_radiationVolqLabel );

  if (RMCRTCommon::d_FLT_DBL == TypeDescription::float_type){
    VarLabel::destroy( d_abskgLabel );
  }

  // when the radiometer class is invoked d_matlSet it deleted twice.  This prevents that.
  if( d_matlSet ) {
    if ( d_matlSet->getReferenceCount() == 1 ){
      d_matlSet->removeReference();
      delete d_matlSet;
    }
  }
}

//______________________________________________________________________
// Register the material index and label names
//______________________________________________________________________
void
RMCRTCommon::registerVarLabels(int   matlIndex,
                               const VarLabel* abskg,
                               const VarLabel* temperature,
                               const VarLabel* celltype,
                               const VarLabel* divQ )
{
  d_matl            = matlIndex;
  d_compAbskgLabel  = abskg;
  d_compTempLabel   = temperature;
  d_cellTypeLabel   = celltype;
  d_divQLabel       = divQ;

  d_abskgBC_tag = d_compAbskgLabel->getName(); // The label name changes when using floats.

  // If using RMCRT:DBL
  const Uintah::TypeDescription* td = d_compAbskgLabel->typeDescription();
  const Uintah::TypeDescription::Type subtype = td->getSubType()->getType();
  if ( RMCRTCommon::d_FLT_DBL == TypeDescription::double_type && subtype == TypeDescription::double_type ){
    d_abskgLabel = d_compAbskgLabel;
  }

  //__________________________________
  //  define the materialSet
  // The constructor can be called twice, so only create matlSet once.
  if (d_matlSet == 0) {
    d_matlSet = scinew MaterialSet();
    vector<int> m;
    m.push_back(matlIndex);
    d_matlSet->addAll(m);
    d_matlSet->addReference();
  }
}

//______________________________________________________________________
//  This task will convert the CCVariable abskg from double -> float
//  If abskg is of type double and the component has
//  specified that RMCRT communicate the all-to-all variables (abskg & sigmaT4)
//  as a float then convert abskg to float
//______________________________________________________________________
void
RMCRTCommon::sched_DoubleToFloat( const LevelP& level,
                                  SchedulerP& sched,
                                  Task::WhichDW myDW,
                                  const int radCalc_freq )
{
  const Uintah::TypeDescription* td = d_compAbskgLabel->typeDescription();
  const Uintah::TypeDescription::Type subtype = td->getSubType()->getType();

  // only run task if a conversion is needed.
  Task* tsk = NULL;
  if ( RMCRTCommon::d_FLT_DBL == TypeDescription::float_type &&  subtype == TypeDescription::double_type ){
    tsk = scinew Task( "RMCRTCommon::DoubleToFloat", this, &RMCRTCommon::DoubleToFloat, myDW, radCalc_freq);
  } else {
    return;
  }

  printSchedule(level, dbg, "RMCRTCommon::DoubleToFloat");

  tsk->requires( myDW,       d_compAbskgLabel, d_gn, 0 );
  tsk->requires( Task::OldDW, d_abskgLabel,    d_gn, 0 );  // for carryforward
  tsk->computes(d_abskgLabel);

  sched->addTask( tsk, level->eachPatch(), d_matlSet );
}
//______________________________________________________________________
//
//______________________________________________________________________
void
RMCRTCommon::DoubleToFloat( const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw,
                            Task::WhichDW which_dw,
                            const int radCalc_freq )
{
  //__________________________________
  //  Carry Forward
  if ( doCarryForward( radCalc_freq ) ) {
    printTask( patches, patches->get(0), dbg, "Doing RMCRTCommon::DoubleToFloat carryForward (abskgRMCRT)" );
    new_dw->transferFrom( old_dw, d_abskgLabel, patches, matls, true );
    return;
  }

  //__________________________________
  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing RMCRTCommon::DoubleToFloat");

    constCCVariable<double> abskg_D;
    CCVariable< float > abskg_F;

    DataWarehouse* myDW = new_dw->getOtherDataWarehouse(which_dw);
    myDW->get(abskg_D,             d_compAbskgLabel, d_matl, patch, Ghost::None, 0);
    new_dw->allocateAndPut(abskg_F, d_abskgLabel, d_matl, patch);

    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      const IntVector& c = *iter;
      abskg_F[c] = (float)abskg_D[c];
    }
  }
}

//______________________________________________________________________
//
//______________________________________________________________________
void
RMCRTCommon::sched_sigmaT4( const LevelP& level,
                           SchedulerP& sched,
                           Task::WhichDW temp_dw,
                           const int radCalc_freq,
                           const bool includeEC )
{
  std::string taskname = "RMCRTCommon::sigmaT4";

  Task* tsk = NULL;
  if ( RMCRTCommon::d_FLT_DBL == TypeDescription::double_type ){
    tsk = scinew Task( taskname, this, &RMCRTCommon::sigmaT4<double>, temp_dw, radCalc_freq, includeEC );
  } else {
    tsk = scinew Task( taskname, this, &RMCRTCommon::sigmaT4<float>, temp_dw, radCalc_freq, includeEC );
  }

  printSchedule(level,dbg,taskname);

  //__________________________________
  // Be careful if you modify this.  This additional logic
  // is needed when restarting from an uda that
  // was previously run without RMCRT.  It's further
  // complicated when the calc_frequency >1  If you change
  // it then test by restarting from an uda that was
  // previously run with Arches + DO with calc_frequency > 1.
  bool old_dwExists = false;
  if( sched->get_dw(0) ){
    old_dwExists = true;
  }

  if(old_dwExists){
    tsk->requires( Task::OldDW, d_sigmaT4Label, d_gn, 0 );
  }

  tsk->requires( temp_dw, d_compTempLabel,    d_gn, 0 );
  tsk->computes(d_sigmaT4Label);

  sched->addTask( tsk, level->eachPatch(), d_matlSet );
}
//______________________________________________________________________
// Compute total intensity over all wave lengths (sigma * Temperature^4/pi)
//______________________________________________________________________
template< class T>
void
RMCRTCommon::sigmaT4( const ProcessorGroup*,
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
  if ( doCarryForward( radCalc_freq ) ) {
    printTask( patches, patches->get(0), dbg, "Doing RMCRTCommon::sigmaT4 carryForward (sigmaT4)" );

    new_dw->transferFrom( old_dw, d_sigmaT4Label, patches, matls, true );
    return;
  }

  //__________________________________
  //  do the work
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing RMCRTCommon::sigmaT4");

    double sigma_over_pi = d_sigma/M_PI;

    constCCVariable<double> temp;
    CCVariable< T > sigmaT4;             // sigma T ^4/pi

    DataWarehouse* temp_dw = new_dw->getOtherDataWarehouse(which_temp_dw);
    temp_dw->get(temp,              d_compTempLabel,  d_matl, patch, Ghost::None, 0);
    new_dw->allocateAndPut(sigmaT4, d_sigmaT4Label,   d_matl, patch);

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
//
//______________________________________________________________________
void
RMCRTCommon::findStepSize(int step[],
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
//
//______________________________________________________________________
void
RMCRTCommon::raySignStep(double sign[],
                         int cellStep[],
                         const Vector& inv_direction_vector){
  // get new step and sign
  for ( int d=0; d<3; d++){
    double me = copysign((double)1.0, inv_direction_vector[d]);  // +- 1
    
    sign[d] = std::max(0.0, me);    // 0, 1
    
    cellStep[d] = int(me);
  }
}

//______________________________________________________________________
//  Compute the ray direction
//______________________________________________________________________
Vector
RMCRTCommon::findRayDirection(MTRand& mTwister,
                             const IntVector& origin,
                             const int iRay )
{
  if( d_isSeedRandom == false ){
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

/*`==========TESTING==========*/
#if ( FIXED_RAY_DIR == 1)
   direction_vector = Vector(0.70711, 0.70711, 0.);
#elif ( FIXED_RAY_DIR == 2 )
   direction_vector = Vector(0.70711, 0.0, 0.70711);
#elif ( FIXED_RAY_DIR == 3 )
   direction_vector = Vector(0.0, 0.70711, 0.70711);
#else
#endif
/*===========TESTING==========`*/

  return direction_vector;
}


//______________________________________________________________________
//  Compute the physical location of a ray's origin
//______________________________________________________________________
void
RMCRTCommon::ray_Origin( MTRand& mTwister,
                         const Point  CC_pos,
                         const Vector dx,
                         const bool   useCCRays,
                         Vector& rayOrigin)
{
  if( useCCRays == false ){
    rayOrigin[0] =  CC_pos.x() - 0.5*dx.x()  + mTwister.rand() * dx.x(); 
    rayOrigin[1] =  CC_pos.y() - 0.5*dx.y()  + mTwister.rand() * dx.y(); 
    rayOrigin[2] =  CC_pos.z() - 0.5*dx.z()  + mTwister.rand() * dx.z();
  }else{
    rayOrigin[0] = CC_pos(0);
    rayOrigin[1] = CC_pos(1);
    rayOrigin[2] = CC_pos(2);
  }
}

//______________________________________________________________________
//  Compute the physical location of the ray
//______________________________________________________________________
void
RMCRTCommon::ray_Origin( MTRand& mTwister,
                         const IntVector origin,
                         const double DyDx,
                         const double DzDx,
                         const bool useCCRays,
                         Vector& rayOrigin)
{
  if( useCCRays == false ){
    rayOrigin[0] =   origin[0] +  mTwister.rand() ;             // FIX ME!!! This is not the physical location of the ray.
    rayOrigin[1] =   origin[1] +  mTwister.rand() * DyDx ;      // this is index space.
    rayOrigin[2] =   origin[2] +  mTwister.rand() * DzDx ;
  }else{
    rayOrigin[0] =   origin[0] +  0.5 ;
    rayOrigin[1] =   origin[1] +  0.5 * DyDx ;
    rayOrigin[2] =   origin[2] +  0.5 * DzDx ;
  }
}

//______________________________________________________________________
//    Core function:
//______________________________________________________________________
void
RMCRTCommon::reflect(double& fs,
                    IntVector& cur,
                    IntVector& prevCell,
                    const double abskg,
                    bool& in_domain,
                    int& step,
                    double& sign,
                    double& ray_direction)
{
  fs = fs * (1 - abskg);

  //put cur back inside the domain
  cur = prevCell;
  in_domain = true;

  // apply reflection condition
  step *= -1;                // begin stepping in opposite direction
  sign *= -1;
  ray_direction *= -1;
  //dbg2 << " REFLECTING " << endl;
}

void
RMCRTCommon::reflect(double& fs,
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
//    Integrate the intensity
//______________________________________________________________________
template <class T >
void
RMCRTCommon::updateSumI (const Level* level, 
                         Vector& ray_direction,
                         Vector& ray_origin,
                         const IntVector& origin,
                         const Vector& Dx,
                         constCCVariable< T >& sigmaT4OverPi,
                         constCCVariable< T >& abskg,
                         constCCVariable<int>& celltype,
                         unsigned long int& nRaySteps,
                         double& sumI,
                         MTRand& mTwister)

{
  IntVector cur = origin;
  IntVector prevCell = cur;
  // Cell stepping direction for ray marching
  int    step[3];                                          
  double sign[3];                 //   is 0 for negative ray direction 

  Vector inv_ray_direction = Vector(1.0)/ray_direction;

/*`==========TESTING==========*/
#if DEBUG == 1
  if( isDbgCell(origin) ) {
    printf("        updateSumI: [%d,%d,%d] ray_dir [%g,%g,%g] ray_loc [%g,%g,%g]\n", origin.x(), origin.y(), origin.z(),ray_direction.x(), ray_direction.y(), ray_direction.z(), ray_origin.x(), ray_origin.y(), ray_origin.z());
  }
#endif
/*===========TESTING==========`*/

  raySignStep(sign, step, ray_direction);
 
  Point CC_pos = level->getCellPosition(origin);
  
  // rayDx is the distance from bottom, left, back, corner of cell to ray
  double rayDx[3];
  rayDx[0] = ray_origin.x() - ( CC_pos.x() - 0.5*Dx[0] );
  rayDx[1] = ray_origin.y() - ( CC_pos.y() - 0.5*Dx[1] );
  rayDx[2] = ray_origin.z() - ( CC_pos.z() - 0.5*Dx[2] );
  
  // tMax is the physical distance from the ray origin to each of the respective planes of intersection
  double tMax[3];
  tMax[0] = (sign[0] * Dx[0] - rayDx[0]) * inv_ray_direction.x();
  tMax[1] = (sign[1] * Dx[1] - rayDx[1]) * inv_ray_direction.y();
  tMax[2] = (sign[2] * Dx[2] - rayDx[2]) * inv_ray_direction.z();
    
  //Length of t to traverse one cell
  Vector tDelta = Abs(inv_ray_direction) * Dx;

  //Initializes the following values for each ray
  bool in_domain     = true;
  double tMax_prev   = 0;
  double intensity   = 1.0;
  double fs          = 1.0;
  int nReflect       = 0;                 // Number of reflections
  double optical_thickness      = 0;
  double expOpticalThick_prev   = 1.0;
  double rayLength              = 0.0;
  Vector ray_location           = ray_origin;


#ifdef RAY_SCATTER
  double scatCoeff = std::max( d_sigmaScat, 1e-99 ); // avoid division by zero  [m^-1]

  // Determine the length at which scattering will occur
  // See CCA/Components/Arches/RMCRT/PaulasAttic/MCRT/ArchesRMCRT/ray.cc
  double scatLength = -log(mTwister.randDblExc() ) / scatCoeff; 
#endif

  //+++++++Begin ray tracing+++++++++++++++++++
  //Threshold while loop
  while ( intensity > d_threshold ){
    DIR dir = NONE;
    
    while (in_domain){

      prevCell = cur;
      double disMin = -9;          // Represents ray segment length.

      T abskg_prev = abskg[prevCell];  // optimization
      T sigmaT4OverPi_prev = sigmaT4OverPi[prevCell];
      
      //__________________________________
      //  Determine which cell the ray will enter next
      dir = NONE;
      if ( tMax[0] < tMax[1] ){        // X < Y
        if ( tMax[0] < tMax[2] ){      // X < Z
          dir = X;
        } else {
          dir = Z;
        }
      } else {
        if( tMax[1] < tMax[2] ){       // Y < Z
          dir = Y;
        } else {
          dir = Z;
        }
      }

      //__________________________________
      //  update marching variables
      cur[dir]   = cur[dir] + step[dir];
      disMin     = (tMax[dir] - tMax_prev);
      tMax_prev  = tMax[dir];
      tMax[dir]  = tMax[dir] + tDelta[dir];
      rayLength += disMin;

      ray_location[0] = ray_location[0] + (disMin  * ray_direction[0]);
      ray_location[1] = ray_location[1] + (disMin  * ray_direction[1]);
      ray_location[2] = ray_location[2] + (disMin  * ray_direction[2]);

      in_domain = (celltype[cur] == d_flowCell);
      
      optical_thickness += abskg_prev*disMin; 

      nRaySteps++;
      
 /*`==========TESTING==========*/
#if ( DEBUG >= 1 )
      if( isDbgCell( origin )){
         printf( "            cur [%d,%d,%d] prev [%d,%d,%d] ", cur.x(), cur.y(), cur.z(), prevCell.x(), prevCell.y(), prevCell.z());
         printf( " dir %d ", dir );
         printf( "tMax [%g,%g,%g] ",tMax[0],tMax[1], tMax[2]);
         printf( "rayLoc [%g,%g,%g] ",ray_location.x(),ray_location.y(), ray_location.z());
         printf( "disMin %g tMax[dir]: %g tMax_prev: %g, Dx[dir]: %g\n",disMin, tMax[dir], tMax_prev, Dx[dir]);

         printf( "            abskg[prev] %g  \t sigmaT4OverPi[prev]: %g \n",abskg[prevCell],  sigmaT4OverPi[prevCell]);
         printf( "            abskg[cur]  %g  \t sigmaT4OverPi[cur]:  %g  \t  cellType: %i\n",abskg[cur], sigmaT4OverPi[cur], celltype[cur]);
         printf( "            optical_thickkness %g \t rayLength: %g\n", optical_thickness, rayLength);
      }
#endif
/*===========TESTING==========`*/     

      //Eqn 3-15(see below reference) while
      //Third term inside the parentheses is accounted for in Inet. Chi is accounted for in Inet calc.

      
/*`==========TESTING==========*/
#ifdef FAST_EXP

      // We need to know the range of optical_thickness before we can select
      // which implementation to use.
  
      double expOpticalThick = d_fastExp.fast_exp(-optical_thickness);
      //double expOpticalThick = exp(-optical_thickness);

      double S_exp    = d_fastExp.Schraudolph_exp(-optical_thickness);
      double fast_exp = d_fastExp.fast_exp(-optical_thickness);
      double exp2     = d_fastExp.exp2(-optical_thickness);
      double exp3     = d_fastExp.exp3(-optical_thickness);
      double exp5     = d_fastExp.exp5(-optical_thickness);
      double exp7     = d_fastExp.exp7(-optical_thickness);
      double exact    = exp(-optical_thickness);
      
      cout << " X: " << -optical_thickness << endl;
      cout << " Sch_exp error:    " << ((S_exp    - exact)/exact ) * 100 << " S_exp:   " << S_exp    << " exact: " << exact << endl;
      cout << " fast_exp error:   " << ((fast_exp - exact)/exact ) * 100 << " fast_exp:" << fast_exp << endl;
      cout << " exp2 error:       " << ((exp2     - exact)/exact ) * 100 << " exp2:    " << exp2    << endl;
      cout << " exp3 error:       " << ((exp3     - exact)/exact ) * 100 << " exp3:    " << exp3    << endl;
      cout << " exp5 error:       " << ((exp5     - exact)/exact ) * 100 << " exp5:    " << exp5    << endl;
      cout << " exp7 error:       " << ((exp7     - exact)/exact ) * 100 << " exp7:    " << exp7    << endl;

#else
      double expOpticalThick = exp(-optical_thickness);
#endif
/*===========TESTING==========`*/

      sumI += sigmaT4OverPi_prev * ( expOpticalThick_prev - expOpticalThick ) * fs;

      expOpticalThick_prev = expOpticalThick;

#ifdef RAY_SCATTER
      if (rayLength > scatLength && in_domain ){
            
        // get new scatLength for each scattering event
        scatLength = -log(mTwister.randDblExc() ) / scatCoeff;

        ray_direction     =  findRayDirection( mTwister, cur );

        inv_ray_direction = Vector(1.0)/ray_direction;

        // get new step and sign
        int stepOld = step[dir];
        raySignStep( sign, step, ray_direction);

        // if sign[dir] changes sign, put ray back into prevCell (back scattering)
        // a sign change only occurs when the product of old and new is negative
        if( step[dir] * stepOld < 0 ){
          cur = prevCell;
        }

        Point CC_pos = level->getCellPosition(cur);
        rayDx[0] = ray_location.x() - ( CC_pos.x() - 0.5*Dx[0] );
        rayDx[1] = ray_location.y() - ( CC_pos.y() - 0.5*Dx[1] );
        rayDx[2] = ray_location.z() - ( CC_pos.z() - 0.5*Dx[2] );
  
        tMax[0] = (sign[0] * Dx[0] - rayDx[0]) * inv_ray_direction.x();
        tMax[1] = (sign[1] * Dx[1] - rayDx[1]) * inv_ray_direction.y();
        tMax[2] = (sign[2] * Dx[2] - rayDx[2]) * inv_ray_direction.z();       

        //Length of t to traverse one cell
        tDelta = Abs(inv_ray_direction) * Dx;
/*`==========TESTING==========*/
#if (DEBUG == 3)
        if( isDbgCell( origin)  ){
          printf( "            Scatter: [%i, %i, %i], rayLength: %g, tmax: %g, %g, %g  tDelta: %g, %g, %g  ray_dir: %g, %g, %g\n",cur.x(), cur.y(), cur.z(),rayLength, tMax[0], tMax[1], tMax[2], tDelta.x(), tDelta.y() , tDelta.z(), ray_direction.x(), ray_direction.y() , ray_direction.z());
          printf( "                    dir: %i sign: [%g, %g, %g], step [%i, %i, %i] cur: [%i, %i, %i], prevCell: [%i, %i, %i]\n", dir, sign[0], sign[1], sign[2], step[0], step[1], step[2], cur[0], cur[1], cur[2], prevCell[0], prevCell[1], prevCell[2] );
          printf( "                    ray_location: [%g, %g, %g]\n", rayLocation[0], rayLocation[1], rayLocation[2] );
//          printf("                     rayDx         [%g, %g, %g]  CC_pos[%g, %g, %g]\n", rayDx[0], rayDx[1], rayDx[2], CC_pos.x(), CC_pos.y(), CC_pos.z());
        }
#endif
/*===========TESTING==========`*/
        tMax_prev = 0;
        rayLength = 0;  // allow for multiple scattering events per ray
      }
#endif

    } //end domain while loop.  ++++++++++++++

    T wallEmissivity = abskg[cur];

    if (wallEmissivity > 1.0){       // Ensure wall emissivity doesn't exceed one.
      wallEmissivity = 1.0;
    }

    intensity = exp(-optical_thickness);

    sumI += wallEmissivity * sigmaT4OverPi[cur] * intensity;

    intensity = intensity * fs;

    // when a ray reaches the end of the domain, we force it to terminate.
    if(!d_allowReflect) intensity = 0;

/*`==========TESTING==========*/
#if DEBUG >0
if( isDbgCell( origin)  ){
   printf( "            cur [%d,%d,%d] intensity: %g expOptThick: %g, fs: %g allowReflect: %i\n",
          cur.x(), cur.y(), cur.z(), intensity,  exp(-optical_thickness), fs, d_allowReflect );

}
#endif
/*===========TESTING==========`*/
    //__________________________________
    //  Reflections
    if ( (intensity > d_threshold) && d_allowReflect){
      reflect( fs, cur, prevCell, abskg[cur], in_domain, step[dir], sign[dir], ray_direction[dir]);
      ++nReflect;
    }
  }  // threshold while loop.
} // end of updateSumI function



//______________________________________________________________________
// Utility task:  move variable from old_dw -> new_dw
//______________________________________________________________________
void
RMCRTCommon::sched_CarryForward_Var ( const LevelP& level,
                                     SchedulerP& sched,
                                     const VarLabel* variable)
{
  string taskname = "        carryForward_Var: " + variable->getName();
  printSchedule(level, dbg, taskname);

  Task* tsk = scinew Task( taskname, this, &RMCRTCommon::carryForward_Var, variable );

  tsk->requires(Task::OldDW, variable,   d_gn, 0);
  tsk->computes(variable);

  sched->addTask( tsk, level->eachPatch(), d_matlSet );
}

//______________________________________________________________________
void
RMCRTCommon::carryForward_Var ( const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw,
                               const VarLabel* variable)
{
  new_dw->transferFrom(old_dw, variable, patches, matls, true);
}


//______________________________________________________________________
//  Trigger a taskgraph recompilation if the *next* timestep is a
//  calculation timestep
//______________________________________________________________________
void
RMCRTCommon::doRecompileTaskgraph( const int radCalc_freq ){

  if( radCalc_freq > 1 ){

    int timestep     = d_sharedState->getCurrentTopLevelTimeStep();
    int nextTimestep = timestep + 1;

    // if the _next_ timestep is a calculation timestep
    if( nextTimestep%radCalc_freq == 0 ){
      // proc0cout << "  RMCRT recompile taskgraph to turn on all-to-all communications" << endl;
      d_sharedState->setRecompileTaskGraph( true );
    }

    // if the _current_ timestep is a calculation timestep
    if( timestep%radCalc_freq == 0 ){
      // proc0cout << "  RMCRT recompile taskgraph to turn off all-to-all communications" << endl;
      d_sharedState->setRecompileTaskGraph( true );
    }
  }
}

//______________________________________________________________________
//  Logic for determing when to carry forward
//______________________________________________________________________
bool
RMCRTCommon::doCarryForward( const int radCalc_freq ){
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();
  bool test = (timestep%radCalc_freq != 0 && timestep != 1);

  return test;
}

//______________________________________________________________________
//
//______________________________________________________________________
bool
RMCRTCommon::isDbgCell( const IntVector me)
{

  if(me == IntVector(0,0,0) ){
    return true;
  }
  for( unsigned int i = 0; i<d_dbgCells.size(); i++) {
    if( me == d_dbgCells[i]) {
      return true;

    }
  }
  return false;
}

//______________________________________________________________________
//  Populate vector with integers which have been randomly shuffled.
//  This is sampling without replacement and can be used to in a
//  Latin-Hyper-Cube sampling scheme.  The algorithm used is the
//  modern fisher-yates shuffle.
//______________________________________________________________________
void
RMCRTCommon::randVector( vector <int> &int_array,
                         MTRand& mTwister,
                         const IntVector& cell ){
  int max= int_array.size();

  for (int i=0; i<max; i++){   // populate sequential array from 0 to max-1
    int_array[i] = i;
  }

  if( d_isSeedRandom == false ){
    mTwister.seed((cell.x() + cell.y() + cell.z()));
  }

  for (int i=max-1; i>0; i--){  // fisher-yates shuffle starting with max-1
    int rand_int =  mTwister.randInt(i);
    int swap = int_array[i];
    int_array[i] = int_array[rand_int];
    int_array[rand_int] = swap;
  }
}

//______________________________________________________________________
//
//______________________________________________________________________
// Explicit template instantiations:

template void
  RMCRTCommon::updateSumI ( const Level*, Vector&, Vector&, const IntVector&, const Vector&, constCCVariable< double >&, constCCVariable<double>&, constCCVariable<int>&, unsigned long int&, double&, MTRand&);

template void
  RMCRTCommon::updateSumI ( const Level*, Vector&, Vector&, const IntVector&, const Vector&, constCCVariable< float >&, constCCVariable<float>&, constCCVariable<int>&, unsigned long int&, double&, MTRand&);

