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
#include <CCA/Components/Models/Radiation/RMCRT/RMCRTCommon.h>

#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Math/MersenneTwister.h>
#include <Core/Util/DOUT.hpp>

#include <fstream>

#define DEBUG -9            // 1: divQ, 2: boundFlux, 3: scattering
#define FIXED_RAY_DIR -9    // Sets ray direction.  1: (0.7071,0.7071, 0), 2: (0.7071, 0, 0.7071), 3: (0, 0.7071, 0.7071)
                            //                     4: (0.7071, 0.7071, 7071), 5: (1,0,0)  6: (0, 1, 0),   7: (0,0,1)
#define SIGN 1              // Multiply the FIXED_RAY_DIRs by value
#define FUZZ 1e-12          // numerical fuzz

//#define FAST_EXP          // This uses a fast approximate exp() function that is
                            // significantly faster.

//______________________________________________________________________
//
using namespace Uintah;

// These are used externally (e.g. Radiamoter.cc), keep them visible
// outside this unit
Dout g_ray_dbg("Ray",     "Radiation Models", "RMCRT Ray general debug stream", false);
Dout g_ray_BC ("Ray_BC",  "Radiation Models", "RMCRT RayBC debug stream", false);

//______________________________________________________________________
// Static variable declarations
// This class is instantiated by ray() and radiometer().
// You only want 1 instance of each of these variables thus we use static variables
//______________________________________________________________________

double      RMCRTCommon::d_threshold;
double      RMCRTCommon::d_sigma;
double      RMCRTCommon::d_sigmaScat;
double      RMCRTCommon::d_maxRayLength;               // max ray length.
bool        RMCRTCommon::d_isSeedRandom;
bool        RMCRTCommon::d_allowReflect;
int         RMCRTCommon::d_matl;
int         RMCRTCommon::d_whichAlgo{singleLevel};
std::string RMCRTCommon::d_abskgBC_tag;
std::map<std::string,Task::WhichDW>    RMCRTCommon::d_abskg_dw;


std::vector<IntVector> RMCRTCommon::d_dbgCells;

MaterialSet* RMCRTCommon::d_matlSet{nullptr};

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
    proc0cout << "  - Using double implementation of RMCRT" << std::endl;
  } else {
    d_sigmaT4Label = VarLabel::create( "sigmaT4",    CCVariable<float>::getTypeDescription() );
    proc0cout << "  - Using float implementation of RMCRT" << std::endl;
  }

  d_boundFluxLabel     = VarLabel::create( "RMCRTboundFlux",   CCVariable<Stencil7>::getTypeDescription() );
  d_radiationVolqLabel = VarLabel::create( "radiationVolq",    CCVariable<double>::getTypeDescription() );

  d_gac          = Ghost::AroundCells;
  d_gn           = Ghost::None;
  d_flowCell     = -1; //<----HARD CODED FLOW CELL
  d_maxRayLength = DBL_MAX;

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

  // when the radiometer class (float) is invoked d_abskgLabel is deleted twice.  This prevents that
 if (RMCRTCommon::d_FLT_DBL == TypeDescription::float_type && d_abskgLabel->getReferenceCount() == 1 ){
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
RMCRTCommon::registerVariables(int   matlIndex,
                               const VarLabel* abskg,
                               const VarLabel* temperature,
                               const VarLabel* celltype,
                               const VarLabel* divQ,
                               const int whichAlgo )
{
  d_matl            = matlIndex;
  d_compAbskgLabel  = abskg;
  d_compTempLabel   = temperature;
  d_cellTypeLabel   = celltype;
  d_divQLabel       = divQ;
  d_whichAlgo       = whichAlgo;

  d_abskgBC_tag = d_compAbskgLabel->getName(); // The label name changes when using floats.

  // define the abskg VarLabel
  const Uintah::TypeDescription* td = d_compAbskgLabel->typeDescription();
  const Uintah::TypeDescription::Type subtype = td->getSubType()->getType();

  if ( RMCRTCommon::d_FLT_DBL == TypeDescription::double_type &&
                      subtype == TypeDescription::double_type ) {
    d_abskgLabel = d_compAbskgLabel;
  }

  if ( RMCRTCommon::d_FLT_DBL == TypeDescription::float_type &&
                      subtype == TypeDescription::float_type ) {
    d_abskgLabel = d_compAbskgLabel;
  }

  if ( RMCRTCommon::d_FLT_DBL == TypeDescription::float_type &&
                      subtype == TypeDescription::double_type ) {
    d_abskgLabel = VarLabel::create( "abskgRMCRT", CCVariable<float>::getTypeDescription() );
  }

  //__________________________________
  //  define the materialSet
  // The constructor can be called twice, so only create matlSet once.
  if (d_matlSet == nullptr) {
    d_matlSet = scinew MaterialSet();
    std::vector<int> m;
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
                                  Task::WhichDW notUsed )
{
  const Uintah::TypeDescription* td = d_compAbskgLabel->typeDescription();
  const Uintah::TypeDescription::Type subtype = td->getSubType()->getType();

  int L = level->getIndex();
  Task::WhichDW abskgDW = get_abskg_whichDW( L, d_compAbskgLabel );

  // only run task if a conversion is needed.
  Task* tsk = nullptr;
  if ( RMCRTCommon::d_FLT_DBL == TypeDescription::float_type &&  subtype == TypeDescription::double_type ){
    tsk = scinew Task( "RMCRTCommon::DoubleToFloat", this, &RMCRTCommon::DoubleToFloat, abskgDW);
  } else {
    return;
  }

  printSchedule(level, g_ray_dbg, "RMCRTCommon::DoubleToFloat");

  tsk->requires( abskgDW,       d_compAbskgLabel, d_gn, 0 );
  tsk->computes(d_abskgLabel);

  // shedule on all taskgraphs not just TG_RMCRT.  The output task needs d_abskgLabel
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
                            Task::WhichDW which_dw )
{
  //__________________________________
  for (int p=0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    printTask(patches, patch, g_ray_dbg, "Doing RMCRTCommon::DoubleToFloat");

    constCCVariable<double> abskg_D;
    CCVariable< float > abskg_F;

    DataWarehouse* myDW = new_dw->getOtherDataWarehouse(which_dw);
    myDW->get(abskg_D, d_compAbskgLabel, d_matl, patch, Ghost::None, 0);
    new_dw->allocateAndPut(abskg_F, d_abskgLabel, d_matl, patch);

    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      const IntVector& c = *iter;
      abskg_F[c] = (float)abskg_D[c];

      // bulletproofing
      if (std::isinf( abskg_F[c] ) || std::isnan( abskg_F[c] ) ) {
        std::ostringstream warn;
        warn<< "RMCRTCommon::DoubleToFloat A non-physical abskg detected (" << abskg_F[c] << ") at cell: " << c << "\n";
        throw InternalError( warn.str(), __FILE__, __LINE__ );
      }
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
                           const bool includeEC )
{
  std::string taskname = "RMCRTCommon::sigmaT4";

  Task* tsk = nullptr;
  if ( RMCRTCommon::d_FLT_DBL == TypeDescription::double_type ) {
    tsk = scinew Task( taskname, this, &RMCRTCommon::sigmaT4<double>, temp_dw, includeEC );
  } else {
    tsk = scinew Task( taskname, this, &RMCRTCommon::sigmaT4<float>, temp_dw, includeEC );
  }

  printSchedule(level, g_ray_dbg, "RMCRTCommon::sched_sigmaT4");

  tsk->requires( temp_dw, d_compTempLabel,    d_gn, 0 );
  tsk->computes( d_sigmaT4Label );

  sched->addTask( tsk, level->eachPatch(), d_matlSet, RMCRTCommon::TG_RMCRT );
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
                      const bool includeEC )
{
  //__________________________________
  //  do the work
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);

    printTask(patches, patch, g_ray_dbg, "Doing RMCRTCommon::sigmaT4");

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
RMCRTCommon::sched_initialize_VarLabel( const LevelP  & level,
                                       SchedulerP     & sched,
                                       const VarLabel * label )
{
  std::string taskname = "RMCRTCommon::initialize_VarLabel";

  Task* tsk = nullptr;
  const TypeDescription* td = label->typeDescription();
  const TypeDescription::Type subtype = td->getSubType()->getType();

  if ( subtype == TypeDescription::double_type ) {
    tsk = scinew Task( taskname, this, &RMCRTCommon::initialize_VarLabel<double>, label );
  } else {
    tsk = scinew Task( taskname, this, &RMCRTCommon::initialize_VarLabel<float>, label );
  }

  std::string mesg = "RMCRTCommon::initialize_VarLabel (" + label->getName() + ")";
  printSchedule( level, g_ray_dbg, mesg );

  tsk->computes( label );

  sched->addTask( tsk, level->eachPatch(), d_matlSet );
}
//______________________________________________________________________
// Initialize a VarLabel = 0
//______________________________________________________________________
template< class T>
void
RMCRTCommon::initialize_VarLabel( const ProcessorGroup *,
                                  const PatchSubset    * patches,
                                  const MaterialSubset *,
                                  DataWarehouse        *,
                                  DataWarehouse        * new_dw,
                                  const VarLabel       * label )
{
  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);

    std::string mesg = "Doing RMCRTCommon::initialize_VarLabel (" + label->getName() + ")";
    printTask(patches, patch, g_ray_dbg, mesg );

    CCVariable< T > var;
    new_dw->allocateAndPut( var, label, d_matl, patch );
    var.initialize( 0.0 );
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

/*`==========DEBUGGING==========*/
#if ( FIXED_RAY_DIR == 1)
   direction_vector = Vector(0.707106781186548, 0.707106781186548, 0.) * Vector(SIGN);
#elif ( FIXED_RAY_DIR == 2 )
   direction_vector = Vector(0.707106781186548, 0.0, 0.707106781186548) * Vector(SIGN);
#elif ( FIXED_RAY_DIR == 3 )
   direction_vector = Vector(0.0, 0.707106781186548, 0.707106781186548) * Vector(SIGN);
#elif ( FIXED_RAY_DIR == 4 )
   direction_vector = Vector(0.707106781186548, 0.707106781186548, 0.707106781186548) * Vector(SIGN);
#elif ( FIXED_RAY_DIR == 5 )
   direction_vector = Vector(1, 0, 0) * Vector(SIGN);
#elif ( FIXED_RAY_DIR == 6 )
   direction_vector = Vector(0, 1, 0) * Vector(SIGN);
#elif ( FIXED_RAY_DIR == 7 )
   direction_vector = Vector(0, 0, 1) * Vector(SIGN);
#else
#endif
/*===========DEBUGGING==========`*/

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
                         Vector& rayOrigin )
{
  if( useCCRays == false ){

    double x = mTwister.rand() * dx.x();
    double y = mTwister.rand() * dx.y();
    double z = mTwister.rand() * dx.z();

    Vector offset(x,y,z);  // Note you HAVE to compute the components separately to ensure that the
                           //  random numbers called in the x,y,z order -Todd

    if ( offset.x() > dx.x() ||
         offset.y() > dx.y() ||
         offset.z() > dx.z() ) {
      std::cout << "  Warning:ray_Origin  The Mersenne twister random number generator has returned garbage (" << offset
                << ") Now forcing the ray origin to be located at the cell-center\n" ;
      offset = Vector( 0.5*dx.x(), 0.5*dx.y(), 0.5*dx.z() );
    }

    rayOrigin[0] =  CC_pos.x() - 0.5*dx.x()  + offset.x();
    rayOrigin[1] =  CC_pos.y() - 0.5*dx.y()  + offset.y();
    rayOrigin[2] =  CC_pos.z() - 0.5*dx.z()  + offset.z();
  }else{
    rayOrigin[0] = CC_pos(0);
    rayOrigin[1] = CC_pos(1);
    rayOrigin[2] = CC_pos(2);
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
  //dbg2 << " REFLECTING " << std::endl;
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
  double rayLength_scatter      = 0.0;    // ray length for each scattering event
  double rayLength              = 0.0;    // total length of the ray
  Vector ray_location           = ray_origin;


#ifdef RAY_SCATTER
  double scatCoeff = std::max( d_sigmaScat, 1e-99 ); // avoid division by zero  [m^-1]

  // Determine the length at which scattering will occur
  // See CCA/Components/Arches/RMCRT/PaulasAttic/MCRT/ArchesRMCRT/ray.cc
  double scatLength = -log(mTwister.randDblExc() ) / scatCoeff;
#endif

  //______________________________________________________________________

  while ( intensity > d_threshold && (rayLength < d_maxRayLength) ){

    DIR dir = NONE;

    while ( in_domain && (rayLength < d_maxRayLength) ){


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

      // occassionally disMin ~ -1e-15ish
      if( disMin > -FUZZ && disMin < FUZZ){
        disMin += FUZZ;
      }

      rayLength += disMin;
      rayLength_scatter += disMin;

      ray_location[0] = ray_location[0] + (disMin  * ray_direction[0]);
      ray_location[1] = ray_location[1] + (disMin  * ray_direction[1]);
      ray_location[2] = ray_location[2] + (disMin  * ray_direction[2]);

      in_domain = (celltype[cur] == d_flowCell);

      optical_thickness += abskg_prev*disMin;

      nRaySteps++;

 /*`==========TESTING==========*/
#if ( DEBUG >= 1 )
      if( isDbgCell( origin )){
         printf( "            cur [%d,%d,%d] prev [%d,%d,%d]", cur.x(), cur.y(), cur.z(), prevCell.x(), prevCell.y(), prevCell.z());
         printf( " dir %d ", dir );
         printf( "tMax [%g,%g,%g] ",tMax[0],tMax[1], tMax[2]);
         printf( "rayLoc [%g,%g,%g] ",ray_location.x(),ray_location.y(), ray_location.z());
         printf( "distanceTraveled %g tMax[dir]: %g tMax_prev: %g, Dx[dir]: %g\n",disMin, tMax[dir], tMax_prev, Dx[dir]);
         printf( "            tDelta [%g,%g,%g] \n",tDelta.x(),tDelta.y(), tDelta.z());

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
      if (rayLength_scatter > scatLength && in_domain ){

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
          Vector mytDelta = tDelta / Dx;
          Vector myrayLoc = ray_location / Dx;
          printf( "            Scatter: [%i, %i, %i], rayLength: %g, tmax: %g, %g, %g  tDelta: %g, %g, %g  ray_dir: %g, %g, %g\n",cur.x(), cur.y(), cur.z(),rayLength, tMax[0] / Dx[0], tMax[1] / Dx[1], tMax[2] / Dx[2], mytDelta.x(), mytDelta.y() , mytDelta.z(), ray_direction.x(), ray_direction.y() , ray_direction.z());
          printf( "                    dir: %i sign: [%g, %g, %g], step [%i, %i, %i] cur: [%i, %i, %i], prevCell: [%i, %i, %i]\n", dir, sign[0], sign[1], sign[2], step[0], step[1], step[2], cur[0], cur[1], cur[2], prevCell[0], prevCell[1], prevCell[2] );
          printf( "                    ray_location: [%g, %g, %g]\n", myrayLoc[0], myrayLoc[1], myrayLoc[2] );
//          printf("                     rayDx         [%g, %g, %g]  CC_pos[%g, %g, %g]\n", rayDx[0], rayDx[1], rayDx[2], CC_pos.x(), CC_pos.y(), CC_pos.z());
        }
#endif
/*===========TESTING==========`*/
        tMax_prev = 0;
        rayLength_scatter = 0;  // allow for multiple scattering events per ray
      }
#endif

      if( rayLength < 0 || std::isnan(rayLength) || std::isinf(rayLength) ) {
        std::ostringstream warn;
        warn<< "ERROR:RMCRTCommon::updateSumI   The ray length is non-physical (" << rayLength << ")"
            << " origin: " << origin << " cur: " << cur << "\n";
        throw InternalError( warn.str(), __FILE__, __LINE__ );
      }
    }  //end domain while loop.  ++++++++++++++
    //______________________________________________________________________


    T wallEmissivity = abskg[cur];

    if (wallEmissivity > 1.0){       // Ensure wall emissivity doesn't exceed one.
      wallEmissivity = 1.0;
    }

    intensity = exp(-optical_thickness);

    sumI += wallEmissivity * sigmaT4OverPi[cur] * intensity;

    intensity = intensity * fs;
//    //__________________________________
//    //  BULLETPROOFING
//    if ( std::isinf(sumI) || std::isnan(sumI) ){
//      printf( "\n\n______________________________________________________________________\n");
//      std::cout <<  " cur: " << cur << " prevCell: " << prevCell << "\n";
//      std::cout <<  " dir: " << dir << " sumI: " << sumI << "\n";
//      std::cout <<  " tMax: " << tMax  << "\n";
//      std::cout <<  " rayLoc:    " << ray_location << "\n";
//      std::cout <<  " tMax[dir]: " << tMax[dir] << " tMax_prev: " << tMax_prev << " Dx[dir]: " << Dx[dir] << "\n";
//      std::cout <<  " tDelta:    " << tDelta << " \n";
//      std::cout <<  "     abskg[prev]: " << abskg[prevCell] << " \t sigmaT4OverPi[prev]: " << sigmaT4OverPi[prevCell]  << "\n";
//      std::cout <<  "     abskg[cur]:  " << abskg[cur]      << " \t sigmaT4OverPi[cur]:  " << sigmaT4OverPi[cur] << "\t  cellType: " <<celltype[cur] << "\n";
//      std::cout <<  "     optical_thickkness: " <<  optical_thickness << " \t rayLength: " << rayLength << "\n";
//
//      IntVector l = abskg.getLowIndex();
//      IntVector h = abskg.getHighIndex();
//      printf( "     abskg:  [%d,%d,%d]  -> [%d,%d,%d] \n", l.x(), l.y(), l.z() , h.x(), h.y(), h.z() );
//
//      std::ostringstream warn;
//        warn<< "ERROR:RMCRTCommon::updateSumI   sumI is non-physical (" << sumI << ")"
//            << " origin: " << origin << " cur: " << cur << "\n";
//        throw InternalError( warn.str(), __FILE__, __LINE__ );
//    }

    // when a ray reaches the end of the domain, we force it to terminate.
    if(!d_allowReflect) intensity = 0;

/*`==========TESTING==========*/
#if DEBUG  >= 0
if( isDbgCell( origin)  ){
   printf( "            cur [%d,%d,%d] intensity: %g expOptThick: %g, fs: %g allowReflect: %i\n",
          cur.x(), cur.y(), cur.z(), intensity,  exp(-optical_thickness), fs, d_allowReflect );

}
#endif
/*===========TESTING==========`*/

    //__________________________________
    //  Reflections
    if ( intensity > d_threshold && d_allowReflect ){
      reflect( fs, cur, prevCell, abskg[cur], in_domain, step[dir], sign[dir], ray_direction[dir]);
      ++nReflect;
    }
  }  // threshold while loop.

} // end of updateSumI function

//______________________________________________________________________
//    Move all computed variables from old_dw -> new_dw
//______________________________________________________________________
void
RMCRTCommon::sched_CarryForward_FineLevelLabels ( const LevelP& level,
                                                  SchedulerP& sched )
{
  std::string schedName = "RMCRTCommon::sched_CarryForward_FineLevelLabels";
  std::string taskName  = "RMCRTCommon::carryForward_FineLevelLabels";
  printSchedule( level, g_ray_dbg, schedName );

  Task* tsk = scinew Task( taskName, this, &RMCRTCommon::carryForward_FineLevelLabels );

  tsk->requires( Task::OldDW, d_divQLabel,          d_gn, 0 );
  tsk->requires( Task::OldDW, d_boundFluxLabel,     d_gn, 0 );
  tsk->requires( Task::OldDW, d_radiationVolqLabel, d_gn, 0 );
  tsk->requires( Task::OldDW, d_sigmaT4Label,       d_gn, 0 );

  tsk->computes( d_divQLabel );
  tsk->computes( d_boundFluxLabel );
  tsk->computes( d_radiationVolqLabel );
  tsk->computes( d_sigmaT4Label );

  sched->addTask( tsk, level->eachPatch(), d_matlSet, RMCRTCommon::TG_CARRY_FORWARD );
}

//______________________________________________________________________
//
void
RMCRTCommon::carryForward_FineLevelLabels(DetailedTask* dtask,
                                    Task::CallBackEvent event,
                                    const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* matls,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw,
                                    void* old_TaskGpuDW,
                                    void* new_TaskGpuDW,
                                    void* stream,
                                    int deviceID)
{
  printTask( patches, patches->get(0), g_ray_dbg, "Doing RMCRTCommon::carryForward_FineLevelLabels" );

  bool replaceVar = true;
  new_dw->transferFrom(old_dw, d_divQLabel,          patches, matls, dtask, replaceVar, nullptr );
  new_dw->transferFrom(old_dw, d_boundFluxLabel,     patches, matls, dtask, replaceVar, nullptr );
  new_dw->transferFrom(old_dw, d_radiationVolqLabel, patches, matls, dtask, replaceVar, nullptr );
  new_dw->transferFrom(old_dw, d_sigmaT4Label,       patches, matls, dtask, replaceVar, nullptr );
}


//______________________________________________________________________
//    Move all computed variables from old_dw -> new_dw
//______________________________________________________________________
void
RMCRTCommon::sched_carryForward_VarLabels ( const LevelP& level,
                                       SchedulerP& sched ,
                                       const std::vector< const VarLabel* > varLabels)
{
  std::string schedName = "RMCRTCommon::sched_carryForward_VarLabels";
  std::string taskName  = "RMCRTCommon::carryForward_VarLabels";
  printSchedule( level, g_ray_dbg, schedName );

  Task* tsk = scinew Task( taskName, this, &RMCRTCommon::carryForward_VarLabels, varLabels );

  for ( auto iter = varLabels.begin(); iter != varLabels.end(); iter++ ){
    tsk->requires( Task::OldDW, *iter, d_gn, 0 );
    tsk->computes( *iter );
  }

  sched->addTask( tsk, level->eachPatch(), d_matlSet, RMCRTCommon::TG_CARRY_FORWARD );
}

//______________________________________________________________________
//
void
RMCRTCommon::carryForward_VarLabels(DetailedTask* dtask,
                                    Task::CallBackEvent event,
                                    const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* matls,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw,
                                    void* old_TaskGpuDW,
                                    void* new_TaskGpuDW,
                                    void* stream,
                                    int deviceID,
                                    const std::vector< const VarLabel* > varLabels)
{
  printTask( patches, patches->get(0), g_ray_dbg, "Doing RMCRTCommon::carryForward_VarLabels" );

  bool replaceVar = true;
  for ( auto iter = varLabels.begin(); iter != varLabels.end(); iter++ ){
    new_dw->transferFrom(old_dw, *iter, patches, matls, dtask, replaceVar, nullptr );
  }
}



//______________________________________________________________________
// Utility task:  move variable from old_dw -> new_dw
//______________________________________________________________________
void
RMCRTCommon::sched_CarryForward_Var ( const LevelP& level,
                                      SchedulerP& sched,
                                      const VarLabel* variable,
                                      const int tg_num /* == -1 */)
{
  std::string schedName = "RMCRTCommon::sched_CarryForward_Var_" + variable->getName();
  std::string taskName  = "RMCRTCommon::carryForward_Var_" + variable->getName();
  printSchedule(level, g_ray_dbg, schedName);

  Task* task = scinew Task( taskName, this, &RMCRTCommon::carryForward_Var, variable );

  task->requires(Task::OldDW, variable,   d_gn, 0);
  task->computes(variable);

  sched->addTask( task, level->eachPatch(), d_matlSet, tg_num);
}

//______________________________________________________________________
void
RMCRTCommon::carryForward_Var ( DetailedTask* dtask,
                                Task::CallBackEvent event,
                                const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw,
                                void* old_TaskGpuDW,
                                void* new_TaskGpuDW,
                                void* stream,
                                int deviceID,
                                const VarLabel* variable )
{
  new_dw->transferFrom(old_dw, variable, patches, matls, dtask, true, nullptr);
}

//______________________________________________________________________
//
//______________________________________________________________________
bool
RMCRTCommon::isDbgCell( const IntVector me)
{
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
RMCRTCommon::randVector( std::vector <int> &int_array,
                         MTRand& mTwister,
                         const IntVector& cell )
{
  int max= int_array.size();

  for (int i=0; i<max; i++){   // populate sequential array from 0 to max-1
    int_array[i] = i;
  }

  if( d_isSeedRandom == false ){
    mTwister.seed((cell.x() + cell.y() + cell.z()));
  }

  for (int i=max-1; i>0; i--){  // fisher-yates shuffle starting with max-1

#ifdef FIXED_RANDOM_NUM
    int rand_int =  0.3*i;
#else
    int rand_int =  mTwister.randInt(i);
#endif
    int swap = int_array[i];
    int_array[i] = int_array[rand_int];
    int_array[rand_int] = swap;
  }
}

//______________________________________________________________________
// For RMCRT algorithms the absorption coefficient can be required from either the old_dw or
// new_dw depending on if RMCRT:float is specified.  On coarse levels abskg _always_ resides
// in the newDW.  If RMCRT:float is used then abskg on the fine level resides in the new_dw.
// This method creates a global map and the key for the  (map d_abskg_dw <string, Task::WhichDW>)
// is the labelName_L-X, the value in the map is the old or new dw.
//_____________________________________________________________________
void
RMCRTCommon::set_abskg_dw_perLevel ( const LevelP& fineLevel,
                                     Task::WhichDW fineLevel_abskg_dw )
{
  int maxLevels = fineLevel->getGrid()->numLevels();
  printSchedule(fineLevel, g_ray_dbg, "RMCRTCommon::set_abskg_dws");

  //__________________________________
  // fineLevel could have two entries.  One for abskg and abskgRMCRT
  std::ostringstream key;
  key << d_compAbskgLabel->getName() << "_L-"<< fineLevel->getIndex();
  d_abskg_dw[key.str()] = fineLevel_abskg_dw;

  //__________________________________
  // fineLevel: FLOAT  abskgRMCRT
  if ( RMCRTCommon::d_FLT_DBL == TypeDescription::float_type ) {
    std::ostringstream key1;
    key1 << d_abskgLabel->getName() << "_L-"<< fineLevel->getIndex();
    d_abskg_dw[key1.str()] = Task::NewDW;
  }

  //__________________________________
  // coarse levels always require from the newDW
  for(int L = 0; L<maxLevels; L++) {

    if ( L != fineLevel->getIndex() ) {
      std::ostringstream key2;
      key2 << d_abskgLabel->getName() << "_L-"<< L;
      d_abskg_dw[key2.str()] = Task::NewDW;
    }
  }

#if 0             // debugging
  for( auto iter = d_abskg_dw.begin(); iter !=  d_abskg_dw.end(); iter++ ){
    std::cout << " key: " << (*iter).first << " value: " << (*iter).second << std::endl;
  }
#endif
}

//______________________________________________________________________
//  return the dw associated for this abskg and level
//______________________________________________________________________
DataWarehouse*
RMCRTCommon::get_abskg_dw ( const int L,
                            const VarLabel* label,
                            DataWarehouse* new_dw)
{
  Task::WhichDW dw = get_abskg_whichDW ( L, label );
  DataWarehouse* abskg_dw = new_dw->getOtherDataWarehouse( dw );

  return abskg_dw;
}

//______________________________________________________________________
//  return the Task::WhichDW for this abskg and level
//______________________________________________________________________
Task::WhichDW
RMCRTCommon::get_abskg_whichDW ( const int L,
                                 const VarLabel* label)
{
  std::ostringstream key;
  key << label->getName() << "_L-"<< L;
  Task::WhichDW abskgDW = d_abskg_dw[key.str()];

//  std::cout << "    key: " << key.str() << " value: " << abskgDW << std::endl;
  return abskgDW;
}

//______________________________________________________________________
//
//______________________________________________________________________
// Explicit template instantiations:

template void
  RMCRTCommon::updateSumI ( const Level*, Vector&, Vector&, const IntVector&, const Vector&, constCCVariable< double >&, constCCVariable<double>&, constCCVariable<int>&, unsigned long int&, double&, MTRand&);

template void
  RMCRTCommon::updateSumI ( const Level*, Vector&, Vector&, const IntVector&, const Vector&, constCCVariable< float >&, constCCVariable<float>&, constCCVariable<int>&, unsigned long int&, double&, MTRand&);

