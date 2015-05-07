/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/BBox.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Math/MersenneTwister.h>
#include <time.h>
#include <fstream>

#include <include/sci_defs/uintah_testdefs.h.in>

#define DEBUG -9 // 1: divQ, 2: boundFlux, 3: scattering
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

  d_gac     = Ghost::AroundCells;
  d_gn      = Ghost::None;
  d_flowCell = -1; //<----HARD CODED FLOW CELL
}

//______________________________________________________________________
// Method: Destructor
//______________________________________________________________________
//
RMCRTCommon::~RMCRTCommon()
{
  VarLabel::destroy( d_sigmaT4Label );
  VarLabel::destroy( d_abskgLabel );

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

  tsk->requires( temp_dw, d_compTempLabel,    d_gn, 0 );
  tsk->requires( Task::OldDW, d_sigmaT4Label, d_gn, 0 );
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
//  Compute the ray direction
//______________________________________________________________________
Vector
RMCRTCommon::findRayDirection(MTRand& mTwister,
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
RMCRTCommon::rayLocation( MTRand& mTwister,
                         const IntVector origin,
                         const double DyDx,
                         const double DzDx,
                         const bool useCCRays,
                         Vector& location)
{
  if( useCCRays == false ){
    location[0] =   origin[0] +  mTwister.rand() ;             // FIX ME!!! This is not the physical location of the ray.
    location[1] =   origin[1] +  mTwister.rand() * DyDx ;      // this is index space.
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
RMCRTCommon::updateSumI ( Vector& ray_direction,
                         Vector& ray_location,
                         const IntVector& origin,
                         const Vector& Dx,
                         constCCVariable< T >& sigmaT4OverPi,
                         constCCVariable< T >& abskg,
                         constCCVariable<int>& celltype,
                         unsigned long int& nRaySteps,
                         double& sumI,
                         MTRand& mTwister)

{
/*`==========TESTING==========*/
#if DEBUG == 1
  if( isDbgCell(origin) ) {
    printf("        updateSumI: [%d,%d,%d] ray_dir [%g,%g,%g] ray_loc [%g,%g,%g]\n", origin.x(), origin.y(), origin.z(),ray_direction.x(), ray_direction.y(), ray_direction.z(), ray_location.x(), ray_location.y(), ray_location.z());
  }
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

       T abskg_prev = abskg[prevCell];  // optimization
       T sigmaT4OverPi_prev = sigmaT4OverPi[prevCell];
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
if( isDbgCell( origin )){
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

       in_domain = (celltype[cur] == d_flowCell);


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
#if DEBUG == 1
if( isDbgCell( origin)  ){
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
  
  for( uint i = 0; i<d_dbgCells.size(); i++) {
    if( me == d_dbgCells[i]) {
      return true;
      
    }
  }
  return false;
}


//______________________________________________________________________
//
//______________________________________________________________________
// Explicit template instantiations:

template void
  RMCRTCommon::updateSumI ( Vector&, Vector&, const IntVector&, const Vector&, constCCVariable< double >&, constCCVariable<double>&, constCCVariable<int>&, unsigned long int&, double&, MTRand&);

template void
  RMCRTCommon::updateSumI ( Vector&, Vector&, const IntVector&, const Vector&, constCCVariable< float >&, constCCVariable<float>&, constCCVariable<int>&, unsigned long int&, double&, MTRand&);

