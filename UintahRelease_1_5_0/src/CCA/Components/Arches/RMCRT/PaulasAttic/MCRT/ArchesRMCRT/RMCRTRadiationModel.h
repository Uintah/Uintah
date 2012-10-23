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
/// RMCRTRadiationModel.h-------------------------------------------------------
/// Reverse Monte Carlo Ray Tracing Radiation Model interface
/// 
/// 
/// 

#ifndef Uintah_Component_Arches_RMCRTRadiationModel_h
#define Uintah_Component_Arches_RMCRTRadiationModel_h

#include <Core/Grid/LevelP.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/CCVariable.h>

namespace Uintah {
class TimeIntegratorLabel; 
class BoundaryCondition; 
class RMCRTRadiationModel {

public:
  // constructor
  RMCRTRadiationModel( const ArchesLabel* label, BoundaryCondition* bc );

  // destructor
  ~RMCRTRadiationModel();
  
  //methods: 
  /** @brief Set any parameters from the input file, initialize constants, etc... */
  void problemSetup( const ProblemSpecP& params );
  
  /** @brief Schedule the solution of the radiative transport equation using RMCRT */
  void sched_solve( const LevelP& level, SchedulerP& sched, const TimeIntegratorLabel* timeLabels );

  /** @brief Actually solve the radiative transport equation using RMCRT */
  void solve( const ProcessorGroup* pc,  
              const PatchSubset* patches, 
              const MaterialSubset*, 
              DataWarehouse* old_dw, 
              DataWarehouse* new_dw,
              const TimeIntegratorLabel* timeLabels );

  /** @brief Interpolate CC temperatures to FC temperatures */ 
  template <class FCV>
  void interpCCTemperatureToFC( constCCVariable<int>& cellType, 
                                FCV& Tx, IntVector dir, IntVector highIdx, 
                                CCVariable<double>& T, const Patch* p );

private:


  // variables:
  int d_constNumRays;
  double d_refTemp, d_uScale, d_refComp;
  bool d_radcal, d_wsgg, d_planckmean, d_patchmean, d_fssk, d_fsck;
  bool d_rr; 
  double d_StopLowerBound; 
  double d_opl;
  int d_ambda; 
  const ArchesLabel* d_lab; 
  int d_mmWallID; 
  int d_wallID; 
  int d_flowID; 
  int i_n, j_n, k_n, theta_n, phi_n;
  string sample_sche;
  
}; // end class RMCRTRadiationModel

template <class FCV> 
void RMCRTRadiationModel::interpCCTemperatureToFC( constCCVariable<int>& cellType,
                                              FCV& Tf, IntVector dir, IntVector highIdx,  
                                              CCVariable<double>& T, const Patch* p )
{

  //NOTE!:
  // Here we have assume piece-wise interpolation if a cell is any kind of boundary.
  // Do we only want to do this for walls?  If yes, one needs to change the logic below. 
  
  // get the index of the current direction
  int mydir = -1; 
  if      ( dir.x() !=0 ) mydir = 0;
  else if ( dir.y() !=0 ) mydir = 1;
  else if ( dir.z() !=0 ) mydir = 2; 
  
   
  for (CellIterator iter=p->getCellIterator(); !iter.done(); iter++){
    
    IntVector c = *iter; 
    IntVector cm1 = *iter - dir;
 
    if ( cellType[c] == d_flowID && cellType[cm1] == d_flowID ) {
      
      Tf[c] = ( T[c] + T[cm1] ) / 2.0;
 
      //    } else if ( cellType[c] != d_flowID | cellType[cm1] != d_flowID ) {
    } else if ( (cellType[c] != d_flowID) || (cellType[cm1] != d_flowID) ) {

      if ( cellType[c] !=d_flowID ) {
        //current cell is a wall
        Tf[c] = T[c]; 
      } else if ( cellType[c] == d_flowID && cellType[cm1] != d_flowID ) {
        //neighbor is a wall and current cell is flow
        Tf[c] = T[cm1]; 
      } else {
        //both are walls 
        Tf[c] = T[c]; 
      }
    }

    bool doBoundary=false; 
    // subtract 1 because of the "extra cells" used in Arches
    if ( mydir == 0 ) {
      if ( c.x() == highIdx.x() - 1) 
        doBoundary = true; 
    }
    else if ( mydir == 1 ) {
      if ( c.y() == highIdx.y() - 1) 
        doBoundary = true; 
    }
    else if ( mydir == 2 ) {
      if ( c.z() == highIdx.z() - 1) 
        doBoundary = true; 
    }
    
    //do + boundary
    if ( doBoundary ) {

      IntVector cp1 = *iter + dir; 
      if ( cellType[cp1] != d_flowID ) {

        Tf[cp1] = T[cp1];
 
      } else {

        Tf[cp1] = ( T[c] + T[cp1] ) / 2.0; 

      }
    }
  } // end interpolation 
}


} // end uintah namespace

#endif
