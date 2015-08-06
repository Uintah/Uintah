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

#ifndef RMCRTCOMMON_H
#define RMCRTCOMMON_H

#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Disclosure/TypeDescription.h>
#include <sci_defs/uintah_defs.h>

#include <iostream>
#include <cmath>
#include <string>
#include <vector>

//==========================================================================
/**
 * @class RMCRTCommon
 * @author Todd Harman
 * @date June, 2014
 *
 * @brief Methods, functions and variables that are common to both the 
 *        radiometer() and RMCRT()  
 *        The variable sigmaT4Pi and abskg can be cast as either doubles or floats
 #        This allows for a significant savings in memory and communication costs
 *
 */
class MTRand;

namespace Uintah{

  class RMCRTCommon  {

    public: 

      RMCRTCommon( TypeDescription::Type FLT_DBL );
      ~RMCRTCommon(); 

       //__________________________________
      //  Helpers
      /** @brief map the component VarLabels to RMCRT class VarLabels */
     void registerVarLabels(int   matl,
                            const VarLabel*  abskg,
                            const VarLabel* temperature,
                            const VarLabel* celltype,
                            const VarLabel* divQ);

      //__________________________________
      // @brief Update the running total of the incident intensity */
      template <class T>
      void  updateSumI ( Vector& ray_direction, // can change if scattering occurs
                         Vector& ray_location,
                         const IntVector& origin,
                         const Vector& Dx,
                         constCCVariable< T >& sigmaT4Pi,
                         constCCVariable< T >& abskg,
                         constCCVariable<int>& celltype,
                         unsigned long int& size,
                         double& sumI,
                         MTRand& mTwister);

      //__________________________________
      /** @brief Schedule compute of blackbody intensity */ 
      void sched_sigmaT4( const LevelP& level, 
                          SchedulerP& sched,
                          Task::WhichDW temp_dw,
                          const int radCalc_freq,
                          const bool includeEC = true );
      //__________________________________
      //
      template< class T>
      void sigmaT4( const ProcessorGroup* pg,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw,
                    Task::WhichDW which_temp_dw,
                    const int radCalc_freq,
                    const bool includeEC );
      
      //__________________________________
      //
      void reflect(double& fs,
                   IntVector& cur,
                   IntVector& prevCell,
                   const double abskg,
                   bool& in_domain,
                   int& step,
                   bool& sign,
                   double& ray_direction);

      //__________________________________
      //
      void findStepSize(int step[],
                        bool sign[],
                        const Vector& inv_direction_vector);
      
      //__________________________________
      //
      void ray_Origin( MTRand& mTwister,
                       const IntVector origin,
                       const double DyDx, 
                       const double DzDx,
                       const bool useCCRays,
                       Vector& rayOrigin);

      //__________________________________
      //
      Vector findRayDirection( MTRand& mTwister,
                               const bool isSeedRandom,
                               const IntVector& = IntVector(-9,-9,-9),
                               const int iRay = -9);

      //__________________________________
      /** @brief populates a vector of integers with a stochastic array without replacement from 0 to n-1 */
      void randVector( std::vector <int> &int_array,
                       MTRand& mTwister);


      //______________________________________________________________________
      //    Carry Foward tasks
      // transfer a variable from old_dw -> new_dw for convience */   
      void sched_CarryForward_Var ( const LevelP& level,
                                    SchedulerP& scheduler,
                                    const VarLabel* variable );

      bool doCarryForward( const int radCalc_freq );

      void carryForward_Var ( const ProcessorGroup*,
                              const PatchSubset* ,
                              const MaterialSubset*,
                              DataWarehouse*,
                              DataWarehouse*,
                              const VarLabel* variable);
      //__________________________________
      // If neede convert abskg double -> float
      void sched_DoubleToFloat( const LevelP& level,
                                SchedulerP& sched,
                                Task::WhichDW myDW,
                                const int radCalc_freq );

      void DoubleToFloat( const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw,
                          Task::WhichDW which_dw,
                          const int radCalc_freq );


      void doRecompileTaskgraph( const int radCalc_freq );
      
      bool isDbgCell( IntVector me);
                            
      //______________________________________________________________________
      //    Public variables that are used by Radiometer & RMCRT classes
      enum DIR {X=0, Y=1, Z=2, NONE=-9}; 
      //           -x      +x       -y       +y     -z     +z
      enum FACE {EAST=0, WEST=1, NORTH=2, SOUTH=3, TOP=4, BOT=5, nFACES=6};     
      double d_sigma_over_pi;                // Stefan Boltzmann divided by pi (W* m-2* K-4)
      int d_flowCell;                       // HARDWIRED 
      Ghost::GhostType d_gn;
      Ghost::GhostType d_gac;

      SimulationStateP d_sharedState;
      TypeDescription::Type d_FLT_DBL;              // Is algorithm based on doubles or floats
      
      static std::vector<IntVector> d_dbgCells;     // cells that we're interogating when DEBUG is on
      
      // This will create only 1 instance for both Ray() and radiometer() classes to use
      static double d_threshold;
      static double d_sigma;
      static double d_sigmaScat;      
      static bool d_isSeedRandom;     
      static bool d_allowReflect;                   // specify as false when doing DOM comparisons 
           
      // These are initialized once in registerVarLabels().
      static int d_matl;      
      static MaterialSet* d_matlSet;
      static std::string d_abskgBC_tag;             // Needed by BC, manages the varLabel name change when using floats
      
      // Varlabels local to RMCRT
      static const VarLabel* d_sigmaT4Label;
      static const VarLabel* d_abskgLabel;
      static const VarLabel* d_divQLabel;
      
      // VarLabels passed to RMCRT by the component
      static const VarLabel* d_compTempLabel;       //  temperature
      static const VarLabel* d_compAbskgLabel;      //  Absorption Coefficitne
      static const VarLabel* d_cellTypeLabel;       //  cell type marker

  }; // class RMCRTCommon

} // namespace Uintah

#endif
