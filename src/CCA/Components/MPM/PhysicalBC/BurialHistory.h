/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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

#ifndef UINTAH_MPM_BURIALHiSTORY_H
#define UINTAH_MPM_BURIALHiSTORY_H

#include <vector>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Exceptions/ProblemSetupException.h>

namespace Uintah {

/**************************************

CLASS
   BurialHistory
   
   Load Curve for MPM
 
GENERAL INFORMATION

   BurialHistory.h

   James Guilkey
   Laird Avenue Consulting
   Department of Mechanical Engineering, University of Utah

KEYWORDS
   BurialHistory

DESCRIPTION
   Stores the burial history for a sandstone sample.  The data stored are:

        time_Ma                 Time in units of millions of years ago
        depth_m                 Burial depth in meters
        temp_C                  Temperature in Celcius
        fluidOverPressure_bar   Fluid Over Pressure in bar
        fluidPressure_bar       Fluid Pressure in bar
        effectiveStress_bar     Effective Stress in bar
        sigma_h_bar             Small horizontal Stress in bar
        sigma_H_bar             Large horizontal Stress in bar
        sigma_V_bar             Vertical Stress in bar
        water_Saturation_pct    Water Saturation in percent
        UintahDissolutionTime   amount of Uintah time to spend in the
                                dissolution portion of this level
        QuartzGrowthVec_fr      Fraction of the total quartz overgrowth
                                to take place at a given period in the burial
                                history
        EndUponCompletion       bool, if true, then simulation exits
                                at the end of the interval

WARNING
   Number of entries for each category needs to be the same. 
   Otherwise behavior is undefined.
  
****************************************/

   class BurialHistory {

   public:
      // Constructor and Destructor
      BurialHistory(/*const ProcessorGroup* myworld*/);
      ~BurialHistory();
      int  populate(ProblemSpecP& ps);
      void outputProblemSpec(ProblemSpecP& ps);

      // Get the number of points on the load curve
      inline int numberOfPointsOnBurialHistory() { 
        return (int) d_time_Ma.size(); 
      }

      // Get the time at a given index
      inline double getTime_Ma(int index) {
       return ((index < (int) d_time_Ma.size()) ? d_time_Ma[index] : 0.0);
      }

      // Get the depth at a given index
      inline double getDepth_m(int index) {
       return ((index < (int) d_time_Ma.size()) ? d_depth_m[index] : 0);
      }

      // Get the temperature associated with a given index
      inline double getTemperature_K(int index) {
       return ((index < (int) d_time_Ma.size()) ? d_temperature_K[index] :
                              d_temperature_K[d_time_Ma.size()]);
      }

      // Get the fluid pressure at a given index
      inline double getFluidPressure_bar(int index) {
       return ((index < (int) d_time_Ma.size()) ?
                                              d_fluidPressure_bar[index] : 0);
      }

      // Get the fluid over pressure at a given index
      inline double getFluidOverPressure_bar(int index) {
       return ((index < (int) d_time_Ma.size()) ?
                                          d_fluidOverPressure_bar[index] : 0);
      }

      // Get the effective stress at a given index
      inline double getEffectiveStress(int index) {
       return ((index < (int) d_time_Ma.size()) ? 
                                            d_effectiveStress_bar[index] : 0);
      }

      // Get the small horizontal stress at a given index
      inline double getSigma_h(int index) {
       return ((index < (int) d_time_Ma.size()) ? 
                                            d_sigma_h_bar[index] : 0);
      }

      // Get the large horizontal stress at a given index
      inline double getSigma_H(int index) {
       return ((index < (int) d_time_Ma.size()) ? 
                                            d_sigma_H_bar[index] : 0);
      }

      // Get the vertical stress at a given index
      inline double getSigma_V(int index) {
       return ((index < (int) d_time_Ma.size()) ? 
                                            d_sigma_V_bar[index] : 0);
      }

      // Get the water saturation percentage at a given index
      inline double getWaterSaturation_pct(int index) {
       return ((index < (int) d_time_Ma.size()) ? 
                                            d_waterSaturation_pct[index] : 0);
      }

      // Find out if the simulation should end after this level is complete
      inline bool getEndOnCompletion(int index) {
       return ((index < (int) d_time_Ma.size()) ? 
                                            d_endOnCompletion[index] : false);
      }

      inline void setTime_Ma(int index, double newTime) {
        d_time_Ma[index] = newTime;
      }

      inline void setCurrentPhaseType(std::string curPhaseType) {
        d_currentPhaseType = curPhaseType;
      }

      // Get the next index
      inline int getNextIndex(double t) {
        int ntimes = static_cast<int>(d_time_Ma.size());
        if (t >= d_time_Ma[ntimes-1]){
          return ntimes;
        }

        for (int ii = 1; ii < ntimes; ++ii) {
          if (t <= d_time_Ma[ii]) {
            return ii;
          } 
        }
        return 0;
      }

      // Get the next index
      inline double getIndexAtPressure(double P) {
        int ntimes = static_cast<int>(d_time_Ma.size());

        for (int ii = ntimes-1; ii > 0; ii--) {
          if (fabs(P) <= fabs(d_effectiveStress_bar[ii])) {
            return ii;
          }
        }
        return 0;
      }

      inline void setCurrentIndex(int index) {
        d_CI = index;
      }

      inline int getCurrentIndex() {
        return d_CI;
      }

      inline double getUintahDissolutionTime(int index) {
        return d_uintahDissolutionTime[index];
      }

      inline double getQuartzGrowthVec_fr(int index) {
        return d_quartzGrowthVec_fr[index];
      }

   private:
      // Prevent copying
      BurialHistory(const BurialHistory&);
      BurialHistory& operator=(const BurialHistory&);
      
      // Private Data
      double d_pressure_conversion_factor;
      double d_ramp_time;
      double d_stableKE;
      double d_settle_time;
      double d_hold_time;
      double d_CI;
      std::string d_currentPhaseType;
      std::vector<double> d_time_Ma;
      std::vector<double> d_depth_m;
      std::vector<double> d_temperature_K;
      std::vector<double> d_fluidOverPressure_bar;
      std::vector<double> d_fluidPressure_bar;
      std::vector<double> d_effectiveStress_bar;
      std::vector<double> d_sigma_h_bar;
      std::vector<double> d_sigma_H_bar;
      std::vector<double> d_sigma_V_bar;
      std::vector<double> d_waterSaturation_pct;
      std::vector<double> d_uintahDissolutionTime;
      std::vector<double> d_quartzGrowthVec_fr;
      std::vector<bool>   d_endOnCompletion;
   };

} // End namespace Uintah

#endif
