/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

#ifndef UINTAH_MPM_LOADCURVE_H
#define UINTAH_MPM_LOADCURVE_H

#include <vector>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Exceptions/ProblemSetupException.h>

namespace Uintah {

/**************************************

CLASS
   LoadCurve
   
   Load Curve for MPM
 
GENERAL INFORMATION

   LoadCurve.h

   Biswajit Banerjee
   Department of Mechanical Engineering, University of Utah
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

KEYWORDS
   LoadCurve

DESCRIPTION
   Stores the load curve data (pressure,temperature,force,
   displacement vs. time)

WARNING
   Number of load and time values should match. 
   Otherwise behavior is undefined.
  
****************************************/

   template<class T>
   class LoadCurve {

   public:
 
      // Constructor and Destructor
      LoadCurve(ProblemSpecP& ps);
      inline ~LoadCurve() { };
      void outputProblemSpec(ProblemSpecP& ps);

      // Get the number of points on the load curve
      inline int numberOfPointsOnLoadCurve() { 
        return (int) d_time.size(); 
      }

      // Get the time
      inline double getTime(int index) {
         return ((index < (int) d_time.size()) ? d_time[index] : 0.0);
      }

      // Get the load
      inline T getLoad(int index) {
         return ((index < (int) d_time.size()) ? d_load[index] : 0);
      }

      // Get the loading phase (ramp, settle, or hold)
      inline std::string getPhase(int index) {
         return ((index < (int) d_time.size()) ? d_phaseType[index] : 0);
      }

      // Get the maxKE associated with a given index
      inline T getMaxKE(int index) {
         return ((index < (int) d_time.size()) ? d_maxKE[index] : 0);
      }

      // Get the burial history index with a given load curve index
      inline T getBHIndex(int index) {
         return ((index < (int) d_time.size()) ? d_BHIndex[index] : 0);
      }

      inline void setTime(int index, double newTime) {
        d_time[index] = newTime;
      }

      inline void setLoad(int index, T newLoad) {
        d_load[index] = newLoad;
      }

      // Get the load curve id
      inline int getID() const {return d_id;}

      // Get the load curve id
      inline int getMatl() const {return d_matl;}

      // Get the load at time t
      inline T getLoad(double t) {

        int ntimes = static_cast<int>(d_time.size());
         if (t >= d_time[ntimes-1]) return d_load[ntimes-1];

         for (int ii = 1; ii < ntimes; ++ii) {
           if (t <= d_time[ii]) {
             double s = (d_time[ii]-t)/((d_time[ii]-d_time[ii-1])+1.e-100);
             return (d_load[ii-1]*s + d_load[ii]*(1.0-s));
           } 
         }

         return d_load[0];
      }

      // Get the next index
      inline int getNextIndex(double t) {
        int ntimes = static_cast<int>(d_time.size());
         if (t >= d_time[ntimes-1]){
           return ntimes;
         }

         for (int ii = 1; ii < ntimes; ++ii) {
           if (t <= d_time[ii]) {
             return ii;
           } 
         }
         return 0;
      }

   private:
      // Prevent creation of empty object
      LoadCurve();

      // Prevent copying
      LoadCurve(const LoadCurve&);
      LoadCurve& operator=(const LoadCurve&);
      
      // Private Data
      // Load curve information 
      std::vector<double> d_time;
      std::vector<T>      d_load;
      std::vector<double> d_maxKE;
      std::vector<std::string> d_phaseType;
      std::vector<int>    d_BHIndex;
      int d_id;
      int d_matl;
      double d_curTime;
      bool d_UBH;
      std::string d_BHE;
   };

   // Construct a load curve from the problem spec
   template<class T>
   LoadCurve<T>::LoadCurve(ProblemSpecP& ps) 
   {
     ProblemSpecP loadCurve = ps->findBlock("load_curve");
     if (!loadCurve){
        throw ProblemSetupException("**ERROR** No load curve specified.", 
                                     __FILE__, __LINE__);
     }
     loadCurve->require("id", d_id);
     loadCurve->getWithDefault("material", d_matl, -99);
     loadCurve->getWithDefault("use_burial_history", d_UBH, false);
     loadCurve->getWithDefault("burial_history_entry", d_BHE,
                                            "effectiveStress_bar");
     if(d_UBH){
      ProblemSpecP root = ps->getRootNode();
      ProblemSpecP TimeBlock = root->findBlock("Time");
      TimeBlock->getWithDefault("currentTime", d_curTime, 0.0);
      
      ProblemSpecP burHist = root->findBlock("BurialHistory");
      if (!burHist){ 
        throw ProblemSetupException("**ERROR** No burial history specified.", 
                                     __FILE__, __LINE__);
      } else {
        double p_c_f, ramp_time, settle_time, hold_time, stableKE;
        burHist->getWithDefault("pressure_conversion_factor", p_c_f,       1.0);
        burHist->getWithDefault("ramp_time",                  ramp_time,   1.0);
        burHist->getWithDefault("settle_time",                settle_time, 2.0);
        burHist->getWithDefault("hold_time",                  hold_time,   1.0);
        burHist->getWithDefault("stableKE",                   stableKE,    2.0e-6);
        std::vector<double> time_Ma;
        std::vector<double> Stress_bar;
        std::vector<double> UintahDissolutionTime;
        std::vector<double> UintahPrecipitationTime;
        for( ProblemSpecP timePoint = burHist->findBlock("time_point");
            timePoint != nullptr;
            timePoint = timePoint->findNextBlock("time_point") ) {
          double time      = 0.0;
          double Stress    = 0.0;
          double UDT       = 0.0;
          double UPT       = 0.0;
          timePoint->require("time_Ma",                 time);
          if(d_BHE == "effectiveStress_bar"){
            timePoint->require("effectiveStress_bar",   Stress);
            proc0cout << "d_BHE = " << d_BHE << std::endl;
          } else if(d_BHE == "sigma_h_bar"){
            timePoint->require("sigma_h_bar",           Stress);
            proc0cout << "d_bHE = " << d_BHE << std::endl;
          } else if(d_BHE == "sigma_H_bar"){
            timePoint->require("sigma_H_bar",           Stress);
            proc0cout << "d_BhE = " << d_BHE << std::endl;
          } else if(d_BHE == "sigma_V_bar"){
            timePoint->require("sigma_V_bar",           Stress);
            proc0cout << "d_BHe = " << d_BHE << std::endl;
          }
          timePoint->getWithDefault("UintahDissolutionTime",   UDT, 0);
          timePoint->getWithDefault("UintahPrecipitationTime", UPT, 0);

          time_Ma.push_back(time);
          Stress_bar.push_back(Stress);
          UintahDissolutionTime.push_back(UDT);
          UintahPrecipitationTime.push_back(UPT);
        }
        int CI = 0;
        burHist->getWithDefault("current_index", CI, time_Ma.size()-1);
        double maxKE = 9.9e99;
        double uintahTime=d_curTime;

        // Fill up the load curve based on burial history data
        d_time.push_back(uintahTime);
        d_load.push_back(-Stress_bar[CI]*p_c_f);
        d_maxKE.push_back(maxKE);
        d_BHIndex.push_back(CI);
        for(int i=CI-1;i>=0;i--){
          // ramp phase
          d_phaseType.push_back("ramp");
          uintahTime+=ramp_time;
          d_time.push_back(uintahTime);
          d_load.push_back(-Stress_bar[i]*p_c_f);
          d_maxKE.push_back(maxKE);
          d_BHIndex.push_back(i);

          // settle down phase
          d_phaseType.push_back("settle");
          uintahTime+=settle_time;
          d_time.push_back(uintahTime);
          d_load.push_back(-Stress_bar[i]*p_c_f);
          d_maxKE.push_back(stableKE);
          d_BHIndex.push_back(i);

          // hold or dissolution phase
          if(UintahDissolutionTime[i] > 0. && UintahPrecipitationTime[i]>0.){
            if(UintahDissolutionTime[i] == UintahPrecipitationTime[i]){
             d_phaseType.push_back("dissolution_and_precipitation");
             uintahTime+=UintahDissolutionTime[i];
             d_time.push_back(uintahTime);
             d_load.push_back(-Stress_bar[i]*p_c_f);
             d_maxKE.push_back(maxKE);
             d_BHIndex.push_back(i);
            } else { // The times aren't equal (might be nearly so) do do them
                     // sequentially
             if(UintahDissolutionTime[i] < UintahPrecipitationTime[i]){
              // First do the precipitation, then the dissolution
              d_phaseType.push_back("precipitation");
              uintahTime+=UintahPrecipitationTime[i];
              d_time.push_back(uintahTime);
              d_load.push_back(-Stress_bar[i]*p_c_f);
              d_maxKE.push_back(maxKE);
              d_BHIndex.push_back(i);

              d_phaseType.push_back("dissolution");
              uintahTime+=UintahDissolutionTime[i];
              d_time.push_back(uintahTime);
              d_load.push_back(-Stress_bar[i]*p_c_f);
              d_maxKE.push_back(maxKE);
              d_BHIndex.push_back(i);
             }
             if(UintahDissolutionTime[i] > UintahPrecipitationTime[i]){
              // First do the precipitation, then the dissolution
              d_phaseType.push_back("dissolution");
              uintahTime+=UintahDissolutionTime[i];
              d_time.push_back(uintahTime);
              d_load.push_back(-Stress_bar[i]*p_c_f);
              d_maxKE.push_back(maxKE);
              d_BHIndex.push_back(i);

              d_phaseType.push_back("precipitation");
              uintahTime+=UintahPrecipitationTime[i];
              d_time.push_back(uintahTime);
              d_load.push_back(-Stress_bar[i]*p_c_f);
              d_maxKE.push_back(maxKE);
              d_BHIndex.push_back(i);
             }
            }
          } else if(UintahDissolutionTime[i] > 0.){
            d_phaseType.push_back("dissolution");
            uintahTime+=UintahDissolutionTime[i];
            d_time.push_back(uintahTime);
            d_load.push_back(-Stress_bar[i]*p_c_f);
            d_maxKE.push_back(maxKE);
            d_BHIndex.push_back(i);
          } else if(UintahPrecipitationTime[i] > 0.){
            d_phaseType.push_back("precipitation");
            uintahTime+=UintahPrecipitationTime[i];
            d_time.push_back(uintahTime);
            d_load.push_back(-Stress_bar[i]*p_c_f);
            d_maxKE.push_back(maxKE);
            d_BHIndex.push_back(i);
          } else {
            d_phaseType.push_back("hold");
            uintahTime+=hold_time;
            d_time.push_back(uintahTime);
            d_load.push_back(-Stress_bar[i]*p_c_f);
            d_maxKE.push_back(maxKE);
            d_BHIndex.push_back(i);
          }
        }
        d_phaseType.push_back("hold");
      }
     } else {
       for( ProblemSpecP timeLoad = loadCurve->findBlock("time_point");
           timeLoad != nullptr;
           timeLoad = timeLoad->findNextBlock("time_point") ) {
         double time = 0.0;
         double maxKE = 9.9e99;
         int BHIndex;
         std::string phaseType;
         T load;
         timeLoad->require(       "time",      time);
         timeLoad->require(       "load",      load);
         timeLoad->getWithDefault("maxKE",     maxKE,     9.9e99);
         timeLoad->getWithDefault("phaseType", phaseType, "ramp");
         timeLoad->getWithDefault("BHIndex",   BHIndex,   0);
         d_time.push_back(time);
         d_load.push_back(load);
         d_maxKE.push_back(maxKE);
         d_phaseType.push_back(phaseType);
         d_BHIndex.push_back(BHIndex);
       }
       for(int i = 1; i<(int)d_time.size(); i++){
         if (d_time[i]==d_time[i-1]){
         throw ProblemSetupException("**ERROR** Identical time entries in Load Curve",
                                        __FILE__, __LINE__);
         }
       }
     }

     for(int i = 1; i<(int)d_time.size(); i++){
       if (d_time[i]==d_time[i-1]){
           throw ProblemSetupException("**ERROR** Identical time entries in Load Curve built from burial history",
                                      __FILE__, __LINE__);
       }
     }
   }

   template<class T>
     void LoadCurve<T>::outputProblemSpec(ProblemSpecP& ps) 
     {
       ProblemSpecP lc_ps = ps->appendChild("load_curve");
       lc_ps->appendElement("id",d_id);
       lc_ps->appendElement("material",d_matl);
       lc_ps->appendElement("use_burial_history", false);
       lc_ps->appendElement("burial_history_entry", d_BHE);
       for (int i = 0; i<(int)d_time.size();i++) {
         ProblemSpecP time_ps = lc_ps->appendChild("time_point");
         time_ps->appendElement("time",      d_time[i]);
         time_ps->appendElement("load",      d_load[i]);
         time_ps->appendElement("maxKE",     d_maxKE[i]);
         time_ps->appendElement("phaseType", d_phaseType[i]);
         time_ps->appendElement("BHIndex",   d_BHIndex[i]);
       }
     }

} // End namespace Uintah

#endif
