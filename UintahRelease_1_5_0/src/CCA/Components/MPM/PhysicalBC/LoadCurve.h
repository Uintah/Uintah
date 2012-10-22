/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

using namespace SCIRun;

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

      // Get the load curve id
      inline int getID() const {return d_id;}

      // Get the load at time t
      inline T getLoad(double t) {

        int ntimes = static_cast<int>(d_time.size());
         if (t >= d_time[ntimes-1]) return d_load[ntimes-1];

         for (int ii = 1; ii < ntimes; ++ii) {
           if (t <= d_time[ii]) {
             double s = (d_time[ii]-t)/(d_time[ii]-d_time[ii-1]);
             return (d_load[ii-1]*s + d_load[ii]*(1.0-s));
           } 
         }

         return d_load[0];
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
      std::vector<T> d_load;
      int d_id;
   };

   // Construct a load curve from the problem spec
   template<class T>
   LoadCurve<T>::LoadCurve(ProblemSpecP& ps) 
   {
      ProblemSpecP loadCurve = ps->findBlock("load_curve");
      if (!loadCurve) 
         throw ProblemSetupException("**ERROR** No load curve specified.", __FILE__, __LINE__);
      loadCurve->require("id", d_id);
      for (ProblemSpecP timeLoad = loadCurve->findBlock("time_point");
           timeLoad != 0;
           timeLoad = timeLoad->findNextBlock("time_point")) {
         double time = 0.0;
         T load;
         timeLoad->require("time", time);
         timeLoad->require("load", load);
         d_time.push_back(time);
         d_load.push_back(load);
      }
   }

   template<class T>
     void LoadCurve<T>::outputProblemSpec(ProblemSpecP& ps) 
     {
       ProblemSpecP lc_ps = ps->appendChild("load_curve");
       lc_ps->appendElement("id",d_id);
       for (int i = 0; i<(int)d_time.size();i++) {
         ProblemSpecP time_ps = lc_ps->appendChild("time_point");
         time_ps->appendElement("time",d_time[i]);
         time_ps->appendElement("load",d_load[i]);
       }

     }
   

} // End namespace Uintah

#endif
