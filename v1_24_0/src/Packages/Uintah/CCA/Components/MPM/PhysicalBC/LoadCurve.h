#ifndef UINTAH_MPM_LOADCURVE_H
#define UINTAH_MPM_LOADCURVE_H

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>

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
   Copyright (C) 2003 University of Utah

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
         throw ProblemSetupException("**ERROR** No load curve specified.");
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

} // End namespace Uintah

#endif
