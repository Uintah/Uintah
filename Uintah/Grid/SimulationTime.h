#ifndef UINTAH_HOMEBREW_SimulationTime_H
#define UINTAH_HOMEBREW_SimulationTime_H

#include <Uintah/Interface/ProblemSpecP.h>

namespace Uintah {
  namespace Grid {
    using Uintah::Interface::ProblemSpecP;
    
    /**************************************
      
      CLASS
        SimulationTime
      
        Short Description...
      
      GENERAL INFORMATION
      
        SimulationTime.h
      
        Steven G. Parker
        Department of Computer Science
        University of Utah
      
        Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
        Copyright (C) 2000 SCI Group
      
      KEYWORDS
        SimulationTime
      
      DESCRIPTION
        Long description...
      
      WARNING
      
      ****************************************/
    
    class SimulationTime {
    public:
      SimulationTime(const ProblemSpecP& params);
      double maxTime;
      double initTime;
      double delt_min;
      double delt_max;

    private:
      SimulationTime(const SimulationTime&);
      SimulationTime& operator=(const SimulationTime&);
      
    };

    
  } // end namespace Grid
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/04/13 06:51:02  sparker
// More implementation to get this to work
//
//

#endif
