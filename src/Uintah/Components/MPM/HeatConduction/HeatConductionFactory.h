#ifndef __HeatConductionFactory__
#define __HeatConductionFactory__

#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Grid/SimulationStateP.h>

namespace Uintah {
namespace MPM {

class HeatConduction;

class HeatConductionFactory {
public:
  static HeatConduction* create(const ProblemSpecP& ps,SimulationStateP& d_sS);
};
      
} // end namespace MPM
} // end namespace Uintah

#endif //__HeatConductionFactory__

// $Log$
// Revision 1.1  2000/06/20 17:59:25  tan
// Heat Conduction model created to move heat conduction part of code from MPM.
// Thus make MPM clean and easy to maintain.
//
