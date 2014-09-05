#ifndef __HeatConductionFactory__
#define __HeatConductionFactory__

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>

namespace Uintah {
class HeatConduction;

class HeatConductionFactory {
public:
  static HeatConduction* create(const ProblemSpecP& ps,SimulationStateP& d_sS);
};
} // End namespace Uintah
      

#endif //__HeatConductionFactory__

