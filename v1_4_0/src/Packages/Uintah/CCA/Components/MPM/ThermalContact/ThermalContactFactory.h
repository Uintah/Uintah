#ifndef __ThermalContactFactory__
#define __ThermalContactFactory__

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>

namespace Uintah {

  class ThermalContact;
  class MPMLabel;

  class ThermalContactFactory {
  public:
   static ThermalContact* create(const ProblemSpecP& ps,SimulationStateP& d_sS,
								MPMLabel* lb);

  };

} // End namespace Uintah
      

#endif //__ThermalContactFactory__

