#ifndef __ThermalContactFactory__
#define __ThermalContactFactory__

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/SimulationStateP.h>

namespace Uintah {

  class ThermalContact;
  class MPMLabel;
  class MPMFlags;

  class ThermalContactFactory {
  public:
   static ThermalContact* create(const ProblemSpecP& ps,SimulationStateP& d_sS,
				 MPMLabel* lb,MPMFlags* flag);

  };

} // End namespace Uintah
      

#endif //__ThermalContactFactory__

