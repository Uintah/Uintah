#ifndef __Uintah_MPM_MPMPhysicalBCFactory__
#define __Uintah_MPM_MPMPhysicalBCFactory__

#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <vector>

namespace Uintah {
  class MPMPhysicalBCFactory
  {
  public:
      static void create(const ProblemSpecP& ps);
      static std::vector<MPMPhysicalBC*> mpmPhysicalBCs;
  };
} // End namespace Uintah


#endif /* __Uintah_MPM_MPMPhysicalBCFactory__ */

