#ifndef __Uintah_MPM_MPMPhysicalBCFactory__
#define __Uintah_MPM_MPMPhysicalBCFactory__

#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  class MPMPhysicalBCFactory
  {
  public:
    static void create(const ProblemSpecP& ps);
    static void clean(); // delete all mpmPhysicalBCs
    static std::vector<MPMPhysicalBC*> mpmPhysicalBCs;
  };
} // End namespace Uintah


#endif /* __Uintah_MPM_MPMPhysicalBCFactory__ */

