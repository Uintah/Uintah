#ifndef __Uintah_MPM_MPMPhysicalBCFactory__
#define __Uintah_MPM_MPMPhysicalBCFactory__

#include <Uintah/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <vector>

namespace Uintah {
namespace MPM {

  class MPMPhysicalBCFactory
  {
  public:
      static void create(const ProblemSpecP& ps);
      static std::vector<MPMPhysicalBC*> mpmPhysicalBCs;
  };

} // end namespace MPM
} // end namespace Uintah

#endif /* __Uintah_MPM_MPMPhysicalBCFactory__ */

// $Log$
// Revision 1.2  2000/08/18 20:30:04  tan
// Fixed some bugs in SerialMPM, mainly in applyPhysicalBC.
//
// Revision 1.1  2000/08/07 00:43:31  tan
// Added MPMPhysicalBC class to handle all kinds of physical boundary conditions
// in MPM.  Currently implemented force boundary conditions.
//
//
