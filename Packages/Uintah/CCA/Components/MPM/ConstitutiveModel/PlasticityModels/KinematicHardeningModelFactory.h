#ifndef _KINEMATIC_HARDENING_MODELFACTORY_H_
#define _KINEMATIC_HARDENING_MODELFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  class KinematicHardeningModel;
  class MPMLabel;

  class KinematicHardeningModelFactory
  {
  public:
    // this function has a switch for all known mat_types
    static KinematicHardeningModel* create(ProblemSpecP& ps);
    static KinematicHardeningModel* createCopy(const KinematicHardeningModel* pm);
  };
} // End namespace Uintah
      
#endif /* _KINEMATIC_HARDENING_MODELFACTORY_H_ */
