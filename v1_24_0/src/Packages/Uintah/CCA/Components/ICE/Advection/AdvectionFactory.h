#ifndef _ADVECTION_FACTORY_H_
#define _ADVECTION_FACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  class Advector;

  class AdvectionFactory
  {
  public:
    // this function has a switch for all known advection_types
    // and calls the proper class' readParameters()
    // addMaterial() calls this
    static Advector* create(ProblemSpecP& ps,
                            std::string& advect_type);
  };

} // End namespace Uintah

#endif /*_ADVECTION_FACTORY_H_ */
