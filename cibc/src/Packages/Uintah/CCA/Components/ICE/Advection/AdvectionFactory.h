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
    static Advector* create(ProblemSpecP& ps,
                            bool& d_useCompatibleFluxes);
  };

} // End namespace Uintah

#endif /*_ADVECTION_FACTORY_H_ */
