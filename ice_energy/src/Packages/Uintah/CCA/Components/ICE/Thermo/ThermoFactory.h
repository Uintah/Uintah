#ifndef Uintah_ThermoFactory_h
#define Uintah_ThermoFactory_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  class ThermoInterface;

  class ThermoFactory
  {
  public:
    // this function has a switch for all known mat_types
    // and calls the proper class' readParameters()
    // addMaterial() calls this
    static ThermoInterface* create(ProblemSpecP& ps);
  };

} // End namespace Uintah

#endif /* Uintah_ThermoFactory_h */

