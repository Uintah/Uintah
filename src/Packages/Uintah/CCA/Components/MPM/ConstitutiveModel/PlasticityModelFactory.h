#ifndef _PLASTICITYMODELFACTORY_H_
#define _PLASTICITYMODELFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  class PlasticityModel;
  class MPMLabel;

  class PlasticityModelFactory
  {
  public:
    // this function has a switch for all known mat_types
    // and calls the proper class' readParameters()
    // addMaterial() calls this
    static PlasticityModel* create(ProblemSpecP& ps);
  };
} // End namespace Uintah
      
#endif /* _PLASTICITYMODELFACTORY_H_ */
