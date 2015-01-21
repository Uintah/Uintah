#ifndef _PLASTICITYMODELFACTORY_H_
#define _PLASTICITYMODELFACTORY_H_

#include <Core/ProblemSpec/ProblemSpecP.h>
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
    static PlasticityModel* create(ProblemSpecP& ps);
    static PlasticityModel* createCopy(const PlasticityModel* pm);
  };
} // End namespace Uintah
      
#endif /* _PLASTICITYMODELFACTORY_H_ */
