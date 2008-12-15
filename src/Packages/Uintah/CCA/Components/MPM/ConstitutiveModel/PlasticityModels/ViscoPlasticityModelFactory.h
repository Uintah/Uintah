#ifndef _VISCOPLASTICITYMODELFACTORY_H_
#define _VISCOPLASTICITYMODELFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  class ViscoPlasticityModel;
  class MPMLabel;

  class ViscoPlasticityModelFactory
  {
  public:
    // this function has a switch for all known mat_types
    static ViscoPlasticityModel* create(ProblemSpecP& ps);
    static ViscoPlasticityModel* createCopy(const ViscoPlasticityModel* pm);
  };
} // End namespace Uintah
      
#endif /* _VISCOPLASTICITYMODELFACTORY_H_ */
