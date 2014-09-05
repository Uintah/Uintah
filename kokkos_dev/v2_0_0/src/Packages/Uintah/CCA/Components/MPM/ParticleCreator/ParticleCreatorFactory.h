#ifndef _PARTICLECREATORFACTORY_H_
#define _PARTICLECREATORFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  class ParticleCreator;
  class MPMLabel;

  class ParticleCreatorFactory
  {
  public:
    // this function has a switch for all known mat_types
    // and calls the proper class' readParameters()
    // addMaterial() calls this
    static ParticleCreator* create(ProblemSpecP& ps, MPMLabel* lb,int n8or27,
				   std::string integrator);
  };
} // End namespace Uintah
      
#endif /* _PARTICLECREATORFACTORY_H_ */
