#ifndef _CONSTITUTIVEMODELFACTORY_H_
#define _CONSTITUTIVEMODELFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  class ConstitutiveModel;
  class MPMLabel;

  class ConstitutiveModelFactory
  {
  public:
    // this function has a switch for all known mat_types
    
    static ConstitutiveModel* create(ProblemSpecP& ps, MPMLabel* lb,int n8or27,
				     std::string integrator);
  };
} // End namespace Uintah
      
#endif /* _CONSTITUTIVEMODELFACTORY_H_ */
