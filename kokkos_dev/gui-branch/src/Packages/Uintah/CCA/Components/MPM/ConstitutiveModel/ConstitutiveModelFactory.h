#ifndef _CONSTITUTIVEMODELFACTORY_H_
#define _CONSTITUTIVEMODELFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  class ConstitutiveModel;
  class MPMLabel;

  class ConstitutiveModelFactory
  {
  public:
    // this function has a switch for all known mat_types
    // and calls the proper class' readParameters()
    // addMaterial() calls this
    static ConstitutiveModel* create(ProblemSpecP& ps, MPMLabel* lb);
  };
} // End namespace Uintah
      
#endif /* _CONSTITUTIVEMODELFACTORY_H_ */
