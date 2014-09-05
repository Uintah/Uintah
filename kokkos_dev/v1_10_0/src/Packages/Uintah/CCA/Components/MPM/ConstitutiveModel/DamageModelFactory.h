#ifndef _DAMAGEMODELFACTORY_H_
#define _DAMAGEMODELFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <string>
namespace Uintah {

  class DamageModel;
  class MPMLabel;

  class DamageModelFactory
  {
  public:
    // this function has a switch for all known mat_types
    // and calls the proper class' readParameters()
    // addMaterial() calls this
    static DamageModel* create(ProblemSpecP& ps);
  };
} // End namespace Uintah
      
#endif /* _DAMAGEMODELFACTORY_H_ */
