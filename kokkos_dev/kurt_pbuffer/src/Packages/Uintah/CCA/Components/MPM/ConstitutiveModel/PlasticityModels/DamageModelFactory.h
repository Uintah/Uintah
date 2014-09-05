#ifndef _DAMAGEMODELFACTORY_H_
#define _DAMAGEMODELFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  class DamageModel;
  class MPMLabel;

  class DamageModelFactory
  {
  public:
    // this function has a switch for all known mat_types
    static DamageModel* create(ProblemSpecP& ps);
    static DamageModel* createCopy(const DamageModel* dm);
  };
} // End namespace Uintah
      
#endif /* _DAMAGEMODELFACTORY_H_ */
