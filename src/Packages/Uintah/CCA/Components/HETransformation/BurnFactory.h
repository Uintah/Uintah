#ifndef _BURNFACTORY_H_
#define _BURNFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  class Burn;

  class BurnFactory
  {
  public:
    static Burn* create(ProblemSpecP& ps);

  };

} // End namespace Uintah

#endif /* _HEBURNFACTORY_H_ */

