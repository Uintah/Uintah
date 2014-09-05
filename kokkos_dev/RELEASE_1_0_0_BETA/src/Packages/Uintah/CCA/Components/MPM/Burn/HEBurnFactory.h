#ifndef _HEBURNFACTORY_H_
#define _HEBURNFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  class HEBurn;

  class HEBurnFactory
  {
  public:
    static HEBurn* create(ProblemSpecP& ps);

  };

} // End namespace Uintah

#endif /* _HEBURNFACTORY_H_ */
