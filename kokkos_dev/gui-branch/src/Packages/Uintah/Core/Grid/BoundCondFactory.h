#ifndef __BOUND_COND_FACTORY_H__
#define __BOUND_COND_FACTORY_H__

// add #include for each ConstitutiveModel here
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/BoundCondData.h>

#include <vector>

namespace Uintah {

class BoundCondFactory
{
public:
  // this function has a switch for all known BC_types
  static void create(const ProblemSpecP& ps,BCData& bcs);
};

} // End namespace Uintah

#endif /* __BOUND_COND_FACTORY_H__ */
