#ifndef __BOUND_COND_FACTORY_H__
#define __BOUND_COND_FACTORY_H__

// add #include for each ConstitutiveModel here
#include <Uintah/Interface/ProblemSpecP.h>
#include <vector>
#include <Uintah/Grid/BoundCondBase.h>

using namespace Uintah;
namespace Uintah {
   
  class BoundCondFactory
    {
    public:
      // this function has a switch for all known BC_types
      static void create(const ProblemSpecP& ps,
			 std::vector<BoundCondBase*>& bcs);
    };
} // end namespace Uintah


#endif /* __BOUND_COND_FACTORY_H__ */
