#ifndef __BOUND_COND_FACTORY_H__
#define __BOUND_COND_FACTORY_H__

// add #include for each ConstitutiveModel here
#include <Uintah/Interface/ProblemSpecP.h>
#include <vector>

using namespace Uintah;
namespace Uintah {
   
  class BoundCond;
      
  class BoundCondFactory
    {
    public:
      // this function has a switch for all known BC_types
      static void create(const ProblemSpecP& ps,
			 std::vector<BoundCond*>& bcs);
    };
} // end namespace Uintah


#endif /* __BOUND_COND_FACTORY_H__ */
