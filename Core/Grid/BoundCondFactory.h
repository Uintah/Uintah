#ifndef __BOUND_COND_FACTORY_H__
#define __BOUND_COND_FACTORY_H__

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/BoundCondData.h>
#include <Packages/Uintah/Core/Grid/BoundCondBase.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

class BoundCondFactory
{
public:
  // this function has a switch for all known BC_types
  static void create(const ProblemSpecP& ps,BoundCondData& bcs);
  static void create(ProblemSpecP& ps,BoundCondBase* &bc, 
		     int& mat_id);
};

} // End namespace Uintah

#endif /* __BOUND_COND_FACTORY_H__ */
