#ifndef __BOUND_COND_FACTORY_H__
#define __BOUND_COND_FACTORY_H__

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/BoundaryConditions/BoundCondBase.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

class BoundCondFactory
{
public:
  // this function has a switch for all known BC_types
  static void create(ProblemSpecP& ps,BoundCondBase* &bc, 
		     int& mat_id);
};

} // End namespace Uintah

#endif /* __BOUND_COND_FACTORY_H__ */
