#ifndef _CRACK_GEOMETRY_FACTORY_H_
#define _CRACK_GEOMETRY_FACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  class CrackGeometry;
  class MPMLabel;
  class MPMFlags;

  class CrackGeometryFactory
  {
  public:
    // this function has a switch for all known mat_types
    
    static CrackGeometry* create(ProblemSpecP& ps);

  };
} // End namespace Uintah
      
#endif /* _CRACK_GEOMETRY_FACTORY_H_ */
