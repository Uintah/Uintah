#ifndef __SHELL_GEOM_FACTORY_H__
#define __SHELL_GEOM_FACTORY_H__

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  class GeometryPiece;

  class ShellGeometryFactory
  {
  public:
    // this function has a switch for all shell go_types
    static void create(const ProblemSpecP& ps,
		       std::vector<GeometryPiece*>& objs);
  };

} // End namespace Uintah

#endif /* __SHELL_GEOM_FACTORY_H__ */
