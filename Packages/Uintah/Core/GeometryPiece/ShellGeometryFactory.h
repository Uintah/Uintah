#ifndef __SHELL_GEOM_FACTORY_H__
#define __SHELL_GEOM_FACTORY_H__

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  class GeometryPiece;

  class ShellGeometryFactory
  {
  public:
    // This function has a switch for all shell go_types
    // It returns a pointer to the piece that it creates.
    static GeometryPiece * create( const ProblemSpecP& ps );
  };

} // End namespace Uintah

#endif /* __SHELL_GEOM_FACTORY_H__ */
