#ifndef __SHELL_GEOM_FACTORY_H__
#define __SHELL_GEOM_FACTORY_H__

#include <Core/ProblemSpec/ProblemSpecP.h>

#include <Core/GeometryPiece/uintahshare.h>
namespace Uintah {

  class GeometryPiece;

  class UINTAHSHARE ShellGeometryFactory
  {
  public:
    // This function has a switch for all shell go_types It returns a
    // pointer to the piece that it creates, NULL if does not know how
    // to create the piece.
    static GeometryPiece * create( ProblemSpecP& ps );
  };

} // End namespace Uintah

#endif /* __SHELL_GEOM_FACTORY_H__ */
