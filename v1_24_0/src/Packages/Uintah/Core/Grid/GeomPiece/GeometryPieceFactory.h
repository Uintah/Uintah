#ifndef __GEOMETRY_OBJECT_FACTORY_H__
#define __GEOMETRY_OBJECT_FACTORY_H__

// add #include for each ConstitutiveModel here
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  class GeometryPiece;

  class GeometryPieceFactory
  {
  public:
    // this function has a switch for all known go_types
    // and calls the proper class' readParameters()
    // addMaterial() calls this
    static void create(const ProblemSpecP& ps,
		       std::vector<GeometryPiece*>& objs);
  };

} // End namespace Uintah

#endif /* __GEOMETRY_PIECE_FACTORY_H__ */
