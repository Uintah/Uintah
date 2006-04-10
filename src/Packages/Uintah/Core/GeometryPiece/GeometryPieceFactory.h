#ifndef __GEOMETRY_OBJECT_FACTORY_H__
#define __GEOMETRY_OBJECT_FACTORY_H__

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include   <vector>
#include   <map>
#include   <string>
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
  private:

    // This variable records all named GeometryPieces, so that if they
    // are referenced a 2nd time, they don't have to be rebuilt.
    //
    // Assuming multiple GeometryPieceFactory's will not exist and if
    // they do, they won't be executing at the same time (in different
    // threads)... If this is not the case, then this variable should
    // be locked...
    static std::map<std::string,GeometryPiece*> namedPieces_;
  };

} // End namespace Uintah

#endif /* __GEOMETRY_PIECE_FACTORY_H__ */
