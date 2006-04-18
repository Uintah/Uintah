
#ifndef Packages_Uintah_CCA_Components_Examples_RegionDB_h
#define Packages_Uintah_CCA_Components_Examples_RegionDB_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/GeometryPiece/GeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GridP.h>

#include <sgi_stl_warnings_off.h>
#include   <map>
#include   <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  class RegionDB {

  public:
    RegionDB();
    ~RegionDB() {}
    void problemSetup(ProblemSpecP& ps, const GridP& grid);
    GeometryPieceP getObject(const std::string& name) const;

  private:
    void addRegion(GeometryPieceP piece);
    void addRegion(GeometryPieceP piece, const std::string& name);

    typedef std::map<std::string, GeometryPieceP> MapType;
    MapType db;
  };
}

#endif
