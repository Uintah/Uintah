
#ifndef Packages_Uintah_CCA_Components_Examples_RegionDB_h
#define Packages_Uintah_CCA_Components_Examples_RegionDB_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <map>
#include <string>

namespace Uintah {
  class GeometryPiece;
  class RegionDB {
  public:
    RegionDB();
    ~RegionDB();
    void problemSetup(ProblemSpecP& ps, const GridP& grid);
    const GeometryPiece* getObject(const std::string& name) const;
  private:
    void addRegion(GeometryPiece* piece);
    void addRegion(GeometryPiece* piece, const std::string& name);
    typedef std::map<std::string, const GeometryPiece*> MapType;
    MapType db;
  };
}

#endif
