
#ifndef UINTAH_HOMEBREW_Level_H
#define UINTAH_HOMEBREW_Level_H

#include "RefCounted.h"
#include "DataWarehouseP.h"
#include "LevelP.h"
#include <SCICore/Geometry/Point.h>
#include <string>
#include <vector>

class Region;

class Task;
class TypeDescription;

class Level : public RefCounted {
public:
    Level();
    virtual ~Level();

    typedef std::vector<Region*>::iterator regionIterator;
    typedef std::vector<Region*>::const_iterator const_regionIterator;
    const_regionIterator regionsBegin() const;
    const_regionIterator regionsEnd() const;
    regionIterator regionsBegin();
    regionIterator regionsEnd();

    Region* addRegion(const SCICore::Geometry::Point& min,
		      const SCICore::Geometry::Point& max,
		      int nx, int ny, int nz);
private:
    Level(const Level&);
    Level& operator=(const Level&);

    std::vector<Region*> regions;
};

#endif
