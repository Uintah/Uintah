#ifndef UINTAH_HOMEBREW_Level_H
#define UINTAH_HOMEBREW_Level_H

#include <Uintah/Grid/RefCounted.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/LevelP.h>
#include <SCICore/Geometry/Point.h>
#include <string>
#include <vector>

namespace Uintah {
namespace Grid {

class Region;
class Task;
class TypeDescription;

/**************************************

CLASS
   Level
   
   Just a container class that manages a set of Regions that
   make up this level.

GENERAL INFORMATION

   Level.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Level

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

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

    std::vector<Region*> d_regions;
};

} // end namespace Grid
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/03/16 22:07:59  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
