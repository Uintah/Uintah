#ifndef UINTAH_GRID_LEVEL_H
#define UINTAH_GRID_LEVEL_H

#include <Uintah/Grid/RefCounted.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Handle.h>
#include <SCICore/Geometry/Point.h>
#include <string>
#include <vector>

namespace Uintah {
namespace Grid {

class Region;
class Task;

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
// Revision 1.6  2000/03/22 00:32:12  sparker
// Added Face-centered variable class
// Added Per-region data class
// Added new task constructor for procedures with arguments
// Use Array3Index more often
//
// Revision 1.5  2000/03/21 01:29:42  dav
// working to make MPM stuff compile successfully
//
// Revision 1.4  2000/03/17 18:45:42  dav
// fixed a few more namespace problems
//
// Revision 1.3  2000/03/16 22:07:59  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
