#ifndef UINTAH_GRID_LEVEL_H
#define UINTAH_GRID_LEVEL_H

#include <Uintah/Grid/RefCounted.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Handle.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/IntVector.h>
#include <string>
#include <vector>

namespace Uintah {

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
      
      Region* addRegion(const SCICore::Geometry::Point& lower,
			const SCICore::Geometry::Point& upper,
			const SCICore::Geometry::IntVector& lowIndex,
			const SCICore::Geometry::IntVector& highIndex);
      
      int numRegions() const;
      long totalCells() const;
      
      void performConsistencyCheck() const;
   private:
      Level(const Level&);
      Level& operator=(const Level&);
      
      std::vector<Region*> d_regions;
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.9  2000/05/10 20:02:59  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made regions have a single uniform index space - still needs work
//
// Revision 1.8  2000/04/26 06:48:49  sparker
// Streamlined namespaces
//
// Revision 1.7  2000/04/12 23:00:47  sparker
// Starting problem setup code
// Other compilation fixes
//
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
