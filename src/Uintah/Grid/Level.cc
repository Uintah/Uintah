/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Handle.h>
#include <Uintah/Grid/Region.h>
#include <Uintah/Exceptions/InvalidGrid.h>
#include <iostream>
using namespace Uintah;
using namespace SCICore::Geometry;
using namespace std;

Level::Level(Grid* grid)
   : grid(grid)
{
}

Level::~Level()
{
  // Delete all of the regions managed by this level
  for(regionIterator iter=d_regions.begin(); iter != d_regions.end(); iter++)
    delete *iter;
}

Level::const_regionIterator Level::regionsBegin() const
{
    return d_regions.begin();
}

Level::const_regionIterator Level::regionsEnd() const
{
    return d_regions.end();
}

Level::regionIterator Level::regionsBegin()
{
    return d_regions.begin();
}

Level::regionIterator Level::regionsEnd()
{
    return d_regions.end();
}

Region* Level::addRegion(const Point& lower, const Point& upper,
			 const IntVector& lowIndex, const IntVector& highIndex)
{
    Region* r = new Region(lower, upper, lowIndex, highIndex);
    d_regions.push_back(r);
    return r;
}

int Level::numRegions() const
{
  return (int)d_regions.size();
}

void Level::performConsistencyCheck() const
{
  for(int i=0;i<d_regions.size();i++){
    Region* r = d_regions[i];
    r->performConsistencyCheck();
  }

  // This is O(n^2) - we should fix it someday if it ever matters
  for(int i=0;i<d_regions.size();i++){
    Region* r1 = d_regions[i];
    for(int j=i+1;j<d_regions.size();j++){
      Region* r2 = d_regions[j];
      if(r1->getBox().overlaps(r2->getBox())){
	cerr << "r1: " << r1 << '\n';
	cerr << "r2: " << r2 << '\n';
	throw InvalidGrid("Two regions overlap");
      }
    }
  }

  // See if abutting boxes have consistent bounds
}

void Level::getIndexRange(BBox& b)
{
  for(int i=0;i<d_regions.size();i++){
    Region* r = d_regions[i];
    IntVector l( r->getNodeLowIndex() );
    IntVector u( r->getNodeHighIndex() );
    Point lower( l.x(), l.y(), l.z() );
    Point upper( u.x(), u.y(), u.z() );
    b.extend(lower);
    b.extend(upper);
  }
}
void Level::getSpatialRange(BBox& b)
{
  for(int i=0;i<d_regions.size();i++){
    Region* r = d_regions[i];
    b.extend(r->getBox().lower());
    b.extend(r->getBox().upper());
  }
}

long Level::totalCells() const
{
  long total=0;
  for(int i=0;i<d_regions.size();i++)
    total+=d_regions[i]->totalCells();
  return total;
}

GridP Level::getGrid() const
{
   return grid;
}

//
// $Log$
// Revision 1.8  2000/05/20 02:36:05  kuzimmer
// Multiple changes for new vis tools and DataArchive
//
// Revision 1.7  2000/05/15 19:39:47  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.6  2000/05/10 20:02:59  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made regions have a single uniform index space - still needs work
//
// Revision 1.5  2000/04/26 06:48:49  sparker
// Streamlined namespaces
//
// Revision 1.4  2000/04/13 06:51:01  sparker
// More implementation to get this to work
//
// Revision 1.3  2000/04/12 23:00:47  sparker
// Starting problem setup code
// Other compilation fixes
//
// Revision 1.2  2000/03/16 22:07:59  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//
