//
// $Id$
//

#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Handle.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Exceptions/InvalidGrid.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream>
#include <PSECore/XMLUtil/XMLUtil.h>
using namespace Uintah;
using namespace SCICore::Geometry;
using namespace std;
using namespace PSECore::XMLUtil;
#include <map>
#include <Uintah/Grid/BoundCondFactory.h>


Level::Level(Grid* grid, const Point& anchor, const Vector& dcell)
   : grid(grid), d_anchor(anchor), d_dcell(dcell)
{
   d_finalized=false;
}

Level::~Level()
{
  // Delete all of the patches managed by this level
  for(patchIterator iter=d_patches.begin(); iter != d_patches.end(); iter++)
    delete *iter;
}

Level::const_patchIterator Level::patchesBegin() const
{
    return d_patches.begin();
}

Level::const_patchIterator Level::patchesEnd() const
{
    return d_patches.end();
}

Level::patchIterator Level::patchesBegin()
{
    return d_patches.begin();
}

Level::patchIterator Level::patchesEnd()
{
    return d_patches.end();
}

Patch* Level::addPatch(const IntVector& lowIndex, const IntVector& highIndex,
		       const IntVector& extraLowIndex, 
		       const IntVector& extraHighIndex)
{
    Patch* r = scinew Patch(this, lowIndex, highIndex,extraLowIndex,
			    extraHighIndex);
    d_patches.push_back(r);
    return r;
}

Patch* Level::addPatch(const IntVector& lowIndex, const IntVector& highIndex,
		       const IntVector& extraLowIndex, 
		       const IntVector& extraHighIndex,
		       int ID)
{
    Patch* r = scinew Patch(this, lowIndex, highIndex, extraLowIndex,
			    extraHighIndex,ID);
    d_patches.push_back(r);
    return r;
}

Patch* Level::getPatchFromPoint(const Point& p)
{
  for(int i=0;i<(int)d_patches.size();i++){
    Patch* r = d_patches[i];
    if( r->getBox().contains( p ) )
      return r;
  }
  return 0;
}

int Level::numPatches() const
{
  return (int)d_patches.size();
}

void Level::performConsistencyCheck() const
{
   if(!d_finalized)
      throw InvalidGrid("Consistency check cannot be performed until Level is finalized");
  for(int i=0;i<(int)d_patches.size();i++){
    Patch* r = d_patches[i];
    r->performConsistencyCheck();
  }

  // This is O(n^2) - we should fix it someday if it ever matters
  for(int i=0;i<(int)d_patches.size();i++){
    Patch* r1 = d_patches[i];
    for(int j=i+1;j<(int)d_patches.size();j++){
      Patch* r2 = d_patches[j];
      if(r1->getBox().overlaps(r2->getBox())){
	cerr << "r1: " << r1 << '\n';
	cerr << "r2: " << r2 << '\n';
	throw InvalidGrid("Two patches overlap");
      }
    }
  }

  // See if abutting boxes have consistent bounds
}

void Level::getIndexRange(BBox& b) const
{
  for(int i=0;i<(int)d_patches.size();i++){
    Patch* r = d_patches[i];
    IntVector l( r->getNodeLowIndex() );
    IntVector u( r->getNodeHighIndex() );
    Point lower( l.x(), l.y(), l.z() );
    Point upper( u.x(), u.y(), u.z() );
    b.extend(lower);
    b.extend(upper);
  }
}

void Level::getIndexRange(IntVector& lowIndex,IntVector& highIndex) const
{
  lowIndex = d_patches[0]->getNodeLowIndex();
  highIndex = d_patches[0]->getNodeHighIndex();
  
  for(int p=1;p<(int)d_patches.size();p++)
  {
    Patch* patch = d_patches[p];
    IntVector l( patch->getNodeLowIndex() );
    IntVector u( patch->getNodeHighIndex() );
    for(int i=0;i<3;i++) {
      if( l(i) < lowIndex(i) ) lowIndex(i) = l(i);
      if( u(i) > highIndex(i) ) highIndex(i) = u(i);
    }
  }
}

void Level::getSpatialRange(BBox& b) const
{
  for(int i=0;i<(int)d_patches.size();i++){
    Patch* r = d_patches[i];
    b.extend(r->getBox().lower());
    b.extend(r->getBox().upper());
  }
}

long Level::totalCells() const
{
  long total=0;
  for(int i=0;i<(int)d_patches.size();i++)
    total+=d_patches[i]->totalCells();
  return total;
}

GridP Level::getGrid() const
{
   return grid;
}

Point Level::getNodePosition(const IntVector& v) const
{
   return d_anchor+d_dcell*v;
}

Point Level::getCellPosition(const IntVector& v) const
{
   return d_anchor+d_dcell*v+d_dcell*0.5;
}

IntVector Level::getCellIndex(const Point& p) const
{
   Vector v((p-d_anchor)/d_dcell);
   return IntVector((int)v.x(), (int)v.y(), (int)v.z());
}

Point Level::positionToIndex(const Point& p) const
{
   return Point((p-d_anchor)/d_dcell);
}

void Level::selectPatches(const IntVector& low, const IntVector& high,
			  std::vector<const Patch*>& neighbors) const
{
   // This sucks - it should be made faster.  -Steve
   for(const_patchIterator iter=d_patches.begin();
       iter != d_patches.end(); iter++){
      const Patch* patch = *iter;
      IntVector l=SCICore::Geometry::Max(patch->getCellLowIndex(), low);
      IntVector u=SCICore::Geometry::Min(patch->getCellHighIndex(), high);
      if(u.x() > l.x() && u.y() > l.y() && u.z() > l.z())
	 neighbors.push_back(*iter);
   }
}

bool Level::containsPoint(const Point& p) const
{
   // This sucks - it should be made faster.  -Steve
   for(const_patchIterator iter=d_patches.begin();
       iter != d_patches.end(); iter++){
      const Patch* patch = *iter;
      if(patch->getBox().contains(p))
	 return true;
   }
   return false;
}

void Level::finalizeLevel()
{
  for(patchIterator iter=d_patches.begin(); iter != d_patches.end(); iter++){
    Patch* patch = *iter;
    // See if there are any neighbors on the 6 faces
    for(Patch::FaceType face = Patch::startFace;
	face <= Patch::endFace; face=Patch::nextFace(face)){
      IntVector l,h;
      patch->getFace(face, 1, l, h);
      std::vector<const Patch*> neighbors;
      selectPatches(l, h, neighbors);
      if(neighbors.size() == 0){
	patch->setBCType(face, Patch::None);
      }
      else {
	patch->setBCType(face, Patch::Neighbor);
      }
    }
  }
  
  d_finalized=true;
}

void Level::assignBCS(const ProblemSpecP& grid_ps)
{

  // Read the bcs for the grid
  ProblemSpecP bc_ps = grid_ps->findBlock("BoundaryConditions");
  if (bc_ps == 0)
    return;
  
  for (ProblemSpecP face_ps = bc_ps->findBlock("Face");
       face_ps != 0; face_ps=face_ps->findNextBlock("Face")) {
    // 
    map<string,string> values;
    face_ps->getAttributes(values);

    Patch::FaceType face_side;
    std::string fc = values["side"];
    if (fc == "x-")
      face_side = Patch::xminus;
    if (fc ==  "x+")
      face_side = Patch::xplus;
    if (fc ==  "y-")
      face_side = Patch::yminus;
    if (fc ==  "y+")
      face_side = Patch::yplus;
    if (fc ==  "z-")
      face_side = Patch::zminus;
    if (fc == "z+")
      face_side = Patch::zplus;

    vector<BoundCondBase*> bcs;
    BoundCondFactory::create(face_ps,bcs);

    for(patchIterator iter=d_patches.begin(); iter != d_patches.end(); 
	iter++){
      Patch* patch = *iter;
      Patch::BCType bc_type = patch->getBCType(face_side);
      if (bc_type == Patch::None) {
	patch->setBCValues(face_side,bcs);
      }
      vector<BoundCondBase*> new_bcs;
      new_bcs = patch->getBCValues(face_side);
    }  // end of patch iterator
  } // end of face_ps

}

//
// $Log$
// Revision 1.24  2000/11/14 03:53:33  jas
// Implemented getExtraCellIterator.
//
// Revision 1.23  2000/11/02 21:25:55  jas
// Rearranged the boundary conditions so there is consistency between ICE
// and MPM.  Added fillFaceFlux for the Neumann BC condition.  BCs are now
// declared differently in the *.ups file.
//
// Revision 1.22  2000/09/25 20:37:42  sparker
// Quiet g++ compiler warnings
// Work around g++ compiler bug instantiating vector<NCVariable<Vector> >
// Added computeVariableExtents to (eventually) simplify data warehouses
//
// Revision 1.21  2000/08/22 18:36:40  bigler
// Added functionality to get a cell's position with the index.
//
// Revision 1.20  2000/08/08 01:32:46  jas
// Changed new to scinew and eliminated some(minor) memory leaks in the scheduler
// stuff.
//
// Revision 1.19  2000/08/02 03:29:33  jas
// Fixed grid bcs problem.
//
// Revision 1.18  2000/07/11 19:32:16  kuzimmer
// Added getPatchFromPoint(const Point& p)
//
// Revision 1.17  2000/07/11 15:53:56  tan
// Changed the following to be const member function.
//       void getIndexRange(SCICore::Geometry::BBox& b) const;
//       void getSpatialRange(SCICore::Geometry::BBox& b) const;
//
// Revision 1.16  2000/07/10 20:20:13  tan
// For IntVector i, i(0)=i.x(),i(1)=i.y(),i(2)=i.z()
//
// Revision 1.15  2000/07/07 03:08:24  tan
// It is better that IndexRange expressed by IntVector instead of BBox,
// and the function is const.
//
// Revision 1.14  2000/06/28 03:29:33  jas
// Got rid of a bunch of ugly code in assignBCS that was commented out.
//
// Revision 1.13  2000/06/27 22:49:03  jas
// Added grid boundary condition support.
//
// Revision 1.12  2000/06/23 19:20:19  jas
// Added in the early makings of Grid bcs.
//
// Revision 1.11  2000/06/15 21:57:16  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.10  2000/05/30 20:19:29  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.9  2000/05/20 08:09:21  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
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
// Made patches have a single uniform index space - still needs work
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
