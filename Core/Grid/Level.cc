
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/ProblemSpec/Handle.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/InvalidGrid.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/FancyAssert.h>
#include <iostream>
#include <Dataflow/XMLUtil/XMLUtil.h>

#include <algorithm>
#include <map>
#include <Packages/Uintah/Core/Grid/BoundCondFactory.h>
#include <Packages/Uintah/Core/Grid/BoundCondBase.h>

#ifdef SELECT_RANGETREE
#include <Packages/Uintah/Core/Grid/PatchRangeTree.h>
#endif

using namespace Uintah;
using namespace SCIRun;
using namespace std;

Level::Level(Grid* grid, const Point& anchor, const Vector& dcell)
   : grid(grid), d_anchor(anchor), d_dcell(dcell),
     d_patchDistribution(-1,-1,-1)
{
  each_patch=0;
  all_patches=0;
#ifdef SELECT_RANGETREE
  d_rangeTree = NULL;
#endif
  d_finalized=false;
  d_extraCells = IntVector(0,0,0);
}

Level::~Level()
{
  // Delete all of the patches managed by this level
  for(patchIterator iter=d_patches.begin(); iter != d_patches.end(); iter++)
    delete *iter;
#if 0
  for(int i=0;i<(int)allbcs.size();i++)
    delete allbcs[i];
#endif

#ifdef SELECT_RANGETREE
  delete d_rangeTree;
#endif
  if(each_patch && each_patch->removeReference())
    delete each_patch;
  if(all_patches && all_patches->removeReference())
    delete all_patches;
}

void Level::setPatchDistributionHint(const IntVector& hint)
{
   if(d_patchDistribution.x() == -1)
      d_patchDistribution = hint;
   else
      // Called more than once, we have to punt
      d_patchDistribution = IntVector(-2,-2,2);
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

Patch* Level::addPatch(const IntVector& lowIndex, 
		       const IntVector& highIndex,
		       const IntVector& inLowIndex, 
		       const IntVector& inHighIndex)
{
    Patch* r = scinew Patch(this, lowIndex,highIndex,inLowIndex, 
			    inHighIndex);
    d_patches.push_back(r);
    return r;
}

Patch* Level::addPatch(const IntVector& lowIndex, 
		       const IntVector& highIndex,
		       const IntVector& inLowIndex, 
		       const IntVector& inHighIndex,
		       int ID)
{
 
    Patch* r = scinew Patch(this, lowIndex,highIndex,inLowIndex, 
			    inHighIndex,ID);
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
	cerr << "r1: " << *r1 << '\n';
	cerr << "r2: " << *r2 << '\n';
	throw InvalidGrid("Two patches overlap");
      }
    }
  }

  // See if abutting boxes have consistent bounds
}

void Level::findNodeIndexRange(IntVector& lowIndex,IntVector& highIndex) const
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

void Level::findCellIndexRange(IntVector& lowIndex,IntVector& highIndex) const
{
  lowIndex = d_patches[0]->getCellLowIndex();
  highIndex = d_patches[0]->getCellHighIndex();
  
  for(int p=1;p<(int)d_patches.size();p++)
  {
    Patch* patch = d_patches[p];
    IntVector l( patch->getCellLowIndex() );
    IntVector u( patch->getCellHighIndex() );
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

void Level::setExtraCells(const IntVector& ec)
{
  d_extraCells = ec;
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
   // This bit of funky looking code is designed to bypass rounding issues
   // for negative numbers.  We need to always round down, but a -0.5 rounds
   // to 0 instead of -1.  So by adding 10000 we get 9999.5, cast to int
   // would be 9999.  Subtract the 10000 and you get -1.

   //////////////////////////////////////////////////////////////////
   // if any member of v gets less than -10000 than this code
   // must be adjusted accordingly
   //////////////////////////////////////////////////////////////////
   return IntVector((int)(v.x()+10000.)-10000,
		    (int)(v.y()+10000.)-10000,
		    (int)(v.z()+10000.)-10000);
}

Point Level::positionToIndex(const Point& p) const
{
   return Point((p-d_anchor)/d_dcell);
}

void Level::selectPatches(const IntVector& low, const IntVector& high,
			  selectType& neighbors) const
{
#ifdef SELECT_LINEAR
   // This sucks - it should be made faster.  -Steve
   for(const_patchIterator iter=d_patches.begin();
       iter != d_patches.end(); iter++){
      const Patch* patch = *iter;
      IntVector l=Max(patch->getCellLowIndex(), low);
      IntVector u=Min(patch->getCellHighIndex(), high);
      if(u.x() > l.x() && u.y() > l.y() && u.z() > l.z())
	neighbors.push_back(*iter);
   }
#else
#ifdef SELECT_GRID
   IntVector start = (low-d_idxLow)*d_gridSize/d_idxSize;
   IntVector end = (high-d_idxLow)*d_gridSize/d_idxSize;
   start=Max(IntVector(0,0,0), start);
   end=Min(d_gridSize-IntVector(1,1,1), end);
   for(int iz=start.z();iz<=end.z();iz++){
      for(int iy=start.y();iy<=end.y();iy++){
	 for(int ix=start.x();ix<=end.x();ix++){
	    int gridIdx = (iz*d_gridSize.y()+iy)*d_gridSize.x()+ix;
	    int s = d_gridStarts[gridIdx];
	    int e = d_gridStarts[gridIdx+1];
	    for(int i=s;i<e;i++){
	       Patch* patch = d_gridPatches[i];
	       IntVector l=Max(patch->getCellLowIndex(), low);
	       IntVector u=Min(patch->getCellHighIndex(), high);
	       if(u.x() > l.x() && u.y() > l.y() && u.z() > l.z())
		  neighbors.push_back(patch);
	    }
	 }
      }
   }
   sort(neighbors.begin(), neighbors.end(), Patch::Compare());
   int i=0;
   int j=0;
   while(j<(int)neighbors.size()) {
      neighbors[i]=neighbors[j];
      j++;
      while(j < (int)neighbors.size() && neighbors[i] == neighbors[j] )
	 j++;
      i++;
   }
   neighbors.resize(i);
#else
#ifdef SELECT_RANGETREE
   d_rangeTree->query(low, high, neighbors);
   sort(neighbors.begin(), neighbors.end(), Patch::Compare());
#else
#error "No selectPatches algorithm defined"
#endif
#endif
#endif

#ifdef CHECK_SELECT
   // Double-check the more advanced selection algorithms against the
   // slow (exhaustive) one.
   vector<const Patch*> tneighbors;
   for(const_patchIterator iter=d_patches.begin();
       iter != d_patches.end(); iter++){
      const Patch* patch = *iter;
      IntVector l=Max(patch->getCellLowIndex(), low);
      IntVector u=Min(patch->getCellHighIndex(), high);
      if(u.x() > l.x() && u.y() > l.y() && u.z() > l.z())
	 tneighbors.push_back(*iter);
   }
   ASSERTEQ(neighbors.size(), tneighbors.size());
   sort(tneighbors.begin(), tneighbors.end(), Patch::Compare());
   for(int i=0;i<(int)neighbors.size();i++)
      ASSERT(neighbors[i] == tneighbors[i]);
#endif
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
  each_patch = scinew PatchSet();
  each_patch->addReference();

  // The compute set requires an array const Patch*, we must copy d_patches
  vector<const Patch*> tmp_patches(d_patches.size());
  for(int i=0;i<(int)d_patches.size();i++)
    tmp_patches[i]=d_patches[i];
  each_patch->addEach(tmp_patches);
  all_patches = scinew PatchSet();
  all_patches->addReference();
  all_patches->addAll(tmp_patches);

#ifdef SELECT_GRID
   if(d_patchDistribution.x() >= 0 && d_patchDistribution.y() >= 0 &&
      d_patchDistribution.z() >= 0){
      d_gridSize = d_patchDistribution;
   } else {
      int np = numPatches();
      int neach = (int)(0.5+pow(np, 1./3.));
      d_gridSize = IntVector(neach, neach, neach);
   }
   getIndexRange(d_idxLow, d_idxHigh);
   d_idxHigh-=IntVector(1,1,1);
   d_idxSize = d_idxHigh-d_idxLow;
   int numCells = d_gridSize.x()*d_gridSize.y()*d_gridSize.z();
   vector<int> counts(numCells+1, 0);
   for(patchIterator iter=d_patches.begin(); iter != d_patches.end(); iter++){
      Patch* patch = *iter;
      IntVector start = (patch->getCellLowIndex()-d_idxLow)*d_gridSize/d_idxSize;
      IntVector end = ((patch->getCellHighIndex()-d_idxLow)*d_gridSize+d_gridSize-IntVector(1,1,1))/d_idxSize;
      for(int iz=start.z();iz<end.z();iz++){
	 for(int iy=start.y();iy<end.y();iy++){
	    for(int ix=start.x();ix<end.x();ix++){
	       int gridIdx = (iz*d_gridSize.y()+iy)*d_gridSize.x()+ix;
	       counts[gridIdx]++;
	    }
	 }
      }
   }
   d_gridStarts.resize(numCells+1);
   int count=0;
   for(int i=0;i<numCells;i++){
      d_gridStarts[i]=count;
      count+=counts[i];
      counts[i]=0;
   }
   d_gridStarts[numCells]=count;
   d_gridPatches.resize(count);
   for(patchIterator iter=d_patches.begin(); iter != d_patches.end(); iter++){
      Patch* patch = *iter;
      IntVector start = (patch->getCellLowIndex()-d_idxLow)*d_gridSize/d_idxSize;
      IntVector end = ((patch->getCellHighIndex()-d_idxLow)*d_gridSize+d_gridSize-IntVector(1,1,1))/d_idxSize;
      for(int iz=start.z();iz<end.z();iz++){
	 for(int iy=start.y();iy<end.y();iy++){
	    for(int ix=start.x();ix<end.x();ix++){
	       int gridIdx = (iz*d_gridSize.y()+iy)*d_gridSize.x()+ix;
	       int pidx = d_gridStarts[gridIdx]+counts[gridIdx];
	       d_gridPatches[pidx]=patch;
	       counts[gridIdx]++;
	    }
	 }
      }
   }
#else
#ifdef SELECT_RANGETREE
   if (d_rangeTree != NULL)
     delete d_rangeTree;
   d_rangeTree = scinew PatchRangeTree(d_patches);
#endif   
#endif
   patchIterator iter;
   int ii;
  for(iter=d_patches.begin(), ii = 0;
      iter != d_patches.end(); iter++, ii++){
    Patch* patch = *iter;
    patch->setLevelIndex( ii );
    //cout << "Patch bounding box = " << patch->getBox() << endl;
    // See if there are any neighbors on the 6 faces
    for(Patch::FaceType face = Patch::startFace;
	face <= Patch::endFace; face=Patch::nextFace(face)){
      IntVector l,h;
      patch->getFace(face, 1, l, h);
      Level::selectType neighbors;
      selectPatches(l, h, neighbors);
      if(neighbors.size() == 0){
	patch->setBCType(face, Patch::None);
      }
      else {
	patch->setBCType(face, Patch::Neighbor);
      }
    }
  }
  
  // There is a possibility that the extraLow and extraHigh indices
  // for a patch are incorrectly determined for unequal number of cells
  // per patch.  This is meant to correct these problems by using the
  // above info about Patch::Neighbor to determine the extra indices.

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
    map<string,string> values;
    face_ps->getAttributes(values);

    Patch::FaceType face_side = Patch::invalidFace;
    std::string fc = values["side"];
    if (fc == "x-")
      face_side = Patch::xminus;
    else if (fc ==  "x+")
      face_side = Patch::xplus;
    else if (fc ==  "y-")
      face_side = Patch::yminus;
    else if (fc ==  "y+")
      face_side = Patch::yplus;
    else if (fc ==  "z-")
      face_side = Patch::zminus;
    else if (fc == "z+")
      face_side = Patch::zplus;

    BCData bc_data;
    BoundCondFactory::create(face_ps,bc_data);
#if 0
    for(int i=0;i<(int)bcs.size();i++)
      allbcs.push_back(bcs[i]);
#endif

    for(patchIterator iter=d_patches.begin(); iter != d_patches.end(); 
	iter++){
      Patch* patch = *iter;
      Patch::BCType bc_type = patch->getBCType(face_side);
      if (bc_type == Patch::None) {
	patch->setBCValues(face_side,bc_data);
      }
#if 0
      if (bc_type == Patch::None) {
	cerr << "face side = " << face_side << endl;
	BoundCondBase* new_bcs = patch->getBCValues(0,"Pressure",face_side);
	cerr << "BC = " << new_bcs->getType() << endl;
      }
#endif
    }  // end of patch iterator
  } // end of face_ps

}

Box Level::getBox(const IntVector& l, const IntVector& h) const
{
   return Box(getNodePosition(l), getNodePosition(h));
}

const PatchSet* Level::eachPatch() const
{
  ASSERT(each_patch != 0);
  return each_patch;
}

const PatchSet* Level::allPatches() const
{
  ASSERT(all_patches != 0);
  return all_patches;
}

const Patch* Level::selectPatchForCellIndex( const IntVector& idx) const
{
  selectType pv;
  IntVector i(1,1,1);
  selectPatches(idx - i,idx + i,pv);
  if(pv.size() == 0)
    return 0;
  else {
    selectType::iterator it;
    for( it = pv.begin(); it != pv.end(); it++)
      if( (*it)->containsCell(idx) )
	return *it;
  }
  return 0;
}
const Patch* Level::selectPatchForNodeIndex( const IntVector& idx) const
{
  selectType pv;
  IntVector i(1,1,1);
  selectPatches(idx - i,idx + i,pv);
  if(pv.size() == 0)
    return 0;
  else {
    selectType::iterator it;
    for( it = pv.begin(); it != pv.end(); it++)
      if( (*it)->containsNode(idx) )
	return *it;
  }
  return 0;
}
