
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Util/Handle.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/InvalidGrid.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCondReader.h>

#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/AtomicCounter.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/ProgressiveWarning.h>

#include <iostream>
#include <algorithm>
#include <map>
#include <math.h>

#define SELECT_RANGETREE
#define BRYAN_SELECT_CACHE

#ifdef SELECT_RANGETREE
#include <Packages/Uintah/Core/Grid/PatchRangeTree.h>
#endif

#ifdef _WIN32
#define rint(x) (int)((x>0) ? x+.5 : x-.5)
#endif

using namespace Uintah;
using namespace SCIRun;
using namespace std;

static AtomicCounter* ids = 0;
static Mutex ids_init("ID init");

static DebugStream bcout("BCTypes", false);

Level::Level(Grid* grid, const Point& anchor, const Vector& dcell, 
             int index, IntVector refinementRatio, int id /*=-1*/)
   : grid(grid), d_anchor(anchor), d_dcell(dcell), d_index(index),
     d_patchDistribution(-1,-1,-1), d_periodicBoundaries(0, 0, 0), d_id(id),
     d_refinementRatio(refinementRatio)
{
  each_patch=0;
  all_patches=0;
#ifdef SELECT_RANGETREE
  d_rangeTree = NULL;
#endif
  d_finalized=false;
  d_extraCells = IntVector(0,0,0);
  if(!ids){
    ids_init.lock();
    if(!ids){
      ids = new AtomicCounter("Patch ID counter", 0);
    }
    ids_init.unlock();
    
  }
  if(d_id == -1)
    d_id = (*ids)++;
  else if(d_id >= *ids)
    ids->set(d_id+1);
}

Level::~Level()
{
  // Delete all of the patches managed by this level
  for(patchIterator iter=d_virtualAndRealPatches.begin(); iter != d_virtualAndRealPatches.end(); iter++)
    delete *iter;

#ifdef SELECT_RANGETREE
  delete d_rangeTree;
#endif
  if(each_patch && each_patch->removeReference())
    delete each_patch;
  if(all_patches && all_patches->removeReference())
    delete all_patches;

#ifdef BRYAN_SELECT_CACHE
  int patches_stored = 0;
  int queries_stored = 0;
  for (selectCache::iterator iter = d_selectCache.begin(); iter != d_selectCache.end(); iter++) {
    queries_stored++;
    patches_stored += iter->second.size();
  }
  //cout << "  Bryan's select cache stored " << queries_stored << " queries and " << patches_stored << " patches\n";
#endif

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
    return d_realPatches.begin();
}

Level::const_patchIterator Level::patchesEnd() const
{
    return d_realPatches.end();
}

Level::patchIterator Level::patchesBegin()
{
    return d_realPatches.begin();
}

Level::patchIterator Level::patchesEnd()
{
    return d_realPatches.end();
}

Level::const_patchIterator Level::allPatchesBegin() const
{
    return d_virtualAndRealPatches.begin();
}

Level::const_patchIterator Level::allPatchesEnd() const
{
    return d_virtualAndRealPatches.end();
}

Patch* Level::addPatch(const IntVector& lowIndex, 
		       const IntVector& highIndex,
		       const IntVector& inLowIndex, 
		       const IntVector& inHighIndex)
{
    Patch* r = scinew Patch(this, lowIndex,highIndex,inLowIndex, 
			    inHighIndex);
    d_realPatches.push_back(r);
    d_virtualAndRealPatches.push_back(r);
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
    d_realPatches.push_back(r);
    d_virtualAndRealPatches.push_back(r);
    return r;
}

Patch* Level::getPatchFromPoint(const Point& p)
{
  for(int i=0;i<(int)d_realPatches.size();i++){
    Patch* r = d_realPatches[i];
    if( r->getBox().contains( p ) )
      return r;
  }
  return 0;
}

int Level::numPatches() const
{
  return (int)d_realPatches.size();
}

void Level::performConsistencyCheck() const
{
   if(!d_finalized)
     SCI_THROW(InvalidGrid("Consistency check cannot be performed until Level is finalized",__FILE__,__LINE__));
  for(int i=0;i<(int)d_virtualAndRealPatches.size();i++){
    Patch* r = d_virtualAndRealPatches[i];
    r->performConsistencyCheck();
  }

  // This is O(n^2) - we should fix it someday if it ever matters
  //   This checks that patches do not overlap
  for(int i=0;i<(int)d_virtualAndRealPatches.size();i++){
    Patch* r1 = d_virtualAndRealPatches[i];
    for(int j=i+1;j<(int)d_virtualAndRealPatches.size();j++){
      Patch* r2 = d_virtualAndRealPatches[j];
      Box b1 = getBox(r1->getInteriorCellLowIndex(), r1->getInteriorCellHighIndex());
      Box b2 = getBox(r2->getInteriorCellLowIndex(), r2->getInteriorCellHighIndex());
      if(b1.overlaps(b2)){
	cerr << "r1: " << *r1 << '\n';
	cerr << "r2: " << *r2 << '\n';
	SCI_THROW(InvalidGrid("Two patches overlap",__FILE__,__LINE__));
      }
    }
  }


}


void Level::findNodeIndexRange(IntVector& lowIndex,IntVector& highIndex) const
{
  lowIndex = d_realPatches[0]->getNodeLowIndex();
  highIndex = d_realPatches[0]->getNodeHighIndex();
  
  for(int p=1;p<(int)d_realPatches.size();p++)
  {
    Patch* patch = d_realPatches[p];
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
  lowIndex = d_realPatches[0]->getCellLowIndex();
  highIndex = d_realPatches[0]->getCellHighIndex();
  
  for(int p=1;p<(int)d_realPatches.size();p++)
  {
    Patch* patch = d_realPatches[p];
    IntVector l( patch->getCellLowIndex() );
    IntVector u( patch->getCellHighIndex() );
    for(int i=0;i<3;i++) {
      if( l(i) < lowIndex(i) ) lowIndex(i) = l(i);
      if( u(i) > highIndex(i) ) highIndex(i) = u(i);
    }
  }
}

void Level::findInteriorCellIndexRange(IntVector& lowIndex,IntVector& highIndex) const
{
  lowIndex = d_realPatches[0]->getInteriorCellLowIndex();
  highIndex = d_realPatches[0]->getInteriorCellHighIndex();
  
  for(int p=1;p<(int)d_realPatches.size();p++)
  {
    Patch* patch = d_realPatches[p];
    IntVector l( patch->getInteriorCellLowIndex() );
    IntVector u( patch->getInteriorCellHighIndex() );
    for(int i=0;i<3;i++) {
      if( l(i) < lowIndex(i) ) lowIndex(i) = l(i);
      if( u(i) > highIndex(i) ) highIndex(i) = u(i);
    }
  }
}

void Level::findInteriorNodeIndexRange(IntVector& lowIndex,IntVector& highIndex) const
{
  lowIndex = d_realPatches[0]->getInteriorNodeLowIndex();
  highIndex = d_realPatches[0]->getInteriorNodeHighIndex();
  
  for(int p=1;p<(int)d_realPatches.size();p++)
  {
    Patch* patch = d_realPatches[p];
    IntVector l( patch->getInteriorNodeLowIndex() );
    IntVector u( patch->getInteriorNodeHighIndex() );
    for(int i=0;i<3;i++) {
      if( l(i) < lowIndex(i) ) lowIndex(i) = l(i);
      if( u(i) > highIndex(i) ) highIndex(i) = u(i);
    }
  }
}

void Level::getSpatialRange(BBox& b) const
{
  for(int i=0;i<(int)d_realPatches.size();i++){
    Patch* r = d_realPatches[i];
    b.extend(r->getBox().lower());
    b.extend(r->getBox().upper());
  }
}

void Level::getInteriorSpatialRange(BBox& b) const
{
  for(int i=0;i<(int)d_realPatches.size();i++){
    Patch* r = d_realPatches[i];
    b.extend(r->getInteriorBox().lower());
    b.extend(r->getInteriorBox().upper());
  }
}

long Level::totalCells() const
{
  long total=0;
  for(int i=0;i<(int)d_realPatches.size();i++)
    total+=d_realPatches[i]->totalCells();
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

const LevelP& Level::getRelativeLevel(int offset) const
{
  return grid->getLevel(d_index + offset);
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
   return IntVector(RoundDown(v.x()), RoundDown(v.y()), RoundDown(v.z()));
}

Point Level::positionToIndex(const Point& p) const
{
   return Point((p-d_anchor)/d_dcell);
}

void Level::selectPatches(const IntVector& low, const IntVector& high,
			  selectType& neighbors) const
{
#ifdef BRYAN_SELECT_CACHE
    
  // look it up in the cache first
  selectCache::const_iterator iter = d_selectCache.find(make_pair(low, high));
  if (iter != d_selectCache.end()) {
    const vector<const Patch*>& cache = iter->second;
    for (unsigned i = 0; i < cache.size(); i++) {
      neighbors.push_back(cache[i]);
    }
    return;
  }
  ASSERT(neighbors.size() == 0);
#endif

#if defined( SELECT_LINEAR )
   // This sucks - it should be made faster.  -Steve
   for(const_patchIterator iter=d_virtualAndRealPatches.begin();
       iter != d_virtualAndRealPatches.end(); iter++){
      const Patch* patch = *iter;
      IntVector l=Max(patch->getCellLowIndex(), low);
      IntVector u=Min(patch->getCellHighIndex(), high);
      if(u.x() > l.x() && u.y() > l.y() && u.z() > l.z())
	neighbors.push_back(*iter);
   }
#elif defined( SELECT_GRID )
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
#elif defined( SELECT_RANGETREE )
   //cout << Parallel::getMPIRank() << " Level Quesy: " << low << " " << high << endl;
   d_rangeTree->query(low, high, neighbors);
   sort(neighbors.begin(), neighbors.end(), Patch::Compare());
#else
#error "No selectPatches algorithm defined"
#endif

#ifdef CHECK_SELECT
   // Double-check the more advanced selection algorithms against the
   // slow (exhaustive) one.
   vector<const Patch*> tneighbors;
   for(const_patchIterator iter=d_virtualAndRealPatches.begin();
       iter != d_virtualAndRealPatches.end(); iter++){
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

#ifdef BRYAN_SELECT_CACHE
   // put it in the cache - start at orig_size in case there was something in
   // neighbors before this query
   vector<const Patch*>& cache = d_selectCache[make_pair(low,high)];
   cache.reserve(6);  // don't reserve too much to save memory, not too little to avoid too much reallocation
   for (int i = 0; i < neighbors.size(); i++) {
     cache.push_back(neighbors[i]);
   }
#endif
}

bool Level::containsPoint(const Point& p) const
{
   // This sucks - it should be made faster.  -Steve
   for(const_patchIterator iter=d_realPatches.begin();
       iter != d_realPatches.end(); iter++){
      const Patch* patch = *iter;
      if(patch->getBox().contains(p))
	 return true;
   }
   return false;
}

bool Level::containsPointInRealCells(const Point& p) const
{
   // This sucks - it should be made faster.  -Steve
   for(const_patchIterator iter=d_realPatches.begin();
       iter != d_realPatches.end(); iter++){
      const Patch* patch = *iter;
      if(patch->containsPointInRealCells(p))
	 return true;
   }
   return false;
}

void Level::finalizeLevel()
{
  each_patch = scinew PatchSet();
  each_patch->addReference();
  
  // The compute set requires an array const Patch*, we must copy d_realPatches
  vector<const Patch*> tmp_patches(d_realPatches.size());
  for(int i=0;i<(int)d_realPatches.size();i++)
    tmp_patches[i]=d_realPatches[i];
  each_patch->addEach(tmp_patches);
  all_patches = scinew PatchSet();
  all_patches->addReference();
  all_patches->addAll(tmp_patches);
  
  all_patches->sortSubsets();
  std::sort(d_realPatches.begin(), d_realPatches.end(), Patch::Compare());
  
  setBCTypes();
}

void Level::finalizeLevel(bool periodicX, bool periodicY, bool periodicZ)
{
  // set each_patch and all_patches before creating virtual patches
  
  each_patch = scinew PatchSet();
  each_patch->addReference();

  // The compute set requires an array const Patch*, we must copy d_realPatches
  vector<const Patch*> tmp_patches(d_realPatches.size());
  for(int i=0;i<(int)d_realPatches.size();i++)
    tmp_patches[i]=d_realPatches[i];
  each_patch->addEach(tmp_patches);
  all_patches = scinew PatchSet();
  all_patches->addReference();
  all_patches->addAll(tmp_patches);

  BBox bbox;
  
  if (d_index > 0)
    grid->getLevel(0)->getSpatialRange(bbox);
  else
    getSpatialRange(bbox);
  Box domain(bbox.min(), bbox.max());
  Vector vextent = positionToIndex(bbox.max()) - positionToIndex(bbox.min());
  IntVector extent((int)rint(vextent.x()), (int)rint(vextent.y()),
		   (int)rint(vextent.z()));
  d_periodicBoundaries = IntVector(periodicX ? 1 : 0, periodicY ? 1 : 0,
				   periodicZ ? 1 : 0);
  IntVector periodicBoundaryRange = d_periodicBoundaries * extent;

  int x,y,z;
  for(int i=0;i<(int)tmp_patches.size();i++) {
    for (x = -d_periodicBoundaries.x(); x <= d_periodicBoundaries.x(); x++) {
      for (y = -d_periodicBoundaries.y(); y <= d_periodicBoundaries.y(); y++) {
	for (z = -d_periodicBoundaries.z(); z <= d_periodicBoundaries.z(); z++)
	{
	  IntVector offset = IntVector(x, y, z) * periodicBoundaryRange;
	  if (offset == IntVector(0,0,0))
	    continue;
	  Box box =
	    getBox(tmp_patches[i]->getLowIndex() + offset - IntVector(1,1,1),
		   tmp_patches[i]->getHighIndex() + offset + IntVector(1,1,1));
	  if (box.overlaps(domain)) {
	    Patch* newPatch = tmp_patches[i]->createVirtualPatch(offset);
	    d_virtualAndRealPatches.push_back(newPatch);
	  }
	}
      }
    }
  }

  all_patches->sortSubsets();
  std::sort(d_realPatches.begin(), d_realPatches.end(), Patch::Compare());
  std::sort(d_virtualAndRealPatches.begin(), d_virtualAndRealPatches.end(),
	    Patch::Compare());
  
  setBCTypes();
}

void Level::setBCTypes()
{
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
   for(patchIterator iter=d_virtualAndRealPatches.begin(); iter != d_virtualAndRealPatches.end(); iter++){
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
   for(patchIterator iter=d_virtualAndRealPatches.begin(); iter != d_virtualAndRealPatches.end(); iter++){
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
   d_rangeTree = scinew PatchRangeTree(d_virtualAndRealPatches);
#endif   
#endif
   patchIterator iter;
   int idx;
  for(iter=d_virtualAndRealPatches.begin(), idx = 0;
      iter != d_virtualAndRealPatches.end(); iter++){
    Patch* patch = *iter;
    if(patch->isVirtual())
      patch->setLevelIndex( -1 );
    else
      patch->setLevelIndex(idx++);
    //cout << "Patch bounding box = " << patch->getBox() << endl;
    // See if there are any neighbors on the 6 faces
    for(Patch::FaceType face = Patch::startFace;
	face <= Patch::endFace; face=Patch::nextFace(face)){
      IntVector l,h;
      patch->getFace(face, IntVector(0,0,0), IntVector(1,1,1), l, h);
      Patch::selectType neighbors;
      selectPatches(l, h, neighbors);
      
      if(neighbors.size() == 0){
	if(d_index != 0){
	  // See if there are any patches on the coarse level at that face
	  IntVector fineLow, fineHigh;
	  patch->getFace(face, IntVector(0,0,0), d_refinementRatio,
			 fineLow, fineHigh);
	  IntVector coarseLow = mapCellToCoarser(fineLow);
	  IntVector coarseHigh = mapCellToCoarser(fineHigh);
	  const LevelP& coarseLevel = getCoarserLevel();
          
          // add 1 to the corresponding index on the plus edges 
          // because the upper corners are sort of one cell off (don't know why)
          if (d_extraCells.x() != 0 && face == Patch::xplus) {
            coarseLow[0] ++;
            coarseHigh[0]++;
          }
          else if (d_extraCells.y() != 0 && face == Patch::yplus) {
            coarseLow[1] ++;
            coarseHigh[1] ++;
          }
          else if (d_extraCells.z() != 0 && face == Patch::zplus) {
            coarseLow[2] ++;
            coarseHigh[2]++;
          }
	  coarseLevel->selectPatches(coarseLow, coarseHigh, neighbors);
	  if(neighbors.size() == 0){
	    patch->setBCType(face, Patch::None);
            bcout << "  Setting Patch " << patch->getID() << " face " << face << " to None\n";
	  } else {
	    patch->setBCType(face, Patch::Coarse);
            bcout << "  Setting Patch " << patch->getID() << " face " << face << " to Coarse\n";
	  }
	} else {
	  patch->setBCType(face, Patch::None);
          bcout << "  Setting Patch " << patch->getID() << " face " << face << " to None\n";
	}
      } else {
	patch->setBCType(face, Patch::Neighbor);
        bcout << "  Setting Patch " << patch->getID() << " face " << face << " to Neighbor\n";
      }
    }
    patch->finalizePatch();
  }
  
  d_finalized=true;
}

void Level::assignBCS(const ProblemSpecP& grid_ps)
{
  ProblemSpecP bc_ps = grid_ps->findBlock("BoundaryConditions");
  if (bc_ps == 0) {
    static ProgressiveWarning warn("No BoundaryConditions specified", -1);
    warn.invoke();
    return;
  }


  BoundCondReader reader;
  reader.read(bc_ps, grid_ps);

  for (Patch::FaceType face_side = Patch::startFace; 
       face_side <= Patch::endFace; face_side=Patch::nextFace(face_side)) {
    
    for(patchIterator iter=d_virtualAndRealPatches.begin(); 
	iter != d_virtualAndRealPatches.end(); iter++){
      Patch* patch = *iter;
      if (patch->getBCType(face_side) == Patch::None) {
	patch->setArrayBCValues(face_side,&(reader.d_BCReaderData[face_side]));
      }
    }  // end of patchIterator
  }
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

const Patch* Level::getPatchByID(int id) const
{
  return d_realPatches[id - d_realPatches[0]->getID()];
}


const LevelP& Level::getCoarserLevel() const
{
  return getRelativeLevel(-1);
}

const LevelP& Level::getFinerLevel() const
{
  return getRelativeLevel(1);
}

bool Level::hasCoarserLevel() const
{
  return getIndex() > 0;
}

bool Level::hasFinerLevel() const
{
  return getIndex() < grid->numLevels()-1;
}

IntVector Level::interpolateCellToCoarser(const IntVector& idx, Vector& weight) const
{
  IntVector i(idx-(d_refinementRatio-IntVector(1,1,1)));
  weight=Vector(double(0.5+i.x()%d_refinementRatio.x())/double(d_refinementRatio.x()),
		  double(0.5+i.y()%d_refinementRatio.y())/double(d_refinementRatio.y()),
		  double(0.5+i.z()%d_refinementRatio.z())/double(d_refinementRatio.z()));
  return i/d_refinementRatio;
}

IntVector Level::interpolateXFaceToCoarser(const IntVector& idx, Vector& weight) const
{
  IntVector i(idx-(d_refinementRatio-IntVector(d_refinementRatio.x(),1,1)));
  weight=Vector(double(i.x()%d_refinementRatio.x())/double(d_refinementRatio.x()),
		double(0.5+i.y()%d_refinementRatio.y())/double(d_refinementRatio.y()),
		double(0.5+i.z()%d_refinementRatio.z())/double(d_refinementRatio.z()));
  return i/d_refinementRatio;
}

IntVector Level::interpolateYFaceToCoarser(const IntVector& idx, Vector& weight) const
{
  IntVector i(idx-(d_refinementRatio-IntVector(1,d_refinementRatio.y(),1)));
  weight=Vector(double(0.5+i.x()%d_refinementRatio.x())/double(d_refinementRatio.x()),
		  double(i.y()%d_refinementRatio.y())/double(d_refinementRatio.y()),
		  double(0.5+i.z()%d_refinementRatio.z())/double(d_refinementRatio.z()));
  return i/d_refinementRatio;
}

IntVector Level::interpolateZFaceToCoarser(const IntVector& idx, Vector& weight) const
{
  IntVector i(idx-(d_refinementRatio-IntVector(1,1,d_refinementRatio.z())));
  weight=Vector(double(0.5+i.x()%d_refinementRatio.x())/double(d_refinementRatio.x()),
		double(0.5+i.y()%d_refinementRatio.y())/double(d_refinementRatio.y()),
		double(i.z()%d_refinementRatio.z())/double(d_refinementRatio.z()));
  return i/d_refinementRatio;
}

IntVector Level::interpolateToCoarser(const IntVector& idx, const IntVector& dir,
			      Vector& weight) const
{
  IntVector d(IntVector(1,1,1)-dir);
  IntVector i(idx-(d_refinementRatio-d-dir*d_refinementRatio));
  Vector o(d.asVector()*0.5);
  weight=Vector(double(o.x()+i.x()%d_refinementRatio.x())/double(d_refinementRatio.x()),
		  double(o.y()+i.y()%d_refinementRatio.y())/double(d_refinementRatio.y()),
		  double(o.z()+i.z()%d_refinementRatio.z())/double(d_refinementRatio.z()));
  return i/d_refinementRatio;
}

IntVector Level::mapCellToCoarser(const IntVector& idx) const
{ 
  IntVector ratio = idx/d_refinementRatio;
  

  // If the fine cell index is negative
  // you must add an offset to get the right
  // coarse cell. -Todd
  IntVector offset(0,0,0);
  if (idx.x()< 0 && d_refinementRatio.x() > 1){
    offset.x((int)fmod((double)idx.x(),(double)d_refinementRatio.x()));
  }
  if (idx.y()< 0 && d_refinementRatio.y() > 1){
    offset.y((int)fmod((double)idx.y(),(double)d_refinementRatio.y()));
  }  
  if (idx.z()< 0 && d_refinementRatio.z() > 1){
    offset.z((int)fmod((double)idx.z(),(double)d_refinementRatio.z()));
  }
  return ratio + offset;
}

IntVector Level::mapCellToFiner(const IntVector& idx) const
{
  IntVector r_ratio = grid->getLevel(d_index+1)->d_refinementRatio;
  IntVector fineCell = idx*r_ratio;
 
  IntVector offset(0,0,0);
  if (idx.x()< 0 && r_ratio.x() > 1){
    offset.x(1);
  }
  if (idx.y()< 0 && r_ratio.y() > 1){   // If the coarse cell index is negative
    offset.y(1);                        // you must add an offset to get the right
  }                                     // fine cell. -Todd
  if (idx.z()< 0 && r_ratio.z() > 1){
    offset.z(1);
  }    
  return fineCell + offset;
}

IntVector Level::mapNodeToCoarser(const IntVector& idx) const
{
  return (idx+d_refinementRatio-IntVector(1,1,1))/d_refinementRatio;
}

IntVector Level::mapNodeToFiner(const IntVector& idx) const
{
  return idx*grid->getLevel(d_index+1)->d_refinementRatio;
}

int Level::getRefinementRatioMaxDim() const {
  return Max(Max(d_refinementRatio.x(), d_refinementRatio.y()), d_refinementRatio.z());
}

namespace Uintah {
  const Level* getLevel(const PatchSubset* subset)
  {
    ASSERT(subset->size()>0);
    const Level* level = subset->get(0)->getLevel();
    for(int i=1;i<subset->size();i++){
      ASSERT(level == subset->get(i)->getLevel());
    }
    return level;
  }

  const Level* getLevel(const PatchSet* set)
  {
    ASSERT(set->size()>0);
    return getLevel(set->getSubset(0));
  }
}

