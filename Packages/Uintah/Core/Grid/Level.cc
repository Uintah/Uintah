
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/ProblemSpec/Handle.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/InvalidGrid.h>
#include <Packages/Uintah/Core/Grid/BoundCondReader.h>
#include <Packages/Uintah/Core/Grid/BoundCondData.h>

#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Thread/AtomicCounter.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>

#include <iostream>
#include <algorithm>
#include <map>
#include <math.h>

#ifdef SELECT_RANGETREE
#include <Packages/Uintah/Core/Grid/PatchRangeTree.h>
#endif

using namespace Uintah;
using namespace SCIRun;
using namespace std;

static AtomicCounter* ids = 0;
static Mutex ids_init("ID init");

Level::Level(Grid* grid, const Point& anchor, const Vector& dcell, 
             int index, int id /*=-1*/)
   : grid(grid), d_anchor(anchor), d_dcell(dcell), d_index(index),
     d_patchDistribution(-1,-1,-1),
     d_periodicBoundaries(0, 0, 0), d_id(id)
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
  refinementRatio = IntVector(2,2,2); // Hardcoded for now...
  d_timeRefinementRatio = 2;
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
     SCI_THROW(InvalidGrid("Consistency check cannot be performed until Level is finalized"));
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
      if(r1->getBox().overlaps(r2->getBox())){
	cerr << "r1: " << *r1 << '\n';
	cerr << "r2: " << *r2 << '\n';
	SCI_THROW(InvalidGrid("Two patches overlap"));
      }
    }
  }

  // Insert code to see if abutting boxes have consistent bounds
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

void Level::getSpatialRange(BBox& b) const
{
  for(int i=0;i<(int)d_realPatches.size();i++){
    Patch* r = d_realPatches[i];
    b.extend(r->getBox().lower());
    b.extend(r->getBox().upper());
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
#ifdef SELECT_LINEAR
   // This sucks - it should be made faster.  -Steve
   for(const_patchIterator iter=d_virtualAndRealPatches.begin();
       iter != d_virtualAndRealPatches.end(); iter++){
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
	  patch->getFace(face, IntVector(0,0,0), refinementRatio,
			 fineLow, fineHigh);
	  IntVector coarseLow = mapCellToCoarser(fineLow);
	  IntVector coarseHigh = mapCellToCoarser(fineHigh);
	  const LevelP& coarseLevel = getCoarserLevel();
	  coarseLevel->selectPatches(coarseLow, coarseHigh, neighbors);
	  if(neighbors.size() == 0){
	    patch->setBCType(face, Patch::None);
	  } else {
	    patch->setBCType(face, Patch::Coarse);
	  }
	} else {
	  patch->setBCType(face, Patch::None);
	}
      } else {
	patch->setBCType(face, Patch::Neighbor);
      }
    }
  }
  
  d_finalized=true;
}

void Level::assignBCS(const ProblemSpecP& grid_ps)
{
  ProblemSpecP bc_ps = grid_ps->findBlock("BoundaryConditions");
  if (bc_ps == 0)
    return;

  BCReader reader;
  reader.read(bc_ps);

  for (Patch::FaceType face_side = Patch::startFace; 
       face_side <= Patch::endFace; face_side=Patch::nextFace(face_side)) {
    
    for(patchIterator iter=d_virtualAndRealPatches.begin(); 
	iter != d_virtualAndRealPatches.end(); iter++){
      Patch* patch = *iter;
      Patch::BCType bc_type = patch->getBCType(face_side);
      BoundCondData bc_data;

      // For old boundary conditions
      reader.getBC(face_side,bc_data);
      if (bc_type == Patch::None) {
	patch->setBCValues(face_side,bc_data);
	patch->setArrayBCValues(face_side,reader.d_BCReaderData[face_side]);
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

const LevelP& Level::getCoarserLevel() const
{
  return getRelativeLevel(-1);
}

const LevelP& Level::getFinerLevel() const
{
  return getRelativeLevel(1);
}

IntVector Level::mapCellToCoarser(const IntVector& idx) const
{
  return idx/refinementRatio;
}

IntVector Level::mapCellToCoarser(const IntVector& idx, Vector& weight) const
{
  IntVector i(idx-(refinementRatio-IntVector(1,1,1)));
  weight=Vector(double(0.5+i.x()%refinementRatio.x())/double(refinementRatio.x()),
		double(0.5+i.y()%refinementRatio.y())/double(refinementRatio.y()),
		double(0.5+i.z()%refinementRatio.z())/double(refinementRatio.z()));
  return i/refinementRatio;
}

IntVector Level::mapXFaceToCoarser(const IntVector& idx, Vector& weight) const
{
  IntVector i(idx-(refinementRatio-IntVector(refinementRatio.x(),1,1)));
  weight=Vector(double(i.x()%refinementRatio.x())/double(refinementRatio.x()),
		double(0.5+i.y()%refinementRatio.y())/double(refinementRatio.y()),
		double(0.5+i.z()%refinementRatio.z())/double(refinementRatio.z()));
  return i/refinementRatio;
}

IntVector Level::mapYFaceToCoarser(const IntVector& idx, Vector& weight) const
{
  IntVector i(idx-(refinementRatio-IntVector(1,refinementRatio.y(),1)));
  weight=Vector(double(0.5+i.x()%refinementRatio.x())/double(refinementRatio.x()),
		double(i.y()%refinementRatio.y())/double(refinementRatio.y()),
		double(0.5+i.z()%refinementRatio.z())/double(refinementRatio.z()));
  return i/refinementRatio;
}

IntVector Level::mapZFaceToCoarser(const IntVector& idx, Vector& weight) const
{
  IntVector i(idx-(refinementRatio-IntVector(1,1,refinementRatio.z())));
  weight=Vector(double(0.5+i.x()%refinementRatio.x())/double(refinementRatio.x()),
		double(0.5+i.y()%refinementRatio.y())/double(refinementRatio.y()),
		double(i.z()%refinementRatio.z())/double(refinementRatio.z()));
  return i/refinementRatio;
}

IntVector Level::mapToCoarser(const IntVector& idx, const IntVector& dir,
			      Vector& weight) const
{
  IntVector d(IntVector(1,1,1)-dir);
  IntVector i(idx-(refinementRatio-d-dir*refinementRatio));
  Vector o(d.asVector()*0.5);
  weight=Vector(double(o.x()+i.x()%refinementRatio.x())/double(refinementRatio.x()),
		double(o.y()+i.y()%refinementRatio.y())/double(refinementRatio.y()),
		double(o.z()+i.z()%refinementRatio.z())/double(refinementRatio.z()));
  return i/refinementRatio;
}

IntVector Level::mapCellToFiner(const IntVector& idx) const
{
  return idx*grid->getLevel(d_index+1)->refinementRatio;
}

IntVector Level::mapNodeToCoarser(const IntVector& idx) const
{
  return (idx+refinementRatio-IntVector(1,1,1))/refinementRatio;
}

IntVector Level::mapNodeToFiner(const IntVector& idx) const
{
  return idx*grid->getLevel(d_index+1)->refinementRatio;
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
}

