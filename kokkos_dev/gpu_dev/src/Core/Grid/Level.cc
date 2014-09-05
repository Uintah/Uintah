/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/

//Allgatherv currently performs poorly on Kraken.  
//This hack changes the Allgatherv to an allgather 
//by padding the digits
//#define AG_HACK  


#include <TauProfilerForSCIRun.h>

#include <Core/Grid/Level.h>
#include <Core/Util/Handle.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/PatchBVH/PatchBVH.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidGrid.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/BoundaryConditions/BoundCondReader.h>

#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Thread/AtomicCounter.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Core/Util/ProgressiveWarning.h>

#include <TauProfilerForSCIRun.h>

#include <iostream>
#include <algorithm>
#include <map>
#include <cmath>

#ifdef _WIN32
#define rint(x) (int)((x>0) ? x+.5 : x-.5)
#endif

using namespace std;
using namespace Uintah;
using namespace SCIRun;


static AtomicCounter ids("Level ID counter",0);
static Mutex ids_init("ID init");

static DebugStream bcout("BCTypes", false);
static DebugStream rgtimes("RGTimes",false);

Level::Level(Grid* grid, const Point& anchor, const Vector& dcell, 
             int index, IntVector refinementRatio, int id /*=-1*/)
   : grid(grid), d_anchor(anchor), d_dcell(dcell), 
     d_spatial_range(Point(DBL_MAX,DBL_MAX,DBL_MAX),Point(DBL_MIN,DBL_MIN,DBL_MIN)),
     d_int_spatial_range(Point(DBL_MAX,DBL_MAX,DBL_MAX),Point(DBL_MIN,DBL_MIN,DBL_MIN)),
     d_index(index),
     d_patchDistribution(-1,-1,-1), d_periodicBoundaries(0, 0, 0), d_id(id),
     d_refinementRatio(refinementRatio),
     d_cachelock("Level Cache Lock")
{
  d_stretched = false;
  each_patch=0;
  all_patches=0;
  d_bvh = NULL;
  d_finalized=false;
  d_extraCells = IntVector(0,0,0);
  d_totalCells = 0;

  if(d_id == -1)
    d_id = ids++;
  else if(d_id >= ids)
    ids.set(d_id+1);
}

Level::~Level()
{
  // Delete all of the patches managed by this level
  for(patchIterator iter=d_virtualAndRealPatches.begin(); iter != d_virtualAndRealPatches.end(); iter++)
    delete *iter;

  delete d_bvh;
  
  if(each_patch && each_patch->removeReference())
    delete each_patch;
  if(all_patches && all_patches->removeReference())
    delete all_patches;

  int patches_stored = 0;
  int queries_stored = 0;
  for (selectCache::iterator iter = d_selectCache.begin(); iter != d_selectCache.end(); iter++) {
    queries_stored++;
    patches_stored += iter->second.size();
  }
}

void Level::setPatchDistributionHint(const IntVector& hint)
{
    if(d_patchDistribution.x() == -1){
      d_patchDistribution = hint;
    }else{
      // Called more than once, we have to punt
      d_patchDistribution = IntVector(-2,-2,2);
    }
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
                       const IntVector& inHighIndex,
                       Grid* grid)
{
    Patch* r = scinew Patch(this, lowIndex,highIndex,inLowIndex, 
                            inHighIndex,getIndex());
    r->setGrid(grid);
    d_realPatches.push_back(r);
    d_virtualAndRealPatches.push_back(r);
    d_int_spatial_range.extend(r->getBox().lower());
    d_int_spatial_range.extend(r->getBox().upper());
    d_spatial_range.extend(r->getExtraBox().lower());
    d_spatial_range.extend(r->getExtraBox().upper());
    return r;
}

Patch* Level::addPatch(const IntVector& lowIndex, 
                       const IntVector& highIndex,
                       const IntVector& inLowIndex, 
                       const IntVector& inHighIndex,
                       Grid* grid,
                       int ID)
{
    Patch* r = scinew Patch(this, lowIndex,highIndex,inLowIndex, 
                            inHighIndex,getIndex(),ID);
    r->setGrid(grid);
    d_realPatches.push_back(r);
    d_virtualAndRealPatches.push_back(r);
    d_int_spatial_range.extend(r->getBox().lower());
    d_int_spatial_range.extend(r->getBox().upper());
    d_spatial_range.extend(r->getExtraBox().lower());
    d_spatial_range.extend(r->getExtraBox().upper());
    return r;
}

const Patch* Level::getPatchFromPoint(const Point& p, const bool includeExtraCells) const
{
  selectType patch;
  IntVector c=getCellIndex(p);
  //point is within the bounding box so query the bvh
  d_bvh->query(c,c+IntVector(1,1,1), patch,includeExtraCells);

  if(patch.size()==0)
    return 0;
  
  ASSERT(patch.size()==1);
  return patch[0];
}

const Patch* Level::getPatchFromIndex(const IntVector& c, const bool includeExtraCells) const
{
  selectType patch;
  
  //point is within the bounding box so query the bvh
  d_bvh->query(c,c+IntVector(1,1,1), patch,includeExtraCells);

  if(patch.size()==0)
    return 0;
  
  ASSERT(patch.size()==1);
  return patch[0];
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
      Box b1 = getBox(r1->getCellLowIndex(), r1->getCellHighIndex());
      Box b2 = getBox(r2->getCellLowIndex(), r2->getCellHighIndex());
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
  Vector l=(d_spatial_range.min()-d_anchor)/d_dcell;
  Vector h=(d_spatial_range.max()-d_anchor)/d_dcell+Vector(1,1,1);

  lowIndex  = roundNearest(l);
  highIndex = roundNearest(h);
}
void Level::findCellIndexRange(IntVector& lowIndex,IntVector& highIndex) const
{
  Vector l=(d_spatial_range.min()-d_anchor)/d_dcell;
  Vector h=(d_spatial_range.max()-d_anchor)/d_dcell; 

  lowIndex  = roundNearest(l);
  highIndex = roundNearest(h);
}

void Level::findInteriorCellIndexRange(IntVector& lowIndex,IntVector& highIndex) const
{
  Vector l=(d_int_spatial_range.min()-d_anchor)/d_dcell;
  Vector h=(d_int_spatial_range.max()-d_anchor)/d_dcell;
  
  lowIndex  = roundNearest(l);
  highIndex = roundNearest(h);
}

void Level::findInteriorNodeIndexRange(IntVector& lowIndex,IntVector& highIndex) const
{
  Vector l=(d_int_spatial_range.min()-d_anchor)/d_dcell;
  Vector h=(d_int_spatial_range.max()-d_anchor)/d_dcell+Vector(1,1,1);
  
  lowIndex  = roundNearest(l);
  highIndex = roundNearest(h);
}

long Level::totalCells() const
{
  return d_totalCells;
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
  if(d_stretched)
    return Point(d_facePosition[0][v.x()], d_facePosition[1][v.y()], d_facePosition[2][v.z()]);
  else
   return d_anchor+d_dcell*v;
}

Point Level::getCellPosition(const IntVector& v) const
{
  if(d_stretched)
    return Point((d_facePosition[0][v.x()]+d_facePosition[0][v.x()+1])*0.5,
                 (d_facePosition[1][v.y()]+d_facePosition[1][v.y()+1])*0.5,
                 (d_facePosition[2][v.z()]+d_facePosition[2][v.z()+1])*0.5);
  else
    return d_anchor+d_dcell*v+d_dcell*0.5;
}

static int binary_search(double x, const OffsetArray1<double>& faces, int low, int high)
{
  while(high-low > 1) {
    int m = (low + high)/2;
    if(x < faces[m])
      high = m;
    else
      low = m;
  }
  return low;
}

IntVector Level::getCellIndex(const Point& p) const
{
  if(d_stretched){
    int x = binary_search(p.x(), d_facePosition[0], d_facePosition[0].low(), d_facePosition[0].high());
    int y = binary_search(p.y(), d_facePosition[1], d_facePosition[1].low(), d_facePosition[1].high());
    int z = binary_search(p.z(), d_facePosition[2], d_facePosition[2].low(), d_facePosition[2].high());
    return IntVector(x, y, z);
  } else {
    Vector v((p-d_anchor)/d_dcell);
    return IntVector(RoundDown(v.x()), RoundDown(v.y()), RoundDown(v.z()));
  }
}

Point Level::positionToIndex(const Point& p) const
{
  if(d_stretched){
    int x = binary_search(p.x(), d_facePosition[0], d_facePosition[0].low(), d_facePosition[0].high());
    int y = binary_search(p.y(), d_facePosition[1], d_facePosition[1].low(), d_facePosition[1].high());
    int z = binary_search(p.z(), d_facePosition[2], d_facePosition[2].low(), d_facePosition[2].high());

    //#if SCI_ASSERTION_LEVEL > 0
    //    if( ( x == d_facePosition[0].high() ) ||
    //        ( y == d_facePosition[1].high() ) ||
    //        ( z == d_facePosition[2].high() ) ) {
    //      static ProgressiveWarning warn( "positionToIndex called with too large a point.", -1 );
    //    }
    //#endif

    // If p.x() == the value of the last position in
    // d_facePosition[0], then the binary_search returns the "high()"
    // value... and the interpolation below segfaults due to trying to
    // go to x+1.  The following check prevents this from happening.
    x = min( x, d_facePosition[0].high()-2 );
    y = min( y, d_facePosition[1].high()-2 );
    z = min( z, d_facePosition[2].high()-2 );

    double xfrac = (p.x() - d_facePosition[0][x]) / (d_facePosition[0][x+1] - d_facePosition[0][x]);
    double yfrac = (p.y() - d_facePosition[1][y]) / (d_facePosition[1][y+1] - d_facePosition[1][y]);
    double zfrac = (p.z() - d_facePosition[2][z]) / (d_facePosition[2][z+1] - d_facePosition[2][z]);
    return Point(x+xfrac, y+yfrac, z+zfrac);
  } else {
    return Point((p-d_anchor)/d_dcell);
  }
}

void Level::selectPatches(const IntVector& low, const IntVector& high,
                          selectType& neighbors, bool withExtraCells, bool cache) const
{
 TAU_PROFILE("Level::selectPatches", " ", TAU_USER);
    
 if(cache){
   // look it up in the cache first
   d_cachelock.readLock();
   selectCache::const_iterator iter = d_selectCache.find(make_pair(low, high));
   if (iter != d_selectCache.end()) {
     const vector<const Patch*>& cache = iter->second;
     for (unsigned i = 0; i < cache.size(); i++) {
       neighbors.push_back(cache[i]);
     }
     d_cachelock.readUnlock();
     return;
   }
   d_cachelock.readUnlock();
   ASSERT(neighbors.size() == 0);
 }

   //cout << Parallel::getMPIRank() << " Level Quesy: " << low << " " << high << endl;
   d_bvh->query(low, high, neighbors, withExtraCells);
   sort(neighbors.begin(), neighbors.end(), Patch::Compare());

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

   if(cache){
     // put it in the cache - start at orig_size in case there was something in
     // neighbors before this query
     d_cachelock.writeLock();
     vector<const Patch*>& cache = d_selectCache[make_pair(low,high)];
     cache.reserve(6);  // don't reserve too much to save memory, not too little to avoid too much reallocation
     for (int i = 0; i < neighbors.size(); i++) {
       cache.push_back(neighbors[i]);
     }
     d_cachelock.writeUnlock();
   }
}

bool Level::containsPointIncludingExtraCells(const Point& p) const
{
  bool includeExtraCells = true;
  return getPatchFromPoint(p, includeExtraCells)!=0;
}

bool Level::containsPoint(const Point& p) const
{
  bool includeExtraCells = false;
  const Patch* patch=getPatchFromPoint(p,includeExtraCells);
  return patch != 0;
}

bool Level::containsCell(const IntVector& idx) const
{
  bool includeExtraCells = false;
  const Patch* patch=getPatchFromIndex(idx,includeExtraCells);
  return patch != 0;
}


void Level::finalizeLevel()
{
  MALLOC_TRACE_TAG_SCOPE("Level::finalizeLevel");
  TAU_PROFILE("Level::finalizeLevel()", " ", TAU_USER);
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
  
  //determines and sets the boundary conditions for the patches
  setBCTypes();

  //finalize the patches
  for(patchIterator iter=d_virtualAndRealPatches.begin();iter!=d_virtualAndRealPatches.end();iter++)
  {
    (*iter)->finalizePatch();
  }

  //compute the number of cells in the level
  d_totalCells=0;
  for(int i=0;i<(int)d_realPatches.size();i++)
  {
    d_totalCells+=d_realPatches[i]->getNumExtraCells();
  }
  
  //compute and store the spatial ranges now that BCTypes are set
  for(int i=0;i<(int)d_realPatches.size();i++){
    Patch* r = d_realPatches[i];
    
    d_spatial_range.extend(r->getExtraBox().lower());
    d_spatial_range.extend(r->getExtraBox().upper());
  }
}

void Level::finalizeLevel(bool periodicX, bool periodicY, bool periodicZ)
{
  MALLOC_TRACE_TAG_SCOPE("Level::finalizeLevel(periodic)");
  TAU_PROFILE("Level::finalizeLevel(periodic)", " ", TAU_USER);

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
    grid->getLevel(0)->getInteriorSpatialRange(bbox);
  else
    getInteriorSpatialRange(bbox);

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
            getBox(tmp_patches[i]->getExtraCellLowIndex() + offset - IntVector(1,1,1),
                   tmp_patches[i]->getExtraCellHighIndex() + offset + IntVector(1,1,1));
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
  
  //finalize the patches
  for(patchIterator iter=d_virtualAndRealPatches.begin();iter!=d_virtualAndRealPatches.end();iter++)
  {
    (*iter)->finalizePatch();
  }

  //compute the number of cells in the level
  d_totalCells=0;
  for(int i=0;i<(int)d_realPatches.size();i++)
  {
    d_totalCells+=d_realPatches[i]->getNumExtraCells();
  }
  
  //compute and store the spatial ranges now that BCTypes are set
  for(int i=0;i<(int)d_realPatches.size();i++){
    Patch* r = d_realPatches[i];
    
    d_spatial_range.extend(r->getExtraBox().lower());
    d_spatial_range.extend(r->getExtraBox().upper());
  }
}
void Level::setBCTypes()
{
  double rtimes[4]={0};
  double start=Time::currentSeconds();

  MALLOC_TRACE_TAG_SCOPE("Level::setBCTypes");
  TAU_PROFILE("Level::setBCTypes", " ", TAU_USER);
  if (d_bvh != NULL)
    delete d_bvh;
  d_bvh = scinew PatchBVH(d_virtualAndRealPatches);
  rtimes[0]+=Time::currentSeconds()-start;
  start=Time::currentSeconds();
  patchIterator iter;
  
  ProcessorGroup *myworld=NULL;
  int numProcs=1;
  int rank=0;
  if (Parallel::isInitialized()) {
    // only sus uses Parallel, but anybody else who uses DataArchive to read data does not
    myworld=Parallel::getRootProcessorGroup();
    numProcs=myworld->size();
    rank=myworld->myrank();
  }

  vector<int> displacements(numProcs,0);
  vector<int> recvcounts(numProcs,0);

  //create recvcounts and displacement arrays
  int div=d_virtualAndRealPatches.size()/numProcs;
  int mod=d_virtualAndRealPatches.size()%numProcs;
  for(int p=0;p<numProcs;p++)
  {     
    if(p<mod)
      recvcounts[p]=div+1;
    else
      recvcounts[p]=div;
  }
  displacements[0]=0;
  for(int p=1;p<numProcs;p++)
  {
    displacements[p]=displacements[p-1]+recvcounts[p-1];
  }
   
  vector<unsigned int> bctypes(d_virtualAndRealPatches.size());
  vector<unsigned int> mybctypes(recvcounts[rank]);

  int idx;
  
  patchIterator startpatch=d_virtualAndRealPatches.begin()+displacements[rank];
  patchIterator endpatch=startpatch+recvcounts[rank];
  //for each of my patches
  for(iter=startpatch,idx=0; iter != endpatch; iter++,idx++){
    Patch* patch = *iter;
    //cout << "Patch bounding box = " << patch->getExtraBox() << endl;
    // See if there are any neighbors on the 6 faces
    int bitfield=0;
    for(Patch::FaceType face = Patch::startFace;
        face <= Patch::endFace; face=Patch::nextFace(face)){
      bitfield<<=2;
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
          
#if 0
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
#endif
          coarseLevel->selectPatches(coarseLow, coarseHigh, neighbors);
          
          if(neighbors.size() == 0){
            bitfield|=Patch::None;
          } else {
            bitfield|=Patch::Coarse;
          }
        } else {
          bitfield|=Patch::None;
        }
      } else {
        bitfield|=Patch::Neighbor;
      }
    }
    mybctypes[idx]=bitfield;
  }
  
  if(numProcs>1)
  {
#ifdef AG_HACK
    int max_size=div;
    if(mod!=0)
      max_size++;
    //make temporary vectors
    vector<unsigned int> bctypes2(max_size*myworld->size());
    vector<unsigned int> mybctypes2(mybctypes);
    mybctypes2.resize(max_size);

    //gather bctypes
    MPI_Allgather(&mybctypes2[0],max_size,MPI_UNSIGNED,&bctypes2[0],max_size,MPI_UNSIGNED,myworld->getComm());
   
    //displacements[p]=displacements[p-1]+recvcounts[p-1];
    //write bctypes2 back into bctypes
    int j=0;
    for(int p=0;p<myworld->size();p++)
    {
      int start=max_size*p;
      int end=start+recvcounts[p];
      for(int i=start;i<end;i++)
        bctypes[j++]=bctypes2[i];
    }
    mybctypes2.clear();
    bctypes2.clear();
#else
    //allgather bctypes
    if(mybctypes.size()==0)
    {
      MPI_Allgatherv(0,0,MPI_UNSIGNED,&bctypes[0],&recvcounts[0],&displacements[0],MPI_UNSIGNED,myworld->getComm());
    }
    else
      MPI_Allgatherv(&mybctypes[0],mybctypes.size(),MPI_UNSIGNED,&bctypes[0],&recvcounts[0],&displacements[0],MPI_UNSIGNED,myworld->getComm());
#endif
  }
  else
  {
     bctypes.swap(mybctypes);
  }
  rtimes[1]+=Time::currentSeconds()-start;
  start=Time::currentSeconds();
  int i;
  //loop through patches
  for(iter=d_virtualAndRealPatches.begin(),i=0,idx=0;iter!=d_virtualAndRealPatches.end();iter++,i++)
  {
    Patch *patch=*iter;
  
    if(patch->isVirtual())
      patch->setLevelIndex( -1 );
    else
      patch->setLevelIndex(idx++);
    
    int bitfield=bctypes[i];
    int mask=3;
    //loop through faces
    for(int j=5;j>=0;j--)
    {
      int bc_type=bitfield&mask;
      if(rank==0)
      {
        switch(bc_type)
        {
          case Patch::None:
            bcout << "  Setting Patch " << patch->getID() << " face " << j << " to None\n";      
            break;
          case Patch::Coarse:
            bcout << "  Setting Patch " << patch->getID() << " face " << j << " to Coarse\n";
            break;
          case Patch::Neighbor:
            bcout << "  Setting Patch " << patch->getID() << " face " << j << " to Neighbor\n";
            break;
        }
      }
      patch->setBCType(Patch::FaceType(j),Patch::BCType(bc_type));
      bitfield>>=2;
    }
  }
  
  //__________________________________
  //  bullet proofing
  for (int dir=0; dir<3; dir++){
    if(d_periodicBoundaries[dir] == 1 && d_extraCells[dir] !=0) {
      ostringstream warn;
      warn<< "\n \n INPUT FILE ERROR: \n You've specified a periodic boundary condition on a face with extra cells specified\n"
          <<" Please set the extra cells on that face to 0";
      throw ProblemSetupException(warn.str(),__FILE__,__LINE__);
    }
  }
  
  
  d_finalized=true;
  
  rtimes[2]+=Time::currentSeconds()-start;
  start=Time::currentSeconds();
  if(rgtimes.active())
  {
    double avg[3]={0};
    MPI_Reduce(&rtimes,&avg,3,MPI_DOUBLE,MPI_SUM,0,myworld->getComm());
    if(myworld->myrank()==0) {
      cout << "SetBCType Avg Times: ";
      for(int i=0;i<3;i++)
      {
        avg[i]/=myworld->size();
        cout << avg[i] << " ";
      }
      cout << endl;
    }
    double max[3]={0};
    MPI_Reduce(&rtimes,&max,3,MPI_DOUBLE,MPI_MAX,0,myworld->getComm());
    if(myworld->myrank()==0) {
      cout << "SetBCType Max Times: ";
      for(int i=0;i<3;i++)
      {
        cout << max[i] << " ";
      }
      cout << endl;
    }
  }

  //recreate BVH with extracells
  if (d_bvh != NULL)
    delete d_bvh;
  d_bvh = scinew PatchBVH(d_virtualAndRealPatches);
  


}

void Level::assignBCS(const ProblemSpecP& grid_ps,LoadBalancer* lb)
{
  TAU_PROFILE("Level::assignBCS()", " ", TAU_USER);
  
  ProblemSpecP bc_ps = grid_ps->findBlock("BoundaryConditions");
  if (bc_ps == 0 ) {
    if ( Parallel::getMPIRank()==0 ){
      static ProgressiveWarning warn("No BoundaryConditions specified", -1);
      warn.invoke();
    }
    return;
  }
  
  BoundCondReader reader;
  reader.read(bc_ps, grid_ps);
    
  for(patchIterator iter=d_virtualAndRealPatches.begin(); 
    iter != d_virtualAndRealPatches.end(); iter++){
    Patch* patch = *iter;
    
    //if we have a lb then only apply bcs this processors patches
    if(lb==0 || lb->getPatchwiseProcessorAssignment(patch)==Parallel::getMPIRank())
    {
      patch->initializeBoundaryConditions();
      for(Patch::FaceType face_side = Patch::startFace; 
          face_side <= Patch::endFace; face_side=Patch::nextFace(face_side)) {
        if (patch->getBCType(face_side) == Patch::None) {
          patch->setArrayBCValues(face_side,&(reader.d_BCReaderData[face_side]));
        }
      }  // end of face iterator
    }
  } //end of patch iterator
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
  selectPatches(idx - i,idx + i,pv,false,false);
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
  selectPatches(idx - i,idx + i,pv,false,false);
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

bool Level::hasCoarserLevel() const
{
  return getIndex() > 0;
}

bool Level::hasFinerLevel() const
{
  return getIndex() < grid->numLevels()-1;
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
//__________________________________
// mapNodeToCoarser:
// Example: 1D grid with refinement ratio = 4
//  Coarse Node index: 10                  11    
//                     |                   |       
//                 ----*----*----*----*----*-----  
//                     |                   |       
//  Fine Node Index    40   41   42   43   44      
//                            
//  What is returned   10   10   10   10   11
IntVector Level::mapNodeToCoarser(const IntVector& idx) const
{
  return (idx+d_refinementRatio-IntVector(1,1,1))/d_refinementRatio;
}

//__________________________________
// mapNodeToFiner:
// Example: 1D grid with refinement ratio = 4
//  Coarse Node index: 10                  11    
//                     |                   |       
//                 ----*----*----*----*----*-----  
//                     |                   |       
//  Fine Node Index    40   41   42   43   44      
//                            
//  What is returned   40                  44
IntVector Level::mapNodeToFiner(const IntVector& idx) const
{
  return idx*grid->getLevel(d_index+1)->d_refinementRatio;
}

// Stretched grid stuff
void Level::getCellWidths(Grid::Axis axis, OffsetArray1<double>& widths) const
{
  const OffsetArray1<double>& faces = d_facePosition[axis];
  widths.resize(faces.low(), faces.high()-1);
  for(int i=faces.low(); i < faces.high()-1; i++)
    widths[i] = faces[i+1] - faces[i];
}
    
void Level::getFacePositions(Grid::Axis axis, OffsetArray1<double>& faces) const
{
  faces = d_facePosition[axis];
}
      
void Level::setStretched(Grid::Axis axis, const OffsetArray1<double>& faces)
{
  d_facePosition[axis] = faces;
  d_stretched = true;
}

int Level::getRefinementRatioMaxDim() const {
  return Max(Max(d_refinementRatio.x(), d_refinementRatio.y()), d_refinementRatio.z());
}

namespace Uintah {
  const Level* getLevel(const PatchSubset* subset)
  {
    ASSERT(subset->size()>0);
    const Level* level = subset->get(0)->getLevel();
#if SCI_ASSERTION_LEVEL>0
    for(int i=1;i<subset->size();i++){
      ASSERT(level == subset->get(i)->getLevel());
    }
#endif
    return level;
  }

  const LevelP& getLevelP(const PatchSubset* subset)
  {
    ASSERT(subset->size()>0);
    const LevelP& level = subset->get(0)->getLevelP();
#if SCI_ASSERTION_LEVEL>0
    for(int i=1;i<subset->size();i++){
      ASSERT(level == subset->get(i)->getLevelP());
    }
#endif
    return level;
  }

  const Level* getLevel(const PatchSet* set)
  {
    ASSERT(set->size()>0);
    return getLevel(set->getSubset(0));
  }
}
