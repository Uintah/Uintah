/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#include <Core/Exceptions/InvalidGrid.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/BBox.h>
#include <Core/Grid/BoundaryConditions/UnstructuredBoundCondReader.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/UnstructuredGrid.h>
#include <Core/Grid/UnstructuredLevel.h>
#include <Core/Grid/UnstructuredPatch.h>
#include <Core/Grid/PatchBVH/UnstructuredPatchBVH.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/OS/ProcessInfo.h> // For Memory Check
#include <Core/Parallel/CrowdMonitor.hpp>
#include <Core/Parallel/MasterLock.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/Timers/Timers.hpp>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/Handle.h>
#include <Core/Util/ProgressiveWarning.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <map>
#include <climits>

using namespace Uintah;

namespace {

  std::atomic<int32_t> ids{0};
  Uintah::MasterLock   ids_init{};
  Uintah::MasterLock   patch_cache_mutex{};

  DebugStream bcout{   "BCTypes",      "Grid_Level", "Grid Level BC debug stream", false };
  DebugStream rgtimes{ "RGTimesLevel", "Grid_Level", "Grid regridder debug stream", false };  
}


//______________________________________________________________________
//
UnstructuredLevel::UnstructuredLevel(       UnstructuredGrid      * grid
            , const Point     & anchor
            , const Vector    & dcell
            , const int         index
            , const IntVector   refinementRatio
            , const int         id /* = -1 */
            )
  : m_grid( grid )
  , m_anchor( anchor )
  , m_dcell( dcell )
  , m_spatial_range( Uintah::Point(DBL_MAX,DBL_MAX,DBL_MAX),Point(DBL_MIN,DBL_MIN,DBL_MIN) )
  , m_int_spatial_range( Uintah::Point(DBL_MAX,DBL_MAX,DBL_MAX),Point(DBL_MIN,DBL_MIN,DBL_MIN) )
  , m_index( index )
  , m_patch_distribution( -1,-1,-1 )
  , m_periodic_boundaries( 0, 0, 0 )
  , m_id( id )
  , m_refinement_ratio( refinementRatio )
{
  if( m_id == -1 ) {
    m_id = ids.fetch_add( 1, std::memory_order_relaxed );
  }
  else if( m_id >= ids ) {
    ids.store( m_id + 1, std::memory_order_relaxed );
  }
}

//______________________________________________________________________
//
UnstructuredLevel::~UnstructuredLevel()
{
  // Delete all of the patches managed by this level
  for (patch_iterator iter = m_virtual_and_real_patches.begin(); iter != m_virtual_and_real_patches.end(); iter++) {
    delete *iter;
  }

  delete m_bvh;

  if (m_each_patch && m_each_patch->removeReference()) {
    delete m_each_patch;
  }
  if (m_all_patches && m_all_patches->removeReference()) {
    delete m_all_patches;
  }

  int patches_stored = 0;
  int queries_stored = 0;
  for (select_cache::iterator iter = m_select_cache.begin(); iter != m_select_cache.end(); iter++) {
    queries_stored++;
    patches_stored += iter->second.size();
  }
}

//______________________________________________________________________
//
void UnstructuredLevel::setPatchDistributionHint( const IntVector & hint )
{
  if (m_patch_distribution.x() == -1) {
    m_patch_distribution = hint;
  } else {
    // Called more than once, we have to punt
    m_patch_distribution = IntVector(-2, -2, 2);
  }
}

//______________________________________________________________________
//
UnstructuredLevel::const_patch_iterator UnstructuredLevel::patchesBegin() const
{
  return m_real_patches.begin();
}

//______________________________________________________________________
//
UnstructuredLevel::const_patch_iterator UnstructuredLevel::patchesEnd() const
{
  return m_real_patches.end();
}

//______________________________________________________________________
//
UnstructuredLevel::patch_iterator UnstructuredLevel::patchesBegin()
{
  return m_real_patches.begin();
}

//______________________________________________________________________
//
UnstructuredLevel::patch_iterator UnstructuredLevel::patchesEnd()
{
  return m_real_patches.end();
}

//______________________________________________________________________
//
UnstructuredLevel::const_patch_iterator UnstructuredLevel::allPatchesBegin() const
{
  return m_virtual_and_real_patches.begin();
}

//______________________________________________________________________
//
UnstructuredLevel::const_patch_iterator UnstructuredLevel::allPatchesEnd() const
{
  return m_virtual_and_real_patches.end();
}

//______________________________________________________________________
//
UnstructuredPatch*
UnstructuredLevel::addPatch( const IntVector & lowIndex
               , const IntVector & highIndex
               , const IntVector & inLowIndex
               , const IntVector & inHighIndex
               ,       UnstructuredGrid      * grid
               )
{
  UnstructuredPatch* r = scinew UnstructuredPatch( this, lowIndex, highIndex, inLowIndex, inHighIndex, getIndex() );
  r->setGrid( grid );
  m_real_patches.push_back( r );
  m_virtual_and_real_patches.push_back( r );

  m_int_spatial_range.extend( r->getBox().lower() );
  m_int_spatial_range.extend( r->getBox().upper() );

  m_spatial_range.extend( r->getExtraBox().lower() );
  m_spatial_range.extend( r->getExtraBox().upper() );

  return r;
}

//______________________________________________________________________
//
UnstructuredPatch*
UnstructuredLevel::addPatch( const IntVector & lowIndex
               , const IntVector & highIndex
               , const IntVector & inLowIndex
               , const IntVector & inHighIndex
               ,       UnstructuredGrid      * grid
               ,       int         ID
               )
{
  UnstructuredPatch* r = scinew UnstructuredPatch(this, lowIndex, highIndex, inLowIndex, inHighIndex, getIndex(), ID);
  r->setGrid(grid);
  m_real_patches.push_back(r);
  m_virtual_and_real_patches.push_back(r);

  m_int_spatial_range.extend(r->getBox().lower());
  m_int_spatial_range.extend(r->getBox().upper());

  m_spatial_range.extend(r->getExtraBox().lower());
  m_spatial_range.extend(r->getExtraBox().upper());

  return r;
}

//______________________________________________________________________
//
const UnstructuredPatch*
UnstructuredLevel::getPatchFromPoint( const Point & p, const bool includeExtraCells ) const
{
  selectType patch;
  IntVector c = getCellIndex(p);
  //point is within the bounding box so query the bvh
  // Comment to get to compile  m_bvh->query(c, c + IntVector(1, 1, 1), patch, includeExtraCells);

  if (patch.size() == 0) {
    return 0;
  }

  ASSERT(patch.size() == 1);
  return patch[0];
}

//______________________________________________________________________
//
const UnstructuredPatch*
UnstructuredLevel::getPatchFromIndex( const IntVector & c, const bool includeExtraCells ) const
{
  selectType patch;

  // Point is within the bounding box so query the bvh.
  // Comment to get to compile  m_bvh->query(c, c + IntVector(1, 1, 1), patch, includeExtraCells);

  if (patch.size() == 0) {
    return 0;
  }

  ASSERT(patch.size() == 1);
  return patch[0];
}

//______________________________________________________________________
//
int
UnstructuredLevel::numPatches() const
{
  return static_cast<int>(m_real_patches.size());
}

//______________________________________________________________________
//
void
UnstructuredLevel::performConsistencyCheck() const
{
  if (!m_finalized) {
    SCI_THROW(InvalidGrid("Consistency check cannot be performed until Level is finalized",__FILE__,__LINE__));
  }

  for (int i = 0; i < (int)m_virtual_and_real_patches.size(); i++) {
    UnstructuredPatch* r = m_virtual_and_real_patches[i];
    r->performConsistencyCheck();
  }

  // This is O(n^2) - we should fix it someday if it ever matters
  //   This checks that patches do not overlap
  for (int i = 0; i < (int)m_virtual_and_real_patches.size(); i++) {
    UnstructuredPatch* r1 = m_virtual_and_real_patches[i];

    for (int j = i + 1; j < (int)m_virtual_and_real_patches.size(); j++) {
      UnstructuredPatch* r2 = m_virtual_and_real_patches[j];
      Box b1 = getBox(r1->getCellLowIndex(), r1->getCellHighIndex());
      Box b2 = getBox(r2->getCellLowIndex(), r2->getCellHighIndex());

      if (b1.overlaps(b2)) {
        std::cerr << "r1: " << *r1 << '\n';
        std::cerr << "r2: " << *r2 << '\n';
        SCI_THROW(InvalidGrid("Two patches overlap",__FILE__,__LINE__));
      }
    }
  }
}

//______________________________________________________________________
//
void
UnstructuredLevel::findNodeIndexRange( IntVector & lowIndex, IntVector & highIndex ) const
{
  Vector l = (m_spatial_range.min() - m_anchor) / m_dcell;
  Vector h = (m_spatial_range.max() - m_anchor) / m_dcell + Vector(1, 1, 1);

  lowIndex  = roundNearest(l);
  highIndex = roundNearest(h);
}

//______________________________________________________________________
//
void UnstructuredLevel::findCellIndexRange( IntVector & lowIndex, IntVector & highIndex ) const
{
  Vector l = (m_spatial_range.min() - m_anchor) / m_dcell;
  Vector h = (m_spatial_range.max() - m_anchor) / m_dcell;

  lowIndex  = roundNearest(l);
  highIndex = roundNearest(h);
}

//______________________________________________________________________
//
void UnstructuredLevel::findInteriorCellIndexRange( IntVector & lowIndex, IntVector & highIndex ) const
{
  Vector l = (m_int_spatial_range.min() - m_anchor) / m_dcell;
  Vector h = (m_int_spatial_range.max() - m_anchor) / m_dcell;

  lowIndex  = roundNearest(l);
  highIndex = roundNearest(h);
}

//______________________________________________________________________
//
void UnstructuredLevel::findInteriorNodeIndexRange( IntVector & lowIndex, IntVector & highIndex) const
{
  Vector l = (m_int_spatial_range.min() - m_anchor) / m_dcell;
  Vector h = (m_int_spatial_range.max() - m_anchor) / m_dcell + Vector(1, 1, 1);

  lowIndex = roundNearest(l);
  highIndex = roundNearest(h);
}

//______________________________________________________________________
//  Compute the variable extents for this variable type
void UnstructuredLevel::computeVariableExtents( const UnstructuredTypeDescription::UnstructuredType TD
                                  , IntVector & lo
                                  , IntVector & hi
                                  ) const
{
  IntVector CCLo;
  IntVector CCHi;
  findCellIndexRange(CCLo, CCHi);

  switch (TD) {
    case UnstructuredTypeDescription::CCVariable :
    case UnstructuredTypeDescription::ParticleVariable :
      lo = CCLo;
      hi = CCHi;
      break;
    case UnstructuredTypeDescription::SFCXVariable :
      lo = CCLo;
      hi = CCHi + IntVector(1, 0, 0);
      break;
    case UnstructuredTypeDescription::SFCYVariable :
      lo = CCLo;
      hi = CCHi + IntVector(0, 1, 0);
      break;
    case UnstructuredTypeDescription::SFCZVariable :
      lo = CCLo;
      hi = CCHi + IntVector(0, 0, 1);
      break;
    case UnstructuredTypeDescription::NCVariable :
      findNodeIndexRange(lo, hi);
      break;
    default :
      std::string me = UnstructuredTypeDescription::toString( TD );
      throw InternalError("  ERROR: UnstructuredLevel::computeVariableExtents type description (" + me + ") not supported", __FILE__, __LINE__);
  }
}

//______________________________________________________________________
//
long
UnstructuredLevel::totalCells() const
{
  return m_total_cells;
}

//______________________________________________________________________
//
long
UnstructuredLevel::getTotalCellsInRegion(const IntVector& lowIndex, const IntVector& highIndex) const {

  // Not all simulations are cubic.  Some simulations might be L shaped, or T shaped, or + shaped, etc.
  // It is not enough to simply do a high - low to figure out the amount of simulation cells.  We instead
  // need to go all patches and see if they exist in this range.  If so, add up their cells.
  // This process is similar to how d_totalCells is computed in UnstructuredLevel::finalizeLevel().

  long cellsInRegion = 0;
  if (m_isNonCubicDomain == false ){
    IntVector diff( highIndex - lowIndex);
    cellsInRegion += diff.x() * diff.y() * diff.z();
    return cellsInRegion;
  }

  for( int i = 0; i < (int)m_real_patches.size(); i++ ) {
    IntVector patchLow  =  m_real_patches[i]->getExtraCellLowIndex();
    IntVector patchHigh =  m_real_patches[i]->getExtraCellHighIndex();

    if( doesIntersect(lowIndex, highIndex, patchLow, patchHigh) ){

      IntVector regionLo = Uintah::Max( lowIndex,  patchLow );
      IntVector regionHi = Uintah::Min( highIndex, patchHigh );
      IntVector diff( regionHi - regionLo );
      cellsInRegion += diff.x() * diff.y() * diff.z();
    }
  }

  return cellsInRegion;
}

//______________________________________________________________________
//
IntVector
UnstructuredLevel::nCellsPatch_max() const       // used by PIDX
{
  return m_numcells_patch_max;
}

//______________________________________________________________________
//
void
UnstructuredLevel::setExtraCells( const IntVector & ec )
{
  m_extra_cells = ec;
}

//______________________________________________________________________
//
UnstructuredGridP
UnstructuredLevel::getGrid() const
{
   return m_grid;
}

//______________________________________________________________________
//
const UnstructuredLevelP &
UnstructuredLevel::getRelativeLevel( int offset ) const
{
  return m_grid->getLevel(m_index + offset);
}

//______________________________________________________________________
//
Uintah::Point
UnstructuredLevel::getNodePosition( const IntVector & v ) const
{
  return m_anchor + m_dcell * v;
}

//______________________________________________________________________
//

Uintah::Point
UnstructuredLevel::getCellPosition( const IntVector & v ) const
{
  return m_anchor + m_dcell * v + m_dcell * 0.5;
}

//______________________________________________________________________
//
IntVector
UnstructuredLevel::getCellIndex( const Point & p ) const
{
  Vector v((p - m_anchor) / m_dcell);
  return IntVector(RoundDown(v.x()), RoundDown(v.y()), RoundDown(v.z()));
}

//______________________________________________________________________
//
Uintah::Point
UnstructuredLevel::positionToIndex( const Point & p ) const
{
  return Point((p-m_anchor)/m_dcell);
}

//______________________________________________________________________
//
void UnstructuredLevel::selectPatches( const IntVector  & low
                         , const IntVector  & high
                         ,       selectType & neighbors
                         ,       bool         withExtraCells /* =false */
                         ,       bool         cache_patches  /* =false */
                         ) const
{
  if (cache_patches) {
    // look it up in the cache first
    patch_cache_mutex.lock();
    {
      select_cache::const_iterator iter = m_select_cache.find(std::make_pair(low, high));

      if (iter != m_select_cache.end()) {
        const std::vector<const UnstructuredPatch*>& cache = iter->second;
        for (unsigned i = 0; i < cache.size(); i++) {
          neighbors.push_back(cache[i]);
        }
        patch_cache_mutex.unlock();
        return;
      }
      ASSERT(neighbors.size() == 0);
    }
    patch_cache_mutex.unlock();
  }


  // Commented out to get to compile  m_bvh->query(low, high, neighbors, withExtraCells);
  std::sort(neighbors.begin(), neighbors.end(), UnstructuredPatch::Compare());


#ifdef CHECK_SELECT
  // Double-check the more advanced selection algorithms against the slow (exhaustive) one.
  std::vector<const UnstructuredPatch*> tneighbors;
  for(const_patch_iterator iter=m_virtual_and_real_patches.begin(); iter != m_virtual_and_real_patches.end(); iter++) {
    const UnstructuredPatch* patch = *iter;

    IntVector l=Max(patch->getCellLowIndex(), low);
    IntVector u=Min(patch->getCellHighIndex(), high);

    if(u.x() > l.x() && u.y() > l.y() && u.z() > l.z()) {
      tneighbors.push_back(*iter);
    }
  }

  ASSERTEQ( neighbors.size(), tneighbors.size() );

  std::sort(tneighbors.begin(), tneighbors.end(), UnstructuredPatch::Compare());
  for( int i = 0; i < (int)neighbors.size(); i++ ) {
    ASSERT(neighbors[i] == tneighbors[i]);
  }
#endif


  if (cache_patches) {
    patch_cache_mutex.lock();
    {
      // put it in the cache - start at orig_size in case there was something in
      // neighbors before this query
      std::vector<const UnstructuredPatch*>& cache = m_select_cache[std::make_pair(low, high)];
      cache.reserve(6);  // don't reserve too much to save memory, not too little to avoid too much reallocation
      for (unsigned int i = 0; i < neighbors.size(); i++) {
        cache.push_back(neighbors[i]);
      }
    }
    patch_cache_mutex.unlock();
  }
}

//______________________________________________________________________
//
bool UnstructuredLevel::containsPointIncludingExtraCells( const Point & p ) const
{
  bool includeExtraCells = true;
  return getPatchFromPoint(p, includeExtraCells) != 0;
}

//______________________________________________________________________
//
bool UnstructuredLevel::containsPoint( const Point & p ) const
{
  bool includeExtraCells = false;
  const UnstructuredPatch* patch = getPatchFromPoint(p, includeExtraCells);
  return patch != 0;
}

//______________________________________________________________________
//
bool UnstructuredLevel::containsCell( const IntVector & idx ) const
{
  bool includeExtraCells = false;
  const UnstructuredPatch* patch = getPatchFromIndex(idx, includeExtraCells);
  return patch != 0;
}

/*______________________________________________________________________
  This method determines if a level is nonCubic or if there are any missing patches.
  Algorithm:
     1) The total volume of the patches must equal the volume of the level.
     The volume of the level is defined by the bounding box.
______________________________________________________________________*/
void
UnstructuredLevel::setIsNonCubicLevel()
{
  double patchesVol = 0.0;
  
  // loop over all patches and sum the patch's volume
  for (int p = 0; p < (int)m_real_patches.size(); p++) {
    const UnstructuredPatch* patch = m_real_patches[p];

    Box box = patch->getBox();
    Vector sides( (box.upper().x() - box.lower().x() ),
                  (box.upper().y() - box.lower().y() ),
                  (box.upper().z() - box.lower().z() ) );

    double volume = sides.x() * sides.y() * sides.z();

    patchesVol += volume;
  }

  // compute the level's volume from the bounding box
  Point loPt = m_int_spatial_range.min();
  Point hiPt = m_int_spatial_range.max();

  Vector levelSides( ( hiPt.x() - loPt.x() ),
                     ( hiPt.y() - loPt.y() ),
                     ( hiPt.z() - loPt.z() ) );

  double levelVol = levelSides.x() * levelSides.y() * levelSides.z();
  
  m_isNonCubicDomain = false;
  double fuzz = 100 * DBL_EPSILON;
  if ( std::fabs( patchesVol - levelVol ) > fuzz ) {
    m_isNonCubicDomain = true;
  }
}

//______________________________________________________________________
//  Loop through all patches on the level and if they overlap with each other then store that info.
//  You need this information when a level has a patch distribution with inside corners
void UnstructuredLevel::setOverlappingPatches()
{ 

  if ( m_isNonCubicDomain == false ){     //  for cubic domains just return
    return;
  }

  for (unsigned i = 0; i < m_real_patches.size(); i++) {
    const UnstructuredPatch* patch = m_real_patches[i];
    Box b1 = patch->getExtraBox();
    IntVector lowEC   = patch->getExtraCellLowIndex();
    IntVector highEC  = patch->getExtraCellHighIndex();
    
    bool includeExtraCells  = true;
    UnstructuredPatch::selectType neighborPatches;
    selectPatches(lowEC, highEC, neighborPatches, includeExtraCells);
    
    for ( unsigned int j = 0; j < neighborPatches.size(); j++) {
      const UnstructuredPatch* neighborPatch = neighborPatches[j];
      
      if ( patch != neighborPatch){
        
        Box b2 = neighborPatch->getExtraBox();

        //  Are the patches overlapping?
        if ( b1.overlaps(b2) ) {
        
          IntVector nLowEC  = neighborPatch->getExtraCellLowIndex();
          IntVector nHighEC = neighborPatch->getExtraCellHighIndex();
          
          // find intersection of the patches
          IntVector intersectLow  = Max( lowEC,  nLowEC );
          IntVector intersectHigh = Min( highEC, nHighEC );
          
          // create overlap
          overlap newOverLap;
          std::pair<int,int> patchIds  = std::make_pair( patch->getID(), neighborPatch->getID() );
          newOverLap.patchIDs  = patchIds;
          newOverLap.lowIndex  = intersectLow;
          newOverLap.highIndex = intersectHigh;

          // only add unique values to the map.
          auto result = m_overLapPatches.find( patchIds );
          if ( result == m_overLapPatches.end() ) {
            m_overLapPatches[patchIds] = newOverLap;
          }
        }
      }
    }
  }

  // debugging 
#if 0
  for (auto itr=m_overLapPatches.begin(); itr!=m_overLapPatches.end(); ++itr) {
    pair<int,int> patchIDs = itr->first;
    overlap me = itr->second;
    std::cout <<  " overlapping patches, UnstructuredLevel: " << getIndex() << " patches: ("  << patchIDs.first << ", " << patchIDs.second <<"), low: " << me.lowIndex << " high: " << me.highIndex << std:: endl;
  }
#endif
}

//______________________________________________________________________
//  This method returns the min/max number of overlapping patch cells that are within a specified
//  region.  Patches will overlap when the domain is non-cubic
std::pair<int,int>
UnstructuredLevel::getOverlapCellsInRegion( const selectType & patches,
                                const IntVector  & regionLow, 
                                const IntVector  & regionHigh) const
{
  // cubic domains never have overlapping patches, just return
  if ( m_isNonCubicDomain == false ){
    return std::make_pair( -9, -9);
  }

  int minOverlapCells   = INT_MAX;
  int totalOverlapCells = 0;
  
  // loop over patches in this region
  for (int i = 0; i < (int) patches.size(); i++) {
    for (int j = i+1; j < (int) patches.size(); j++) {    //  the i+1 prevents examining the transposed key pairs, i.e. 8,9 and 9,8

      int Id         = patches[i]->getID();
      int neighborId = patches[j]->getID();
      std::pair<int,int> patchIds = std::make_pair( Id, neighborId );

      auto result = m_overLapPatches.find( patchIds );
      overlap ol  = result->second;

      // does the overlapping patches intersect with the region extents?
      if ( doesIntersect( ol.lowIndex, ol.highIndex, regionLow, regionHigh ) ){
        IntVector intrsctLow   = Uintah::Max( ol.lowIndex,  regionLow );
        IntVector intrsctHigh  = Uintah::Min( ol.highIndex, regionHigh );

        IntVector diff    = IntVector ( intrsctHigh - intrsctLow );
        int nOverlapCells = std::abs( diff.x() * diff.y() * diff.z() );
       
        minOverlapCells    = std::min( minOverlapCells, nOverlapCells );
        totalOverlapCells += nOverlapCells;
        
#if 0   // debugging
        std::cout << "  getOverlapCellsInRegion  patches: " << ol.patchIDs.first << ", " << ol.patchIDs.second
                  << "\n   region:      " << regionLow   << ",              " << regionHigh        
                  << "\n   ol.low:      " << ol.lowIndex << " ol.high:      " << ol.highIndex  << " nCells: " << ol.nCells
                  << "\n   intrsct.low: " << ol.lowIndex << " intrsct.high: " << ol.highIndex 
                  << " overlapCells: " << nOverlapCells  << " minOverlapCells: " << minOverlapCells << " totalOverlapCells: " << totalOverlapCells << std::endl;
#endif
      }
    }
  }
  std::pair<int,int> overLapCells_minMax = std::make_pair(minOverlapCells,totalOverlapCells);
  return overLapCells_minMax; 
}


//______________________________________________________________________
//
void
UnstructuredLevel::finalizeLevel()
{
  m_each_patch = scinew UnstructuredPatchSet();
  m_each_patch->addReference();

  // The compute set requires an array const Patch*, we must copy d_realPatches
  std::vector<const UnstructuredPatch*> tmp_patches(m_real_patches.size());
  for (int i = 0; i < (int)m_real_patches.size(); i++) {
    tmp_patches[i] = m_real_patches[i];
  }

  m_each_patch->addEach(tmp_patches);

  m_all_patches = scinew UnstructuredPatchSet();
  m_all_patches->addReference();
  m_all_patches->addAll(tmp_patches);

  m_all_patches->sortSubsets();
  std::sort(m_real_patches.begin(), m_real_patches.end(), UnstructuredPatch::Compare());

  // determines and sets the boundary conditions for the patches
  setBCTypes();

  // finalize the patches - Currently, finalizePatch() does nothing... empty method - APH 09/10/15
  for (patch_iterator iter = m_virtual_and_real_patches.begin(); iter != m_virtual_and_real_patches.end(); iter++) {
    (*iter)->finalizePatch();
  }

  // compute the number of cells in the level
  m_total_cells = 0;
  for (int i = 0; i < (int)m_real_patches.size(); i++) {
    m_total_cells += m_real_patches[i]->getNumExtraCells();
  }

  // compute the max number of cells over all patches  Needed by PIDX
  m_numcells_patch_max = IntVector(0, 0, 0);
  int nCells = 0;
  for (int i = 0; i < (int)m_real_patches.size(); i++) {

    if (m_real_patches[i]->getNumExtraCells() > nCells) {
      IntVector lo = m_real_patches[i]->getExtraCellLowIndex();
      IntVector hi = m_real_patches[i]->getExtraCellHighIndex();
      m_numcells_patch_max = hi - lo;
    }
  }

  // compute and store the spatial ranges now that BCTypes are set
  for (int i = 0; i < (int)m_real_patches.size(); i++) {
    UnstructuredPatch* r = m_real_patches[i];
    m_spatial_range.extend(r->getExtraBox().lower());
    m_spatial_range.extend(r->getExtraBox().upper());
  }
  
  // determine if this level is cubic
  setIsNonCubicLevel();
  
  // Loop through all patches and find the patches that overlap.  Needed
  // when patches layouts have inside corners.
  setOverlappingPatches();
  
}

//______________________________________________________________________
//
void
UnstructuredLevel::finalizeLevel( bool periodicX, bool periodicY, bool periodicZ )
{
  // set each_patch and all_patches before creating virtual patches
  m_each_patch = scinew UnstructuredPatchSet();
  m_each_patch->addReference();

  // The compute set requires an array const Patch*, we must copy d_realPatches
  std::vector<const UnstructuredPatch*> tmp_patches(m_real_patches.size());

  for (int i = 0; i < (int)m_real_patches.size(); i++) {
    tmp_patches[i] = m_real_patches[i];
  }

  m_each_patch->addEach(tmp_patches);

  m_all_patches = scinew UnstructuredPatchSet();
  m_all_patches->addReference();
  m_all_patches->addAll(tmp_patches);

  BBox bbox;

  if (m_index > 0) {
    m_grid->getLevel(0)->getInteriorSpatialRange(bbox);
  } else {
    getInteriorSpatialRange(bbox);
  }

  Box domain(bbox.min(), bbox.max());
  Vector vextent = positionToIndex(bbox.max()) - positionToIndex(bbox.min());
  IntVector extent((int)rint(vextent.x()), (int)rint(vextent.y()), (int)rint(vextent.z()));

  m_periodic_boundaries = IntVector( periodicX ? 1 : 0, periodicY ? 1 : 0, periodicZ ? 1 : 0 );
  IntVector periodicBoundaryRange = m_periodic_boundaries * extent;

  int x, y, z;
  for (int i = 0; i < (int)tmp_patches.size(); i++) {

    for (x = -m_periodic_boundaries.x(); x <= m_periodic_boundaries.x(); x++) {
      for (y = -m_periodic_boundaries.y(); y <= m_periodic_boundaries.y(); y++) {
        for (z = -m_periodic_boundaries.z(); z <= m_periodic_boundaries.z(); z++) {

          IntVector offset = IntVector(x, y, z) * periodicBoundaryRange;
          if (offset == IntVector(0, 0, 0)) {
            continue;
          }

          Box box = getBox(tmp_patches[i]->getExtraCellLowIndex() + offset - IntVector(1, 1, 1),
                           tmp_patches[i]->getExtraCellHighIndex() + offset + IntVector(1, 1, 1));

          if (box.overlaps(domain)) {
            UnstructuredPatch* newPatch = tmp_patches[i]->createVirtualPatch(offset);
            m_virtual_and_real_patches.push_back(newPatch);
          }
        }
      }
    }
  }

  m_all_patches->sortSubsets();
  std::sort(m_real_patches.begin(), m_real_patches.end(), UnstructuredPatch::Compare());
  std::sort(m_virtual_and_real_patches.begin(), m_virtual_and_real_patches.end(), UnstructuredPatch::Compare());

  setBCTypes();

  //finalize the patches
  for (patch_iterator iter = m_virtual_and_real_patches.begin(); iter != m_virtual_and_real_patches.end(); iter++) {
    (*iter)->finalizePatch();
  }

  //compute the number of cells in the level
  m_total_cells = 0;
  for (int i = 0; i < (int)m_real_patches.size(); i++) {
    m_total_cells += m_real_patches[i]->getNumExtraCells();
  }

  //compute the max number of cells over all patches  Needed by PIDX
  m_numcells_patch_max = IntVector(0, 0, 0);
  int nCells = 0;

  for (int i = 0; i < (int)m_real_patches.size(); i++) {
    if (m_real_patches[i]->getNumExtraCells() > nCells) {
      IntVector lo = m_real_patches[i]->getExtraCellLowIndex();
      IntVector hi = m_real_patches[i]->getExtraCellHighIndex();

      m_numcells_patch_max = hi - lo;
    }
  }

  //compute and store the spatial ranges now that BCTypes are set
  for (int i = 0; i < (int)m_real_patches.size(); i++) {
    UnstructuredPatch* r = m_real_patches[i];

    m_spatial_range.extend(r->getExtraBox().lower());
    m_spatial_range.extend(r->getExtraBox().upper());
  }
  
  // determine if this level is cubic
  setIsNonCubicLevel();
    
  // Loop through all patches and find the patches that overlap.  Needed
  // when patch layouts have inside corners.
  setOverlappingPatches();
}

//______________________________________________________________________
//
void
UnstructuredLevel::setBCTypes()
{
  Timers::Simple timer;
  timer.start();

  const int nTimes = 3;
  double rtimes[ nTimes ] = { 0 };

  if (m_bvh != nullptr) {
    delete m_bvh;
  }

  // Comment out so it will compile  m_bvh = scinew PatchBVH(m_virtual_and_real_patches);

  rtimes[0] += timer().seconds();
  timer.reset( true );

  patch_iterator iter;

  ProcessorGroup *myworld = nullptr;
  int numProcs = 1;
  int rank = 0;

  if (Parallel::isInitialized()) {
    // only sus uses Parallel, but anybody else who uses DataArchive
    // to read data does not
    myworld = Parallel::getRootProcessorGroup();
    numProcs = myworld->nRanks();
    rank = myworld->myRank();
  }

  std::vector<int> displacements(numProcs, 0);
  std::vector<int> recvcounts(numProcs, 0);

  //create recvcounts and displacement arrays
  int div = m_virtual_and_real_patches.size() / numProcs;
  int mod = m_virtual_and_real_patches.size() % numProcs;

  for (int p = 0; p < numProcs; p++) {
    if (p < mod) {
      recvcounts[p] = div + 1;
    } else {
      recvcounts[p] = div;
    }
  }

  displacements[0] = 0;
  for (int p = 1; p < numProcs; p++) {
    displacements[p] = displacements[p - 1] + recvcounts[p - 1];
  }

  std::vector<unsigned int> bctypes(m_virtual_and_real_patches.size());
  std::vector<unsigned int> mybctypes(recvcounts[rank]);

  int idx;

  patch_iterator startpatch = m_virtual_and_real_patches.begin() + displacements[rank];
  patch_iterator endpatch = startpatch + recvcounts[rank];

  // for each of my patches
  for (iter = startpatch, idx = 0; iter != endpatch; iter++, idx++) {
    UnstructuredPatch* patch = *iter;
    // See if there are any neighbors on the 6 faces
    int bitfield = 0;

    for (UnstructuredPatch::FaceType face = UnstructuredPatch::startFace; face <= UnstructuredPatch::endFace; face = UnstructuredPatch::nextFace(face)) {
      bitfield <<= 2;
      IntVector l, h;

      patch->getFace(face, IntVector(0, 0, 0), IntVector(1, 1, 1), l, h);

      UnstructuredPatch::selectType neighbors;
      selectPatches(l, h, neighbors);

      if (neighbors.size() == 0) {
        if (m_index != 0) {
          // See if there are any patches on the coarse level at that face
          IntVector fineLow, fineHigh;
          patch->getFace(face, IntVector(0, 0, 0), m_refinement_ratio, fineLow, fineHigh);

          IntVector coarseLow = mapCellToCoarser(fineLow);
          IntVector coarseHigh = mapCellToCoarser(fineHigh);
          const UnstructuredLevelP& coarseLevel = getCoarserLevel();

#if 0
          // add 1 to the corresponding index on the plus edges
          // because the upper corners are sort of one cell off (don't know why)
          if (m_extra_cells.x() != 0 && face == UnstructuredPatch::xplus) {
            coarseLow[0] ++;
            coarseHigh[0]++;
          }
          else if (m_extra_cells.y() != 0 && face == UnstructuredPatch::yplus) {
            coarseLow[1] ++;
            coarseHigh[1] ++;
          }
          else if (m_extra_cells.z() != 0 && face == UnstructuredPatch::zplus) {
            coarseLow[2] ++;
            coarseHigh[2]++;
          }
#endif
          coarseLevel->selectPatches(coarseLow, coarseHigh, neighbors);

          if (neighbors.size() == 0) {
            bitfield |= UnstructuredPatch::None;
          } else {
            bitfield |= UnstructuredPatch::Coarse;
          }
        } else {
          bitfield |= UnstructuredPatch::None;
        }
      } else {
        bitfield |= UnstructuredPatch::Neighbor;
      }
    }
    mybctypes[idx] = bitfield;
  }

  if (numProcs > 1) {
    // allgather bctypes
    if (mybctypes.size() == 0) {
      Uintah::MPI::Allgatherv(0, 0, MPI_UNSIGNED, &bctypes[0], &recvcounts[0], &displacements[0], MPI_UNSIGNED, myworld->getComm());
    } else {
      Uintah::MPI::Allgatherv(&mybctypes[0], mybctypes.size(), MPI_UNSIGNED, &bctypes[0], &recvcounts[0], &displacements[0],
                              MPI_UNSIGNED, myworld->getComm());
    }
  } else {
    bctypes.swap(mybctypes);
  }

  rtimes[1] += timer().seconds();
  timer.reset( true );

  int i;
  // loop through patches
  for (iter = m_virtual_and_real_patches.begin(), i = 0, idx = 0; iter != m_virtual_and_real_patches.end(); iter++, i++) {
    UnstructuredPatch *patch = *iter;

    if (patch->isVirtual()) {
      patch->setLevelIndex(-1);
    } else {
      patch->setLevelIndex(idx++);
    }

    int bitfield = bctypes[i];
    int mask = 3;

    // loop through faces
    for (int j = 5; j >= 0; j--) {

      int bc_type = bitfield & mask;

      if (rank == 0) {
        switch (bc_type) {
          case UnstructuredPatch::None :
            bcout << "  Setting UnstructuredPatch " << patch->getID() << " face " << j << " to None\n";
            break;
          case UnstructuredPatch::Coarse :
            bcout << "  Setting UnstructuredPatch " << patch->getID() << " face " << j << " to Coarse\n";
            break;
          case UnstructuredPatch::Neighbor :
            bcout << "  Setting UnstructuredPatch " << patch->getID() << " face " << j << " to Neighbor\n";
            break;
        }
      }
      patch->setBCType(UnstructuredPatch::FaceType(j), UnstructuredPatch::BCType(bc_type));
      bitfield >>= 2;
    }
  }

  //__________________________________
  //  bullet proofing
  for( int dir = 0; dir < 3; dir++ ) {
    if( m_periodic_boundaries[dir] == 1 && m_extra_cells[dir] != 0 ) {
      std::ostringstream warn;
      warn << "\n \n INPUT FILE ERROR: \n You've specified a periodic boundary condition on a face with extra cells specified\n"
           << " Please set the extra cells on that face to 0";
      throw ProblemSetupException( warn.str(), __FILE__, __LINE__ );
    }
  }

  m_finalized = true;

  rtimes[2] += timer().seconds();
  timer.reset( true );

  if (rgtimes.active()) {
    double avg[ nTimes ] = { 0 };
    Uintah::MPI::Reduce(rtimes, avg, nTimes, MPI_DOUBLE, MPI_SUM, 0, myworld->getComm());

    if (myworld->myRank() == 0) {

      std::cout << "SetBCType Avg Times: ";
      for (int i = 0; i < nTimes; i++) {
        avg[i] /= myworld->nRanks();
        std::cout << avg[i] << " ";
      }
      std::cout << std::endl;
    }

    double max[nTimes] = { 0 };
    Uintah::MPI::Reduce(rtimes, max, nTimes, MPI_DOUBLE, MPI_MAX, 0, myworld->getComm());

    if (myworld->myRank() == 0) {
      std::cout << "SetBCType Max Times: ";
      for (int i = 0; i < nTimes; i++) {
        std::cout << max[i] << " ";
      }
      std::cout << std::endl;
    }
  }

  // recreate BVH with extracells
  if (m_bvh != nullptr) {
    delete m_bvh;
  }
  // Comment out so it will compile m_bvh = scinew PatchBVH(m_virtual_and_real_patches);
}

//______________________________________________________________________
//
void
UnstructuredLevel::assignBCS( const ProblemSpecP & grid_ps, UnstructuredLoadBalancer * lb )
{
  ProblemSpecP bc_ps = grid_ps->findBlock("BoundaryConditions");
  if( bc_ps == nullptr ) {
    if( Parallel::getMPIRank() == 0 ) {
      static ProgressiveWarning warn( "No BoundaryConditions specified", -1 );
      warn.invoke();
    }
    return;
  }

  UnstructuredBoundCondReader reader;

  reader.read(bc_ps, grid_ps, this);

  for (patch_iterator iter = m_virtual_and_real_patches.begin(); iter != m_virtual_and_real_patches.end(); iter++) {
    UnstructuredPatch* patch = *iter;

    // If we have a lb, then only apply bcs this processors patches.
    if (lb == 0 || lb->getPatchwiseProcessorAssignment(patch) == Parallel::getMPIRank()) {

      patch->initializeBoundaryConditions();

      for (UnstructuredPatch::FaceType face_side = UnstructuredPatch::startFace; face_side <= UnstructuredPatch::endFace; face_side = UnstructuredPatch::nextFace(face_side)) {
        if (patch->getBCType(face_side) == UnstructuredPatch::None) {
          patch->setArrayBCValues(face_side, &(reader.d_BCReaderData[face_side]));
        }
        patch->setInteriorBndArrayBCValues(face_side, &(reader.d_interiorBndBCReaderData[face_side]));
      }  // end of face iterator
    }
  }  //end of patch iterator
}

//______________________________________________________________________
//
Box
UnstructuredLevel::getBox( const IntVector & l, const IntVector & h ) const
{
  return Box(getNodePosition(l), getNodePosition(h));
}

//______________________________________________________________________
//
const UnstructuredPatchSet*
UnstructuredLevel::eachPatch() const
{
  ASSERT(m_each_patch != nullptr);
  return m_each_patch;
}

//______________________________________________________________________
//
const UnstructuredPatchSet*
UnstructuredLevel::allPatches() const
{
  ASSERT(m_all_patches != nullptr);
  return m_all_patches;
}

//______________________________________________________________________
//
const UnstructuredPatch*
UnstructuredLevel::selectPatchForCellIndex( const IntVector & idx ) const
{
  selectType pv;
  IntVector i(1, 1, 1);
  selectPatches(idx - i, idx + i, pv, false, false);

  if (pv.size() == 0) {
    return nullptr;
  } else {
    selectType::iterator it;

    for (it = pv.begin(); it != pv.end(); it++) {
      if ((*it)->containsCell(idx)) {
        return *it;
      }
    }
  }
  return nullptr;
}

//______________________________________________________________________
//
const UnstructuredPatch*
UnstructuredLevel::selectPatchForNodeIndex( const IntVector & idx ) const
{
  selectType pv;
  IntVector i(1, 1, 1);

  selectPatches(idx - i, idx + i, pv, false, false);

  if (pv.size() == 0) {
    return nullptr;
  } else {
    selectType::iterator it;
    for (it = pv.begin(); it != pv.end(); it++) {
      if ((*it)->containsNode(idx)) {
        return *it;
      }
    }
  }
  return nullptr;
}

//______________________________________________________________________
//
const UnstructuredLevelP &
UnstructuredLevel::getCoarserLevel() const
{
  return getRelativeLevel( -1 );
}

//______________________________________________________________________
//
const UnstructuredLevelP &
UnstructuredLevel::getFinerLevel() const
{
  return getRelativeLevel( 1 );
}

//______________________________________________________________________
//
bool
UnstructuredLevel::hasCoarserLevel() const
{
  return getIndex() > 0;
}

//______________________________________________________________________
//
bool
UnstructuredLevel::hasFinerLevel() const
{
  return getIndex() < ( m_grid->numLevels() - 1 );
}

//______________________________________________________________________
//
IntVector
UnstructuredLevel::mapCellToCoarser( const IntVector & idx, int level_offset ) const
{
  IntVector refinementRatio = m_refinement_ratio;
  while (--level_offset) {
    refinementRatio = refinementRatio * m_grid->getLevel(m_index - level_offset)->m_refinement_ratio;
  }
  IntVector ratio = idx / refinementRatio;

  // If the fine cell index is negative you must add an offset to get the right coarse cell. -Todd
  IntVector offset(0, 0, 0);
  if (idx.x() < 0 && refinementRatio.x() > 1) {
    offset.x((int)fmod((double)idx.x(), (double)refinementRatio.x()));
  }

  if (idx.y() < 0 && refinementRatio.y() > 1) {
    offset.y((int)fmod((double)idx.y(), (double)refinementRatio.y()));
  }

  if (idx.z() < 0 && refinementRatio.z() > 1) {
    offset.z((int)fmod((double)idx.z(), (double)refinementRatio.z()));
  }
  return ratio + offset;
}

//______________________________________________________________________
//
IntVector
UnstructuredLevel::mapCellToFiner( const IntVector & idx ) const
{
  IntVector r_ratio = m_grid->getLevel(m_index + 1)->m_refinement_ratio;
  IntVector fineCell = idx * r_ratio;

  IntVector offset(0, 0, 0);
  if (idx.x() < 0 && r_ratio.x() > 1) {
    offset.x(1);
  }

  if (idx.y() < 0 && r_ratio.y() > 1) {  // If the coarse cell index is negative
    offset.y(1);                      // you must add an offset to get the right
  }                                   // fine cell. -Todd

  if (idx.z() < 0 && r_ratio.z() > 1) {
    offset.z(1);
  }
  return fineCell + offset;
}

//______________________________________________________________________
//Provides the (x-,y-,z-) corner of a fine cell given a coarser coordinate
//If any of the coordinates are negative, assume the fine cell coordiantes
//went too far into the negative and adjust forward by 1
//This adjusting approach means that for L-shaped domains the results are
//not always consistent with what is expected.
//(Note: Does this adjustment mean this only works in a 2:1 refinement ratio
//and only on cubic domains? -- Brad P)
IntVector
UnstructuredLevel::mapCellToFinest( const IntVector & idx ) const
{

  IntVector r_ratio  = IntVector(1,1,1);
  for (int i=m_index; i< m_grid->numLevels()-1; i++){
    r_ratio  = r_ratio*m_grid->getLevel(i+1)->getRefinementRatio();
  }

  IntVector fineCell = idx * r_ratio;
  IntVector offset(0,0,0);
  if (idx.x()< 0 && r_ratio.x() > 1){
    offset.x(1);
  }

  if (idx.y()< 0 && r_ratio.y() > 1){ // If the coarse cell index is negative
    offset.y(1);                      // you must add an offset to get the right
  }                                   // fine cell. -Todd

  if (idx.z()< 0 && r_ratio.z() > 1){
    offset.z(1);
  }

  return fineCell + offset;
}

//______________________________________________________________________
//Provides the x-,y-,z- corner of a fine cell given a coarser coordinate.
//This does not attempt to adjust the cell in the + direction if it goes
//negative. It is left up to the caller of this method to determine if
//those coordinates are too far past any level boundary.
IntVector
UnstructuredLevel::mapCellToFinestNoAdjustments( const IntVector & idx ) const
{

      IntVector r_ratio  = IntVector(1,1,1);
      for (int i=m_index; i< m_grid->numLevels()-1; i++){
       r_ratio  = r_ratio*m_grid->getLevel(i+1)->getRefinementRatio();
      }
      return idx * r_ratio;
}

//______________________________________________________________________
//
// mapNodeToCoarser:
// Example: 1D grid with refinement ratio = 4
//  Coarse Node index: 10                  11
//                     |                   |
//                 ----*----*----*----*----*-----
//                     |                   |
//  Fine Node Index    40   41   42   43   44
//
//  What is returned   10   10   10   10   11

IntVector
UnstructuredLevel::mapNodeToCoarser( const IntVector & idx ) const
{
  return ( idx + m_refinement_ratio - IntVector(1,1,1) ) / m_refinement_ratio;
}

//______________________________________________________________________
//
// mapNodeToFiner:
// Example: 1D grid with refinement ratio = 4
//  Coarse Node index: 10                  11
//                     |                   |
//                 ----*----*----*----*----*-----
//                     |                   |
//  Fine Node Index    40   41   42   43   44
//
//  What is returned   40                  44

IntVector
UnstructuredLevel::mapNodeToFiner( const IntVector & idx ) const
{
  return idx * m_grid->getLevel(m_index + 1)->m_refinement_ratio;
}

//______________________________________________________________________
//
int
UnstructuredLevel::getRefinementRatioMaxDim() const
{
  return Max( Max(m_refinement_ratio.x(), m_refinement_ratio.y()), m_refinement_ratio.z() );
}



//______________________________________________________________________
//
namespace Uintah {

const UnstructuredLevel* getLevel( const UnstructuredPatchSubset * subset )
{
  ASSERT(subset->size() > 0);
  const UnstructuredLevel* level = subset->get(0)->getLevel();
#if SCI_ASSERTION_LEVEL>0
  for (int i = 1; i < subset->size(); i++) {
    ASSERT(level == subset->get(i)->getLevel());
  }
#endif
  return level;
}
//______________________________________________________________________
//
const UnstructuredLevelP& getLevelP( const UnstructuredPatchSubset * subset )
{
  ASSERT(subset->size() > 0);
  const UnstructuredLevelP& level = subset->get(0)->getLevelP();
#if SCI_ASSERTION_LEVEL>0
  for (int i = 1; i < subset->size(); i++) {
    ASSERT(level == subset->get(i)->getLevelP());
  }
#endif
  return level;
}

//______________________________________________________________________
//
const UnstructuredLevel* getLevel( const UnstructuredPatchSet * set )
{
  ASSERT(set->size() > 0);
  return getLevel(set->getSubset(0));
}

//______________________________________________________________________
// We may need to put coutLocks around this?
std::ostream& operator<<( std::ostream & out, const UnstructuredLevel & level )
{
  IntVector lo, hi;
  level.findCellIndexRange(lo, hi);

  out << "(UnstructuredLevel " << level.getIndex() << ", numPatches: " << level.numPatches()
      << ", cellIndexRange: " << lo << ", " << hi << ", " << *(level.allPatches()) << ")";

  return out;
}

} // end namespace Uintah

