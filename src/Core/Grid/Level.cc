/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <Core/Grid/Level.h>

#include <Core/Exceptions/InvalidGrid.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/BBox.h>
#include <Core/Grid/BoundaryConditions/BoundCondReader.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/PatchBVH/PatchBVH.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/OS/ProcessInfo.h> // For Memory Check
#include <Core/Parallel/CrowdMonitor.hpp>
#include <Core/Parallel/MasterLock.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DOUT.hpp>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/Handle.h>
#include <Core/Util/ProgressiveWarning.h>
#include <Core/Util/Timers/Timers.hpp>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>

using namespace Uintah;

namespace {

  std::atomic<int32_t> g_ids{0};
  Uintah::MasterLock   g_patch_cache_mutex{};

  Dout g_bc_dbg{   "BCTypes",      "Level", "Level BC debug info"        , false };
  Dout g_rg_times{ "RGTimesLevel", "Level", "Level regridder timing info", false };

}

//______________________________________________________________________
//
Level::Level(       Grid      * grid
            , const Point     & anchor
            , const Vector    & dcell
            , const int         index
            , const IntVector   refinementRatio
            , const int         id /* = -1 */
            )
  : m_grid{ grid }
  , m_anchor{ anchor }
  , m_dcell{ dcell }
  , m_index{ index }
  , m_id{ id }
  , m_refinement_ratio{ refinementRatio }
{
  if( m_id == -1 ) {
    m_id = g_ids.fetch_add( 1, std::memory_order_relaxed );
  }
  else if( m_id >= g_ids ) {
    g_ids.store( m_id + 1, std::memory_order_relaxed );
  }
}

//______________________________________________________________________
//
Level::~Level()
{
  // Delete all of the patches managed by this level
  for (patch_iterator iter = m_virtual_and_real_patches.begin(); iter != m_virtual_and_real_patches.end(); ++iter) {
    delete *iter;
  }

  delete m_bvh;

  if (m_each_patch && m_each_patch->removeReference()) {
    delete m_each_patch;
  }
  if (m_all_patches && m_all_patches->removeReference()) {
    delete m_all_patches;
  }
}

//______________________________________________________________________
//
void Level::setPatchDistributionHint( const IntVector & hint )
{
  if (m_patch_distribution.x() == -1) {
    m_patch_distribution = hint;
  }
  else {
    // Called more than once, we have to punt
    m_patch_distribution = IntVector(-2, -2, 2);
  }
}

//______________________________________________________________________
//
Level::const_patch_iterator Level::patchesBegin() const
{
  return m_real_patches.cbegin();
}

//______________________________________________________________________
//
Level::const_patch_iterator Level::patchesEnd() const
{
  return m_real_patches.cend();
}

//______________________________________________________________________
//
Level::patch_iterator Level::patchesBegin()
{
  return m_real_patches.begin();
}

//______________________________________________________________________
//
Level::patch_iterator Level::patchesEnd()
{
  return m_real_patches.end();
}

//______________________________________________________________________
//
Level::const_patch_iterator Level::allPatchesBegin() const
{
  return m_virtual_and_real_patches.begin();
}

//______________________________________________________________________
//
Level::const_patch_iterator Level::allPatchesEnd() const
{
  return m_virtual_and_real_patches.end();
}

//______________________________________________________________________
//
Patch*
Level::addPatch( const IntVector & lowIndex
               , const IntVector & highIndex
               , const IntVector & inLowIndex
               , const IntVector & inHighIndex
               ,       Grid      * grid
               )
{
  Patch* r = scinew Patch( this, lowIndex, highIndex, inLowIndex, inHighIndex, getIndex() );
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
Patch*
Level::addPatch( const IntVector & lowIndex
               , const IntVector & highIndex
               , const IntVector & inLowIndex
               , const IntVector & inHighIndex
               ,       Grid      * grid
               ,       int         ID
               )
{
  Patch* r = scinew Patch(this, lowIndex, highIndex, inLowIndex, inHighIndex, getIndex(), ID);
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
const Patch*
Level::getPatchFromPoint( const Point & p, const bool includeExtraCells ) const
{
  selectType patch;
  IntVector c = getCellIndex(p);
  // Point is within the bounding box so query the BVH
  m_bvh->query(c, c + IntVector(1, 1, 1), patch, includeExtraCells);

  if (patch.size() == 0) {
    return 0;
  }

  ASSERT(patch.size() == 1);
  return patch[0];
}

//______________________________________________________________________
//
const Patch*
Level::getPatchFromIndex( const IntVector & c, const bool includeExtraCells ) const
{
  selectType patch;

  // Point is within the bounding box so query the BVH.
  m_bvh->query(c, c + IntVector(1, 1, 1), patch, includeExtraCells);

  if (patch.size() == 0) {
    return 0;
  }

  ASSERT(patch.size() == 1);
  return patch[0];
}

//______________________________________________________________________
//
int
Level::numPatches() const
{
  return static_cast<int>(m_real_patches.size());
}

//______________________________________________________________________
//
void
Level::performConsistencyCheck() const
{
  if (!m_finalized) {
    SCI_THROW(InvalidGrid("Consistency check cannot be performed until Level is finalized",__FILE__,__LINE__));
  }

  for (size_t i = 0; i < m_virtual_and_real_patches.size(); ++i) {
    Patch* r = m_virtual_and_real_patches[i];
    r->performConsistencyCheck();
  }

  // This is O(n^2) - we should fix it someday if it ever matters
  //   This checks that patches do not overlap
  for (size_t i = 0; i < m_virtual_and_real_patches.size(); ++i) {
    Patch* r1 = m_virtual_and_real_patches[i];

    for (size_t j = i + 1; j < m_virtual_and_real_patches.size(); j++) {
      Patch* r2 = m_virtual_and_real_patches[j];
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
Level::findNodeIndexRange( IntVector & lowIndex, IntVector & highIndex ) const
{
  Vector l = (m_spatial_range.min() - m_anchor) / m_dcell;
  Vector h = (m_spatial_range.max() - m_anchor) / m_dcell + Vector(1, 1, 1);

  lowIndex  = roundNearest(l);
  highIndex = roundNearest(h);
}

//______________________________________________________________________
//

void
Level::findCellIndexRange( IntVector & lowIndex, IntVector & highIndex ) const
{
  Vector l = (m_spatial_range.min() - m_anchor) / m_dcell;
  Vector h = (m_spatial_range.max() - m_anchor) / m_dcell;

  lowIndex  = roundNearest(l);
  highIndex = roundNearest(h);
}

//______________________________________________________________________
//
void Level::findInteriorCellIndexRange( IntVector & lowIndex, IntVector & highIndex ) const
{
  Vector l = (m_int_spatial_range.min() - m_anchor) / m_dcell;
  Vector h = (m_int_spatial_range.max() - m_anchor) / m_dcell;

  lowIndex  = roundNearest(l);
  highIndex = roundNearest(h);
}

//______________________________________________________________________
//
void Level::findInteriorNodeIndexRange( IntVector & lowIndex, IntVector & highIndex) const
{
  Vector l = (m_int_spatial_range.min() - m_anchor) / m_dcell;
  Vector h = (m_int_spatial_range.max() - m_anchor) / m_dcell + Vector(1, 1, 1);

  lowIndex = roundNearest(l);
  highIndex = roundNearest(h);
}

//______________________________________________________________________
//  Compute the variable extents for this variable type
void
Level::computeVariableExtents( const TypeDescription::Type   type
                               ,     IntVector             & lo
                               ,     IntVector             & hi ) const
{
  IntVector CCLo;
  IntVector CCHi;
  findCellIndexRange( CCLo, CCHi );

  // Fix me: better way to calc this var? Or to use it below?
  IntVector not_periodic( !m_periodic_boundaries[0], !m_periodic_boundaries[1], !m_periodic_boundaries[2] );

  switch( type ) {
    case TypeDescription::CCVariable :
    case TypeDescription::ParticleVariable :
      lo = CCLo;
      hi = CCHi;
      break;
    case TypeDescription::SFCXVariable :
      lo = CCLo;
      hi = CCHi + ( IntVector(1, 0, 0) * not_periodic );
      break;
    case TypeDescription::SFCYVariable :
      lo = CCLo;
      hi = CCHi + ( IntVector(0, 1, 0) * not_periodic );
      break;
    case TypeDescription::SFCZVariable :
      lo = CCLo;
      hi = CCHi + ( IntVector(0, 0, 1) * not_periodic );
      break;
    case TypeDescription::NCVariable :
      // Dav's fix: findInteriorCellIndexRange( lo, hi );
      findNodeIndexRange(lo, hi);
      break;
    default :
      std::string me = TypeDescription::toString( type );
      throw InternalError("  ERROR: Level::computeVariableExtents type description (" + me + ") not supported", __FILE__, __LINE__);
  }
}

//______________________________________________________________________
//
long
Level::totalCells() const
{
  return m_total_cells;
}

//______________________________________________________________________
//
long
Level::getTotalCellsInRegion(const TypeDescription::Type varType,
                             const IntVector& boundaryLayer,
                             const IntVector& lowIndex, 
                             const IntVector& highIndex) const {

  // Not all simulations are cubic.  Some simulations might be L shaped, or T shaped, or + shaped, etc.
  // It is not enough to simply do a high - low to figure out the amount of simulation cells.  We instead
  // need to go all patches and see if they exist in this range.  If so, add up their cells.
  // This process is similar to how d_totalCells is computed in Level::finalizeLevel().

  long cellsInRegion = 0;
  if (m_isNonCubicDomain == false ){
    IntVector diff( highIndex - lowIndex);
    cellsInRegion += diff.x() * diff.y() * diff.z();
    return cellsInRegion;
  }

  Patch::VariableBasis basis = Patch::translateTypeToBasis(varType, false);

  // scjmc - using also virtual patches to take into account periodic boundaries on non cubic levels such as refined amr levels
  for( size_t i = 0; i < m_virtual_and_real_patches.size(); ++i ) {
    IntVector patchLow  =  m_virtual_and_real_patches[i]->getExtraLowIndex(  basis, boundaryLayer );
    IntVector patchHigh =  m_virtual_and_real_patches[i]->getExtraHighIndex( basis, boundaryLayer );

    if (doesIntersect(lowIndex, highIndex, patchLow, patchHigh)) {

      IntVector regionLo = Uintah::Max(lowIndex, patchLow);
      IntVector regionHi = Uintah::Min(highIndex, patchHigh);
      IntVector diff(regionHi - regionLo);
      cellsInRegion += diff.x() * diff.y() * diff.z();
    }
  }

  return cellsInRegion;
}

//______________________________________________________________________
//
IntVector
Level::nCellsPatch_max() const       // used by PIDX
{
  return m_numcells_patch_max;
}

//______________________________________________________________________
//
void
Level::setExtraCells( const IntVector & ec )
{
  m_extra_cells = ec;
}

//______________________________________________________________________
//
GridP
Level::getGrid() const
{
   return m_grid;
}

//______________________________________________________________________
//
const LevelP &
Level::getRelativeLevel( int offset ) const
{
  return m_grid->getLevel(m_index + offset);
}

//______________________________________________________________________
//
Uintah::Point
Level::getNodePosition( const IntVector & v ) const
{
  return m_anchor + m_dcell * v;
}

//______________________________________________________________________
//

Uintah::Point
Level::getCellPosition( const IntVector & v ) const
{
  return m_anchor + m_dcell * v + m_dcell * 0.5;
}

//______________________________________________________________________
//
IntVector
Level::getCellIndex( const Point & p ) const
{
  Vector v((p - m_anchor) / m_dcell);
  return IntVector(RoundDown(v.x()), RoundDown(v.y()), RoundDown(v.z()));
}

//______________________________________________________________________
//
Uintah::Point
Level::positionToIndex( const Point & p ) const
{
  return Point((p-m_anchor)/m_dcell);
}

//______________________________________________________________________
//
void Level::selectPatches( const IntVector  & low
                         , const IntVector  & high
                         ,       selectType & neighbors
                         ,       bool         withExtraCells /* =false */
                         ,       bool         cache_patches  /* =false */
                         ) const
{
  if (cache_patches) {
    std::lock_guard<Uintah::MasterLock> cache_lock(g_patch_cache_mutex);

    // look it up in the cache first
    select_cache::const_iterator iter = m_select_cache.find(std::make_pair(low, high));
    if (iter != m_select_cache.cend()) {
      const std::vector<const Patch*>& cache = iter->second;
      for (size_t i = 0; i < cache.size(); ++i) {
        neighbors.push_back(cache[i]);
      }
      return;
    }
    ASSERT(neighbors.size() == 0);
  }


  m_bvh->query(low, high, neighbors, withExtraCells);

  std::sort(neighbors.begin(), neighbors.end(), Patch::Compare());


#ifdef CHECK_SELECT
  // Double-check the more advanced selection algorithms against the slow (exhaustive) one.
  std::vector<const Patch*> check_select_neighbors;
  for (const_patch_iterator iter = m_virtual_and_real_patches.begin(); iter != m_virtual_and_real_patches.end(); ++iter) {
    const Patch* patch = *iter;

    IntVector l = Max(patch->getCellLowIndex(), low);
    IntVector u = Min(patch->getCellHighIndex(), high);

    if (u.x() > l.x() && u.y() > l.y() && u.z() > l.z()) {
      check_select_neighbors.push_back(*iter);
    }
  }

  ASSERTEQ(neighbors.size(), check_select_neighbors.size());

  std::sort(check_select_neighbors.begin(), check_select_neighbors.end(), Patch::Compare());
  for (size_t i = 0; i < neighbors.size(); ++i) {
    ASSERT(neighbors[i] == check_select_neighbors[i]);
  }
#endif


  if (cache_patches) {
    std::lock_guard<Uintah::MasterLock> cache_lock(g_patch_cache_mutex);

    // put it in the cache - start at orig_size in case there was something in neighbors before this query
    std::vector<const Patch*>& cache = m_select_cache[std::make_pair(low, high)];
    cache.reserve(6);  // don't reserve too much to save memory, not too little to avoid too much reallocation
    for (size_t i = 0; i < neighbors.size(); ++i) {
      cache.push_back(neighbors[i]);
    }
  }
}

//______________________________________________________________________
//
bool Level::containsPointIncludingExtraCells( const Point & p ) const
{
  bool includeExtraCells = true;
  return getPatchFromPoint(p, includeExtraCells) != nullptr;
}

//______________________________________________________________________
//
bool Level::containsPoint( const Point & p ) const
{
  bool includeExtraCells = false;
  const Patch* patch = getPatchFromPoint(p, includeExtraCells);
  return patch != nullptr;
}

//______________________________________________________________________
//
bool Level::containsCell( const IntVector & idx ) const
{
  bool includeExtraCells = false;
  const Patch* patch = getPatchFromIndex(idx, includeExtraCells);
  return patch != nullptr;
}

/*______________________________________________________________________
  This method determines if a level is nonCubic or if there are any missing patches.
  Algorithm:
     1) The total volume of the patches must equal the volume of the level.
     The volume of the level is defined by the bounding box.
______________________________________________________________________*/
void
Level::setIsNonCubicLevel()
{
  double patchesVol = 0.0;
  
  // loop over all patches and sum the patch's volume
  for (size_t p = 0; p < m_real_patches.size(); ++p) {
    const Patch* patch = m_real_patches[p];

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
  double fuzz = 0.5 * cellVolume(); // 100 * DBL_EPSILON;
  if ( std::fabs( patchesVol - levelVol ) > fuzz ) {
    m_isNonCubicDomain = true;
  }
}

//______________________________________________________________________
//  Loop through all patches on the level and if they overlap with each other then store that info.
//  You need this information when a level has a patch distribution with inside corners
void Level::setOverlappingPatches()
{ 

  if ( m_isNonCubicDomain == false ){     //  for cubic domains just return
    return;
  }

  for (size_t i = 0; i < m_real_patches.size(); ++i) {
    const Patch* patch = m_real_patches[i];
    Box b1 = patch->getExtraBox();
    IntVector lowEC   = patch->getExtraCellLowIndex();
    IntVector highEC  = patch->getExtraCellHighIndex();
    
    bool includeExtraCells  = true;
    Patch::selectType neighborPatches;
    selectPatches(lowEC, highEC, neighborPatches, includeExtraCells);
    
    for ( size_t j = 0; j < neighborPatches.size(); ++j) {
      const Patch* neighborPatch = neighborPatches[j];
      
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
    std::cout <<  " overlapping patches, Level: " << getIndex() << " patches: ("  << patchIDs.first << ", " << patchIDs.second <<"), low: " << me.lowIndex << " high: " << me.highIndex << std:: endl;
  }
#endif
}

//______________________________________________________________________
//  This method returns the min/max number of overlapping patch cells that are within a specified
//  region.  Patches will overlap when the domain is non-cubic
std::pair<int,int>
Level::getOverlapCellsInRegion( const selectType & patches,
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
  for (size_t i = 0; i < patches.size(); ++i) {
    for (size_t j = i+1; j < patches.size(); ++j) {    //  the i+1 prevents examining the transposed key pairs, i.e. 8,9 and 9,8

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
Level::finalizeLevel()
{
  m_each_patch = scinew PatchSet();
  m_each_patch->addReference();

  // The compute set requires an array const Patch*, we must copy d_realPatches
  std::vector<const Patch*> tmp_patches(m_real_patches.size());
  for (size_t i = 0; i < m_real_patches.size(); ++i) {
    tmp_patches[i] = m_real_patches[i];
  }

  m_each_patch->addEach(tmp_patches);

  m_all_patches = scinew PatchSet();
  m_all_patches->addReference();
  m_all_patches->addAll(tmp_patches);

  m_all_patches->sortSubsets();
  std::sort(m_real_patches.begin(), m_real_patches.end(), Patch::Compare());

  // determines and sets the boundary conditions for the patches
  setBCTypes();

  // compute the number of cells in the level
  m_total_cells = 0;
  for (size_t i = 0; i < m_real_patches.size(); ++i) {
    m_total_cells += m_real_patches[i]->getNumExtraCells();
  }

  // compute the max number of cells over all patches  Needed by PIDX
  m_numcells_patch_max = IntVector(0, 0, 0);
  int nCells = 0;
  for (size_t i = 0; i < m_real_patches.size(); ++i) {

    if (m_real_patches[i]->getNumExtraCells() > nCells) {
      IntVector lo = m_real_patches[i]->getExtraCellLowIndex();
      IntVector hi = m_real_patches[i]->getExtraCellHighIndex();
      m_numcells_patch_max = hi - lo;
    }
  }

  // compute and store the spatial ranges now that BCTypes are set
  for (size_t i = 0; i < m_real_patches.size(); ++i) {
    Patch* r = m_real_patches[i];
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
Level::finalizeLevel( bool periodicX, bool periodicY, bool periodicZ )
{
  // set each_patch and all_patches before creating virtual patches
  m_each_patch = scinew PatchSet();
  m_each_patch->addReference();

  // The compute set requires an array const Patch*, we must copy d_realPatches
  std::vector<const Patch*> tmp_patches(m_real_patches.size());

  for (size_t i = 0; i < m_real_patches.size(); ++i) {
    tmp_patches[i] = m_real_patches[i];
  }

  m_each_patch->addEach(tmp_patches);

  m_all_patches = scinew PatchSet();
  m_all_patches->addReference();
  m_all_patches->addAll(tmp_patches);

  BBox bbox;

  if (m_index > 0) {
    m_grid->getLevel(0)->getInteriorSpatialRange(bbox);
  }
  else {
    getInteriorSpatialRange(bbox);
  }

  Box domain(bbox.min(), bbox.max());
  Vector vextent = positionToIndex(bbox.max()) - positionToIndex(bbox.min());
  IntVector extent(static_cast<int>(rint(vextent.x())),
                   static_cast<int>(rint(vextent.y())),
                   static_cast<int>(rint(vextent.z())));

  m_periodic_boundaries = IntVector( periodicX ? 1 : 0, periodicY ? 1 : 0, periodicZ ? 1 : 0 );
  IntVector periodicBoundaryRange = m_periodic_boundaries * extent;

  int x, y, z;
  for (size_t i = 0; i < tmp_patches.size(); ++i) {

    for (x = -m_periodic_boundaries.x(); x <= m_periodic_boundaries.x(); ++x) {
      for (y = -m_periodic_boundaries.y(); y <= m_periodic_boundaries.y(); ++y) {
        for (z = -m_periodic_boundaries.z(); z <= m_periodic_boundaries.z(); ++z) {

          IntVector offset = IntVector(x, y, z) * periodicBoundaryRange;
          if (offset == IntVector(0, 0, 0)) {
            continue;
          }

          Box box = getBox(tmp_patches[i]->getExtraCellLowIndex() + offset - IntVector(1, 1, 1),
                           tmp_patches[i]->getExtraCellHighIndex() + offset + IntVector(1, 1, 1));

          if (box.overlaps(domain)) {
            Patch* newPatch = tmp_patches[i]->createVirtualPatch(offset);
            m_virtual_and_real_patches.push_back(newPatch);
          }
        }
      }
    }
  }

  m_all_patches->sortSubsets();
  std::sort(m_real_patches.begin(), m_real_patches.end(), Patch::Compare());
  std::sort(m_virtual_and_real_patches.begin(), m_virtual_and_real_patches.end(), Patch::Compare());

  // defind boundary conditions
  setBCTypes();

  //compute the number of cells in the level
  m_total_cells = 0;
  for (size_t i = 0; i < m_real_patches.size(); ++i) {
    m_total_cells += m_real_patches[i]->getNumExtraCells();
  }

  //compute the max number of cells over all patches  Needed by PIDX
  m_numcells_patch_max = IntVector(0, 0, 0);
  int nCells = 0;

  for (size_t i = 0; i < m_real_patches.size(); ++i) {
    if (m_real_patches[i]->getNumExtraCells() > nCells) {
      IntVector lo = m_real_patches[i]->getExtraCellLowIndex();
      IntVector hi = m_real_patches[i]->getExtraCellHighIndex();

      m_numcells_patch_max = hi - lo;
    }
  }

  //compute and store the spatial ranges now that BCTypes are set
  for (size_t i = 0; i < m_real_patches.size(); ++i) {
    Patch* r = m_real_patches[i];

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
Level::setBCTypes()
{
  Timers::Simple timer;
  timer.start();

  const int nTimes = 3;
  double rtimes[ nTimes ] = { 0 };

  if (m_bvh != nullptr) {
    delete m_bvh;
  }

  m_bvh = scinew PatchBVH(m_virtual_and_real_patches);

  rtimes[0] += timer().seconds();
  timer.reset( true );

  patch_iterator iter;

  ProcessorGroup *myworld = nullptr;
  int numProcs = 1;
  int rank = 0;

  if (Parallel::isInitialized()) {
    // only sus uses Parallel, but anybody else who uses DataArchive to read data does not
    myworld = Parallel::getRootProcessorGroup();
    numProcs = myworld->nRanks();
    rank = myworld->myRank();
  }

  std::vector<int> displacements(numProcs, 0);
  std::vector<int> recvcounts(numProcs, 0);

  //create recvcounts and displacement arrays
  int div = m_virtual_and_real_patches.size() / numProcs;
  int mod = m_virtual_and_real_patches.size() % numProcs;

  for (int p = 0; p < numProcs; ++p) {
    if (p < mod) {
      recvcounts[p] = div + 1;
    }
    else {
      recvcounts[p] = div;
    }
  }

  displacements[0] = 0;
  for (int p = 1; p < numProcs; ++p) {
    displacements[p] = displacements[p - 1] + recvcounts[p - 1];
  }

  std::vector<unsigned int> bctypes(m_virtual_and_real_patches.size());
  std::vector<unsigned int> mybctypes(recvcounts[rank]);

  int idx;

  patch_iterator startpatch = m_virtual_and_real_patches.begin() + displacements[rank];
  patch_iterator endpatch = startpatch + recvcounts[rank];

  // for each of my patches
  for (iter = startpatch, idx = 0; iter != endpatch; ++iter, ++idx) {
    Patch* patch = *iter;
    // See if there are any neighbors on the 6 faces
    int bitfield = 0;

    for (Patch::FaceType face = Patch::startFace; face <= Patch::endFace; face = Patch::nextFace(face)) {
      bitfield <<= 2;
      IntVector l, h;

      patch->getFace(face, IntVector(0, 0, 0), IntVector(1, 1, 1), l, h);

      Patch::selectType neighbors;
      selectPatches(l, h, neighbors);

      if (neighbors.size() == 0) {
        if (m_index != 0) {
          // See if there are any patches on the coarse level at that face
          IntVector fineLow, fineHigh;
          patch->getFace(face, IntVector(0, 0, 0), m_refinement_ratio, fineLow, fineHigh);

          IntVector coarseLow = mapCellToCoarser(fineLow);
          IntVector coarseHigh = mapCellToCoarser(fineHigh);
          const LevelP& coarseLevel = getCoarserLevel();

#if 0
          // add 1 to the corresponding index on the plus edges
          // because the upper corners are sort of one cell off (don't know why)
          if (m_extra_cells.x() != 0 && face == Patch::xplus) {
            coarseLow[0] ++;
            coarseHigh[0]++;
          }
          else if (m_extra_cells.y() != 0 && face == Patch::yplus) {
            coarseLow[1] ++;
            coarseHigh[1] ++;
          }
          else if (m_extra_cells.z() != 0 && face == Patch::zplus) {
            coarseLow[2] ++;
            coarseHigh[2]++;
          }
#endif
          coarseLevel->selectPatches(coarseLow, coarseHigh, neighbors);

          if (neighbors.size() == 0) {
            bitfield |= Patch::None;
          } else {
            bitfield |= Patch::Coarse;
          }
        } else {
          bitfield |= Patch::None;
        }
      } else {
        bitfield |= Patch::Neighbor;
      }
    }
    mybctypes[idx] = bitfield;
  }

  if (numProcs > 1) {
    // allgather bctypes
    if (mybctypes.size() == 0) {
      Uintah::MPI::Allgatherv(0, 0, MPI_UNSIGNED, &bctypes[0], &recvcounts[0], &displacements[0], MPI_UNSIGNED, myworld->getComm());
    }
    else {
      Uintah::MPI::Allgatherv(&mybctypes[0], mybctypes.size(), MPI_UNSIGNED, &bctypes[0], &recvcounts[0], &displacements[0],
                              MPI_UNSIGNED, myworld->getComm());
    }
  }
  else {
    bctypes.swap(mybctypes);
  }

  rtimes[1] += timer().seconds();
  timer.reset( true );

  int i;
  // loop through patches
  for (iter = m_virtual_and_real_patches.begin(), i = 0, idx = 0; iter != m_virtual_and_real_patches.end(); ++iter, ++i) {
    Patch *patch = *iter;

    if (patch->isVirtual()) {
      patch->setLevelIndex(-1);
    } else {
      patch->setLevelIndex(idx++);
    }

    int bitfield = bctypes[i];
    int mask = 3;

    // loop through faces
    for (int j = 5; j >= 0; --j) {

      int bc_type = bitfield & mask;

      if (rank == 0) {
        switch (bc_type) {
          case Patch::None :
            DOUT(g_bc_dbg, "  Setting Patch " << patch->getID() << " face " << j << " to None");
            break;
          case Patch::Coarse :
            DOUT(g_bc_dbg, "  Setting Patch " << patch->getID() << " face " << j << " to Coarse");
            break;
          case Patch::Neighbor :
            DOUT(g_bc_dbg, "  Setting Patch " << patch->getID() << " face " << j << " to Neighbor");
            break;
        }
      }
      patch->setBCType(Patch::FaceType(j), Patch::BCType(bc_type));
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

  if (g_rg_times.active()) {
    std::ostringstream mesg;
    double avg[ nTimes ] = { 0 };
    Uintah::MPI::Reduce(rtimes, avg, nTimes, MPI_DOUBLE, MPI_SUM, 0, myworld->getComm());

    if (myworld->myRank() == 0) {

      mesg << "SetBCType Avg Times: ";
      for (int i = 0; i < nTimes; ++i) {
        avg[i] /= myworld->nRanks();
        mesg << avg[i] << " ";
      }
      mesg << std::endl;
    }

    double max[nTimes] = { 0 };
    Uintah::MPI::Reduce(rtimes, max, nTimes, MPI_DOUBLE, MPI_MAX, 0, myworld->getComm());

    if (myworld->myRank() == 0) {
      mesg << "SetBCType Max Times: ";
      for (int i = 0; i < nTimes; ++i) {
        mesg << max[i] << " ";
      }
      mesg << std::endl;
    }
    DOUT(true, mesg.str());
  }

  // recreate BVH with extra cells
  if (m_bvh != nullptr) {
    delete m_bvh;
  }
  m_bvh = scinew PatchBVH(m_virtual_and_real_patches);
}

//______________________________________________________________________
//
void
Level::assignBCS( const ProblemSpecP & grid_ps, LoadBalancer * lb )
{
  ProblemSpecP bc_ps = grid_ps->findBlock("BoundaryConditions");
  if( bc_ps == nullptr ) {
    if( Parallel::getMPIRank() == 0 ) {
      static ProgressiveWarning warn( "No BoundaryConditions specified", -1 );
      warn.invoke();
    }
    return;
  }

  BoundCondReader reader;

  reader.read(bc_ps, grid_ps, this);

  for (patch_iterator iter = m_virtual_and_real_patches.begin(); iter != m_virtual_and_real_patches.end(); iter++) {
    Patch* patch = *iter;

    // If we have a lb, then only apply bcs this processors patches.
    if (lb == 0 || lb->getPatchwiseProcessorAssignment(patch) == Parallel::getMPIRank()) {

      patch->initializeBoundaryConditions();

      for (Patch::FaceType face_side = Patch::startFace; face_side <= Patch::endFace; face_side = Patch::nextFace(face_side)) {
        if (patch->getBCType(face_side) == Patch::None) {
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
Level::getBox( const IntVector & l, const IntVector & h ) const
{
  return Box(getNodePosition(l), getNodePosition(h));
}

//______________________________________________________________________
//
const PatchSet*
Level::eachPatch() const
{
  ASSERT(m_each_patch != nullptr);
  return m_each_patch;
}

//______________________________________________________________________
//
const PatchSet*
Level::allPatches() const
{
  ASSERT(m_all_patches != nullptr);
  return m_all_patches;
}

//______________________________________________________________________
//
const Patch*
Level::selectPatchForCellIndex( const IntVector & idx ) const
{
  selectType pv;
  IntVector i(1, 1, 1);
  selectPatches(idx - i, idx + i, pv, false, false);

  if (pv.size() == 0) {
    return nullptr;
  }
  else {
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
const Patch*
Level::selectPatchForNodeIndex( const IntVector & idx ) const
{
  selectType pv;
  IntVector i(1, 1, 1);

  selectPatches(idx - i, idx + i, pv, false, false);

  if (pv.size() == 0) {
    return nullptr;
  }
  else {
    selectType::iterator iter;
    for (iter = pv.begin(); iter != pv.end(); ++iter) {
      if ((*iter)->containsNode(idx)) {
        return *iter;
      }
    }
  }
  return nullptr;
}

//______________________________________________________________________
//
const LevelP &
Level::getCoarserLevel() const
{
  return getRelativeLevel( -1 );
}

//______________________________________________________________________
//
const LevelP &
Level::getFinerLevel() const
{
  return getRelativeLevel( 1 );
}

//______________________________________________________________________
//
bool
Level::hasCoarserLevel() const
{
  return getIndex() > 0;
}

//______________________________________________________________________
//
bool
Level::hasFinerLevel() const
{
  return getIndex() < ( m_grid->numLevels() - 1 );
}

//______________________________________________________________________
//
IntVector
Level::mapCellToCoarser( const IntVector & idx, int level_offset ) const
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
Level::mapCellToFiner( const IntVector & idx ) const
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
Level::mapCellToFinest( const IntVector & idx ) const
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
Level::mapCellToFinestNoAdjustments( const IntVector & idx ) const
{

      IntVector r_ratio  = IntVector(1,1,1);
      for (int i=m_index; i< m_grid->numLevels()-1; ++i){
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
Level::mapNodeToCoarser( const IntVector & idx ) const
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
Level::mapNodeToFiner( const IntVector & idx ) const
{
  return idx * m_grid->getLevel(m_index + 1)->m_refinement_ratio;
}

//______________________________________________________________________
//
int
Level::getRefinementRatioMaxDim() const
{
  return Max( Max(m_refinement_ratio.x(), m_refinement_ratio.y()), m_refinement_ratio.z() );
}



//______________________________________________________________________
//
namespace Uintah {

const Level* getLevel( const PatchSubset * subset )
{
  ASSERT(subset->size() > 0);
  const Level* level = subset->get(0)->getLevel();
#if SCI_ASSERTION_LEVEL>0
  for (int i = 1; i < subset->size(); ++i) {
    ASSERT(level == subset->get(i)->getLevel());
  }
#endif
  return level;
}
//______________________________________________________________________
//
const LevelP& getLevelP( const PatchSubset * subset )
{
  ASSERT(subset->size() > 0);
  const LevelP& level = subset->get(0)->getLevelP();
#if SCI_ASSERTION_LEVEL>0
  for (int i = 1; i < subset->size(); ++i) {
    ASSERT(level == subset->get(i)->getLevelP());
  }
#endif
  return level;
}

//______________________________________________________________________
//
const Level* getLevel( const PatchSet * set )
{
  ASSERT(set->size() > 0);
  return getLevel(set->getSubset(0));
}

//______________________________________________________________________
// We may need to put a lock around this?
std::ostream& operator<<( std::ostream & out, const Level & level )
{
  IntVector lo, hi;
  level.findCellIndexRange(lo, hi);

  out << "(Level " << level.getIndex() << ", numPatches: " << level.numPatches()
      << ", cellIndexRange: " << lo << ", " << hi << ", " << *(level.allPatches()) << ")";

  return out;
}

} // end namespace Uintah

