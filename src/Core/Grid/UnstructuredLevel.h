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

#ifndef CORE_GRID_UNSTRUCTURED_LEVEL_H
#define CORE_GRID_UNSTRUCTURED_LEVEL_H

#include <CCA/Ports/UnstructuredLoadBalancer.h>

#include <Core/Disclosure/UnstructuredTypeDescription.h>
#include <Core/Grid/UnstructuredGridP.h>
#include <Core/Grid/UnstructuredGrid.h>
#include <Core/Grid/UnstructuredLevelP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Containers/OffsetArray1.h>
#include <Core/Util/Handle.h>
#include <Core/Util/RefCounted.h>

#ifdef max
// some uintah 3p utilities define max, so undefine it before BBox chokes on it.
#  undef max
#endif

#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <vector>
#include <map>

namespace Uintah {

  class UnstructuredPatchBVH;
  class BoundCondBase;
  class Box;
  class UnstructuredPatch;
  class UnstructuredTask;
/**************************************

CLASS
   Level
   
   Just a container class that manages a set of Patches that
   make up this level.

GENERAL INFORMATION

   Level.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   Level

DESCRIPTION

  
****************************************/

class UnstructuredLevel : public RefCounted {

public:

  UnstructuredLevel(UnstructuredGrid      * grid
       , const Point     & anchor
       , const Vector    & dcell
       , const int         index
       , const IntVector   refinementRatio
       , const int         id = -1
       );

  virtual ~UnstructuredLevel();
  
  void setPatchDistributionHint( const IntVector & patchDistribution );

  void setBCTypes();
     
  using       patch_iterator = std::vector<UnstructuredPatch*>::iterator;
  using const_patch_iterator = std::vector<UnstructuredPatch*>::const_iterator;
  const_patch_iterator patchesBegin() const;
  patch_iterator       patchesBegin();

  const_patch_iterator patchesEnd() const;
  patch_iterator       patchesEnd();

  const UnstructuredPatch* getPatch( int index ) const { return m_real_patches[index]; }

  // go through the virtual ones too
  const_patch_iterator allPatchesBegin() const;
  const_patch_iterator allPatchesEnd() const;
      
  UnstructuredPatch* addPatch( const IntVector & extraLowIndex
			       , const IntVector & extraHighIndex
			       , const IntVector & lowIndex
			       , const IntVector & highIndex
			       , UnstructuredGrid * grid
                 );
      
  UnstructuredPatch* addPatch( const IntVector & extraLowIndex
                 , const IntVector & extraHighIndex
                 , const IntVector & lowIndex
                 , const IntVector & highIndex
                 ,       UnstructuredGrid      * grid
                 ,       int         ID
                 );

  // Move up and down the hierarchy
  const UnstructuredLevelP& getCoarserLevel() const;
  const UnstructuredLevelP& getFinerLevel() const;
  bool          hasCoarserLevel() const;
  bool          hasFinerLevel() const;

  IntVector     mapNodeToCoarser( const IntVector & idx ) const;
  IntVector     mapNodeToFiner(   const IntVector & idx ) const;
  IntVector     mapCellToCoarser( const IntVector & idx, int level_offset=1 ) const;
  IntVector     mapCellToFiner(   const IntVector & idx ) const;
  IntVector     mapCellToFinest(const IntVector& idx) const;
  IntVector     mapCellToFinestNoAdjustments( const IntVector & idx ) const;


  //////////
  // Find a patch containing the point, return 0 if non exists
  const UnstructuredPatch* getPatchFromPoint( const Point &, const bool includeExtraCells ) const;

  //////////
  // Find a patch containing the cell or node, return 0 if non exists
  const UnstructuredPatch* getPatchFromIndex( const IntVector &, const bool includeExtraCells ) const;

  void finalizeLevel();
  void finalizeLevel( bool periodicX, bool periodicY, bool periodicZ );
  void assignBCS( const ProblemSpecP & ps, UnstructuredLoadBalancer * lb );
      
  int numPatches() const;
  long totalCells() const;
  long getTotalCellsInRegion(const IntVector& lowIndex, const IntVector& highIndex) const;
  IntVector nCellsPatch_max() const;

  void getSpatialRange( BBox & b ) const { b.extend(m_spatial_range); };
  void getInteriorSpatialRange( BBox & b ) const { b.extend(m_int_spatial_range); };
  
  // methods to identify if this is non-cubic level
  bool isNonCubic() const { return m_isNonCubicDomain; };
  
  
  void findIndexRange(     IntVector & lowIndex, IntVector & highIndex ) const { findNodeIndexRange(lowIndex, highIndex); }
  void findNodeIndexRange( IntVector & lowIndex, IntVector & highIndex ) const;
  void findCellIndexRange( IntVector & lowIndex, IntVector & highIndex ) const;

  void findInteriorIndexRange(     IntVector & lowIndex, IntVector & highIndex ) const { findInteriorNodeIndexRange(lowIndex, highIndex); }
  void findInteriorNodeIndexRange( IntVector & lowIndex, IntVector & highIndex ) const;
  void findInteriorCellIndexRange( IntVector & lowIndex, IntVector & highIndex ) const;
                                  
  void computeVariableExtents( const UnstructuredTypeDescription::UnstructuredType TD
                             , IntVector& lo
                             , IntVector& hi
                             ) const;
      
  void performConsistencyCheck() const;

  UnstructuredGridP getGrid() const;

  const UnstructuredLevelP & getRelativeLevel( int offset ) const;

  // Grid spacing
  Vector dCell() const { return m_dcell; }

  // Returns the cell volume dx*dy*dz.
  double cellVolume() const {
    return m_dcell.x()*m_dcell.y()*m_dcell.z();
  }

  // Returns the cell area dx*dy, dx*dz, or dy*dz.
  double cellArea( Vector unitNormal ) const {
    Vector areas(m_dcell.y() * m_dcell.z(), m_dcell.x() * m_dcell.z(), m_dcell.x() * m_dcell.y());
    return Dot(areas, unitNormal);
  }

  Point getAnchor() const { return m_anchor; }

  void      setExtraCells( const IntVector & ec );
  IntVector getExtraCells() const { return m_extra_cells; }

  Point     getNodePosition( const IntVector & ) const;
  Point     getCellPosition( const IntVector & ) const;

  IntVector getCellIndex(    const Point & ) const;
  Point     positionToIndex( const Point & ) const;

  Box getBox( const IntVector &, const IntVector & ) const;

  using selectType = std::vector<const UnstructuredPatch*>;


  void selectPatches( const IntVector  &
                    , const IntVector  &
                    ,       selectType &
                    ,       bool withExtraCells = false
                    ,       bool cache_patches  = false
                    ) const;

  bool containsPointIncludingExtraCells( const Point & ) const;
  bool containsPoint( const Point & ) const;
  bool containsCell(  const IntVector & ) const;

  // IntVector whose elements are each 1 or 0 specifying whether there
  // are periodic boundaries in each dimension (1 means periodic).
  IntVector getPeriodicBoundaries() const { return m_periodic_boundaries; }

  // The eachPatch() function returns a PatchSet containing patches on
  // this level with one patch per PatchSubSet.  Eg: { {1}, {2}, {3} }
  const UnstructuredPatchSet* eachPatch() const;
  const UnstructuredPatchSet* allPatches() const;
  
  const UnstructuredPatch* selectPatchForCellIndex( const IntVector & idx ) const;
  const UnstructuredPatch* selectPatchForNodeIndex( const IntVector & idx ) const;
  
  // getID() returns a unique identifier so if the grid is rebuilt the new
  // levels will have different id numbers (like a  serial number).
  inline int getID() const { return m_id; }

  // getIndex() returns the relative position of the level - 0 is coarsest,
  // 1 is next and so forth.
  inline int getIndex() const { return m_index; }
  inline IntVector getRefinementRatio() const { return m_refinement_ratio; }
  int getRefinementRatioMaxDim() const;

  friend std::ostream& operator<<( std::ostream& out, const UnstructuredLevel& level );

  //__________________________________
  //  overlapping patches:  Used to keep track of patches that overlap in non-cubic levels
  struct overlap{
    std::pair <int,int> patchIDs{-9,-9};        // overlapping patch IDs
    IntVector lowIndex{ IntVector(-9,-9,-9)};   // low/high index of overlap
    IntVector highIndex{IntVector(-9,-9,-9)};
  };
  // for a set of patches and region return the min/max number of overlapping cells
  std::pair<int,int> getOverlapCellsInRegion( const selectType & patches,
                                              const IntVector  & regionLow, 
                                              const IntVector  & regionHigh) const;

private:

  // eliminate copy, assignment and move
  UnstructuredLevel( const UnstructuredLevel & )            = delete;
  UnstructuredLevel& operator=( const UnstructuredLevel & ) = delete;
  UnstructuredLevel( UnstructuredLevel && )                 = delete;
  UnstructuredLevel& operator=( UnstructuredLevel && )      = delete;
      
  UnstructuredGrid    * m_grid;
  Point     m_anchor;
  Vector    m_dcell;

  // The spatial range of the level.
  BBox      m_spatial_range;
  BBox      m_int_spatial_range;

  bool      m_isNonCubicDomain{false};                    // is level non cubic level
  void      setIsNonCubicLevel();
  
  bool      m_finalized{false};
  int       m_index;                                      // number of the level
  IntVector m_patch_distribution;
  IntVector m_periodic_boundaries;

  UnstructuredPatchSet* m_each_patch{nullptr};
  UnstructuredPatchSet* m_all_patches{nullptr};

  long      m_total_cells{0};
  IntVector m_extra_cells{IntVector(0,0,0)};
  IntVector m_numcells_patch_max{IntVector(0,0,0)};

  std::vector<UnstructuredPatch*> m_real_patches;                    // only real patches
  std::vector<UnstructuredPatch*> m_virtual_and_real_patches;        // real and virtual

  int       m_id;
  IntVector m_refinement_ratio;

  class IntVectorCompare {
  public:
    bool operator() (const std::pair<IntVector, IntVector>&a, const std::pair<IntVector, IntVector>&b) const 
    {
      return (a.first < b.first) || (!(b.first < a.first) && a.second < b.second);
    }
  };

  
  using select_cache =  std::map<std::pair<IntVector, IntVector>, std::vector<const UnstructuredPatch*>, IntVectorCompare>;
  mutable select_cache m_select_cache; // we like const Levels in most places :)

  UnstructuredPatchBVH * m_bvh{nullptr};

  // overlapping patches   
  std::map< std::pair<int, int>, overlap > m_overLapPatches;
  void setOverlappingPatches();
};

const UnstructuredLevel  * getLevel(  const UnstructuredPatchSubset * subset );
const UnstructuredLevel  * getLevel(  const UnstructuredPatchSet    * set );
const UnstructuredLevelP & getLevelP( const UnstructuredPatchSubset * subset );

} // End namespace Uintah

#endif // CORE_GRID_UNSTRUCTURED_LEVEL_H
