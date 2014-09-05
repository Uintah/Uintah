#ifndef UINTAH_GRID_LEVEL_H
#define UINTAH_GRID_LEVEL_H

#include <Packages/Uintah/Core/Util/RefCounted.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Util/Handle.h>

#ifdef max
// some uintah 3p utilities define max, so undefine it before BBox chokes on it.
#undef max
#endif

#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/fixedvector.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <map>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  using SCIRun::Vector;
  using SCIRun::Point;
  using SCIRun::IntVector;
  using SCIRun::BBox;

  class PatchRangeTree;
  class BoundCondBase;
  class Box;
  class Patch;
  class Task;
   
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
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Level

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class Level : public RefCounted {
public:
  Level(Grid* grid, const Point& anchor, const Vector& dcell, int index, 
        IntVector refinementRatio,
        int id = -1);
  virtual ~Level();
  
  void setPatchDistributionHint(const IntVector& patchDistribution);
  void setBCTypes();
     
  typedef std::vector<Patch*>::iterator patchIterator;
  typedef std::vector<Patch*>::const_iterator const_patchIterator;
  const_patchIterator patchesBegin() const;
  const_patchIterator patchesEnd() const;
  patchIterator patchesBegin();
  patchIterator patchesEnd();

  // go through the virtual ones too
  const_patchIterator allPatchesBegin() const;
  const_patchIterator allPatchesEnd() const;
      
  Patch* addPatch(const IntVector& extraLowIndex,
                  const IntVector& extraHighIndex,
                  const IntVector& lowIndex,
                  const IntVector& highIndex);
      
  Patch* addPatch(const IntVector& extraLowIndex,
                  const IntVector& extraHighIndex,
                  const IntVector& lowIndex,
                  const IntVector& highIndex,
                  int ID);

  // Move up and down the hierarchy
  const LevelP& getCoarserLevel() const;
  const LevelP& getFinerLevel() const;
  bool hasCoarserLevel() const;
  bool hasFinerLevel() const;
  IntVector mapNodeToCoarser(const IntVector& idx) const;
  IntVector mapNodeToFiner(const IntVector& idx) const;
  IntVector mapCellToCoarser(const IntVector& idx) const;
  IntVector mapCellToFiner(const IntVector& idx) const;
  IntVector interpolateCellToCoarser(const IntVector& idx, Vector& weight) const;
  IntVector interpolateXFaceToCoarser(const IntVector& idx, Vector& weight) const;
  IntVector interpolateYFaceToCoarser(const IntVector& idx, Vector& weight) const;
  IntVector interpolateZFaceToCoarser(const IntVector& idx, Vector& weight) const;
  IntVector interpolateToCoarser(const IntVector& idx, const IntVector& dir,
                                 Vector& weight) const;

  //////////
  // Find a patch containing the point, return 0 if non exists
  Patch* getPatchFromPoint( const Point& );

  void finalizeLevel();
  void finalizeLevel(bool periodicX, bool periodicY, bool periodicZ);
  void assignBCS(const ProblemSpecP& ps);
      
  int numPatches() const;
  long totalCells() const;

  void getSpatialRange(BBox& b) const;

  void findIndexRange(IntVector& lowIndex, IntVector& highIndex) const
  { findNodeIndexRange(lowIndex, highIndex); }
  void findNodeIndexRange(IntVector& lowIndex, IntVector& highIndex) const;
  void findCellIndexRange(IntVector& lowIndex, IntVector& highIndex) const;
      
  void performConsistencyCheck() const;
  GridP getGrid() const;

  const LevelP& getFineLevel() const
  { return getRelativeLevel(1); }
  const LevelP& getCoarseLevel() const
  { return getRelativeLevel(-1); }
     
  const LevelP& getRelativeLevel(int offset) const;

  Vector dCell() const {
    return d_dcell;
  }
  Point getAnchor() const {
    return d_anchor;
  }

  void setExtraCells(const IntVector& ec);
  IntVector getExtraCells() const {
    return d_extraCells;
  }

  Point getNodePosition(const IntVector&) const;
  Point getCellPosition(const IntVector&) const;
  IntVector getCellIndex(const Point&) const;
  Point positionToIndex(const Point&) const;

  Box getBox(const IntVector&, const IntVector&) const;

  static const int MAX_PATCH_SELECT = 32;
  typedef fixedvector<const Patch*, MAX_PATCH_SELECT> selectType;
      

  void selectPatches(const IntVector&, const IntVector&,
                     selectType&) const;

  bool containsPoint(const Point&) const;
  bool containsPointInRealCells(const Point&) const;

  // IntVector whose elements are each 1 or 0 specifying whether there
  // are periodic boundaries in each dimension (1 means periodic).
  IntVector getPeriodicBoundaries() const
  { return d_periodicBoundaries; }

  // The eachPatch() function returns a PatchSet containing patches on
  // this level with one patch per PatchSubSet.  Eg: { {1}, {2}, {3} }
  const PatchSet* eachPatch() const;
  const PatchSet* allPatches() const;
  const Patch* selectPatchForCellIndex( const IntVector& idx) const;
  const Patch* selectPatchForNodeIndex( const IntVector& idx) const;
  const Patch* getPatchByID(int id) const;
  inline int getID() const {
    return d_id;
  }
  inline int timeRefinementRatio() const {
    return d_timeRefinementRatio;
  }
  void setTimeRefinementRatio(int trr);
  inline int getIndex() const {
    return d_index;
  }
  inline IntVector getRefinementRatio() const {
    return d_refinementRatio;
  }

  //! Use this when you're done setting the delt, and this function
  //! will compensate for whichever level you're on
  double adjustDelt(double delt) const;
private:
  Level(const Level&);
  Level& operator=(const Level&);
      
  std::vector<Patch*> d_patches;

  Grid* grid;
  Point d_anchor;
  Vector d_dcell;
  bool d_finalized;
  int d_index; // number of the level
  IntVector d_patchDistribution;
  IntVector d_periodicBoundaries;

  PatchSet* each_patch;
  PatchSet* all_patches;

  IntVector d_extraCells;

  std::vector<Patch*> d_realPatches; // only real patches
  std::vector<Patch*> d_virtualAndRealPatches; // real and virtual

  int d_id;
  IntVector d_refinementRatio;

      // should technically be a "grid" parameter, but here we have a way
      // to check if it's been finalized.
  int d_timeRefinementRatio;

  // vars for select_grid - don't ifdef them here, so if we change it
  // we don't have to compile everything
  IntVector d_idxLow;
  IntVector d_idxHigh;
  IntVector d_idxSize;
  IntVector d_gridSize;
  vector<int> d_gridStarts;
  vector<Patch*> d_gridPatches;

  // vars for select_rangetree

  class IntVectorCompare {
  public:
    bool operator() (const std::pair<IntVector, IntVector>&a, const std::pair<IntVector, IntVector>&b) const 
    {
      return (a.first < b.first) || (!(b.first < a.first) && a.second < b.second);
    }
  };

  typedef std::map<std::pair<IntVector, IntVector>, vector<const Patch*>, IntVectorCompare> selectCache;
  mutable selectCache d_selectCache; // we like const Levels in most places :) 
  PatchRangeTree* d_rangeTree;
};

  const Level* getLevel(const PatchSubset* subset);
  const Level* getLevel(const PatchSet* set);
} // End namespace Uintah

#endif
