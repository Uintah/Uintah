/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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


#ifndef UINTAH_GRID_LEVEL_H
#define UINTAH_GRID_LEVEL_H

#include <Core/Util/RefCounted.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/LevelP.h>
#include <Core/Util/Handle.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Containers/OffsetArray1.h>

#ifdef max
// some uintah 3p utilities define max, so undefine it before BBox chokes on it.
#undef max
#endif

#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/fixedvector.h>
#include <Core/Grid/Variables/ComputeSet.h>

#include <vector>
#include <map>

#include <Core/Grid/uintahshare.h>
namespace Uintah {

  using SCIRun::Vector;
  using SCIRun::Point;
  using SCIRun::IntVector;
  using SCIRun::BBox;
  using SCIRun::OffsetArray1;

  class PatchBVH;
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

class UINTAHSHARE Level : public RefCounted {
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

  const Patch* getPatch(int index) const { return d_realPatches[index]; }

  // go through the virtual ones too
  const_patchIterator allPatchesBegin() const;
  const_patchIterator allPatchesEnd() const;
      
  Patch* addPatch(const IntVector& extraLowIndex,
                  const IntVector& extraHighIndex,
                  const IntVector& lowIndex,
                  const IntVector& highIndex,
                  Grid* grid);
      
  Patch* addPatch(const IntVector& extraLowIndex,
                  const IntVector& extraHighIndex,
                  const IntVector& lowIndex,
                  const IntVector& highIndex,
                  Grid* grid,
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

  //////////
  // Find a patch containing the point, return 0 if non exists
  Patch* getPatchFromPoint( const Point& );

  void finalizeLevel();
  void finalizeLevel(bool periodicX, bool periodicY, bool periodicZ);
  void assignBCS(const ProblemSpecP& ps, LoadBalancer* lb);
      
  int numPatches() const;
  long totalCells() const;

  void getSpatialRange(BBox& b) const;
  void getInteriorSpatialRange(BBox& b) const;
  void findIndexRange(IntVector& lowIndex, IntVector& highIndex) const
  { findNodeIndexRange(lowIndex, highIndex); }
  void findNodeIndexRange(IntVector& lowIndex, IntVector& highIndex) const;
  void findCellIndexRange(IntVector& lowIndex, IntVector& highIndex) const;

  void findInteriorIndexRange(IntVector& lowIndex, IntVector& highIndex) const
  { findInteriorNodeIndexRange(lowIndex, highIndex);}
  void findInteriorNodeIndexRange(IntVector& lowIndex,
                                  IntVector& highIndex) const;
  void findInteriorCellIndexRange(IntVector& lowIndex,
                                  IntVector& highIndex) const;
      
  void performConsistencyCheck() const;
  GridP getGrid() const;

  const LevelP& getFineLevel() const
  { return getRelativeLevel(1); }
  const LevelP& getCoarseLevel() const
  { return getRelativeLevel(-1); }
     
  const LevelP& getRelativeLevel(int offset) const;


  // Grid spacing
  Vector dCell() const {
    return d_dcell;
  }
  Point getAnchor() const {
    return d_anchor;
  }

  // For stretched grids
  bool isStretched() const {
    return d_stretched;
  }
  void getCellWidths(Grid::Axis axis, OffsetArray1<double>& widths) const;
  void getFacePositions(Grid::Axis axis, OffsetArray1<double>& faces) const;
  void setStretched(Grid::Axis axis, const OffsetArray1<double>& faces);

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
                     selectType&, bool cache=true) const;

  bool containsPoint(const Point&) const;
  bool containsPointInRealCells(const Point&) const;
  bool containsCell(const IntVector&) const;

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
  
  //getID() returns a unique identifier so if the grid is rebuilt the new 
  //levels will have different id numbers (like a  serial number).  
  inline int getID() const {
    return d_id;
  }

 //getIndex() returns the relative position of the level - 0 is coarsest, 1 is  
 //next and so forth.  
  inline int getIndex() const {
    return d_index;
  }
  inline IntVector getRefinementRatio() const {
    return d_refinementRatio;
  }
  int getRefinementRatioMaxDim() const;

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

  // vars for select_grid - don't ifdef them here, so if we change it
  // we don't have to compile everything
  IntVector d_idxLow;
  IntVector d_idxHigh;
  IntVector d_idxSize;
  IntVector d_gridSize;
  vector<int> d_gridStarts;
  vector<Patch*> d_gridPatches;

  // For stretched frids
  bool d_stretched;

  // This is three different arrays containing the x,y,z coordinate of the face position
  // be sized to d_idxSize[axis] + 1.  Used only for stretched grids
  OffsetArray1<double> d_facePosition[3];

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
  PatchBVH* d_bvh;
};

  UINTAHSHARE const Level* getLevel(const PatchSubset* subset);
  UINTAHSHARE const Level* getLevel(const PatchSet* set);
} // End namespace Uintah

#endif
