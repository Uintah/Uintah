#ifndef UINTAH_GRID_LEVEL_H
#define UINTAH_GRID_LEVEL_H

#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/ProblemSpec/Handle.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/fixedvector.h>

#include <string>
#include <vector>

#define SELECT_RANGETREE

namespace Uintah {

#ifdef SELECT_RANGETREE
class PatchRangeTree;
#endif
  
using namespace SCIRun;

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
      Level(Grid* grid, const Point& anchor, const Vector& dcell);
      virtual ~Level();
      
      void setPatchDistributionHint(const IntVector& patchDistribution);
      typedef std::vector<Patch*>::iterator patchIterator;
      typedef std::vector<Patch*>::const_iterator const_patchIterator;
      const_patchIterator patchesBegin() const;
      const_patchIterator patchesEnd() const;
      patchIterator patchesBegin();
      patchIterator patchesEnd();
      
      Patch* addPatch(const IntVector& lowIndex,
		      const IntVector& highIndex,
		      const IntVector& extraLowIndex,
		      const IntVector& extrahighIndex);
      
      Patch* addPatch(const IntVector& lowIndex,
		      const IntVector& highIndex,
		      const IntVector& extraLowIndex,
		      const IntVector& extrahighIndex,
		      int ID);

     //////////
     // Find a patch containing the point, return 0 if non exists
      Patch* getPatchFromPoint( const Point& );

      void finalizeLevel();
      void assignBCS(const ProblemSpecP& ps);
      
      int numPatches() const;
      long totalCells() const;

      void getIndexRange(BBox& b) const;
      void getSpatialRange(BBox& b) const;

      // it is better that IndexRange expressed by IntVector
      // instead of BBox, and the function is const. --tan
      void getIndexRange(IntVector& lowIndex,IntVector& highIndex) const;
      
      void performConsistencyCheck() const;
      GridP getGrid() const;

      Vector dCell() const {
	 return d_dcell;
      }
      Point getAnchor() const {
	 return d_anchor;
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

      const Patch* selectPatch( const IntVector& idx) const
      {
	selectType pv;
	selectPatches(idx,idx,pv);
	// may need to check if pv is empty
	return pv[0];
      }

      bool containsPoint(const Point&) const;
   private:
      Level(const Level&);
      Level& operator=(const Level&);
      
      std::vector<Patch*> d_patches;
     
      Grid* grid;
      Point d_anchor;
      Vector d_dcell;
      bool d_finalized;
      IntVector d_patchDistribution;

#ifdef SELECT_GRID
      IntVector d_idxLow;
      IntVector d_idxHigh;
      IntVector d_idxSize;
      IntVector d_gridSize;
      vector<int> d_gridStarts;
      vector<Patch*> d_gridPatches;
#else
#ifdef SELECT_RANGETREE
      PatchRangeTree* d_rangeTree;
#endif
#endif
   };
} // End namespace Uintah

#endif
