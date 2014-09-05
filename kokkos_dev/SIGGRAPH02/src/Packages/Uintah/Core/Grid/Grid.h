#ifndef UINTAH_HOMEBREW_GRID_H
#define UINTAH_HOMEBREW_GRID_H

#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/ProblemSpec/Handle.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>

#include <Core/Geometry/BBox.h>

#include <vector>

namespace Uintah {

using namespace SCIRun;

/**************************************

CLASS
   Grid
   
   This class manages the grid used to solve the CFD and MPM problems.

GENERAL INFORMATION

   Grid.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Grid, Level

DESCRIPTION
   This class basicly manages the pointers to the levels that make
   up the grid.  
  
WARNING
  
****************************************/

  class Grid : public RefCounted {
  public:
    Grid();
    virtual ~Grid();
    
    //////////
    // Returns the number of levels in this grid.
    int     numLevels() const;
    
    //////////
    // Returns a "Handle" to the "idx"th level 
    const LevelP& getLevel(int idx) const;
    
    //////////
    // Adds a level to the grid.
    Level* addLevel(const Point& anchor, const Vector& dcell);
    
    void performConsistencyCheck() const;
    void printStatistics() const;
    
    //////////
    // Computes the physical boundaries for the grid
    void getSpatialRange(BBox& b) const;
    
  private:
    std::vector<LevelP> d_levels;
    
    Grid(const Grid&);
    Grid& operator=(const Grid&);
  };

} // End namespace Uintah

#endif
