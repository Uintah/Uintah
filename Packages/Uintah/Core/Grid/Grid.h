#ifndef UINTAH_HOMEBREW_GRID_H
#define UINTAH_HOMEBREW_GRID_H

#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Util/Handle.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Util/RefCounted.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <list>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
  class BBox;
  class Point;
  class Vector;
}

#include <Packages/Uintah/Core/Grid/share.h>
namespace Uintah {

  class ProcessorGroup;

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

  class SCISHARE Grid : public RefCounted {
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
    Level* addLevel(const SCIRun::Point& anchor,
                    const SCIRun::Vector& dcell, int id=-1);
   
    void performConsistencyCheck() const;
    void printStatistics() const;

    //////////
    // Computes the physical boundaries for the grid (including extra cells)
    void getSpatialRange(SCIRun::BBox& b) const;

    ////////// 
    // Returns the boundary of the grid exactly (without
    // extra cells).  The value returned is the same value
    // as found in the .ups file.
    void getInteriorSpatialRange(SCIRun::BBox& b) const;
    
    //////////
    // Computes the length of the grid
    void getLength(SCIRun::Vector& length,
                   const string flag = "plusExtraCells") const;
    
    //////////
    // Problem setup functions called from simulation controller
    void problemSetup(const ProblemSpecP& params, const ProcessorGroup *pg, bool do_amr); 

    // For comparing grids - level and patch structure must be equal
    bool operator==(const Grid& othergrid) const;

    SCISHARE friend std::ostream& operator<<(std::ostream& out, const Uintah::Grid& grid);

    // Used in Level and Patch for stretched grids
    enum Axis {
      XAxis, YAxis, ZAxis
    };

  private:
    std::vector<LevelP> d_levels;
    
    Grid& operator=(const Grid&);

    // For automatic patch layout.  run_partition() will initialize the values of
    // af_, bf_, cf_, and nf_, then start the recursive call.  You should never
    // explicitly call partition(), only run_partition().
    IntVector run_partition3D(std::list<int> primes);
    void partition3D(std::list<int> primes, int a, int b, int c);
    
    IntVector run_partition2D(std::list<int> primes);
    void partition2D(std::list<int> primes, int a, int b);

    // The current (final) values of a,b,c, and norm for the partitian function.
    // Used to hold data between recursive calls.
    int af_;
    int bf_;
    int cf_;
    double nf_;

    // Temporary storage of the resolution values for use in the norm functions
    // In the 3D case, (a,b,c) corresponds to (x,y,z).  However in the 2D case
    // we do not know which dimensions (a,b) are, so the calling method will
    // place the results into the correct dimensions.
    int ares_;
    int bres_;
    int cres_;

    // Arbitrary desired maxium value for the norm.  If the norm of the best possible
    // patch layout exceeds this number, a warning message will be printed suggestion
    // the user run on a different number of processors.
    // static const double PATCH_TOLERANCE_ = 3;  

    // whether or not this is a 'lockstep' AMR grid.  This is stored here since the time refinement
    // ratio is to be defined by the level's refinement ratio, and the level can query this from the Grid.
    // Can potentially change if the Switcher changes from a lockstep component to a W-cycle component
    bool d_lockstepAMRGrid;
  };

} // End namespace Uintah

#endif
