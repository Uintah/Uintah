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
    Level* addLevel(const SCIRun::Point& anchor,
                    const SCIRun::Vector& dcell, int id=-1);
    
    void performConsistencyCheck() const;
    void printStatistics() const;

    //////////
    // Computes the physical boundaries for the grid
    void getSpatialRange(SCIRun::BBox& b) const;
    
    //////////
    // Computes the length of the grid
    void getLength(SCIRun::Vector& length,
                   const string flag = "plusExtraCells") const;
    
    //////////
    // Problem setup functions called from simulation controller
    void problemSetup(const ProblemSpecP& params, const ProcessorGroup *pg, bool do_amr); 

    // For comparing grids - level and patch structure must be equal
    bool operator==(const Grid& othergrid);

    friend std::ostream& operator<<(std::ostream& out, const Uintah::Grid& grid);

  private:
    std::vector<LevelP> d_levels;
    
    Grid(const Grid&);
    Grid& operator=(const Grid&);

    // For automatic patch layout.  run_partition() will initialize the values of
    // af_, bf_, cf_, and nf_, then start the recursive call.  You should never
    // explicitly call partition(), only run_partition().
    IntVector run_partition(std::list<int> primes);
    void partition(std::list<int> primes, int a, int b, int c);
    
    // The current (final) values of a,b,c, and norm for the partitian function.
    // Used to hold data between recursive calls.
    int af_;
    int bf_;
    int cf_;
    double nf_;

    // Arbitrary desired maxium value for the norm.  If the norm of the best possible
    // patch layout exceeds this number, a warning message will be printed suggestion
    // the user run on a different number of processors.
    // static const double PATCH_TOLERANCE_ = 3;  
  };

} // End namespace Uintah

#endif
